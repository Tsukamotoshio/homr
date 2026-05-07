import argparse
import glob
import hashlib
import os
import sys
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort

from homr import color_adjust, download_utils
from homr.autocrop import autocrop
from homr.bar_line_detection import (
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions, MultiStaff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.segmentation.config import segnet_path_onnx, segnet_path_onnx_fp16
from homr.segmentation.inference_segnet import extract
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.staff_position_save_load import load_staff_positions, save_staff_positions
from homr.title_detection import detect_title, download_ocr_weights
from homr.transformer.configs import Config, default_config
from homr.type_definitions import NDArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines


class InvalidProgramArgumentException(Exception):
    """Raise this exception for issues which the user can address."""


class GpuSupport(Enum):
    No = "no"
    AUTO = "auto"
    FORCE = "force"


def get_predictions(
    original: NDArray,
    preprocessed: NDArray,
    img_path: str,
    enable_cache: bool,
    use_gpu_inference: bool,
    batch_size: int = 8,
) -> InputPredictions:
    result = extract(
        preprocessed,
        img_path,
        step_size=320,
        use_cache=enable_cache,
        use_gpu_inference=use_gpu_inference,
        batch_size=batch_size,
    )
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool, use_gpu_inference: bool,
    segnet_batch_size: int = 8,
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidProgramArgumentException(
            "The file format is not supported, please provide a JPG or PNG image file:" + image_path
        )
    image = autocrop(image)
    image = resize_image(image)
    preprocessed = color_adjust.apply_clahe(image)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache, use_gpu_inference,
                                   batch_size=segnet_batch_size)
    debug = Debug(predictions.original, image_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines)


@dataclass
class ProcessingConfig:
    enable_debug: bool
    enable_cache: bool
    write_staff_positions: bool
    read_staff_positions: bool
    selected_staff: int
    use_gpu_inference: bool
    segnet_batch_size: int = 8  # SegNet 每批推理的 patch 数；弱机可降低以减少内存峰值


def process_image(
    image_path: str,
    config: ProcessingConfig,
    xml_generator_args: XmlGeneratorArguments,
) -> None:
    eprint("Processing " + image_path)
    xml_file = replace_extension(image_path, ".musicxml")
    debug_cleanup: Debug | None = None
    _xml_written = False  # track whether xml.write() has already succeeded
    try:
        if config.read_staff_positions:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read " + image_path)
            image = resize_image(image)
            debug = Debug(image, image_path, config.enable_debug)
            staff_position_files = replace_extension(image_path, ".txt")
            multi_staffs = load_staff_positions(
                debug, image, staff_position_files, config.selected_staff
            )
            title = ""
        else:
            multi_staffs, image, debug, title_future = detect_staffs_in_image(image_path, config)
        debug_cleanup = debug

        transformer_config = Config()
        transformer_config.use_gpu_inference = config.use_gpu_inference

        result_staffs = parse_staffs(
            debug,
            multi_staffs,
            image,
            selected_staff=config.selected_staff,
            config=transformer_config,
        )

        title = title_future.result(60)
        eprint("Found title:", title)

        eprint("Writing XML", result_staffs)
        xml = generate_xml(xml_generator_args, result_staffs, title)
        xml.write(xml_file)
        _xml_written = True  # file is now complete; keep it even if later cleanup throws

        eprint("Finished parsing " + str(len(result_staffs)) + " staves")
        teaser_file = replace_extension(image_path, "_teaser.png")
        if config.write_staff_positions:
            staff_position_files = replace_extension(image_path, ".txt")
            save_staff_positions(multi_staffs, image.shape, staff_position_files)
        debug.write_teaser(teaser_file, multi_staffs)
        debug.clean_debug_files_from_previous_runs()

        eprint("Result was written to", xml_file)
    except:
        # Only remove the output file if the XML write itself never completed.
        # If _xml_written is True the file is valid; the exception came from
        # post-write housekeeping (e.g. CUDA session destruction) and the
        # caller can still use the output.
        if not _xml_written and os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        if debug_cleanup is not None:
            debug_cleanup.clean_debug_files_from_previous_runs()


def detect_staffs_in_image(
    image_path: str, config: ProcessingConfig
) -> tuple[list[MultiStaff], NDArray, Debug, Future[str]]:
    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache, config.use_gpu_inference,
        segnet_batch_size=config.segnet_batch_size,
    )
    symbols = predict_symbols(debug, predictions)

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
    eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
    eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")
    if len(noteheads_with_stems) == 0:
        raise Exception("No noteheads found")

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    eprint("Average note head height: " + str(average_note_head_height))

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
    eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

    debug.write_bounding_boxes(
        "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
    )
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    if len(staffs) == 0:
        raise Exception("No staffs found")
    title_future = detect_title(debug, staffs[0])
    debug.write_bounding_boxes_alternating_colors("staffs", staffs)

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    debug.write_threshold_image("brace_dot", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    debug.write_all_bounding_boxes_alternating_colors("notes", multi_staffs, notes)

    return multi_staffs, predictions.preprocessed, debug, title_future


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


_WEIGHT_BASE_URLS = [
    # ModelScope 镜像（大陆访问优先）
    "https://modelscope.cn/models/Tsukamotoshio/homr/resolve/master/",
    # 官方 GitHub Releases（备用）
    "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/",
]

# Canonical filenames of all 6 HOMR ONNX weights (CPU + GPU/fp16 sets).
# Order matches download priority — segmentation first (smallest), then encoder,
# then decoder (largest). Sequential download lets users see fast initial progress.
_WEIGHT_FILES = [
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx",
    "encoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1.onnx",
    "encoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1_fp16.onnx",
    "decoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1.onnx",
    "decoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1_fp16.onnx",
]

# SHA256 hash of each weight file (lowercase hex, no prefix).
# Computed from the canonical files served by ModelScope / GitHub releases.
# Both mirrors must serve byte-identical files; if a future weight version is
# uploaded with different bytes, regenerate these hashes from one mirror and
# verify the other matches before shipping.
_WEIGHT_HASHES: dict[str, str] = {
    'segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx':
        '6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d',
    'segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx':
        '60f495496cb41473c0521d0811d8f44b9d5cff892d287974a8aebb3eaee2fa83',
    'encoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1.onnx':
        'de27b33554d89cc9aed2128188fc24c9ba69c1209cea7686cb9c344a72076c37',
    'encoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1_fp16.onnx':
        'a11c8b80485e0c57c5967c082108b9103ed7d52f7f9d31304484ee95e6b96745',
    'decoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1.onnx':
        'fb678135e00c5777071f04906e178019a80fb42bb3d4210b859604e3368f1739',
    'decoder_pytorch_model_331-e10346542968cc71fbcce0c0696f3ac963f11ae1_fp16.onnx':
        '0c5549700a06733ec60e1bf0f0852f29300495c8fdeaee657341d2042ad5935e',
}
# Sanity check: every file in _WEIGHT_FILES must have a hash entry.
assert set(_WEIGHT_HASHES.keys()) == set(_WEIGHT_FILES), (
    'WEIGHT_HASHES out of sync with _WEIGHT_FILES — one was added without the other'
)


def verify_sha256(path: str, expected: str) -> bool:
    """Return True if the file at `path` has SHA256 == `expected`.

    Empty `expected` short-circuits to True (used during early dev before hashes
    are filled in). Raises FileNotFoundError if the file is missing.
    """
    if not expected:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected.lower()


def _download_from_any_source(onnx_path: str, dest_model_path: str) -> None:
    """依次尝试多个下载源下载单个模型文件，成功即返回，全部失败则抛出最后一个异常。

    ModelScope 源直接提供 .onnx 文件；GitHub 源提供 .zip 需解压。
    """
    base_name = os.path.basename(onnx_path).split(".")[0]
    dest_dir = os.path.dirname(dest_model_path)
    last_err: Exception = RuntimeError("No sources configured")

    for base_url in _WEIGHT_BASE_URLS:
        is_modelscope = "modelscope.cn" in base_url
        try:
            if is_modelscope:
                # ModelScope：直接下载 .onnx（按目录结构拼路径）
                rel = os.path.relpath(dest_model_path,
                                      os.path.dirname(os.path.dirname(dest_model_path)))
                url = base_url + rel.replace("\\", "/")
                download_utils.download_file(url, dest_model_path)
            else:
                # GitHub：下载 .zip 再解压
                zip_name = base_name + ".zip"
                downloaded_zip = os.path.join(dest_dir, zip_name)
                try:
                    download_utils.download_file(base_url + zip_name, downloaded_zip)
                    download_utils.unzip_file(downloaded_zip, dest_dir)
                finally:
                    if os.path.exists(downloaded_zip):
                        os.remove(downloaded_zip)
            return
        except Exception as e:
            eprint(f"\nSource {base_url} failed: {e}, trying next...")
            last_err = e
    raise last_err


def download_weights(use_gpu_inference: bool) -> None:
    if use_gpu_inference:
        models = [
            segnet_path_onnx_fp16,
            default_config.filepaths.encoder_path_fp16,
            default_config.filepaths.decoder_path_fp16,
        ]
        missing_models = [model for model in models if not os.path.exists(model)]
    else:
        models = [
            segnet_path_onnx,
            default_config.filepaths.encoder_path,
            default_config.filepaths.decoder_path,
        ]
        missing_models = [model for model in models if not os.path.exists(model)]

    if len(missing_models) == 0:
        return

    eprint("Downloading", len(missing_models), "models - this is only required once")
    for model in missing_models:
        if not os.path.exists(model):
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            _download_from_any_source(model, model)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="homer", description="An optical music recognition (OMR) system"
    )
    parser.add_argument("image", type=str, nargs="?", help="Path to the image to process")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Downloads the models if they are missing and then exits. "
        + "You don't have to call init before processing images, "
        + "it's only useful if you want to prepare for example a Docker image.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--cache", action="store_true", help="Read an existing cache file or create a new one"
    )
    parser.add_argument(
        "--output-large-page",
        action="store_true",
        help="Adds instructions to the musicxml so that it gets rendered on larger pages",
    )
    parser.add_argument(
        "--output-metronome", type=int, help="Adds a metronome to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--output-tempo", type=int, help="Adds a tempo to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--write-staff-positions",
        action="store_true",
        help="Writes the position of all detected staffs to a txt file.",
    )
    parser.add_argument(
        "--read-staff-positions",
        action="store_true",
        help="Reads the position of all staffs from a txt file instead"
        + " of running the built-in staff detection.",
    )
    parser.add_argument(
        "--gpu",
        type=GpuSupport,
        choices=list(GpuSupport),
        default=GpuSupport.AUTO,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    has_gpu_support = "CUDAExecutionProvider" in ort.get_available_providers()

    use_gpu_inference = (
        args.gpu == GpuSupport.AUTO and has_gpu_support
    ) or args.gpu == GpuSupport.FORCE

    download_weights(use_gpu_inference)
    if args.init:
        download_ocr_weights()
        eprint("Init finished")
        return

    config = ProcessingConfig(
        args.debug,
        args.cache,
        args.write_staff_positions,
        args.read_staff_positions,
        -1,
        use_gpu_inference,
    )

    xml_generator_args = XmlGeneratorArguments(
        args.output_large_page, args.output_metronome, args.output_tempo
    )
    if args.debug:
        eprint(f"Using Log Level {2} for OnnxRuntime")
        ort.set_default_logger_severity(2)
    else:
        ort.set_default_logger_severity(3)

    if not args.image:
        eprint("No image provided")
        parser.print_help()
        sys.exit(1)
    elif os.path.isfile(args.image):
        try:
            process_image(args.image, config, xml_generator_args)
        except InvalidProgramArgumentException as e:
            eprint(str(e))
            sys.exit(2)
    elif os.path.isdir(args.image):
        image_files = get_all_image_files_in_folder(args.image)
        eprint("Processing", len(image_files), "files:", image_files)
        error_files = []
        for image_file in image_files:
            eprint("=========================================")
            try:
                process_image(image_file, config, xml_generator_args)
                eprint("Finished", image_file)
            except Exception as e:
                eprint(f"An error occurred while processing {image_file}: {e}")
                error_files.append(image_file)
        if len(error_files) > 0:
            eprint("Errors occurred while processing the following files:", error_files)
    else:
        eprint(f"{args.image} is not a valid file or directory")
        sys.exit(2)


if __name__ == "__main__":
    main()

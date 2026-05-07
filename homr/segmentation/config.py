import os

script_location = os.path.dirname(os.path.realpath(__file__))

# When HOMR_MODELS_DIR is set (by the parent app), weights live there.
# Otherwise fall back to the module-relative location (legacy / dev / tests).
# `or script_location` (not get(..., default)) ensures empty-string env values
# also fall through, instead of silently joining against an empty path.
_models_dir = os.environ.get("HOMR_MODELS_DIR") or script_location

model_name = "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f"

segnet_path_onnx = os.path.join(_models_dir, f"{model_name}.onnx")

segnet_path_onnx_fp16 = os.path.join(_models_dir, f"{model_name}_fp16.onnx")

# Torch path is dev-only; left at original location.
segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    f"{model_name}.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version

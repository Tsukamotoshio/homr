import os

import numpy as np
import onnxruntime as ort

from homr.onnx_providers import (
    coreml_available,
    coreml_mlprogram_providers,
    gpu_providers,
)
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.type_definitions import NDArray

# 限制 onnxruntime CPU 推理线程数（使用不超过一半的逻辑核心），
# 避免推理阶段占满所有 CPU 核心导致系统响应迟缓。
_ORT_INTRA_THREADS = max(1, (os.cpu_count() or 4) - 2)


class Encoder:
    def __init__(self, config: Config) -> None:
        self.use_gpu = False
        _sess_opts = ort.SessionOptions()
        _sess_opts.log_severity_level = 3  # 仅显示 ERROR，抑制 WARNING/INFO（含 Conv Fallback 等）
        _sess_opts.intra_op_num_threads = _ORT_INTRA_THREADS  # 限制单算子并行线程
        _sess_opts.inter_op_num_threads = 1                   # 算子间串行执行
        if config.use_gpu_inference:
            try:
                providers, device = gpu_providers({"cudnn_conv_algo_search": "DEFAULT"})
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    sess_options=_sess_opts,
                    providers=providers,
                )
                self.fp16 = True
                # CoreML/DML bind IO on the CPU even though compute runs on the GPU/ANE.
                self.use_gpu = device == "cuda"

            except Exception as ex:
                eprint(ex)
                eprint("Going on without GPU support")
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    sess_options=_sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self.fp16 = True

        elif config.use_coreml_encoder and coreml_available():
            try:
                # CPUAndGPU skips the (slow) ANE specialization: it halves the
                # session creation time vs "ALL" and inference is even slightly
                # faster (measured on an M1).
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    providers=coreml_mlprogram_providers(
                        config.filepaths.encoder_path_fp16, compute_units="CPUAndGPU"
                    ),
                )
                self.fp16 = True
                # use_gpu stays False: CoreML binds IO on the CPU even though
                # the compute runs on the GPU/ANE.
            except Exception as ex:
                eprint(ex)
                eprint("Could not create the CoreML encoder session, using the CPU instead")
                self.encoder = ort.InferenceSession(config.filepaths.encoder_path)
                self.fp16 = False

        else:
            self.encoder = ort.InferenceSession(
                config.filepaths.encoder_path,
                sess_options=_sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self.fp16 = False

        self.io_binding = self.encoder.io_binding()
        self.device_id = 0

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def __del__(self) -> None:
        # 确保 io_binding 在 encoder session 之前释放，防止 use-after-free 崩溃
        try:
            if hasattr(self, 'io_binding'):
                del self.io_binding
        except Exception:
            pass
        try:
            if hasattr(self, 'encoder'):
                del self.encoder
        except Exception:
            pass

    def generate(self, x: NDArray) -> NDArray:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", x.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", x.astype(np.float32))

        self.io_binding.bind_output("output", "cuda" if self.use_gpu else "cpu", self.device_id)
        self.encoder.run_with_iobinding(self.io_binding)
        return self.io_binding.get_outputs()[0].numpy()

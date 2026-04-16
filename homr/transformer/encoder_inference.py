import os

import numpy as np
import onnxruntime as ort
from onnxruntime import OrtValue

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.type_definitions import NDArray

# 限制 onnxruntime CPU 推理线程数（使用不超过一半的逻辑核心），
# 避免推理阶段占满所有 CPU 核心导致系统响应迟缓。
_ORT_INTRA_THREADS = max(1, (os.cpu_count() or 4) // 2)


class Encoder:
    def __init__(self, config: Config) -> None:
        self.use_gpu = False
        _sess_opts = ort.SessionOptions()
        _sess_opts.log_severity_level = 3  # 仅显示 ERROR，抑制 WARNING/INFO（含 Conv Fallback 等）
        _sess_opts.intra_op_num_threads = _ORT_INTRA_THREADS  # 限制单算子并行线程
        _sess_opts.inter_op_num_threads = 1                   # 算子间串行执行
        if config.use_gpu_inference:
            try:
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    sess_options=_sess_opts,
                    providers=["DmlExecutionProvider"],
                )
                self.fp16 = True
                # DML: use_gpu stays False — io_binding outputs remain on CPU for compatibility
            except Exception:
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    sess_options=_sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self.fp16 = True

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

    def generate(self, x: NDArray) -> list[OrtValue]:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", x.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", x.astype(np.float32))

        self.io_binding.bind_output("output", "cuda" if self.use_gpu else "cpu", self.device_id)
        self.encoder.run_with_iobinding(self.io_binding)
        return self.io_binding.get_outputs()[0].numpy()

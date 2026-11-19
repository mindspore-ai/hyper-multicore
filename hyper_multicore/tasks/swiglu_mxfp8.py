# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Fused SwiGLU / gradient + dual-axis MXFP8 quantization task binding."""
from dataclasses import dataclass
from .quantization_mxfp8 import Mxfp8QuantFillConfig, MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK

@dataclass
class Mxfp8SwiGLUFillConfig(Mxfp8QuantFillConfig):
    def __post_init__(self):
        if self.kind not in (MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK):
            raise ValueError("Fused SwiGLU quantization requires a forward or backward task kind")

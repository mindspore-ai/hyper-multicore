# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Compose the pinned MX quantization kernels without modifying cached Git sources."""

from pathlib import Path
import shutil

QUANT_KERNELS = {
    'dynamic': 'dynamic_mx_quant_with_dual_axis',
    'swiglu': 'swiglu_mx_quant_with_dual_axis',
    'swiglu_row': 'swiglu_mx_quant',
    'swiglu_grad': 'swiglu_backward_mx_quant_with_dual_axis',
}
QUANT_SOURCE_PATHS = tuple(f'quant/{name}/op_kernel/arch35' for name in QUANT_KERNELS.values()) + (
    'common/inc/op_kernel/platform.h', 'common/inc/op_kernel/kernel_utils.h',
)


def compose_quant_sources(ops_nn_source: Path, destination: Path) -> None:
    """Place each kernel and its relative common-header includes under one closure."""
    for name, upstream in QUANT_KERNELS.items():
        target = destination / name
        shutil.copytree(ops_nn_source / 'quant' / upstream / 'op_kernel/arch35', target / 'arch35')
        (target / 'inc').mkdir()
        for header in ('platform.h', 'kernel_utils.h'):
            shutil.copy2(ops_nn_source / 'common/inc/op_kernel' / header, target / 'inc' / header)

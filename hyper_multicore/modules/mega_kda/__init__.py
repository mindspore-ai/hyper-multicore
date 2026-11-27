# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Whole-layer MegaKda persistent megakernel module (CP=1, Ulysses, AllGather, Prefix-Sum)."""

from hyper_multicore.modules.mega_kda.graph import (
    CP_MODE_ALLGATHER,
    CP_MODE_PREFIX_SUM,
    CP_MODE_ULYSSES,
    MegaKdaScheduleSpec,
    build_mega_kda_runtime_config,
    resolve_cp_mode,
    resolve_schedule_spec,
)
from hyper_multicore.modules.mega_kda.module import MegaKda

__all__ = [
    "CP_MODE_ULYSSES",
    "CP_MODE_ALLGATHER",
    "CP_MODE_PREFIX_SUM",
    "MegaKda",
    "MegaKdaScheduleSpec",
    "resolve_cp_mode",
    "resolve_schedule_spec",
    "build_mega_kda_runtime_config",
]

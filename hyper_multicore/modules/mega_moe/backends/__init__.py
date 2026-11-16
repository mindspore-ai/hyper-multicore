# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Two implemented precision backends behind the shared public expert module."""
from functools import lru_cache
from importlib import import_module

@lru_cache(maxsize=2)
def get_backend(precision):
    if precision not in ('bf16', 'mxfp8'):
        raise ValueError("MegaMoE precision must be bf16 or mxfp8")
    return import_module(f'{__name__}.{precision}.backend')

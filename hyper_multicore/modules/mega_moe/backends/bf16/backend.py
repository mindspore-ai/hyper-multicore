# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""BF16 backend: use standard tensor weights and the BF16 plan/ABI."""
from .resources import BF16Resources as Resources
from .execution import forward

__all__ = ["Resources", "forward"]

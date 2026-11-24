# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Stable model numerical options owned by the MegaMoE component."""

import math
import struct
from dataclasses import dataclass
from typing import Literal


def _is_finite_number(value: object) -> bool:
    """Accept integers and finite floats, excluding booleans."""
    return ((isinstance(value, int) and not isinstance(value, bool))
            or (isinstance(value, float) and math.isfinite(value)))


def _check_positive_float32(name: str, value: float) -> None:
    """Check that a positive parameter stays finite and positive in float32."""
    if _is_finite_number(value) and value > 0:
        try:
            encoded = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error):
            pass
        else:
            if math.isfinite(encoded) and encoded > 0:
                return
    raise ValueError(f"{name} must be a finite positive float32-representable number, got {value!r}")


@dataclass(frozen=True, kw_only=True)  # pylint: disable=unexpected-keyword-arg
class SwiGLUParameters:
    """SwiGLU clipping; None disables clipping on both gate and up branches.

    The gate is clipped above clamp_limit, before or after SiLU. The up
    branch is clipped to [-clamp_limit, clamp_limit].
    """

    clamp_limit: float | None = None
    clamp_after_silu: bool = False

    def __post_init__(self) -> None:
        if self.clamp_limit is not None:
            _check_positive_float32("clamp_limit", self.clamp_limit)
        if not isinstance(self.clamp_after_silu, bool):
            raise TypeError("clamp_after_silu must be a boolean")


@dataclass(frozen=True, kw_only=True)  # pylint: disable=unexpected-keyword-arg
class SiTUGLUParameters:
    """SiTUGLU parameters; nonpositive linear_beta disables up-branch tanh.

    Describes the algorithm only. Execution requires an available backend.
    """

    beta: float
    linear_beta: float = 0.0

    def __post_init__(self) -> None:
        _check_positive_float32("beta", self.beta)
        if not _is_finite_number(self.linear_beta):
            raise ValueError("linear_beta must be a finite number")
        if self.linear_beta > 0:
            _check_positive_float32("linear_beta", self.linear_beta)

    def validate_bf16(self) -> None:
        """Check the FP32 reciprocal used by the BF16 device implementation."""
        min_normal = 2.0 ** -126
        for name, bound in (("beta", self.beta), ("linear_beta", self.linear_beta)):
            if bound > 0 and not min_normal <= bound <= 1.0 / min_normal:
                raise ValueError(f"SiTUGLU {name} and its reciprocal must be normal FP32 values")


@dataclass(frozen=True)
class MegaMoeNumericsOptions:
    """Probability placement and partial weight-gradient precision.

    Numerical choices are fixed for the executor. Before-FC2 weighting rounds
    activations to BF16, multiplies FP32 probabilities, then rounds to BF16
    again. Activation-specific parameters belong to the activation object.
    """

    partial_weight_grad_dtype: Literal["native", "float32"] | None = None
    apply_probs_before_fc2: bool = False

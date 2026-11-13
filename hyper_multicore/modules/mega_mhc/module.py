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
"""Model-facing MegaMhc module for shifted BF16 mHC."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._precision import preserve_fp32_parameters
from ..mega_mhc_common import (
    MHC_MAPPINGS,
    MHC_SINKHORN_ITERS,
    MHC_STREAMS,
    MegaMhcOutputs,
    validate_mega_mhc_hidden_size,
)
from .function import hyper_mega_mhc
from .graph import DEFAULT_TOKEN_TILE


class MegaMhc(torch.nn.Module):
    """Own the current layer's mapping predictor and input RMSNorm weight.

    Note:
        This trainable module requires the native backward hidden-size contract.
        Forward-only callers can use ``hyper_mega_mhc(..., need_backward=False)``.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        num_iters: int = MHC_SINKHORN_ITERS,
        token_tile: int = DEFAULT_TOKEN_TILE,
        device: Any | None = None,
    ) -> None:
        """Initialize the fixed-four-stream shifted mHC parameters.

        Args:
            hidden_size: Stream width H, a positive multiple of 128, at most 5760.
                The module enforces this training limit even in eval/no-grad mode.
            hc_eps: Epsilon used by pre-mix and Sinkhorn mapping prediction.
            norm_eps: Epsilon used by RMS normalization.
            num_iters: Sinkhorn iteration count; must be 20.
            token_tile: Positive integer tokens per scheduler task; defaults to 32.
            device: Parameter device. None uses Torch's default device; execution
                requires moving all parameters and inputs to the same NPU.

        Note:
            phi [24, 4*H], alpha [3] and bias [24] use FP32; norm_weight [H]
            uses BF16. Module dtype conversions preserve the FP32 predictor
            parameters and their existing gradients. Runtime tensors require NPU
            and the native payload.

        Raises:
            ValueError: If hidden_size, num_iters or token_tile is unsupported.
        """
        super().__init__()
        validate_mega_mhc_hidden_size(hidden_size, need_backward=True)
        if num_iters != MHC_SINKHORN_ITERS:
            raise ValueError(f"MegaMhc requires num_iters={MHC_SINKHORN_ITERS}, got {num_iters}.")
        if not isinstance(token_tile, int) or isinstance(token_tile, bool) or token_tile <= 0:
            raise ValueError(f"token_tile must be a positive integer, got {token_tile!r}.")
        self.hidden_size = hidden_size
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.num_iters = num_iters
        self.token_tile = token_tile
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.phi = torch.nn.Parameter(
            torch.empty((MHC_MAPPINGS, MHC_STREAMS * hidden_size), **factory_kwargs)
        )
        self.alpha = torch.nn.Parameter(torch.ones((3,), **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.zeros((MHC_MAPPINGS,), **factory_kwargs))
        self.norm_weight = torch.nn.Parameter(
            torch.ones((hidden_size,), device=device, dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(
            self.phi,
            mean=0.0,
            std=(MHC_STREAMS * hidden_size) ** -0.5,
        )

    def _apply(self, fn: Callable, recurse: bool = True) -> MegaMhc:
        """Keep mapping predictors FP32 when the enclosing model changes dtype or device."""
        return super()._apply(preserve_fp32_parameters(self, fn, ("phi", "alpha", "bias")), recurse=recurse)

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> MegaMhc:
        """Bind an existing four-stream, shifted mHC module without replacing its Parameters.

        Args:
            module: Source with hidden_size, hc_eps, norm_eps, num_iters and Parameters
                phi [24, 4H], alpha [3], bias [24], norm_weight [H]. Predictor Parameters must be FP32.

        Returns:
            Boundary sharing all original Parameters and mathematical configuration.
        """
        configuration = {}
        for name in ("hidden_size", "hc_eps", "norm_eps", "num_iters"):
            if not hasattr(module, name):
                raise ValueError(f"source mHC module must expose {name}")
            configuration[name] = getattr(module, name)
        phi = getattr(module, "phi", None)
        if not isinstance(phi, torch.nn.Parameter):
            raise TypeError("source phi must be an FP32 Parameter [24, 4H]")
        result = cls(**configuration, device=phi.device)
        for name in ("phi", "alpha", "bias", "norm_weight"):
            parameter = getattr(module, name, None)
            expected = getattr(result, name)
            if (not isinstance(parameter, torch.nn.Parameter) or parameter.shape != expected.shape
                    or parameter.dtype != expected.dtype or parameter.device != expected.device):
                raise ValueError(f"source {name} must match shape={tuple(expected.shape)}, dtype={expected.dtype}")
            setattr(result, name, parameter)
        return result

    def forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MegaMhcOutputs:
        """Advance one shifted mHC block boundary and predict the next mappings.

        Args:
            previous_output: BF16 previous block output with shape [*S, H].
            residual: BF16 residual streams with shape [*S, 4, H]. S contains
                one or more token dimensions, all inputs share the same NPU.
            previous_pre_mix: FP32 coefficients [*S, 4] used to form block_input.
            previous_post_mix: FP32 coefficients [*S, 4] that distribute
                previous_output into the residual streams.
            previous_residual_mix: FP32 residual mixing matrices [*S, 4, 4].

        Returns:
            A five-tensor tuple, in order: new_residual (BF16, [*S, 4, H]),
            next_pre_mix (FP32, [*S, 4]), next_post_mix (FP32, [*S, 4]),
            next_residual_mix (FP32, [*S, 4, 4]), block_input (BF16, [*S, H]).
            The new mappings are for the next boundary; block_input uses
            previous_pre_mix to combine new_residual, then applies RMSNorm.

        Note:
            Training uses native autograd. The flattened token count must be
            at least max(40, configured AIV core count) when backward is needed;
            forward-only execution requires at least one token. Cross-stream
            producers must establish dependencies before this call.

        Raises:
            TypeError: If tensors, dtypes or the NPU device type are invalid.
            ValueError: If shapes, devices, hidden size or training token count
                violate the native contract.
        """
        return hyper_mega_mhc(
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            self.phi,
            self.alpha,
            self.bias,
            self.norm_weight,
            hc_eps=self.hc_eps,
            norm_eps=self.norm_eps,
            num_iters=self.num_iters,
            token_tile=self.token_tile,
        )

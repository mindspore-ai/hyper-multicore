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
"""Functional entry point for the BF16 HyperMegaMhc scheduler kernel."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from hyper_multicore._core_spec import get_device_core_spec
from hyper_multicore.modules.mega_mhc_grad.common import MegaMhcGradCache
from hyper_multicore.modules.mega_mhc_grad.function import hyper_mega_mhc_grad
from hyper_multicore.torch import ops as multicore_ops

from ..mega_mhc_common import (
    MHC_SINKHORN_ITERS,
    MegaMhcOutputs,
    validate_mega_mhc_core_spec,
    validate_mega_mhc_native_inputs,
    validate_mega_mhc_token_count,
)
from .graph import DEFAULT_TOKEN_TILE
from .plan import MegaMhcPlan, build_mega_mhc_plan


@lru_cache(maxsize=16)
def _get_cached_plan(
    token_count: int,
    hidden_size: int,
    device_index: int,
    num_cube_cores: int,
    token_tile: int,
) -> MegaMhcPlan:
    """Reuse immutable RuntimeConfig tensors for recurring static shapes."""
    return build_mega_mhc_plan(
        token_count,
        hidden_size,
        torch.device("npu", device_index),
        num_cube_cores,
        token_tile,
    )


def _restore_forward_outputs(
    outputs: tuple[torch.Tensor, ...],
    leading_shape: tuple[int, ...],
    hidden_size: int,
) -> tuple[MegaMhcOutputs, MegaMhcGradCache]:
    """Restore public dimensions and retain the native-layout backward cache."""
    (
        new_residual,
        next_pre,
        next_post,
        next_res,
        block_input,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        mixed_input,
        rms_rstd,
    ) = outputs
    public_outputs = (
        new_residual.reshape(*leading_shape, 4, hidden_size),
        next_pre.reshape(*leading_shape, 4),
        next_post.reshape(*leading_shape, 4),
        next_res.reshape(*leading_shape, 4, 4),
        block_input.reshape(*leading_shape, hidden_size),
    )
    cache = MegaMhcGradCache(
        new_residual=new_residual,
        hc_before_norm=hc_before_norm,
        inv_rms=inv_rms,
        sum_out=sum_out,
        norm_out=norm_out,
        mixed_input=mixed_input,
        rms_rstd=rms_rstd,
    )
    return public_outputs, cache


def _hyper_mega_mhc_forward(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = MHC_SINKHORN_ITERS,
    need_backward: bool = True,
    token_tile: int = DEFAULT_TOKEN_TILE,
) -> tuple[MegaMhcOutputs, MegaMhcGradCache]:
    """Run one shifted boundary and retain the native backward cache.

    Args:
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
        need_backward: Whether to retain native backward caches.
        token_tile: Number of tokens assigned to each logical task.
    """
    validate_mega_mhc_native_inputs(
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
        num_iters=num_iters,
        need_backward=need_backward,
        token_tile=token_tile,
    )
    leading_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    token_count = residual.numel() // (4 * hidden_size)
    device_index = residual.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    core_spec = get_device_core_spec(device_index)
    validate_mega_mhc_core_spec(core_spec)
    num_cube_cores, num_vector_cores = core_spec.num_cube_cores, core_spec.num_vector_cores
    if need_backward:
        validate_mega_mhc_token_count(token_count, num_vector_cores)
    plan = _get_cached_plan(
        token_count,
        hidden_size,
        device_index,
        num_cube_cores,
        token_tile,
    )
    events = torch.zeros(plan.event_capacity * 4, dtype=torch.uint8, device=residual.device)
    options = {}
    if plan.runtime.is_active:
        options["diagnostics"] = plan.runtime
    outputs = multicore_ops.hyper_mega_mhc(
        previous_output.reshape(1, token_count, hidden_size).contiguous(),
        residual.reshape(1, token_count, 4, hidden_size).contiguous(),
        previous_pre_mix.reshape(1, token_count, 4).contiguous(),
        previous_post_mix.reshape(1, token_count, 4).contiguous(),
        previous_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
        phi.contiguous(),
        alpha.contiguous(),
        bias.contiguous(),
        norm_weight.contiguous(),
        plan.runtime.runtime_config,
        events,
        events,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        num_iters=num_iters,
        need_backward=need_backward,
        **options,
    )
    return _restore_forward_outputs(outputs, leading_shape, hidden_size)


class _HyperMegaMhcFunction(torch.autograd.Function):
    """Connect the fused forward and backward NPU operators to autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_eps: float,
        norm_eps: float,
        num_iters: int,
        token_tile: int,
    ) -> MegaMhcOutputs:
        """Launch HyperMegaMhc and save its explicit native cache.

        Args:
            ctx: Autograd context retaining only the state required for backward.
            previous_output: Previous block output tensor.
            residual: Residual streams supplied to the current block.
            previous_pre_mix: Previous block's input mixing coefficients.
            previous_post_mix: Previous block's output mixing coefficients.
            previous_residual_mix: Previous block's residual mixing matrix.
            phi: Projection weight used to generate mixing coefficients.
            alpha: Scaling parameters for the projected mixing coefficients.
            bias: Bias parameters for the projected mixing coefficients.
            norm_weight: RMS normalization weight.
            hc_eps: Numerical epsilon used by mixing coefficient normalization.
            norm_eps: Numerical epsilon used by RMS normalization.
            num_iters: Number of Sinkhorn normalization iterations.
            token_tile: Number of token rows assigned to a task tile.
        """
        outputs, cache = _hyper_mega_mhc_forward(
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            phi,
            alpha,
            bias,
            norm_weight,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            num_iters=num_iters,
            need_backward=True,
            token_tile=token_tile,
        )
        ctx.save_for_backward(
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            phi,
            alpha,
            bias,
            norm_weight,
            cache.new_residual,
            cache.hc_before_norm,
            cache.inv_rms,
            cache.sum_out,
            cache.norm_out,
            cache.mixed_input,
            cache.rms_rstd,
        )
        ctx.hc_eps = hc_eps
        ctx.token_tile = token_tile
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Differentiate the boundary with HyperMegaMhcGrad.

        Args:
            ctx: Autograd context retaining only the state required for backward.
            grad_outputs: Upstream gradients for the differentiable forward results.
        """
        (
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            phi,
            alpha,
            bias,
            norm_weight,
            new_residual,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
            mixed_input,
            rms_rstd,
        ) = ctx.saved_tensors
        cache = MegaMhcGradCache(
            new_residual=new_residual,
            hc_before_norm=hc_before_norm,
            inv_rms=inv_rms,
            sum_out=sum_out,
            norm_out=norm_out,
            mixed_input=mixed_input,
            rms_rstd=rms_rstd,
        )
        gradients = hyper_mega_mhc_grad(
            *grad_outputs,
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            phi,
            alpha,
            bias,
            norm_weight,
            cache,
            hc_eps=ctx.hc_eps,
            token_tile=ctx.token_tile,
        )
        return (*gradients, None, None, None, None)


def hyper_mega_mhc(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = MHC_SINKHORN_ITERS,
    need_backward: bool = True,
    token_tile: int = DEFAULT_TOKEN_TILE,
) -> MegaMhcOutputs:
    """Run the fused shifted boundary with its matching autograd backward."""
    differentiable_inputs = (
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
    )
    if need_backward and torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in differentiable_inputs
    ):
        return _HyperMegaMhcFunction.apply(
            *differentiable_inputs,
            hc_eps,
            norm_eps,
            num_iters,
            token_tile,
        )
    outputs, _ = _hyper_mega_mhc_forward(
        *differentiable_inputs,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        num_iters=num_iters,
        need_backward=False,
        token_tile=token_tile,
    )
    return outputs

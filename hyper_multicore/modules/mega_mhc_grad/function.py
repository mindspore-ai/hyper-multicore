# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Functional entry point for the BF16 HyperMegaMhcGrad mega kernel."""

from __future__ import annotations

from functools import lru_cache

import torch

from hyper_multicore._core_spec import get_device_core_spec
from hyper_multicore.torch import ops as multicore_ops

from ..mega_mhc_common import (
    validate_mega_mhc_core_spec, validate_mega_mhc_native_inputs, validate_mega_mhc_token_count,
)
from .common import MegaMhcGradCache, MegaMhcGradOutputs, validate_mega_mhc_grad_tensors
from .graph import DEFAULT_GRAD_TOKEN_TILE
from .plan import MegaMhcGradPlan, build_mega_mhc_grad_plan


@lru_cache(maxsize=16)
def _get_cached_plan(
    token_count: int,
    hidden_size: int,
    device_index: int,
    num_cube_cores: int,
    num_vector_cores: int,
    token_tile: int,
) -> MegaMhcGradPlan:
    """Reuse immutable RuntimeConfig tensors for recurring static shapes."""
    return build_mega_mhc_grad_plan(
        token_count,
        hidden_size,
        torch.device("npu", device_index),
        num_cube_cores,
        num_vector_cores,
        token_tile,
    )


def _get_plan_for_shape(
    token_count: int,
    hidden_size: int,
    device: torch.device,
    token_tile: int,
) -> MegaMhcGradPlan:
    """Read current Device core limits and retrieve the immutable plan for this shape."""
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    core_spec = get_device_core_spec(device_index)
    validate_mega_mhc_core_spec(core_spec)
    num_cube_cores, num_vector_cores = core_spec.num_cube_cores, core_spec.num_vector_cores
    validate_mega_mhc_token_count(token_count, num_vector_cores)
    return _get_cached_plan(
        token_count,
        hidden_size,
        device_index,
        num_cube_cores,
        num_vector_cores,
        token_tile,
    )


def _restore_output_shapes(
    outputs: tuple[torch.Tensor, ...],
    leading_shape: tuple[int, ...],
    hidden_size: int,
) -> MegaMhcGradOutputs:
    """Restore public leading dimensions and reorder native outputs.

    Args:
        outputs: Native operator outputs in ACLNN order.
        leading_shape: Original activation leading dimensions.
        hidden_size: Hidden dimension of every token.

    Returns:
        Public gradient tuple in forward-input order.
    """
    (
        grad_residual,
        grad_phi,
        grad_alpha,
        grad_bias,
        grad_previous_output,
        grad_previous_pre,
        grad_previous_post,
        grad_previous_residual,
        grad_norm_weight,
    ) = outputs
    return (
        grad_previous_output.reshape(*leading_shape, hidden_size),
        grad_residual.reshape(*leading_shape, 4, hidden_size),
        grad_previous_pre.reshape(*leading_shape, 4),
        grad_previous_post.reshape(*leading_shape, 4),
        grad_previous_residual.reshape(*leading_shape, 4, 4),
        grad_phi,
        grad_alpha,
        grad_bias,
        grad_norm_weight,
    )


def hyper_mega_mhc_grad(
    grad_new_residual: torch.Tensor,
    grad_next_pre_mix: torch.Tensor,
    grad_next_post_mix: torch.Tensor,
    grad_next_residual_mix: torch.Tensor,
    grad_block_input: torch.Tensor,
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    cache: MegaMhcGradCache,
    *,
    hc_eps: float = 1e-6,
    token_tile: int = DEFAULT_GRAD_TOKEN_TILE,
) -> MegaMhcGradOutputs:
    """Execute the fused Prepare/PhiRms/PrevXPost backward pipeline.

    Args:
        grad_new_residual: Gradient of the updated residual branches.
        grad_next_pre_mix: Gradient of the next pre-mix coefficients.
        grad_next_post_mix: Gradient of the next post-mix coefficients.
        grad_next_residual_mix: Gradient of the next residual-mix coefficients.
        grad_block_input: Gradient of the normalized block input.
        previous_output: Previous block output activation.
        residual: Current residual branches.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weight.
        alpha: Projection scale.
        bias: Projection bias.
        norm_weight: RMSNorm weight.
        cache: Forward intermediates required by the backward kernel.
        hc_eps: RMSNorm epsilon.
        token_tile: Number of tokens assigned to each AIV task.

    Returns:
        Gradients for the five block inputs and four trainable tensors.
    """
    validate_mega_mhc_native_inputs(
        previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix,
        phi, alpha, bias, norm_weight, need_backward=True, token_tile=token_tile,
    )
    validate_mega_mhc_grad_tensors(
        (grad_new_residual, grad_next_pre_mix, grad_next_post_mix, grad_next_residual_mix, grad_block_input),
        residual,
        cache,
    )
    leading_shape, hidden_size = residual.shape[:-2], residual.shape[-1]
    token_count = residual.numel() // (4 * hidden_size)
    plan = _get_plan_for_shape(token_count, hidden_size, residual.device, token_tile)
    events = torch.zeros(plan.event_capacity * 4, dtype=torch.uint8, device=residual.device)
    kernel_inputs = (
        grad_block_input.reshape(1, token_count, hidden_size).contiguous(),
        grad_next_post_mix.reshape(1, token_count, 4).contiguous(),
        grad_next_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
        cache.new_residual.reshape(1, token_count, 4, hidden_size).contiguous(),
        phi.contiguous(),
        alpha.contiguous(),
        bias.contiguous(),
        previous_pre_mix.reshape(1, token_count, 4).contiguous(),
        cache.hc_before_norm.contiguous(),
        cache.inv_rms.contiguous(),
        cache.sum_out.contiguous(),
        cache.norm_out.contiguous(),
        grad_next_pre_mix.reshape(1, token_count, 4).contiguous(),
        cache.mixed_input.reshape(1, token_count, hidden_size).contiguous(),
        cache.rms_rstd.contiguous(),
        norm_weight.contiguous(),
        grad_new_residual.reshape(1, token_count, 4, hidden_size).contiguous(),
        residual.reshape(1, token_count, 4, hidden_size).contiguous(),
        previous_output.reshape(1, token_count, hidden_size).contiguous(),
        previous_post_mix.reshape(1, token_count, 4).contiguous(),
        previous_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
    )
    options = {"hc_eps": hc_eps}
    if plan.runtime.is_active:
        options["diagnostics"] = plan.runtime
    outputs = multicore_ops.hyper_mega_mhc_grad(
        *kernel_inputs, plan.runtime.runtime_config, events, events, **options,
    )
    return _restore_output_shapes(outputs, leading_shape, hidden_size)

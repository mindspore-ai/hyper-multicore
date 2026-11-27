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
"""Torch projection and autograd bridge for native HyperMegaGate Route."""

from __future__ import annotations

from types import ModuleType

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional


from .plan import MegaGatePlan


_NATIVE_OPS: ModuleType | None = None


def _native_ops() -> ModuleType:
    """Load the optional torch_npu adapter only when Device work is requested."""
    global _NATIVE_OPS  # pylint: disable=global-statement
    if _NATIVE_OPS is None:
        from hyper_multicore.torch import ops  # pylint: disable=import-outside-toplevel

        _NATIVE_OPS = ops
    return _NATIVE_OPS


def _launch_route(
    logits: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor,
    use_vision_bias: bool,
    plan: MegaGatePlan,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the eventless Route pipeline on FP32 logits."""
    diagnostics = plan.route_diagnostics
    options = {}
    if diagnostics.requires_diagnostic_launch:
        options["diagnostics"] = diagnostics
    return _native_ops()._mega_gate_route(  # pylint: disable=protected-access
        logits, text_bias, vision_bias, image_mask,
        diagnostics.runtime_config_tensor, diagnostics.runtime_config_tensor,
        top_k=top_k, routed_scaling_factor=routed_scaling_factor, use_vision_bias=use_vision_bias, **options,
    )


def _launch_route_grad(
    plan: MegaGatePlan,
    logits: torch.Tensor,
    route_scores: torch.Tensor,
    selected_scores: torch.Tensor,
    normalization_denominator: torch.Tensor,
    expert_indices: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Launch RouteGrad without combining the direct logits gradient."""
    diagnostics = plan.route_grad_k1_diagnostics if top_k == 1 else plan.route_grad_diagnostics
    inputs = (logits, route_scores, selected_scores, normalization_denominator, expert_indices, grad_routing_weights)
    if not diagnostics.requires_diagnostic_launch:
        return _native_ops()._mega_gate_route_grad(  # pylint: disable=protected-access
            *inputs, diagnostics.runtime_config_tensor, diagnostics.runtime_config_tensor,
            top_k=top_k, routed_scaling_factor=routed_scaling_factor,
        )
    # Positions 0..6 mirror the internal RouteGrad MegaKernel ABI. The zero-task
    # ACLNN tail node only uses Host positions 9/10, never Device workspace pointers.
    parameters = [selected_scores, normalization_denominator, grad_routing_weights, route_scores, expert_indices,
                  diagnostics.runtime_config_tensor, diagnostics.runtime_config_tensor, None, None, None, logits]
    with diagnostics.launch(parameters, direction="backward") as call:
        result = _native_ops()._mega_gate_route_grad(  # pylint: disable=protected-access
            *inputs, call.runtime_config_tensor, call.diagnostic_buffer,
            top_k=top_k, routed_scaling_factor=routed_scaling_factor,
        )
        parameters[9] = result
        return result


class _MegaGateRouteFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Differentiate only the native Route pipeline with respect to logits."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        text_bias: torch.Tensor,
        vision_bias: torch.Tensor,
        image_mask: torch.Tensor,
        use_vision_bias: bool,
        plan: MegaGatePlan,
        top_k: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Route while retaining only the state required by RouteGrad."""
        ctx.set_materialize_grads(False)
        routing_weights, expert_indices, route_scores, selected_scores, normalization_denominator = _launch_route(
            logits,
            text_bias.detach(),
            vision_bias.detach(),
            image_mask,
            use_vision_bias,
            plan,
            top_k,
            routed_scaling_factor,
        )
        ctx.save_for_backward(
            logits,
            expert_indices,
            route_scores,
            selected_scores,
            normalization_denominator,
        )
        ctx.top_k = top_k
        ctx.routed_scaling_factor = routed_scaling_factor
        ctx.plan = plan
        ctx.mark_non_differentiable(expert_indices)
        return routing_weights, expert_indices

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_routing_weights: torch.Tensor | None,
        grad_expert_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None]:
        """Return the Route contribution to the FP32 logits gradient."""
        del grad_expert_indices
        if grad_routing_weights is None or not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None, None

        logits, expert_indices, route_scores, selected_scores, normalization_denominator = ctx.saved_tensors
        grad_logits = _launch_route_grad(
            ctx.plan,
            logits,
            route_scores,
            selected_scores,
            normalization_denominator,
            expert_indices,
            grad_routing_weights.contiguous(),
            top_k=ctx.top_k,
            routed_scaling_factor=ctx.routed_scaling_factor,
        )
        return grad_logits, None, None, None, None, None, None, None


def mega_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
    plan: MegaGatePlan,
    *,
    vision_bias: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Torch projection followed by the native HyperMegaGate Route."""
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = functional.linear(  # pylint: disable=not-callable
        flattened.float(), weight.float()
    )
    route_logits = logits.float()
    use_vision_bias = vision_bias is not None and image_mask is not None
    # FSDP may present BF16 bias parameters; golden promotes them when adding FP32 scores.
    correction_bias = correction_bias.detach().float()
    resolved_vision_bias = vision_bias.detach().float() if use_vision_bias else correction_bias
    resolved_image_mask = plan.vision_mask_placeholder if not use_vision_bias else image_mask
    if not torch.is_grad_enabled() or not logits.requires_grad:
        routing_weights, expert_indices = _launch_route(
            route_logits,
            correction_bias,
            resolved_vision_bias,
            resolved_image_mask,
            use_vision_bias,
            plan,
            top_k,
            routed_scaling_factor,
        )[:2]
        return logits, routing_weights, expert_indices
    routing_weights, expert_indices = _MegaGateRouteFunction.apply(
        route_logits,
        correction_bias,
        resolved_vision_bias,
        resolved_image_mask,
        use_vision_bias,
        plan,
        top_k,
        routed_scaling_factor,
    )
    return logits, routing_weights, expert_indices

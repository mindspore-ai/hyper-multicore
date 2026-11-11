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
"""Differentiable execution for independently selectable route endpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from hyper_multicore.torch import ops as multicore_ops

from .route import (
    PreparedTopKRoute,
    RouteMetadata,
    _build_route_indices,
    restore_topk_output,
)
from .plan import MegaMoePlan, record_plan_stream
from .function import _native_source, _release_completed_graph
from .workspace import MegaMoeWorkspace
from .shared import shared_backward_buffers, shared_forward_buffers


_MEGA_MOE_FUNCTION = None


@dataclass(frozen=True)
class _ForwardBuffers:
    """Forward tensors prepared around the fused multicore launch."""

    dispatch: Any
    source: Any
    combine: Any
    gmm_workspace: Any
    up_proj: Any
    activation: Any
    down_proj: Any
    final_out: Any
    permute_src_index: Any
    events: Any
    shared_up_proj: Any | None
    shared_activation: Any | None
    shared_output: Any | None
    shared_ptrs: Any | None


@dataclass(frozen=True)
class _SavedForwardState:
    """Named view of tensors retained by the autograd forward pass."""

    dispatch: Any
    up_proj: Any
    activation: Any
    weight1: Any
    weight2: Any
    group_list: Any
    dispatch_src_off: Any
    dispatch_target_off: Any
    dispatch_size: Any
    combine_src_off: Any
    combine_target_off: Any
    combine_size: Any
    topk_ids: Any
    topk_weights: Any
    reduce_route_index: Any
    permute_src_index: Any
    expert_output: Any
    shared_input: Any | None
    shared_gate_up_weight: Any | None
    shared_down_weight: Any | None
    shared_up_proj: Any | None
    shared_activation: Any | None
    up_proj_reusable: bool = False
    expert_output_reusable: bool = False


@dataclass(frozen=True)
class _BackwardRouteState:
    """Route-dependent tensors selected before the backward kernel launch."""

    kernel_grad_output: Any
    permute_src_index: Any
    route_weight: Any
    grad_topk_weights: Any | None
    dispatch: Any
    combine: Any
    grad_x: Any
    route_graph: bool


@dataclass(frozen=True)
class _BackwardBuffers:
    """Backward outputs, intermediates and device workspaces."""

    grad_weight1: Any
    grad_weight2: Any
    act_grad: Any
    swiglu_grad: Any
    gate_dx: Any
    gmm_workspace: Any
    swiglu_workspace: Any
    events: Any
    shared_dx: Any | None
    shared_dw1: Any | None
    shared_dw2: Any | None
    shared_ptrs: Any | None
    shared_tensors: tuple[Any, ...]
    shared_gradients_in_scratch: bool


def _workspace_tensor(tensor: Any | None, name: str) -> Any:
    """Return an initialized workspace tensor or raise an invariant error."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe workspace {name} is not initialized.")
    return tensor


def _native_operand(tensor: Any) -> Any:
    """Keep empty-rank ABI pointers non-null without changing task extents."""
    if tensor.numel():
        return tensor
    if tensor.ndim == 2 and tensor.shape[1]:
        return _native_source(tensor)
    return tensor.new_empty((1,))


def _route_tensor(tensor: Any | None, name: str) -> Any:
    """Return a route tensor required by the selected fusion mode."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe fused route is missing {name}.")
    return tensor


def _allocate_forward_intermediates(
    torch_module: Any,
    spec: Any,
    capacity: int,
    reference: Any,
) -> tuple[Any, Any, Any]:
    """Allocate forward tensors at the route's actual bounded capacity."""
    up_proj = torch_module.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=reference.dtype,
        device=reference.device,
    )
    activation = torch_module.empty(
        (capacity, spec.intermediate_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    down_proj = torch_module.empty(
        (max(capacity, spec.routed_slots) if spec.forward_unpermute_fused else capacity, spec.hidden_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    return up_proj, activation, down_proj


def _can_reuse_saved_tensors() -> bool:
    """Reject hook-managed saved storage without changing the caller's hook state."""
    disable_hooks = getattr(getattr(torch.autograd, "graph", None), "disable_saved_tensors_hooks", None)
    if not callable(disable_hooks):
        return False
    try:
        with disable_hooks("MegaMoe backward storage reuse requires inactive saved tensor hooks."):
            pass
    except Exception:
        return False
    return True


def _is_final_backward() -> bool:
    """Require an explicit final first-order engine task before consuming storage."""
    if torch.is_grad_enabled():
        return False
    keep_graph = getattr(getattr(torch._C, "_autograd", None), "_get_current_graph_task_keep_graph", None)
    if not callable(keep_graph):
        return False
    try:
        return keep_graph() is False
    except Exception:  # An unavailable engine query must preserve saved tensors.
        return False


def _claim_saved_tensor(ctx: Any, index: int) -> bool:
    """Reject late hooks after unpacking, immediately before saved tensors clear."""
    if getattr(ctx, "saved_tensors_reusable", False) is not True or not _is_final_backward():
        return False
    try:
        # Registration rejects existing hooks without replacing them. These
        # pass-through hooks only live until the final backward clears its saves.
        ctx._raw_saved_tensors[index].register_hooks(lambda tensor: tensor, lambda tensor: tensor)
    except Exception:
        return False
    return True


def _reusable_swiglu_input(up_proj: Any, spec: Any, capacity: int, grad_output: Any) -> Any | None:
    """Consume the owned BF16 input only during a graph's final first-order backward."""
    if not _is_final_backward():
        return None
    if (capacity <= 0 or tuple(up_proj.shape) != (capacity, spec.intermediate_size * 2)
            or up_proj.dtype != torch.bfloat16 or grad_output.dtype != up_proj.dtype
            or up_proj.device != grad_output.device or not up_proj.is_contiguous()):
        return None
    # Routed SwiGLUGrad is the sole original-data reader and loads each BF16
    # tile before writing that tile. Later routed/shared readers use its result.
    return up_proj


def _allocate_backward_intermediates(
    torch_module: Any,
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
    reusable_dispatch: Any | None = None,
    reusable_up: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate backward tensors that the device kernel fully initializes."""
    grad_weight2 = torch_module.zeros_like(weight2)
    act_grad = torch_module.empty(
        (capacity, spec.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = reusable_up if reusable_up is not None else torch_module.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = reusable_dispatch if reusable_dispatch is not None else torch_module.empty(
        (capacity, spec.hidden_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    grad_weight1 = torch_module.zeros_like(weight1)
    return grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx


def _unpermute_grad(
    expert_output: Any,
    grad_output: Any,
    reduce_route_index: Any,
    topk_weights: Any,
    expert_grad: Any | None = None,
) -> tuple[Any, Any]:
    """Differentiate the external weighted-unpermute endpoint with ACLNN."""

    if expert_grad is None:
        expert_grad = torch.empty_like(expert_output)
    grad_weights = torch.empty_like(topk_weights)
    if grad_output.shape[0] == 0:
        return expert_grad, grad_weights
    multicore_ops.mega_moe_unpermute_grad_out(
        expert_output, grad_output, reduce_route_index.reshape(-1).to(torch.int32),
        topk_weights, expert_grad, grad_weights,
    )
    return expert_grad, grad_weights


def _permute_grad(
    token_template: Any,
    expert_grad: Any,
    topk_ids: Any,
    reduce_route_index: Any,
) -> Any:
    """Differentiate token permute using a shape/dtype template for its output."""

    if token_template.shape[0] == 0:
        return token_template.clone()
    return multicore_ops.moe_token_permute_grad(
        expert_grad, reduce_route_index.reshape(-1).to(torch.int32), token_template.shape[0], topk_ids.shape[1],
    )


def _save_forward_state(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    dispatch: Any,
    up_proj: Any,
    activation: Any,
    weight1: Any,
    weight2: Any,
    metadata: RouteMetadata,
    topk_ids: Any,
    topk_weights: Any,
    route: PreparedTopKRoute,
    expert_output: Any,
    shared_input: Any | None = None,
    shared_gate_up_weight: Any | None = None,
    shared_down_weight: Any | None = None,
    shared_up_proj: Any | None = None,
    shared_activation: Any | None = None,
) -> None:
    """Save owned forward tensors, route indices and static execution state."""
    ctx.plan = plan
    ctx.workspace = workspace
    # Hooks can unpack tensors that alias another graph or user-visible storage.
    ctx.saved_tensors_reusable = _can_reuse_saved_tensors()
    ctx.save_for_backward(
        dispatch,
        up_proj,
        activation,
        weight1,
        weight2,
        metadata.group_list,
        metadata.dispatch_src_off,
        metadata.dispatch_target_off,
        metadata.dispatch_size,
        metadata.combine_src_off,
        metadata.combine_target_off,
        metadata.combine_size,
        topk_ids,
        topk_weights,
        route.unpermute_mapping,
        (
            route.permute_src_index
            if route.permute_src_index is not None
            else metadata.dispatch_size
        ),
        expert_output,
        shared_input if shared_input is not None else metadata.dispatch_size,
        shared_gate_up_weight if shared_gate_up_weight is not None else metadata.dispatch_size,
        shared_down_weight if shared_down_weight is not None else metadata.dispatch_size,
        shared_up_proj if shared_up_proj is not None else metadata.dispatch_size,
        shared_activation if shared_activation is not None else metadata.dispatch_size,
    )
    workspace.track_graph(ctx)


def _publish_pull_route(workspace: MegaMoeWorkspace, indices: Any, weights: Any | None = None) -> Any:
    """Publish owned scalar routing metadata before the device-ready handshake."""
    metadata = _workspace_tensor(workspace.route_metadata, "route_metadata")
    metadata = metadata[:indices.numel()]
    metadata[:, 0].copy_(indices)
    if weights is not None:
        ordered = weights.reshape(-1).index_select(0, indices)
        metadata[:, 1].copy_(ordered.view(torch.int32))
    return metadata


def _prepare_forward_buffers(
    torch_module: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    metadata: RouteMetadata,
    hidden_states: Any,
    route: PreparedTopKRoute,
    shared_gate_up_weight: Any | None = None,
    shared_down_weight: Any | None = None,
) -> _ForwardBuffers:
    """Allocate and select tensors consumed by the forward custom operator."""
    spec = plan.spec
    capacity = metadata.expert_capacity
    pull = spec.dispatch_mode == "pull"
    dispatch = torch_module.empty(
        (capacity, spec.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device,
    ) if pull else _workspace_tensor(
        workspace.expert_buffer,
        "expert_buffer",
    ).narrow(0, 0, capacity)
    up_proj, activation, down_proj = _allocate_forward_intermediates(
        torch_module,
        spec,
        capacity,
        hidden_states,
    )
    combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    final_out = (
        torch_module.empty(
            (spec.local_num_tokens, spec.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if spec.forward_unpermute_fused
        else combine.reshape(-1).narrow(0, 0, 1)
    )
    permute_src_index = (
        _route_tensor(route.permute_src_index, "permute_src_index")
        if spec.forward_permute_fused
        else route.tokens_per_expert
    )
    source_rows = spec.local_num_tokens if spec.forward_permute_fused else spec.routed_slots
    source = _workspace_tensor(workspace.source_buffer, "source_buffer")[:source_rows] if pull else None
    if pull and spec.forward_permute_fused:
        permute_src_index = _publish_pull_route(workspace, permute_src_index)
    shared_up_proj = shared_activation = shared_output = shared_ptrs = None
    if spec.forward_shared_expert or spec.backward_shared_expert:
        if shared_gate_up_weight is None or shared_down_weight is None:
            raise RuntimeError("Shared expert requires both shared weights")
        shared_up_proj, shared_activation, shared_output, shared_ptrs = shared_forward_buffers(
            hidden_states, shared_gate_up_weight, shared_down_weight, spec.shared_intermediate_size,
            output=final_out if spec.forward_unpermute_fused else None,
        )
        if not spec.forward_shared_expert:
            torch_module.mm(hidden_states, shared_gate_up_weight.t(), out=shared_up_proj)
            gate, up = shared_up_proj.chunk(2, dim=-1)
            torch_module.mul(F.silu(gate), up, out=shared_activation)
            torch_module.mm(shared_activation, shared_down_weight.t(), out=shared_output)
    return _ForwardBuffers(
        dispatch=dispatch,
        source=source,
        combine=combine,
        gmm_workspace=_workspace_tensor(
            workspace.gmm_workspace,
            "gmm_workspace",
        ),
        up_proj=up_proj,
        activation=activation,
        down_proj=down_proj,
        final_out=final_out,
        permute_src_index=permute_src_index,
        events=workspace.prepare_event_counters(forward=True),
        shared_up_proj=shared_up_proj,
        shared_activation=shared_activation,
        shared_output=shared_output,
        shared_ptrs=shared_ptrs,
    )


def _forward_expert_output(spec: Any, buffers: _ForwardBuffers, save_for_backward: bool) -> Any:
    """Save expert outputs without retaining surplus down-projection rows."""
    output = (buffers.down_proj.narrow(0, 0, spec.routed_slots)
              if spec.forward_unpermute_fused else buffers.combine[:spec.routed_slots])
    # Fused backward retains this snapshot while allocating GMM scratch.
    # A narrow view would keep the larger receive allocation alive throughout.
    compact = (spec.forward_unpermute_fused and spec.backward_permute_fused
               and buffers.down_proj.shape[0] > spec.routed_slots)
    if save_for_backward and (not spec.forward_unpermute_fused or compact):
        return output.clone()
    return output


def _launch_forward(
    plan: MegaMoePlan,
    metadata: RouteMetadata,
    hidden_states: Any,
    kernel_input: Any,
    weight1: Any,
    weight2: Any,
    shared_gate_up_weight: Any | None,
    shared_down_weight: Any | None,
    route: PreparedTopKRoute,
    topk_weights: Any,
    buffers: _ForwardBuffers,
) -> None:
    """Launch the forward graph with route-mode-independent ABI arguments."""
    spec = plan.spec
    kernel_input = kernel_input.contiguous()
    if spec.dispatch_mode == "pull":
        if kernel_input.data_ptr() != buffers.source.data_ptr():
            buffers.source.copy_(kernel_input)
        kernel_input = buffers.source
    record_plan_stream(plan, plan.fwd_runtime_config)
    parameters = (
        buffers.dispatch,
        metadata.dispatch_target_off * spec.hidden_size,
        kernel_input,
        metadata.dispatch_src_off * spec.hidden_size,
        metadata.dispatch_size * spec.hidden_size,
        weight1.contiguous(),
        metadata.group_list,
        buffers.up_proj,
        buffers.activation,
        weight2.contiguous(),
        metadata.group_list,
        buffers.down_proj,
        buffers.combine,
        metadata.combine_target_off * spec.hidden_size,
        metadata.combine_src_off * spec.hidden_size,
        metadata.combine_size * spec.hidden_size,
        buffers.gmm_workspace,
        plan.up_proj_tiling,
        plan.swiglu_tiling,
        plan.down_proj_tiling,
        plan.fwd_runtime_config,
        buffers.events,
        kernel_input if spec.forward_permute_fused else hidden_states,
        buffers.permute_src_index,
        route.unpermute_mapping,
        topk_weights,
        buffers.final_out,
    )
    if spec.forward_shared_expert or spec.backward_shared_expert:
        parameters += (buffers.shared_ptrs, plan.shared_forward_runtime)
        parameters += (
            hidden_states, shared_gate_up_weight, shared_down_weight,
            buffers.shared_up_proj, buffers.shared_activation, buffers.shared_output,
        )
    parameters = tuple(_native_operand(tensor) for tensor in parameters)
    multicore_ops.launch_route_fused(
        parameters, (spec.rank_id, spec.ep_size, spec.num_experts, spec.hidden_size, spec.plan_tokens),
        plan.fwd_diagnostics, forward=True,
    )


def _load_forward_state(ctx: Any) -> _SavedForwardState:
    """Expose saved tensors by name without repacking device storage."""
    # Preserve normal version checks before claiming storage with pass-through hooks.
    tensors = ctx.saved_tensors
    spec = ctx.plan.spec
    writes_snapshot = spec.backward_permute_fused and spec.backward_unpermute_fused and not spec.backward_shared_expert
    saved = _SavedForwardState(
        *tensors, up_proj_reusable=_claim_saved_tensor(ctx, 1),
        expert_output_reusable=_claim_saved_tensor(ctx, 16) if writes_snapshot else False,
    )
    if not (spec.forward_shared_expert or spec.backward_shared_expert):
        return replace(
            saved, shared_input=None, shared_gate_up_weight=None, shared_down_weight=None,
            shared_up_proj=None, shared_activation=None,
        )
    return saved


def _prepare_fused_backward_route(
    plan: MegaMoePlan,
    saved: _SavedForwardState,
) -> tuple[Any, Any]:
    """Return expert-major route slots and token-major weights for dispatch."""
    if plan.spec.forward_permute_fused:
        permute_src_index = saved.permute_src_index
    else:
        route_slots = _route_tensor(plan.route_slots, "route-slot metadata")
        permute_src_index, _ = _build_route_indices(
            route_slots,
            saved.topk_ids,
        )
    route_weight = saved.topk_weights.detach().reshape(-1).clone()
    return permute_src_index, route_weight


def _prepare_backward_route(
    ctx: Any,
    grad_output: Any,
    saved: _SavedForwardState,
) -> _BackwardRouteState:
    """Select external or fused route gradients and their destination buffers."""
    spec = ctx.plan.spec
    workspace = ctx.workspace
    if spec.backward_permute_fused:
        kernel_grad_output = grad_output.contiguous()
        permute_src_index, route_weight = _prepare_fused_backward_route(
            ctx.plan,
            saved,
        )
        grad_topk_weights = None
        if spec.dispatch_mode == "pull":
            _publish_pull_route(workspace, permute_src_index, saved.topk_weights)
            permute_src_index = _workspace_tensor(workspace.route_metadata, "route_metadata")
            source = _workspace_tensor(workspace.source_buffer, "source_buffer")[:spec.local_num_tokens]
            source.copy_(kernel_grad_output)
            kernel_grad_output = source
    else:
        source = (_workspace_tensor(workspace.source_buffer, "source_buffer")[:spec.routed_slots]
                  if spec.dispatch_mode == "pull" else None)
        kernel_grad_output, grad_topk_weights = _unpermute_grad(
            saved.expert_output,
            grad_output,
            saved.reduce_route_index,
            saved.topk_weights,
            expert_grad=source,
        )
        kernel_grad_output = kernel_grad_output.contiguous()
        # This stable ABI slot has no consumer in the non-permute graph.
        route_weight = saved.topk_weights.reshape(-1)[:1]
        permute_src_index = saved.dispatch_size

    capacity = saved.dispatch.shape[0]
    dispatch = torch.empty(
        (capacity, spec.hidden_size), dtype=grad_output.dtype, device=grad_output.device,
    ) if spec.dispatch_mode == "pull" else _workspace_tensor(
        workspace.expert_buffer,
        "expert_buffer",
    ).narrow(0, 0, capacity)
    route_graph = spec.backward_permute_fused or spec.backward_unpermute_fused
    if spec.backward_permute_fused:
        # dTopK reads the snapshot while Combine writes the separate return buffer.
        writes_snapshot = spec.backward_unpermute_fused and not spec.backward_shared_expert
        grad_x = (saved.expert_output.clone() if writes_snapshot and not saved.expert_output_reusable
                  else saved.expert_output)
    elif route_graph:
        grad_x = torch.empty(
            (spec.local_num_tokens, spec.hidden_size),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    return _BackwardRouteState(
        kernel_grad_output=kernel_grad_output,
        permute_src_index=permute_src_index,
        route_weight=route_weight,
        grad_topk_weights=grad_topk_weights,
        dispatch=dispatch,
        combine=_workspace_tensor(workspace.routed_buffer, "routed_buffer"),
        grad_x=grad_x,
        route_graph=route_graph,
    )


def _prepare_backward_buffers(
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    grad_output: Any,
) -> _BackwardBuffers:
    """Allocate backward intermediates and select persistent workspaces."""
    intermediates = _allocate_backward_intermediates(
        torch,
        plan.spec,
        saved.dispatch.shape[0],
        route_state.kernel_grad_output,
        saved.weight1,
        saved.weight2,
        (route_state.dispatch.narrow(0, 0, saved.dispatch.shape[0])
         if plan.reuse_backward_dispatch else None),
        reusable_up=_reusable_swiglu_input(
            saved.up_proj, plan.spec, saved.dispatch.shape[0], route_state.kernel_grad_output,
        ) if saved.up_proj_reusable is True else None,
    )
    grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx = intermediates
    shared_dx = shared_dw1 = shared_dw2 = shared_ptrs = None
    shared_values = ()
    shared_gradients_in_scratch = False
    if plan.spec.backward_shared_expert:
        if (saved.shared_input is None or saved.shared_gate_up_weight is None
                or saved.shared_down_weight is None or saved.shared_up_proj is None
                or saved.shared_activation is None):
            raise RuntimeError("Native shared backward requires saved shared tensors")
        shared_values, shared_ptrs, shared_gradients_in_scratch = shared_backward_buffers(
            grad_output,
            (saved.shared_input, saved.shared_gate_up_weight, saved.shared_down_weight,
             saved.shared_up_proj, saved.shared_activation),
            scratch=(act_grad, swiglu_grad),
            integrated_output=plan.spec.backward_unpermute_fused,
        )
        shared_dw2, shared_dw1 = shared_values[2], shared_values[10]
        shared_dx = shared_values[11] if plan.spec.backward_unpermute_fused else shared_values[8]
    return _BackwardBuffers(
        grad_weight1=grad_weight1,
        grad_weight2=grad_weight2,
        act_grad=act_grad,
        swiglu_grad=swiglu_grad,
        gate_dx=gate_dx,
        gmm_workspace=_workspace_tensor(
            workspace.gmm_workspace,
            "gmm_workspace",
        ),
        swiglu_workspace=_workspace_tensor(
            workspace.swiglu_grad_workspace,
            "swiglu_grad_workspace",
        ),
        events=workspace.prepare_event_counters(forward=False),
        shared_dx=shared_dx,
        shared_dw1=shared_dw1,
        shared_dw2=shared_dw2,
        shared_ptrs=shared_ptrs,
        shared_tensors=shared_values,
        shared_gradients_in_scratch=shared_gradients_in_scratch,
    )


def _launch_backward(
    ctx: Any,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    buffers: _BackwardBuffers,
) -> None:
    """Launch the backward graph with route-mode-independent ABI arguments."""
    plan = ctx.plan
    spec = plan.spec
    record_plan_stream(plan, plan.bwd_runtime_config)
    parameters = (
        route_state.dispatch,
        saved.dispatch_target_off * spec.hidden_size,
        route_state.kernel_grad_output,
        saved.dispatch_src_off * spec.hidden_size,
        saved.dispatch_size * spec.hidden_size,
        saved.activation,
        buffers.grad_weight2,
        saved.weight2,
        buffers.act_grad,
        saved.up_proj,
        buffers.swiglu_grad,
        saved.weight1,
        buffers.gate_dx,
        route_state.grad_x,
        saved.combine_target_off * spec.hidden_size,
        saved.combine_src_off * spec.hidden_size,
        saved.combine_size * spec.hidden_size,
        saved.dispatch,
        buffers.grad_weight1,
        saved.group_list,
        plan.act_grad_tiling,
        plan.gate_grad_tiling,
        plan.w1_grad_tiling,
        plan.w2_grad_tiling,
        plan.swiglu_grad_tiling,
        buffers.gmm_workspace,
        buffers.swiglu_workspace,
        plan.bwd_runtime_config,
        buffers.events,
        route_state.permute_src_index,
        route_state.route_weight,
        saved.reduce_route_index,
        route_state.combine,
    )
    if spec.backward_shared_expert:
        parameters += (buffers.shared_ptrs, plan.shared_backward_runtime)
        parameters += buffers.shared_tensors
    parameters = tuple(_native_operand(tensor) for tensor in parameters)
    multicore_ops.launch_route_fused(
        parameters, (spec.rank_id, spec.ep_size, spec.num_experts, spec.hidden_size, spec.plan_tokens),
        plan.bwd_diagnostics, forward=False,
    )


def _finish_backward(
    ctx: Any,
    grad_output: Any,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    buffers: _BackwardBuffers,
) -> tuple[Any, ...]:
    """Restore public gradient layouts after the backward custom operator."""
    # pylint: disable=not-callable
    spec = ctx.plan.spec
    grad_topk_weights = route_state.grad_topk_weights
    if spec.backward_permute_fused:
        grad_topk_weights = route_state.route_weight.reshape_as(
            saved.topk_weights
        ).to(saved.topk_weights.dtype)
    if spec.backward_unpermute_fused:
        grad_hidden = (buffers.shared_dx if spec.backward_shared_expert
                       else route_state.grad_x.narrow(0, 0, spec.local_num_tokens).clone())
    else:
        expert_grad = (
            route_state.combine.narrow(0, 0, spec.routed_slots)
            if route_state.route_graph
            else route_state.grad_x.narrow(0, 0, spec.routed_slots)
        )
        grad_hidden = _permute_grad(
            grad_output,
            expert_grad,
            saved.topk_ids,
            saved.reduce_route_index,
        )
    shared_grad_x = shared_grad_w1 = shared_grad_w2 = None
    if spec.backward_shared_expert:
        shared_grad_x, shared_grad_w1, shared_grad_w2 = (
            buffers.shared_dx, buffers.shared_dw1, buffers.shared_dw2,
        )
    elif spec.forward_shared_expert:
        shared_input = saved.shared_input.detach().requires_grad_(True)
        shared_w1 = saved.shared_gate_up_weight.detach().requires_grad_(True)
        shared_w2 = saved.shared_down_weight.detach().requires_grad_(True)
        with torch.enable_grad():
            gate_up = F.linear(shared_input, shared_w1)
            gate, up = gate_up.chunk(2, dim=-1)
            shared_value = F.linear(F.silu(gate) * up, shared_w2)
        shared_grad_x, shared_grad_w1, shared_grad_w2 = torch.autograd.grad(
            shared_value, (shared_input, shared_w1, shared_w2), grad_output,
            allow_unused=True,
        )
    if shared_grad_x is not None and not (spec.backward_shared_expert and spec.backward_unpermute_fused):
        grad_hidden = grad_hidden + shared_grad_x
    return (
        grad_hidden,
        None,
        grad_topk_weights.to(ctx.topk_dtype),
        buffers.grad_weight1,
        buffers.grad_weight2,
        shared_grad_w1,
        shared_grad_w2,
        None,
        None,
        None,
        None,
        None,
    )


def _forward_mega_moe(
    ctx: Any, hidden_states: Any, topk_ids: Any, topk_weights: Any,
    weight1: Any, weight2: Any, shared_gate_up_weight: Any | None, shared_down_weight: Any | None,
    routed_tokens: Any | None, route: PreparedTopKRoute, plan: MegaMoePlan,
    workspace: MegaMoeWorkspace, save_for_backward: bool,
) -> Any:
    """Run the selected graph and retain owned tensors under its workspace lease."""
    spec = plan.spec
    metadata = route.metadata
    if metadata is None:
        raise RuntimeError("MegaMoe route metadata is unavailable.")
    ctx.topk_dtype = topk_weights.dtype
    topk_weights = topk_weights.float().contiguous()
    kernel_input = (
        hidden_states
        if spec.forward_permute_fused
        else _route_tensor(routed_tokens, "routed_tokens")
    )
    if not spec.auto_tokens:
        workspace.ensure(spec, hidden_states.dtype, hidden_states.device)
    pull = spec.dispatch_mode == "pull"
    if not pull:
        workspace.claim()
    try:
        buffers = _prepare_forward_buffers(
            torch, plan, workspace, metadata, hidden_states, route,
            shared_gate_up_weight, shared_down_weight,
        )
        _launch_forward(
            plan, metadata, hidden_states, kernel_input, weight1, weight2,
            shared_gate_up_weight, shared_down_weight, route, topk_weights, buffers,
        )
        expert_output = _forward_expert_output(spec, buffers, save_for_backward)
        output = (
            buffers.final_out
            if spec.forward_unpermute_fused
            else restore_topk_output(
                expert_output,
                route.unpermute_mapping,
                topk_weights,
            )
        )
        if (spec.forward_shared_expert or spec.backward_shared_expert) and not spec.forward_unpermute_fused:
            output = output + buffers.shared_output
        if not save_for_backward:
            return output
        if pull:
            saved_dispatch = buffers.dispatch
        elif spec.forward_unpermute_fused:
            saved_dispatch = buffers.dispatch.clone()
        else:
            # Down projection is dead after Combine; retain the received
            # expert input there without a second capacity-sized allocation.
            saved_dispatch = buffers.down_proj
            saved_dispatch.copy_(buffers.dispatch)
        _save_forward_state(
            ctx, plan, workspace, saved_dispatch, buffers.up_proj, buffers.activation,
            weight1, weight2, metadata, topk_ids, topk_weights, route, expert_output,
            hidden_states, shared_gate_up_weight, shared_down_weight,
            buffers.shared_up_proj, buffers.shared_activation,
        )
        return output
    finally:
        if not pull:
            workspace.release()


def _get_mega_moe_function() -> Any:
    """Create the Torch autograd.Function lazily inside the Torch backend."""
    # pylint: disable=global-statement
    global _MEGA_MOE_FUNCTION
    if _MEGA_MOE_FUNCTION is not None:
        return _MEGA_MOE_FUNCTION

    class _MegaMoeFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
        """Autograd bridge for legacy and independently fused route endpoints."""

        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            topk_ids: Any,
            topk_weights: Any,
            weight1: Any,
            weight2: Any,
            shared_gate_up_weight: Any | None,
            shared_down_weight: Any | None,
            routed_tokens: Any | None,
            route: PreparedTopKRoute,
            plan: MegaMoePlan,
            workspace: MegaMoeWorkspace,
            save_for_backward: bool,
        ) -> Any:
            """Run forward with owned saved tensors and a serialized workspace lease."""
            return _forward_mega_moe(
                ctx, hidden_states, topk_ids, topk_weights, weight1, weight2,
                shared_gate_up_weight, shared_down_weight, routed_tokens,
                route, plan, workspace, save_for_backward,
            )

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            """Launch the selected backward route endpoints and expert graph.

            Args:
                ctx: Saved forward tensors and workspace lifetime state.
                grad_output: Token-major output gradient from the following layer.
            """
            workspace = ctx.workspace
            saved = _load_forward_state(ctx)
            ctx.maybe_clear_saved_tensors()
            workspace.claim()
            try:
                route_state = _prepare_backward_route(ctx, grad_output, saved)
                if not ctx.plan.spec.backward_permute_fused:
                    # The external route gradient has consumed this snapshot.
                    # Retained autograd graphs keep their own saved reference.
                    saved = replace(saved, expert_output=None)
                buffers = _prepare_backward_buffers(
                    ctx.plan,
                    workspace,
                    saved,
                    route_state,
                    grad_output,
                )
                _launch_backward(ctx, saved, route_state, buffers)
                # Release device scratch before allocating the owned token dX.
                buffers = replace(buffers, act_grad=None, swiglu_grad=None, gate_dx=None, shared_tensors=())
                if buffers.shared_gradients_in_scratch:
                    # Return compact owned gradients after releasing the other routed scratch.
                    buffers = replace(buffers, shared_dw1=buffers.shared_dw1.clone(),
                                      shared_dw2=buffers.shared_dw2.clone())
                gradients = _finish_backward(
                    ctx,
                    grad_output,
                    saved,
                    route_state,
                    buffers,
                )
                _release_completed_graph(ctx, workspace)
                return gradients
            finally:
                workspace.release()

    _MEGA_MOE_FUNCTION = _MegaMoeFunction
    return _MEGA_MOE_FUNCTION


def execute_mega_moe(
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    weight1: Any,
    weight2: Any,
    shared_gate_up_weight: Any | None,
    shared_down_weight: Any | None,
    route: PreparedTopKRoute,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
) -> Any:
    """Execute Top-K routing endpoints and expert compute as one autograd unit.

    Args:
        hidden_states: Original token-major activations.
        topk_ids: Expert selected by each token and route.
        topk_weights: Router probabilities with autograd tracking.
        weight1: Current local gate/up weights.
        weight2: Current local down-projection weights.
        route: Owned scalar indices and communication metadata.
        plan: Selected forward/backward graphs and tiling tensors.
        workspace: Shared communication workspace with per-call leases.
    """
    function = _get_mega_moe_function()
    return function.apply(
        hidden_states,
        topk_ids,
        topk_weights,
        weight1.contiguous(),
        weight2.contiguous(),
        shared_gate_up_weight.contiguous() if shared_gate_up_weight is not None else None,
        shared_down_weight.contiguous() if shared_down_weight is not None else None,
        route.routed_tokens,
        route,
        plan,
        workspace,
        torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad
            for tensor in (hidden_states, topk_weights, weight1, weight2,
                           shared_gate_up_weight, shared_down_weight)
        ),
    )

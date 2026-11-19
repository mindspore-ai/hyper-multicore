# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Orchestrate native MX expert graphs without retaining gathered weight views."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch_npu

from hyper_multicore.torch import ops as multicore_ops

from .autograd import MXFP8ForwardState
from .diagnostics import current_mx_profile_scope, mx_profile_capture_enabled, profile_mx_launch
from .launch import launch_mxfp8_experts, prepare_mxfp8_launch
from .weights import MXFP8PhysicalWeight
from ...route import PreparedTopKRoute


@dataclass(frozen=True)
class _ExecutionMetadata:
    plan: Any
    hidden_shape: tuple[int, ...]
    source_rows: int
    probs_dtype: torch.dtype
    all_experts_active: bool = False


def _quantized_buffers(rows: int, columns: int, experts: int, device: Any, *,
                       row_only: bool = False) -> tuple[torch.Tensor, ...]:
    """Allocate packed scales only for the axes consumed by the graph."""
    row = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device)
    row_scale = torch.empty((rows, columns // 64, 2), dtype=torch.uint8, device=device)
    if row_only:
        # Dormant buffer positions preserve the native launch layout.
        return row, row_scale, row, row_scale
    column = torch.empty_like(row)
    column_scale = torch.empty(((rows + 63) // 64 + experts, columns, 2), dtype=torch.uint8, device=device)
    return row, row_scale, column, column_scale


class MXFP8Execution:
    """Own one autograd invocation's orchestration while borrowing serial resources."""

    def __init__(self, resources: Any, *, nograd_no_clone: bool = True,
                 placement=None, weight_prefetch=None, gradient_pool=None) -> None:
        self.resources = resources
        self.nograd_no_clone = nograd_no_clone
        self._profile_scope = None
        self.placement = placement
        self.weight_prefetch = weight_prefetch
        self.gradient_pool = gradient_pool

    def _launch(self, buffers, runtime, events, plan, direction="forward"):
        """Submit the graph with exact buffer bounds and optional profiling."""
        if events.numel() < runtime.event_capacity * 4:
            raise ValueError("MX event storage is smaller than the graph")
        launch_graph = partial(launch_mxfp8_experts, direction=direction)
        if mx_profile_capture_enabled():
            scope = current_mx_profile_scope() or self._profile_scope
            if direction == 'forward':
                self._profile_scope = scope
            elif scope is not None:
                scope = {**scope, 'phase': 'backward'}
            profile_mx_launch(buffers, runtime, events, plan.spec, direction,
                              prepare_mxfp8_launch, launch_graph, scope=scope)
            return
        launch = prepare_mxfp8_launch(buffers, runtime.runtime, events, plan.spec)
        # Disabled diagnostics never read this binding, matching the BF16 adapter.
        launch_graph(buffers, runtime.runtime, events, events, launch)

    def forward(self, hidden: torch.Tensor, route: PreparedTopKRoute, probs: torch.Tensor,
                weight1: MXFP8PhysicalWeight, weight2: MXFP8PhysicalWeight,
                retain_backward: bool) -> MXFP8ForwardState:
        """Prepare MX operands and restore the forward output to token order."""
        resources, workspace = self.resources, self.resources.workspace
        spec, device = resources.spec, hidden.device
        metadata = route.metadata
        transport = metadata.for_kernel(spec.hidden_size)
        pull = spec.dispatch_mode == "pull"
        # Symmetric transport keeps its reservation. Rank-local activations
        # and their retained backward storage only need this route's live rows.
        capacity = metadata.expert_capacity
        plan_capacity = capacity if pull else workspace.expert_capacity
        plan_options = dict(tokens=metadata.plan_tokens)
        if self.placement is not None:
            plan_options['home_experts'] = weight1.logical_shape[0]
        plan = resources.plan_for_capacity(plan_capacity, device, **plan_options)
        if pull:
            dispatch_buffer = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
            source_buffer = workspace.source_buffer
        else:
            source_buffer = route.routed_tokens
        row_only = not retain_backward
        forward = plan.forward_without_grad if row_only else plan.forward
        x_row, x_scale, x_col, x_col_scale = _quantized_buffers(
            capacity, spec.hidden_size, spec.local_experts, device, row_only=row_only)
        a_row, a_scale, a_col, a_col_scale = _quantized_buffers(
            capacity, spec.intermediate_size, spec.local_experts, device, row_only=row_only)
        up = torch.empty((capacity, 2 * spec.intermediate_size), dtype=torch.bfloat16, device=device)
        down = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        if not pull:
            workspace.claim()
        try:
            if not pull:
                dispatch_buffer = workspace.receive_buffer
            routed_buffer = workspace.lease_owner.routed_buffer
            buffers = [
                dispatch_buffer, transport.dispatch_target_off, source_buffer,
                transport.dispatch_src_off, transport.dispatch_size,
                weight1.col_data, metadata.group_list, up, a_row, weight2.col_data, metadata.group_list, down,
                routed_buffer, transport.combine_target_off,
                transport.combine_src_off, transport.combine_size,
                x_row, x_scale, x_col, x_col_scale, a_scale, a_col, a_col_scale,
                weight1.col_scale, weight2.col_scale, forward.gmm, forward.quant,
            ]
            events = workspace.prepare_event_counters(forward=True)
            if self.placement is None:
                self._launch(buffers, forward, events, plan)
            else:
                with self.weight_prefetch.lease((weight1, weight2), self.placement,
                                                backward=False) as (guests, ready):
                    self._launch([*buffers, *guests, ready], forward, events, plan)
            expert_output = routed_buffer[:route.routed_tokens.shape[0]]
            if retain_backward or not self.nograd_no_clone:
                expert_output = expert_output.clone()
            probabilities = probs.float().contiguous()
            output = torch_npu.npu_moe_token_unpermute(expert_output, route.unpermute_mapping, probs=probabilities)
            if (self.nograd_no_clone and not retain_backward
                    and output.untyped_storage().data_ptr() == routed_buffer.untyped_storage().data_ptr()):
                raise RuntimeError('No-grad MX unpermute output must not alias shared routed storage.')
            saved = ((x_col, x_col_scale, a_col, a_col_scale, up, expert_output, probabilities,
                      route.unpermute_mapping, metadata.group_list, transport.dispatch_target_off,
                      transport.dispatch_src_off, transport.dispatch_size, transport.combine_target_off,
                      transport.combine_src_off, transport.combine_size) if retain_backward else ())
            return MXFP8ForwardState(
                output.reshape_as(hidden), saved,
                _ExecutionMetadata(plan, tuple(hidden.shape), route.routed_tokens.shape[0], probs.dtype,
                                   metadata.all_experts_active), workspace,
            )
        finally:
            if not pull:
                workspace.release()

    def _allocate_weight_gradients(self, weights, dtype, device, all_experts_active):
        """Initialize only outputs that the full-K TN producers can skip."""
        if self.placement is None:
            allocate = torch.empty if all_experts_active else torch.zeros
            return tuple(allocate(weight.logical_shape, dtype=dtype, device=device) for weight in weights)
        owners = tuple(torch.empty(weight.logical_shape, dtype=dtype, device=device) for weight in weights)
        counts = self.placement.owner_rows[:owners[0].shape[0]]
        for slot, count in enumerate(counts):
            if count == 0:
                for weight in owners:
                    weight[slot].zero_()
        return owners

    def backward(self, gradient: torch.Tensor, saved: tuple[torch.Tensor, ...], metadata: _ExecutionMetadata,
                 weight1: MXFP8PhysicalWeight, weight2: MXFP8PhysicalWeight) -> tuple[torch.Tensor, ...]:
        """Run the backward graph on the saved route and current weights."""
        workspace, plan = self.resources.workspace, metadata.plan
        spec, device = plan.spec, gradient.device
        (x_col, x_col_scale, a_col, a_col_scale, up, expert_output, probs, mapping, group_list,
         dispatch_target, dispatch_source, dispatch_size, combine_target, combine_source, combine_size) = saved
        capacity = x_col.shape[0]
        dy_row, dy_scale, dy_col, dy_col_scale = _quantized_buffers(
            capacity, spec.hidden_size, spec.local_experts, device)
        du_row, du_scale, du_col, du_col_scale = _quantized_buffers(
            capacity, 2 * spec.intermediate_size, spec.local_experts, device)
        pull = spec.dispatch_mode == 'pull'
        source = (workspace.source_buffer[:metadata.source_rows] if pull else
                  torch.empty((metadata.source_rows, spec.hidden_size), dtype=torch.bfloat16, device=device))
        dispatch = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device) if pull else None
        grad_probs = torch.empty_like(probs)
        da = torch.empty((capacity, spec.intermediate_size), dtype=torch.bfloat16, device=device)
        dx = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        partial_dtype = torch.float32 if spec.partial_weight_grad_dtype == "float32" else torch.bfloat16
        dw1, dw2 = self._allocate_weight_gradients((weight1, weight2), partial_dtype, device,
                                                   metadata.all_experts_active)
        workspace.claim()
        try:
            if not pull:
                dispatch = workspace.receive_buffer
            routed_buffer = workspace.lease_owner.routed_buffer
            multicore_ops.mega_moe_unpermute_grad_out(
                expert_output, gradient.reshape(-1, spec.hidden_size).contiguous(), mapping, probs, source, grad_probs,
            )
            buffers = [
                dispatch, dispatch_target, workspace.source_buffer if pull else source,
                dispatch_source, dispatch_size,
                a_col, dw2, weight2.row_data, da, up, du_row, weight1.row_data, dx,
                routed_buffer, combine_target, combine_source,
                combine_size, x_col, dw1, group_list, dy_row, dy_scale, dy_col, dy_col_scale,
                a_col_scale, weight2.row_scale, du_scale, du_col, du_col_scale, weight1.row_scale, x_col_scale,
                plan.backward.gmm, plan.backward.quant,
            ]
            events = workspace.prepare_event_counters(forward=False)
            if self.placement is None:
                self._launch(buffers, plan.backward, events, plan, direction="backward")
            else:
                with self.weight_prefetch.lease((weight1, weight2), self.placement,
                                                backward=True) as (guests, ready), \
                        self.gradient_pool.native_return(self.placement, plan.backward, (dw1, dw2)) as returns:
                    self._launch([*buffers, *guests, ready, *returns], plan.backward, events, plan, direction="backward")
            grad_hidden = multicore_ops.moe_token_permute_grad(
                routed_buffer[:metadata.source_rows], mapping,
                gradient.numel() // spec.hidden_size, spec.top_k,
            )
            return grad_hidden.reshape(metadata.hidden_shape), grad_probs.to(metadata.probs_dtype), dw1, dw2
        finally:
            workspace.release()

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


def _quantized_buffers(rows: int, columns: int, experts: int, device: Any) -> tuple[torch.Tensor, ...]:
    """Allocate both data axes and preserve grouped packed-column padding."""
    return (
        torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((rows, columns // 64, 2), dtype=torch.uint8, device=device),
        torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device),
        torch.empty(((rows + 63) // 64 + experts, columns, 2), dtype=torch.uint8, device=device),
    )


class MXFP8Execution:
    """Own one autograd invocation's orchestration while borrowing serial resources."""

    def __init__(self, resources: Any, *, nograd_no_clone: bool = True) -> None:
        self.resources = resources
        self.nograd_no_clone = nograd_no_clone
        self._profile_scope = None

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
        launch = prepare_mxfp8_launch(buffers, runtime.runtime, events, plan.spec,
                                     cache=self.resources.launch_cache)
        # Disabled diagnostics never read this binding, matching the BF16 adapter.
        launch_graph(buffers, runtime.runtime, events, events, launch)

    def forward(self, hidden: torch.Tensor, route: PreparedTopKRoute, probs: torch.Tensor,
                weight1: MXFP8PhysicalWeight, weight2: MXFP8PhysicalWeight,
                retain_backward: bool) -> MXFP8ForwardState:
        """Prepare MX operands and restore the forward output to token order."""
        resources, workspace = self.resources, self.resources.workspace
        spec, device = resources.spec, hidden.device
        metadata = route.metadata
        kernel_metadata = metadata.for_kernel(spec.hidden_size)
        pull = spec.dispatch_mode == "pull"
        capacity = metadata.expert_capacity if pull else workspace.expert_capacity
        plan = resources.plan_for_capacity(capacity, device, tokens=metadata.plan_tokens)
        if pull:
            dispatch_buffer = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
            source_buffer = workspace.source_buffer
        else:
            dispatch_buffer = workspace.expert_buffer
            source_buffer = route.routed_tokens
        x_row, x_scale, x_col, x_col_scale = _quantized_buffers(capacity, spec.hidden_size, spec.local_experts, device)
        a_row, a_scale, a_col, a_col_scale = _quantized_buffers(
            capacity, spec.intermediate_size, spec.local_experts, device)
        up = torch.empty((capacity, 2 * spec.intermediate_size), dtype=torch.bfloat16, device=device)
        down = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        if not pull:
            workspace.claim()
        try:
            buffers = [
                dispatch_buffer, kernel_metadata.dispatch_target_off, source_buffer,
                kernel_metadata.dispatch_src_off, kernel_metadata.dispatch_size,
                weight1.col_data, metadata.group_list, up, a_row, weight2.col_data, metadata.group_list, down,
                workspace.routed_buffer, kernel_metadata.combine_target_off,
                kernel_metadata.combine_src_off, kernel_metadata.combine_size,
                x_row, x_scale, x_col, x_col_scale, a_scale, a_col, a_col_scale,
                weight1.col_scale, weight2.col_scale, plan.forward.gmm, plan.forward.quant,
            ]
            events = workspace.prepare_event_counters(forward=True)
            self._launch(buffers, plan.forward, events, plan)
            expert_output = workspace.routed_buffer[:route.routed_tokens.shape[0]]
            if retain_backward or not self.nograd_no_clone:
                expert_output = expert_output.clone()
            probabilities = probs.float().contiguous()
            output = torch_npu.npu_moe_token_unpermute(expert_output, route.unpermute_mapping, probs=probabilities)
            if (self.nograd_no_clone and not retain_backward
                    and output.untyped_storage().data_ptr() == workspace.routed_buffer.untyped_storage().data_ptr()):
                raise RuntimeError('No-grad MX unpermute output must not alias shared routed storage.')
            saved = ((x_col, x_col_scale, a_col, a_col_scale, up, expert_output, probabilities,
                      route.unpermute_mapping, metadata.group_list, kernel_metadata.dispatch_target_off,
                      kernel_metadata.dispatch_src_off, kernel_metadata.dispatch_size, kernel_metadata.combine_target_off,
                      kernel_metadata.combine_src_off, kernel_metadata.combine_size) if retain_backward else ())
            return MXFP8ForwardState(
                output.reshape_as(hidden), saved,
                _ExecutionMetadata(plan, tuple(hidden.shape), route.routed_tokens.shape[0], probs.dtype), workspace,
            )
        finally:
            if not pull:
                workspace.release()

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
        dispatch = (torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
                    if pull else workspace.expert_buffer)
        grad_probs = torch.empty_like(probs)
        da = torch.empty((capacity, spec.intermediate_size), dtype=torch.bfloat16, device=device)
        dx = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        partial_dtype = torch.float32 if spec.partial_weight_grad_dtype == "float32" else torch.bfloat16
        dw1 = torch.zeros(weight1.logical_shape, dtype=partial_dtype, device=device)
        dw2 = torch.zeros(weight2.logical_shape, dtype=partial_dtype, device=device)
        workspace.claim()
        try:
            multicore_ops.mega_moe_unpermute_grad_out(
                expert_output, gradient.reshape(-1, spec.hidden_size).contiguous(), mapping, probs, source, grad_probs,
            )
            buffers = [
                dispatch, dispatch_target, workspace.source_buffer if pull else source,
                dispatch_source, dispatch_size,
                a_col, dw2, weight2.row_data, da, up, du_row, weight1.row_data, dx,
                workspace.routed_buffer, combine_target, combine_source,
                combine_size, x_col, dw1, group_list, dy_row, dy_scale, dy_col, dy_col_scale,
                a_col_scale, weight2.row_scale, du_scale, du_col, du_col_scale, weight1.row_scale, x_col_scale,
                plan.backward.gmm, plan.backward.quant,
            ]
            events = workspace.prepare_event_counters(forward=False)
            self._launch(buffers, plan.backward, events, plan, direction="backward")
            grad_hidden = multicore_ops.moe_token_permute_grad(
                workspace.routed_buffer[:metadata.source_rows], mapping,
                gradient.numel() // spec.hidden_size, spec.top_k,
            )
            return grad_hidden.reshape(metadata.hidden_shape), grad_probs.to(metadata.probs_dtype), dw1, dw2
        finally:
            workspace.release()

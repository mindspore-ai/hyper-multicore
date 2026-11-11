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
"""Build the static graph and tiling plan for managed MegaMoe execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_multicore.modules.mega_moe.backward.graph import (
    build_backward_graph, build_route_backward_graph,
)
from hyper_multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_multicore.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from hyper_multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_multicore.modules.mega_moe.forward.graph import (
    build_forward_graph, build_route_forward_graph,
)
from hyper_multicore.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from hyper_multicore.scheduler.config import TaskSplitValue, mega_moe_event_capacity
from hyper_multicore.scheduler.graph import ComputeGraph

from .diagnostics import _mega_moe_owner_options, bind_shared_diagnostic_ports
from .spec import MegaMoeSpec
from .tensor_dump import attach_mega_moe_l1_view_resolvers
from .shared import shared_tiling_bytes


@dataclass(frozen=True)
class MegaMoePlan:
    """Rank-local schedules and tiling tensors for forward and backward."""

    spec: MegaMoeSpec
    forward_graph: ComputeGraph
    fwd_diagnostics: MegaKernelDiagnostics
    up_proj_tiling: Any
    swiglu_tiling: Any
    down_proj_tiling: Any
    backward_graph: ComputeGraph
    bwd_diagnostics: MegaKernelDiagnostics
    act_grad_tiling: Any
    gate_grad_tiling: Any
    w1_grad_tiling: Any
    w2_grad_tiling: Any
    swiglu_grad_tiling: Any
    reuse_backward_dispatch: bool = False
    route_slots: Any | None = None
    shared_forward_runtime: Any | None = None
    shared_backward_runtime: Any | None = None

    @property
    def fwd_runtime_config(self) -> Any:
        """Return the original ordinary forward RuntimeConfig."""
        return self.fwd_diagnostics.runtime_config

    @property
    def bwd_runtime_config(self) -> Any:
        """Return the original ordinary backward RuntimeConfig."""
        return self.bwd_diagnostics.runtime_config


def record_plan_stream(plan: MegaMoePlan, runtime_config: Any) -> None:
    """Protect cached tiling and runtime storage from cross-stream eviction."""
    tensors = [value for value in vars(plan).values() if isinstance(value, torch.Tensor)]
    tensors.append(runtime_config)
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
            tensor.record_stream(torch.npu.current_stream(tensor.device))


def _tensor_from_bytes(data: bytes, device: Any) -> torch.Tensor:
    """Copy serialized runtime data into a device uint8 tensor."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def _resize_swiglu_tiling(data: bytes, num_cube_cores: int) -> bytes:
    """Resize the baseline 24-core repeated struct for the active workers."""
    baseline_workers = 49
    if len(data) % baseline_workers:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    struct_size = len(data) // baseline_workers
    return data[:struct_size] * (2 * num_cube_cores + 1)


def _build_task_values(spec: MegaMoeSpec) -> TaskSplitValue:
    """Translate rank-local token shape into legacy scheduler coordinates.

    Args:
        spec: Validated local-token and expert topology specification.

    Returns:
        Scheduler values whose single partition is the rank-local input.
    """
    values = TaskSplitValue(
        tp=1,
        ep=spec.ep_size,
        seq_size=spec.plan_tokens,
        all_expert_num=spec.num_experts,
        top_k=spec.top_k,
        dispatch_mode=spec.dispatch_mode,
        event_capacity_floor=mega_moe_event_capacity(
            spec.num_experts, spec.ep_size, shared_expert=bool(spec.shared_intermediate_size),
            route_fusion=spec.forward_fusion_mode != "legacy" or spec.backward_fusion_mode != "legacy",
        ),
    )
    values.local_num_tokens = spec.local_num_tokens
    return values


def _build_runtime_artifacts(spec: MegaMoeSpec) -> tuple[Any, Any, Any, Any]:
    """Build forward/backward graphs and their serialized RuntimeConfig objects."""
    task_values = _build_task_values(spec)
    forward_builder = build_route_forward_graph if spec.forward_permute_fused else build_forward_graph
    forward_graph = forward_builder(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        expert_weight_layout=spec.expert_weight_layout,
        integrated_unpermute=spec.forward_unpermute_fused,
        shared_intermediate_size=spec.shared_intermediate_size,
        fuse_shared_expert=spec.forward_shared_expert,
        integrated_shared_output=spec.forward_shared_expert or spec.backward_shared_expert,
    )
    forward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(forward_graph)
    forward_data = build_forward_config(
        forward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    task_values = _build_task_values(spec)
    route_backward = spec.backward_permute_fused or spec.backward_unpermute_fused
    backward_options = ({"integrated_permute": spec.backward_permute_fused,
                         "integrated_unpermute": spec.backward_unpermute_fused} if route_backward else {})
    backward_builder = build_route_backward_graph if route_backward else build_backward_graph
    backward_graph = backward_builder(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        expert_weight_layout=spec.expert_weight_layout,
        shared_intermediate_size=spec.shared_intermediate_size,
        fuse_shared_expert=spec.backward_shared_expert,
        **backward_options,
    )
    backward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(backward_graph)
    backward_data = build_backward_config(
        backward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    return forward_graph, forward_data, backward_graph, backward_data


def _prepare_diagnostics(
    spec: MegaMoeSpec,
    forward_graph: ComputeGraph,
    forward_config: Any,
    backward_graph: ComputeGraph,
    backward_config: Any,
    device: Any,
) -> tuple[MegaKernelDiagnostics, MegaKernelDiagnostics]:
    """Bind forward/backward plans to the public diagnostics integration API."""
    topology = _build_task_values(spec)
    topology.rank_id = spec.rank_id
    route_abi = (spec.forward_fusion_mode != "legacy" or spec.backward_fusion_mode != "legacy"
                 or spec.forward_shared_expert or spec.backward_shared_expert)
    options = {
        "kernel_name": "mega_moe",
        "device": device,
        "rank": dist.get_rank() if dist.is_initialized() else spec.rank_id,
        **_mega_moe_owner_options(topology, num_cube_cores=spec.num_cube_cores),
    }
    forward_options = options.copy()
    backward_options = options.copy()
    if spec.forward_shared_expert or (spec.backward_shared_expert and spec.forward_unpermute_fused):
        forward_options["graph_adapter"] = lambda graph, config: bind_shared_diagnostic_ports(
            graph, config, forward=True,
        )
    if spec.backward_shared_expert:
        backward_options["graph_adapter"] = lambda graph, config: bind_shared_diagnostic_ports(
            graph, config, forward=False,
        )
    return (
        prepare_mega_kernel_diagnostics(
            forward_graph, forward_config,
            parameter_count=(35 if spec.forward_shared_expert or spec.backward_shared_expert
                             else (27 if route_abi else 16)), **forward_options,
        ),
        prepare_mega_kernel_diagnostics(
            backward_graph, backward_config,
            parameter_count=((47 if spec.backward_unpermute_fused else 46) if spec.backward_shared_expert
                             else (33 if route_abi else 20)), **backward_options,
        ),
    )


def build_mega_moe_plan(spec: MegaMoeSpec, device: Any) -> MegaMoePlan:
    """Build trimmed forward/backward runtime images for the selected fusion modes.

    Args:
        spec: Validated local-token and expert topology specification.
        device: NPU device receiving serialized descriptors and tiling tensors.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(spec)
    fwd_diagnostics, bwd_diagnostics = _prepare_diagnostics(
        spec, forward_graph, forward_config, backward_graph, backward_config, device,
    )
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": spec.local_experts,
        "num_cube_cores": spec.num_cube_cores,
    }
    return MegaMoePlan(
        spec=spec,
        route_slots=(torch.arange(spec.routed_slots, dtype=torch.float32, device=device).reshape(-1, 1)
                     if (spec.backward_permute_fused
                         or spec.forward_permute_fused and not spec.dynamic_tokens) else None),
        shared_forward_runtime=(
            _tensor_from_bytes(shared_tiling_bytes(spec, backward=False), device)
            if spec.forward_shared_expert or spec.backward_shared_expert else None
        ),
        shared_backward_runtime=(
            _tensor_from_bytes(shared_tiling_bytes(spec, backward=True), device)
            if spec.backward_shared_expert else None
        ),
        reuse_backward_dispatch=can_reuse_backward_dispatch(
            backward_config, spec.local_experts, spec.num_cube_cores,
        ),
        forward_graph=forward_graph,
        fwd_diagnostics=fwd_diagnostics,
        up_proj_tiling=_tensor_from_bytes(
            get_up_proj_tiling_bytes(forward_graph.get_op("up_proj").split_value, **gmm_options),
            device,
        ),
        swiglu_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_tiling_bytes(
                    forward_graph.get_op("swiglu").split_value,
                    intermediate_size=spec.intermediate_size,
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
        down_proj_tiling=_tensor_from_bytes(
            get_down_proj_tiling_bytes(forward_graph.get_op("down_proj").split_value, **gmm_options),
            device,
        ),
        backward_graph=backward_graph,
        bwd_diagnostics=bwd_diagnostics,
        act_grad_tiling=_tensor_from_bytes(
            get_act_grad_tiling_bytes(backward_graph.get_op("act_grad").split_value, **gmm_options),
            device,
        ),
        gate_grad_tiling=_tensor_from_bytes(
            get_gate_grad_tiling_bytes(backward_graph.get_op("gate_grad").split_value, **gmm_options),
            device,
        ),
        w1_grad_tiling=_tensor_from_bytes(
            get_w1_grad_tiling_bytes(backward_graph.get_op("w1_grad").split_value,
                                    expert_weight_layout=spec.expert_weight_layout, **gmm_options),
            device,
        ),
        w2_grad_tiling=_tensor_from_bytes(
            get_w2_grad_tiling_bytes(backward_graph.get_op("w2_grad").split_value,
                                    expert_weight_layout=spec.expert_weight_layout, **gmm_options),
            device,
        ),
        swiglu_grad_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_grad_tiling_bytes(
                    backward_graph.get_op("swiglu_grad").split_value,
                    intermediate_size=spec.intermediate_size,
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
    )

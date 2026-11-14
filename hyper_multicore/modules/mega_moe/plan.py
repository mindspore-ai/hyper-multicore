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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
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
    build_forward_graph,
)
from hyper_multicore.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from hyper_multicore.scheduler.config import (
    FAST_DEPENDENCY_POLL_INTERVAL_US,
    TaskSplitValue,
)
from hyper_multicore.scheduler.graph import ComputeGraph

from .diagnostics import _mega_moe_owner_options, _adapt_mega_moe_diagnostic_graph
from .probability import (
    PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD,
)

from .schedule_policy import (
    MegaMoeSchedulePolicy, compact_graph,
)
from .spec import MegaMoeSpec
from .tensor_dump import attach_mega_moe_l1_view_resolvers


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
    weight_grad_dtype: torch.dtype | None = None
    _recorded_streams: set = field(default_factory=set, repr=False, compare=False)

    @property
    def fwd_runtime_config(self) -> Any:
        """Return the original ordinary forward RuntimeConfig."""
        return self.fwd_diagnostics.runtime_config

    @property
    def bwd_runtime_config(self) -> Any:
        """Return the original ordinary backward RuntimeConfig."""
        return self.bwd_diagnostics.runtime_config


    def record_stream(self, runtime_config: Any, *, stream: Any = None) -> None:
        """Register immutable plan storage once per stream, preserving allocator lifetime."""
        if not isinstance(runtime_config, torch.Tensor) or runtime_config.device.type != "npu":
            return
        if stream is None:
            stream = torch.npu.current_stream(runtime_config.device)
        key = (runtime_config.device, stream.npu_stream)
        if key not in self._recorded_streams:
            tensors = [value for value in vars(self).values() if isinstance(value, torch.Tensor)]
            tensors.extend((self.fwd_runtime_config, self.bwd_runtime_config))
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
                    tensor.record_stream(stream)
            self._recorded_streams.add(key)
        if runtime_config is not self.fwd_runtime_config and runtime_config is not self.bwd_runtime_config:
            # Diagnostic buffers are per invocation and cannot share the ordinary cache.
            runtime_config.record_stream(stream)


# Keep the existing free-function call sites without another Python wrapper call.
record_plan_stream = MegaMoePlan.record_stream


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
    return TaskSplitValue(
        tp=1,
        ep=spec.ep_size,
        seq_size=spec.plan_tokens,
        all_expert_num=spec.num_experts,
        top_k=spec.top_k,
        dispatch_mode=spec.dispatch_mode,
    )


def _build_runtime_artifacts(
    spec: MegaMoeSpec, weight_grad_dtype: torch.dtype | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Build forward/backward graphs and their serialized RuntimeConfig objects."""
    task_values = _build_task_values(spec)
    forward_graph = build_forward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        clamp_after_activation=spec.clamp_after_activation,
        apply_probs_before_fc2=spec.apply_probs_before_fc2,
        expert_weight_layout=spec.expert_weight_layout,
    )
    compact = (spec.dispatch_mode == "push" and spec.intermediate_size < 2048
               and compact_graph(forward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule))
    forward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(forward_graph)
    forward_data = build_forward_config(
        forward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    backward_graph = build_backward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        clamp_after_activation=spec.clamp_after_activation,
        apply_probs_before_fc2=spec.apply_probs_before_fc2,
        expert_weight_layout=spec.expert_weight_layout,
    )
    if weight_grad_dtype == torch.float32:
        for name in ("w1_grad", "w2_grad"):
            backward_graph.get_op(name).outputs[0].dtype_size = 4
    if compact:
        compact_graph(backward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule)
    backward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(backward_graph)
    backward_data = build_backward_config(
        backward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    policy = MegaMoeSchedulePolicy(
        vector_ready_queue=bool(compact),
        pipeline_vector=bool(compact) and not spec.batch_experts,
        pipeline_cube=bool(compact) and not spec.batch_experts,
        batch_experts=spec.batch_experts,
        dependency_poll_interval=FAST_DEPENDENCY_POLL_INTERVAL_US,
    )
    for graph, config in ((forward_graph, forward_data), (backward_graph, backward_data)):
        policy.apply(config, graph, task_values, spec.num_cube_cores)
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
    options = {
        "kernel_name": "mega_moe",
        "graph_adapter": _adapt_mega_moe_diagnostic_graph,
        "device": device,
        "rank": dist.get_rank() if dist.is_initialized() else spec.rank_id,
        **_mega_moe_owner_options(topology, num_cube_cores=spec.num_cube_cores),
    }
    return (
        prepare_mega_kernel_diagnostics(
            forward_graph, forward_config,
            parameter_count=PROBABILITY_POSITION_FORWARD + 1 if spec.apply_probs_before_fc2 else 16, **options),
        prepare_mega_kernel_diagnostics(
            backward_graph, backward_config,
            parameter_count=PROBABILITY_POSITION_BACKWARD + 1 if spec.apply_probs_before_fc2 else 20, **options),
    )


def build_mega_moe_plan(
    spec: MegaMoeSpec, device: Any, *, weight_grad_dtype: torch.dtype | None = None,
) -> MegaMoePlan:
    """Build trimmed dense forward/backward runtime images without fusion slots.

    Args:
        spec: Validated local-token and expert topology specification.
        device: NPU device receiving serialized descriptors and tiling tensors.
        weight_grad_dtype: Keep partial expert weight gradients in FP32.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    if weight_grad_dtype not in (None, torch.float32):
        raise ValueError("Partial expert gradients support only the native dtype or FP32")
    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(
        spec, weight_grad_dtype)
    fwd_diagnostics, bwd_diagnostics = _prepare_diagnostics(
        spec, forward_graph, forward_config, backward_graph, backward_config, device,
    )
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": (1 if not spec.batch_experts and hasattr(forward_graph.get_op("dispatch"), "compact_lanes")
                       else spec.local_experts),
        "num_cube_cores": spec.num_cube_cores,
    }
    return MegaMoePlan(
        spec=spec,
        weight_grad_dtype=weight_grad_dtype,
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

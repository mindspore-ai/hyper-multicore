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
"""Dynamic task-view resolution for MegaMoe L1 Tensor dumps."""

from __future__ import annotations

from math import prod
from typing import Any

from hyper_multicore.tasks.gmm import GroupedMatmulTaskMode
from hyper_multicore.scheduler.config import (
    TaskType,
)
from hyper_multicore.diagnostics.dump.l1 import L1TensorView, L1TensorViewContext

from .probability import PROBABILITY_DTYPE_BYTES, probability_capture_plane


def _group_counts(context: L1TensorViewContext) -> tuple[int, int, int]:
    """Resolve this task's EP-local Expert token range from cumulative routing counts."""
    dynamic_data = context.runtime_config.dynamic_data
    group_count = int(dynamic_data.dynamic_group_size)
    if group_count <= 0:
        raise ValueError(f"MegaMoe L1 dump requires a positive dynamic group size, got {group_count}")
    group_position = int(dynamic_data.dynamic_input_position)
    cumulative = context.parameter_cpu(group_position).reshape(-1).tolist()
    if len(cumulative) < group_count:
        raise ValueError(
            f"MegaMoe L1 group list is too short: expected={group_count}, got={len(cumulative)}"
        )
    owner_id = context.owner_id
    if owner_id is None:
        raise ValueError(f"MegaMoe L1 stage {context.stage!r} has no resolved expert owner")
    # Export rank is job-global; Expert IDs and routing offsets belong to this EP group.
    owner_context = context.owner_context
    ep_rank = context.rank if owner_context is None else owner_context.topology.rank_id
    local_owner = owner_id - ep_rank * group_count
    if not 0 <= local_owner < group_count:
        raise ValueError(
            f"MegaMoe L1 owner {owner_id} is outside EP rank {ep_rank} local expert range"
        )
    end = int(cumulative[local_owner])
    start = 0 if local_owner == 0 else int(cumulative[local_owner - 1])
    if end < start:
        raise ValueError(f"MegaMoe L1 group list is not cumulative at expert {local_owner}")
    return local_owner, start, end


def _resolve_grouped_matmul(context: L1TensorViewContext) -> L1TensorView:
    """Resolve a grouped-matmul operand without capturing collaborative partial outputs."""
    if context.io == "output":
        raise ValueError(
            f"MegaMoe L1 cannot snapshot {context.stage!r} output atomically across Cube cores; "
            "select the downstream stage input after its dependency instead"
        )
    if context.task_desc.extra_value_1 in (GroupedMatmulTaskMode.BATCHED, GroupedMatmulTaskMode.BATCHED_READY):
        shape = list(context.tensor_spec.shape)
        if context.tensor_desc.dynamic_shape:
            data = context.runtime_config.dynamic_data
            counts = context.parameter_cpu(int(data.dynamic_input_position)).reshape(-1)
            shape[0] = int(counts[int(data.dynamic_group_size) - 1])
        return context.make_view(shape, source_offset_bytes=0, note="batched GMM input across local experts")
    local_owner, start, end = _group_counts(context)
    shape = list(context.tensor_spec.shape)
    if context.tensor_desc.dynamic_shape:
        shape[0] = end - start
        source_offset_bytes = start * int(shape[1]) * int(context.tensor_desc.data_type)
        return context.make_view(
            shape,
            source_offset_bytes=source_offset_bytes,
            note="expert-local dynamic GMM operand captured after dependency",
        )
    group_count = int(context.runtime_config.dynamic_data.dynamic_group_size)
    if len(shape) >= 3 and shape[0] == group_count:
        expert_elements = prod(shape[1:])
        return context.make_view(
            shape[1:],
            source_offset_bytes=local_owner * expert_elements * int(context.tensor_desc.data_type),
            note="expert-local weight slice",
        )
    return context.default_view


def _swiglu_rows(context: L1TensorViewContext) -> tuple[int, int]:
    """Resolve the first contiguous tile owned by this vector task."""
    _, start, end = _group_counts(context)
    groups = int(context.runtime_config.dynamic_data.dynamic_group_size)
    lanes = int(context.task_desc.task_split_num) // groups
    if lanes <= 0:
        raise ValueError("MegaMoe L1 SwiGLU task count per expert must be positive")
    row_start = int(context.task_desc.task_index) % lanes * int(context.task_desc.task_split_value)
    rows = min(int(context.task_desc.task_split_value), max(end - start - row_start, 0))
    return start + row_start, rows


def _resolve_swiglu(context: L1TensorViewContext) -> L1TensorView:
    """Resolve a SwiGLU task's contiguous expert-local token rows."""
    source_row, rows = _swiglu_rows(context)
    shape = list(context.tensor_spec.shape)
    shape[0] = rows
    return context.make_view(
        shape, source_offset_bytes=source_row * int(shape[1]) * int(context.tensor_desc.data_type) if rows else 0,
        note="expert-local SwiGLU tile captured at its task boundary",
    )


def _alltoall_rows(context: L1TensorViewContext) -> tuple[int, int, int]:
    """Resolve the local payload span independently of the selected operand."""
    is_pull = int(context.task_desc.task_type) == TaskType.TASK_SHMEM_GET_MEM
    payload = context.operator.outputs[0] if is_pull else context.operator.inputs[1]
    hidden_size = int(payload.shape[1])
    expert_count = int(context.operator.inputs[3].shape[0])
    tasks_per_expert = int(context.task_desc.task_split_num) // expert_count
    if tasks_per_expert <= 0:
        raise ValueError("MegaMoe L1 AllToAll task count per expert must be positive")
    expert = int(context.task_desc.task_index) // tasks_per_expert
    tile = int(context.task_desc.task_index) % tasks_per_expert
    sizes = context.parameter_cpu(int(context.operator.inputs[3].param_position)).reshape(-1)
    offsets = context.parameter_cpu(int(context.operator.inputs[0 if is_pull else 2].param_position)).reshape(-1)
    tile_start = tile * int(context.task_desc.task_split_value) * hidden_size
    count = max(0, min(int(context.task_desc.task_split_value) * hidden_size, int(sizes[expert]) - tile_start))
    source_offset = int(offsets[expert]) + tile_start
    if count % hidden_size or source_offset % hidden_size:
        raise ValueError("MegaMoe L1 AllToAll payload and offset must be row-aligned")
    return source_offset // hidden_size, count // hidden_size, hidden_size


def _resolve_alltoall(context: L1TensorViewContext) -> L1TensorView:
    """Resolve local communication operands while rejecting remote-address reads."""
    is_pull = int(context.task_desc.task_type) == TaskType.TASK_SHMEM_GET_MEM
    if is_pull and context.io == "input" and context.index == 1:
        raise ValueError("MegaMoe pull sources belong to a remote rank; select the local dispatch output instead")
    if context.io == "output" and not is_pull:
        raise ValueError(
            f"MegaMoe L1 stage {context.stage!r} output belongs to a possibly remote rank; "
            "select its local consumer input instead"
        )
    if context.io == "input" and context.index != 1:
        return context.make_view((1,), note="task-local AllToAll metadata scalar")
    source_row, rows, hidden_size = _alltoall_rows(context)
    return context.make_view(
        (rows, hidden_size),
        source_offset_bytes=source_row * hidden_size * int(context.tensor_desc.data_type) if rows else 0,
        note="local pull receive tile" if is_pull else "local AllToAll send tile before remote transfer",
    )


def _resolve_probability_workspace(context: L1TensorViewContext) -> L1TensorView:
    """Capture only the stable local scalar slice used or produced by this task."""
    task_type = int(context.task_desc.task_type)
    if task_type in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        source_row, rows = _swiglu_rows(context)
    else:
        source_row, rows, _ = _alltoall_rows(context)
    plane = probability_capture_plane(context.operator)
    stride = int(context.tensor_spec.shape[1])
    return context.make_view(
        (rows,), source_offset_bytes=(plane * stride + source_row) * PROBABILITY_DTYPE_BYTES if rows else 0,
        note="task-local routing probability or probability-gradient slice",
    )


def resolve_mega_moe_l1_tensor_view(context: Any) -> L1TensorView:
    """Resolve exact task-local Tensor views for MegaMoe dynamic operators.

    Args:
        context: Selected invocation or operand context supplied by the diagnostic framework.
    """
    if not isinstance(context, L1TensorViewContext):
        raise TypeError(
            f"MegaMoe L1 view context must be L1TensorViewContext, got {type(context).__name__}"
        )
    if context.tensor_spec.name == "probability_workspace":
        return _resolve_probability_workspace(context)
    task_type = int(context.task_desc.task_type)
    if task_type == TaskType.TASK_GROUPED_MATMUL:
        return _resolve_grouped_matmul(context)
    if task_type in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        return _resolve_swiglu(context)
    if task_type in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
        return _resolve_alltoall(context)
    return context.default_view


def attach_mega_moe_l1_view_resolvers(graph: Any) -> None:
    """Attach one shared dynamic-view resolver to every MegaMoe graph stage.

    Args:
        graph: Business compute graph whose scheduling storage remains unchanged.
    """
    for operator in graph.topological_sort():
        operator.diagnostic_view_resolver = resolve_mega_moe_l1_tensor_view

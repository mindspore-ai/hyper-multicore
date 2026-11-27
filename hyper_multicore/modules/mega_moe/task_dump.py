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
"""Dynamic task-view resolution for MegaMoe Task Tensor dumps."""

from __future__ import annotations

from functools import cache
from itertools import groupby
import logging
from math import prod
from typing import Any

from hyper_multicore.diagnostics.common.graph import operator_stage_name
from hyper_multicore.diagnostics.dumper import TaskDumpTensorView, TaskDumpTensorViewContext, UnsupportedTaskDumpPort
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.graph import OpType
from hyper_multicore.tasks.gmm import GroupedMatmulTaskMode

from .probability import PROBABILITY_DTYPE_BYTES, probability_capture_plane


@cache
def _warn_compact_task_dump() -> None:
    """Log compact capture limitations on first use in this process."""
    logging.getLogger(__name__).warning(
        "MegaMoE compact Task dump: communication ranges can differ from the executed task, "
        "and SwiGLU captures only the first tile of each persistent task."
    )


def _group_counts(context: TaskDumpTensorViewContext) -> tuple[int, int, int]:
    """Resolve this task's EP-local Expert token range from cumulative routing counts."""
    dynamic_data = context.diagnostic_runtime_config.dynamic_data
    group_count = int(dynamic_data.dynamic_group_size)
    if group_count <= 0:
        raise ValueError(f"MegaMoe Task dump requires a positive dynamic group size, got {group_count}")
    group_position = int(dynamic_data.dynamic_input_position)
    cumulative = context.parameter_cpu(group_position).reshape(-1).tolist()
    if len(cumulative) < group_count:
        raise ValueError(
            f"MegaMoe Task group list is too short: expected={group_count}, got={len(cumulative)}"
        )
    owner_id = context.owner_id
    if owner_id is None:
        raise ValueError(f"MegaMoe Task stage {context.stage!r} has no resolved expert owner")
    # Routing offsets use the EP schedule slot even when the exported owner is a borrowed expert.
    owner_context = context.owner_context
    ep_rank = context.rank if owner_context is None else owner_context.topology.rank_id
    local_owner = owner_id - ep_rank * group_count
    if not 0 <= local_owner < group_count:
        raise ValueError(
            f"MegaMoe Task owner {owner_id} is outside EP rank {ep_rank} local expert range"
        )
    end = int(cumulative[local_owner])
    start = 0 if local_owner == 0 else int(cumulative[local_owner - 1])
    if end < start:
        raise ValueError(f"MegaMoe Task group list is not cumulative at expert {local_owner}")
    return local_owner, start, end


def _check_batched_activation_ready(context: TaskDumpTensorViewContext, groups: int) -> None:
    """Check whether this activation still has per-expert writers at the BEFORE boundary."""
    if groups <= 1 or context.task_desc.extra_value_1 != GroupedMatmulTaskMode.BATCHED:
        return
    for producer in context.operator.predecessors:
        for index, output in enumerate(producer.outputs):
            if output.param_position != context.tensor_spec.param_position:
                continue
            port = f"{operator_stage_name(producer)}.output[{index}]"
            hint = f"select Kernel dump of {port} for the full activation"
            if producer.op_type == OpType.SWIGLU:
                hint += f", or Task dump of {port} for its after snapshot"
            raise UnsupportedTaskDumpPort(
                f"{context.stage}.input[{context.index}] can still be written by other experts "
                f"at the batched GMM before snapshot; {hint}"
            )


def _batched_gmm_input(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Resolve ready activation rows or the physical weight bank and its assigned slots."""
    shape = list(context.tensor_spec.shape)
    data = context.diagnostic_runtime_config.dynamic_data
    groups = int(data.dynamic_group_size)
    note = "batched GMM input across local experts"
    if context.tensor_desc.dynamic_shape:
        _check_batched_activation_ready(context, groups)
        counts = context.parameter_cpu(int(data.dynamic_input_position)).reshape(-1)
        shape[0] = int(counts[groups - 1])
    elif context.owner_ids is not None and len(shape) >= 3 and shape[0] == groups:
        rank = context.owner_context.topology.rank_id
        owners = context.owner_ids[rank * groups:(rank + 1) * groups]
        if None in owners:
            stride = prod(shape[1:])
            ranges = []
            start = 0
            for assigned, run in groupby(owners, key=lambda owner: owner is not None):
                end = start + sum(1 for _ in run) * stride
                if assigned and start < end:
                    ranges.append((start, end))
                start = end
            return context.make_view(shape, source_offset_bytes=0, note=note, coverage="valid_elements",
                                     valid_ranges=tuple(ranges))
    return context.make_view(shape, source_offset_bytes=0, note=note)


def _resolve_grouped_matmul(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Resolve a grouped-matmul operand without dumping collaborative partial outputs."""
    if context.io == "output":
        raise UnsupportedTaskDumpPort(
            f"MegaMoe Task cannot snapshot {context.stage!r} output atomically across Cube cores; "
            "select the downstream stage input after its dependency instead"
        )
    if context.task_desc.extra_value_1 in (GroupedMatmulTaskMode.BATCHED, GroupedMatmulTaskMode.BATCHED_READY):
        return _batched_gmm_input(context)
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
    group_count = int(context.diagnostic_runtime_config.dynamic_data.dynamic_group_size)
    if len(shape) >= 3 and shape[0] == group_count:
        expert_elements = prod(shape[1:])
        return context.make_view(
            shape[1:],
            source_offset_bytes=local_owner * expert_elements * int(context.tensor_desc.data_type),
            note="expert-local weight slice",
        )
    return context.default_view


def _swiglu_rows(context: TaskDumpTensorViewContext) -> tuple[int, int, str]:
    """Resolve the first contiguous tile owned by this vector task."""
    _, start, end = _group_counts(context)
    groups = int(context.diagnostic_runtime_config.dynamic_data.dynamic_group_size)
    lanes = int(context.task_desc.task_split_num) // groups
    if lanes <= 0:
        raise ValueError("MegaMoe Task SwiGLU task count per expert must be positive")
    row_start = int(context.task_desc.task_index) % lanes * int(context.task_desc.task_split_value)
    rows = min(int(context.task_desc.task_split_value), max(end - start - row_start, 0))
    coverage = "first_tile" if hasattr(context.operator, "compact_lanes") else "task_view"
    return start + row_start, rows, coverage


def _resolve_swiglu(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Resolve a SwiGLU task's contiguous expert-local token rows."""
    source_row, rows, coverage = _swiglu_rows(context)
    shape = list(context.tensor_spec.shape)
    shape[0] = rows
    return context.make_view(
        shape, source_offset_bytes=source_row * int(shape[1]) * int(context.tensor_desc.data_type) if rows else 0,
        note="expert-local SwiGLU tile captured at its task boundary",
        coverage=coverage,
    )


def _alltoall_rows(context: TaskDumpTensorViewContext) -> tuple[int, int, int, str]:
    """Resolve the local payload span independently of the selected operand."""
    is_pull = int(context.task_desc.task_type) == TaskType.TASK_SHMEM_GET_MEM
    payload = context.operator.outputs[0] if is_pull else context.operator.inputs[1]
    hidden_size = int(payload.shape[1])
    expert_count = int(context.operator.inputs[3].shape[0])
    tasks_per_expert = int(context.task_desc.task_split_num) // expert_count
    if tasks_per_expert <= 0:
        raise ValueError("MegaMoe Task AllToAll task count per expert must be positive")
    expert = int(context.task_desc.task_index) // tasks_per_expert
    tile = int(context.task_desc.task_index) % tasks_per_expert
    sizes = context.parameter_cpu(int(context.operator.inputs[3].param_position)).reshape(-1)
    offsets = context.parameter_cpu(int(context.operator.inputs[0 if is_pull else 2].param_position)).reshape(-1)
    tile_start = tile * int(context.task_desc.task_split_value) * hidden_size
    count = max(0, min(int(context.task_desc.task_split_value) * hidden_size, int(sizes[expert]) - tile_start))
    source_offset = int(offsets[expert]) + tile_start
    if count % hidden_size or source_offset % hidden_size:
        raise ValueError("MegaMoe Task AllToAll payload and offset must be row-aligned")
    coverage = "range_unknown" if hasattr(context.operator, "compact_lanes") else "task_view"
    return source_offset // hidden_size, count // hidden_size, hidden_size, coverage


def _resolve_alltoall(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Resolve local communication operands while rejecting remote-address reads."""
    is_pull = int(context.task_desc.task_type) == TaskType.TASK_SHMEM_GET_MEM
    if is_pull and context.io == "input" and context.index == 1:
        raise UnsupportedTaskDumpPort(
            "MegaMoe pull sources belong to a remote rank; select the local dispatch output instead"
        )
    if context.io == "output" and not is_pull:
        raise UnsupportedTaskDumpPort(
            f"MegaMoe Task stage {context.stage!r} output belongs to a possibly remote rank; "
            "select its local consumer input instead"
        )
    if context.io == "input" and context.index != 1:
        return context.make_view((1,), note="task-local AllToAll metadata scalar")
    source_row, rows, hidden_size, coverage = _alltoall_rows(context)
    return context.make_view(
        (rows, hidden_size),
        source_offset_bytes=source_row * hidden_size * int(context.tensor_desc.data_type) if rows else 0,
        note="local pull receive tile" if is_pull else "local AllToAll send tile before remote transfer",
        coverage=coverage,
    )


def _resolve_probability_workspace(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Capture only the stable local scalar slice used or produced by this task."""
    task_type = int(context.task_desc.task_type)
    if task_type in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        source_row, rows, coverage = _swiglu_rows(context)
    else:
        source_row, rows, _, coverage = _alltoall_rows(context)
    plane = probability_capture_plane(context.operator)
    stride = int(context.tensor_spec.shape[1])
    return context.make_view(
        (rows,), source_offset_bytes=(plane * stride + source_row) * PROBABILITY_DTYPE_BYTES if rows else 0,
        note="task-local routing probability or probability-gradient slice",
        coverage=coverage,
    )


def resolve_mega_moe_task_dump_tensor_view(context: Any) -> TaskDumpTensorView:
    """Resolve task-local Tensor views for MegaMoe dynamic operators.

    Args:
        context: Selected invocation or operand context supplied by the diagnostic framework.
    """
    if not isinstance(context, TaskDumpTensorViewContext):
        raise TypeError(
            f"MegaMoe Task view context must be TaskDumpTensorViewContext, got {type(context).__name__}"
        )
    if hasattr(context.operator, "compact_lanes"):
        _warn_compact_task_dump()
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


def attach_view_resolver(graph: Any) -> None:
    """Attach one shared dynamic-view resolver to every MegaMoe graph stage.

    Args:
        graph: Business compute graph whose scheduling storage remains unchanged.
    """
    for operator in graph.topological_sort():
        operator.task_dump_tensor_view_resolver = resolve_mega_moe_task_dump_tensor_view

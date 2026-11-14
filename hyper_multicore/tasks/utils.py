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
"""
Shared utilities for all FillConfig implementations.

Counter helpers (advance_tsv_*) mirror the C++ tsv state advancement.
add_terminate / add_dynamic_data are called from gen_runtime_data after
the main fill loop.
"""
from typing import Any
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.scheduler.config import (
    TaskSplitValue,
    TaskAiCoreType,
    TensorDescC, TaskDescC, RuntimeConfigC,
    TaskType, DynamicType,
    MAX_TENSOR_DIMS,
)


# ── TensorDescC quick constructor ─────────────────────────────────────────────

def make_tensor_desc(
    tensor_type:     int = 0,
    data_type:       int = 2,
    input_position:  int = 0,
    base_ptr_offset: int = 0,
    transpose_flag:  int = 0,
    dynamic_shape:   int = 0,
    dynamic_dim:     int = 0,
    shape:           list = None,
) -> TensorDescC:
    """Build a TensorDescC with the given fields."""
    t = TensorDescC()
    t.tensor_type     = tensor_type
    t.data_type       = data_type
    t.input_position  = input_position
    t.base_ptr_offset = base_ptr_offset
    t.transpose_flag  = transpose_flag
    t.dynamic_shape   = dynamic_shape
    t.dynamic_dim     = dynamic_dim
    if shape:
        for k, v in enumerate(shape[:MAX_TENSOR_DIMS]):
            t.dim[k] = v
    return t


# ── tsv counter helpers ───────────────────────────────────────────────────────

def advance_tsv_vector(tsv, task_num: int, event_group_size: int) -> None:
    """Advance counters after a vector op with event grouping."""
    tsv.pre_task_num        += task_num
    tsv.pre_vector_task_num += task_num
    tsv.pre_pre_event_num    = tsv.pre_event_num
    tsv.pre_event_num       += event_group_size


def advance_tsv_cube(tsv, task_num: int, event_group_size: int) -> None:
    """Advance counters after a cube op with event grouping."""
    tsv.pre_task_num      += task_num
    tsv.pre_cube_task_num += task_num
    tsv.pre_pre_event_num  = tsv.pre_event_num
    tsv.pre_event_num     += event_group_size


def advance_tsv_cube_only(tsv, task_num: int) -> None:
    """Advance only task/cube counters (no event update). Used by bwd GMM1."""
    tsv.pre_task_num      += task_num
    tsv.pre_cube_task_num += task_num


def advance_tsv_vector_only(tsv, task_num: int) -> None:
    """Advance only task/vector counters (no event update). Used by bwd combine."""
    tsv.pre_task_num        += task_num
    tsv.pre_vector_task_num += task_num


# ── Post-fill utilities ───────────────────────────────────────────────────────

def add_terminate(cfg: RuntimeConfigC, tsv, trigger_count: int) -> None:
    """
    Append a terminate task to cfg.

    trigger_count:
      forward:  combine_op.task_num // tsv.ep * tsv.ep
      backward: w2_grad_op.task_num + w1_grad_op.task_num + combine_op.task_num // tsv.ep * tsv.ep
    """
    cfg.all_event_num_triggers[tsv.all_event_num] = trigger_count
    task = TaskDescC()
    task.task_type       = TaskType.TASK_TERMINATE
    task.dependent_event = tsv.pre_pre_event_num + 1
    task.trigger_event   = tsv.pre_event_num + 1
    cfg.all_event_num_triggers[task.trigger_event] = 1

    cfg.all_tasks[tsv.pre_task_num]                 = task
    cfg.vector_task_indices[tsv.pre_vector_task_num] = tsv.pre_task_num
    cfg.task_index_num[1] += 1


def add_dynamic_data(cfg: RuntimeConfigC, tsv, dynamic_input_position: int) -> None:
    """
    Write the dynamic data record into cfg.

    dynamic_input_position: forward = 6, backward = 19.
    """
    cfg.dynamic_data.dynamic_type           = DynamicType.DYNAMIC_DSV3_MOE
    cfg.dynamic_data.dynamic_input_position = dynamic_input_position
    cfg.dynamic_data.dynamic_group_size     = tsv.single_rank_expert_num
    cfg.dynamic_data.dynamic_max_seq_len    = -1


def fill_expert_compute_tasks(cfg: Any, op: OperatorNode, tsv: TaskSplitValue,
                        kind: int, workers: int, first: bool, indices: list[int], *, cube: bool) -> None:
    for index, record_index in enumerate(indices):
        expert = index // workers
        task = TaskDescC()
        task.task_type = kind
        task.task_aicore_type = (TaskAiCoreType.TASK_AICORE_CUBE if cube else TaskAiCoreType.TASK_AICORE_VECTOR)
        task.num_inputs, task.num_outputs = len(op.inputs), len(op.outputs)
        for slot, spec in enumerate(op.inputs):
            task.inputs[slot] = make_tensor_desc(input_position=spec.param_position,
                                                data_type=spec.dtype_size, shape=spec.shape)
        for slot, spec in enumerate(op.outputs):
            task.outputs[slot] = make_tensor_desc(input_position=spec.param_position,
                                                 data_type=spec.dtype_size, shape=spec.shape)
        rank_offset = tsv.rank_id * tsv.single_rank_expert_num if first else 0
        task.dependent_event = tsv.pre_pre_event_num + rank_offset + expert + 1
        task.trigger_event = tsv.pre_event_num + expert + 1
        cfg.all_event_num_triggers[task.trigger_event] = workers
        task.task_index, task.task_split_num = index, len(indices)
        task.task_split_value, task.extra_value_0 = 1, record_index
        task.tiling_data_position = 0xFFFFFFFF
        task_index = tsv.pre_task_num + index
        cfg.all_tasks[task_index] = task
        queue = cfg.cube_task_indices if cube else cfg.vector_task_indices
        offset = tsv.pre_cube_task_num if cube else tsv.pre_vector_task_num
        queue[offset + index] = task_index
    cfg.task_index_num[0 if cube else 1] += len(indices)
    advance = advance_tsv_cube if cube else advance_tsv_vector
    advance(tsv, len(indices), tsv.single_rank_expert_num)

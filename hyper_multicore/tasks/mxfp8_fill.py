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
"""MX task descriptor filling; graph dependencies and scheduling stay in the module."""
from dataclasses import dataclass
from typing import Any
from hyper_multicore.scheduler.config import TaskDescC, TaskAiCoreType, TaskSplitValue
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.mxfp8 import MX_GMM_TASK_TYPE, make_mx_gmm_descriptor
from hyper_multicore.tasks.mxfp8_quant import MxQuantDescriptor
from hyper_multicore.tasks.utils import advance_tsv_cube, advance_tsv_vector, make_tensor_desc


def _fill_compute_stage(cfg: Any, op: OperatorNode, tsv: TaskSplitValue,
                        kind: int, workers: int, first: bool, indices: list[int]) -> None:
    cube = kind == MX_GMM_TASK_TYPE
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

@dataclass
class _GmmStage:
    layout: str
    m: int
    n: int
    k: int
    x_scale: int
    w_scale: int
    workers: int
    records: list

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(make_mx_gmm_descriptor(
                    layout=self.layout, m=self.m, n=self.n, k=self.k,
                    x_scale_position=self.x_scale, w_scale_position=self.w_scale,
                    cube_count=self.workers, cube_index=worker, group_list_offset=expert * 8,
                    w_offset=0 if self.layout == 'TN' else expert * self.n * self.k,
                    w_scale_offset=0 if self.layout == 'TN' else expert * self.n * self.k // 32,
                    y_offset=expert * self.m * self.n * 2 if self.layout == 'TN' else 0,
                ))
        _fill_compute_stage(cfg, op, tsv, MX_GMM_TASK_TYPE, self.workers, False, indices)

@dataclass
class _QuantStage:
    kind: int
    columns: int
    workers: int
    first: bool
    records: list

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(MxQuantDescriptor(
                    columns=self.columns, expert_index=expert, worker_index=worker,
                    worker_count=self.workers, task_type=self.kind,
                ))
        _fill_compute_stage(cfg, op, tsv, self.kind, self.workers, self.first, indices)


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
"""Compare the optimized writer to independently constructed scalar ABI records."""
import ctypes
from types import SimpleNamespace

import pytest

from hyper_multicore.scheduler.config import TaskAiCoreType, TaskDescC, TaskType, TensorDescC
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.tasks.alltoall import AllToAllFillConfig, AllToAllType
from hyper_multicore.tasks.gmm import GmmFillConfig
from hyper_multicore.tasks.task_base import _TASK_DTYPE


@pytest.mark.parametrize("layout", [TaskDescC, TensorDescC])
def test_bulk_field_offsets(layout):
    dtype = _TASK_DTYPE if layout is TaskDescC else _TASK_DTYPE["inputs"].base
    assert dtype.itemsize == ctypes.sizeof(layout)
    for name, field in layout._fields_:  # pylint: disable=protected-access
        field_dtype, offset = dtype.fields[name]
        assert offset == getattr(layout, name).offset
        assert field_dtype.itemsize == ctypes.sizeof(field)


def _scalar_task(kind, index, n):
    """Construct one independent scalar ABI reference descriptor."""
    gmm = kind == "gmm"
    expert = index // 2
    task = TaskDescC()
    task.task_type = (TaskType.TASK_GROUPED_MATMUL if gmm else
                      TaskType.TASK_SHMEM_GET_MEM if kind == "pull" else TaskType.TASK_SHMEM_PUT_MEM_SIGNAL)
    task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
    task.num_inputs, task.num_outputs = n, 1
    task.task_index, task.task_split_num, task.task_split_value = index, 4, 64
    task.tiling_data_position = 9 if gmm else 0xffffffff
    task.dependent_event = 4 + expert if gmm else 4 if kind == "combine" else 3
    task.trigger_event = 15 if kind == "combine" else 9 if kind == "pull" else 8 + expert
    for pos in range(n):
        item = task.inputs[pos]
        item.input_position, item.data_type = pos + 1, 2
        if gmm:
            item.tensor_type = int(pos < n - 1)
            item.dynamic_shape = int(pos == 0)
            item.dim[0], item.dim[1] = 4, 16
            if pos == 0:
                item.base_ptr_offset = expert * 4096 * 16
        elif pos == 1:
            item.tensor_type = item.dynamic_shape = 1
        else:
            item.base_ptr_offset = expert
    out = task.outputs[0]
    out.input_position, out.data_type = n + 1, 2
    out.tensor_type = out.dynamic_shape = 1
    if gmm:
        out.dim[0], out.dim[1] = 4, 16
        out.base_ptr_offset = expert * 4096 * 16
    return task


@pytest.mark.parametrize("kind", ["gmm", "push", "pull", "combine"])
def test_bulk_matches_scalar_abi(kind):
    """Compare every emitted descriptor byte with the independent scalar writer."""
    gmm = kind == "gmm"
    cfg = allocate_runtime_config(16)
    state = SimpleNamespace(pre_task_num=2, pre_cube_task_num=1, pre_vector_task_num=1,
                            all_expert_num=2, single_rank_expert_num=1, rank_id=1, ep=2,
                            pre_pre_event_num=3, pre_event_num=7, all_event_num=15, dispatch_mode=kind)
    operand = SimpleNamespace(dtype_size=2, tensor_type=1, is_dynamic=True, transpose=False, shape=(4, 16))
    n = 3 if gmm else 4
    op = SimpleNamespace(task_num=4, split_value=64, tiling_position=9, inputs=[operand]*n,
                         outputs=[operand], param_positions=list(range(1,n+2)))
    if gmm:
        GmmFillConfig(num_cube_cores=2, advance="cube_only").fill(cfg, op, state)
    else:
        moe_type = AllToAllType.COMBINE if kind == "combine" else AllToAllType.DISPATCH
        AllToAllFillConfig(moe_type=moe_type, advance="vector_only").fill(cfg, op, state)

    for index in range(4):
        task = _scalar_task(kind, index, n)
        assert bytes(cfg.all_tasks[2+index]) == bytes(task)
        queue = cfg.cube_task_indices if gmm else cfg.vector_task_indices
        assert queue[1+index] == 2+index

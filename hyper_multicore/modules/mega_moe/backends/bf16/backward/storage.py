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
"""Check the physical backward schedule before sharing transient storage."""

import numpy as np

from hyper_multicore.scheduler.config import RuntimeConfigC, TaskDescC, TaskType


_DISPATCH = 0
_WEIGHT2_GRAD = 6
_ACTIVATION_GRAD = 8
_SWIGLU_GRAD = 10
_INPUT_GRAD = 12


def _worker_read_order(outputs, task_indices, task_types, num_experts, num_cores):
    """Each worker must run exactly one W2, activation and input gradient per expert, in order."""
    positions = []
    for output in (_WEIGHT2_GRAD, _ACTIVATION_GRAD, _INPUT_GRAD):
        index = np.flatnonzero(outputs == output)
        if len(index) != num_experts * num_cores or np.any(task_types[index] != TaskType.TASK_GROUPED_MATMUL):
            return False
        experts = task_indices[index] // num_cores
        if np.any(experts >= num_experts):
            return False
        # Worker assignment is round-robin by queue position, independently of task_index.
        slot = experts * num_cores + index % num_cores
        by_slot = np.full(num_experts * num_cores, -1, dtype=np.intp)
        by_slot[slot] = index
        if np.any(by_slot < 0):
            return False
        positions.append(by_slot)
    w2, activation, dx = positions[0], positions[1], positions[2]
    return bool(np.all(w2 < activation) and np.all(activation < dx))


def _joined_events(tasks, experts, task_ids, producers, thresholds, num_experts):
    """Return the single fully joined completion event for each expert or None."""
    if np.size(task_ids) == 0 or np.any(experts >= num_experts):
        return None
    counts = np.bincount(experts, minlength=num_experts)
    if np.any(counts == 0):
        return None
    triggers = tasks["trigger_event"][task_ids]
    events = np.empty(num_experts, dtype=triggers.dtype)
    events[experts] = triggers
    if (np.any(events[experts] != triggers) or np.any(events >= len(thresholds))
            or np.any(producers[events] != counts) or np.any(thresholds[events] != counts)):
        return None
    return events


def can_reuse_backward_dispatch(cfg: RuntimeConfigC, num_experts: int, num_cores: int) -> bool:
    """Validate the physical reader order and event joins before aliasing dY storage as dX."""
    if cfg.task_index_num[2] or num_experts <= 0 or num_cores <= 0:
        return False
    tasks = np.frombuffer(cfg.all_tasks, dtype=np.dtype(TaskDescC))
    cube = np.ctypeslib.as_array(cfg.cube_task_indices)[:cfg.task_index_num[0]]
    vector = np.ctypeslib.as_array(cfg.vector_task_indices)[:cfg.task_index_num[1]]
    active = np.concatenate((cube, vector))
    # Read scalar fields from the wire layout. Indexing the record array itself
    # would copy whole task descriptors, including unused input and output ports.
    out = tasks["outputs"][:, 0]["input_position"]
    kinds = tasks["task_type"]
    allowed = (kinds[active] == TaskType.TASK_GROUPED_MATMUL) & np.isin(out[active], (_WEIGHT2_GRAD, _ACTIVATION_GRAD))
    other = active[~allowed]
    ports = tasks["inputs"]["input_position"]
    for port in range(ports.shape[1]):
        if np.any((tasks["num_inputs"][other] > port) & (ports[other, port] == _DISPATCH)):
            return False

    indices = tasks["task_index"]
    cube_out = out[cube]
    if not _worker_read_order(cube_out, indices[cube], kinds[cube], num_experts, num_cores):
        return False
    acts = cube[cube_out == _ACTIVATION_GRAD]
    gates = cube[cube_out == _INPUT_GRAD]
    swiglu = vector[kinds[vector] == TaskType.TASK_SWI_GLU_GRAD]
    splits = tasks["task_split_num"][swiglu]
    if np.any(splits < num_experts) or np.any(splits % num_experts) or np.any(out[swiglu] != _SWIGLU_GRAD):
        return False
    experts = indices[swiglu] // (splits // num_experts)
    thresholds = np.ctypeslib.as_array(cfg.all_event_num_triggers)
    triggers = tasks["trigger_event"][active]
    if np.any(triggers >= len(thresholds)):
        return False
    producers = np.bincount(triggers, minlength=len(thresholds))
    act_event = _joined_events(tasks, indices[acts] // num_cores, acts, producers, thresholds, num_experts)
    swiglu_event = _joined_events(tasks, experts, swiglu, producers, thresholds, num_experts)
    if act_event is None or swiglu_event is None:
        return False
    dependency = tasks["dependent_event"]
    return bool(np.all(dependency[swiglu] == act_event[experts])
                and np.all(dependency[gates] == swiglu_event[indices[gates] // num_cores]))

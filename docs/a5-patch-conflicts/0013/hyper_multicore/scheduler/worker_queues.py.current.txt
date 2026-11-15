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
"""Explicit per-worker queues; policies and operator stage names stay outside runtime."""
from collections import Counter
import ctypes
from functools import lru_cache

from .config import RuntimeConfigC


class WorkerQueueC(ctypes.Structure):
    """A contiguous interval in one task-kind queue and its bounded ready lookahead."""
    _fields_ = [(name, ctypes.c_uint32) for name in ("begin", "end", "ready_depth", "reserved")]


@lru_cache(maxsize=16)
def _queued_type(base: type, workers: int) -> type:
    return type("Queued" + base.__name__, (base,), {"_fields_": [("worker_queues", WorkerQueueC * workers)]})


def install_worker_queues(cfg: RuntimeConfigC, cubes: list[list[int]], vectors: list[list[int]], *,
                          cube_ready_depth: int = 0, vector_ready_depths: list[int] | None = None) -> RuntimeConfigC:
    """Return a runtime with complete, unique work and no padding tasks.

    Per-worker ready depths are explicit: communication owners can remain FIFO
    while independent math workers use local lookahead. No worker parity or
    operator type is assumed by this representation.
    """
    if cfg.num_workers % 2 or len(cubes) != cfg.num_workers // 2 or len(vectors) != cfg.num_workers:
        raise ValueError("Worker queues must match the runtime's AIC/AIV resources")
    depths = [0] * len(vectors) if vector_ready_depths is None else vector_ready_depths
    if len(depths) != len(vectors) or any(d not in (0, 1, 8) for d in [cube_ready_depth, *depths]):
        raise ValueError("Worker ready depths must be 0, 1 or 8")
    total_workers = len(cubes) + len(vectors)
    result = _queued_type(type(cfg), total_workers)()
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(cfg), ctypes.sizeof(cfg))
    result.__dict__.update(cfg.__dict__)
    queue_record = 0
    for kind, queues, indices, per_worker_depths in (
        (0, cubes, result.cube_task_indices, [cube_ready_depth] * len(cubes)),
        (1, vectors, result.vector_task_indices, depths),
    ):
        old = list((cfg.cube_task_indices if kind == 0 else cfg.vector_task_indices)[:cfg.task_index_num[kind]])
        flat = [task for queue in queues for task in queue]
        if Counter(flat) != Counter(old) or len(flat) != len(set(flat)):
            raise ValueError("Worker queues must cover the original work exactly once")
        begin = 0
        for queue, depth in zip(queues, per_worker_depths):
            end = begin + len(queue)
            for position, task in enumerate(queue, begin):
                indices[position] = task
            result.worker_queues[queue_record] = WorkerQueueC(begin, end, depth, 0)
            queue_record += 1
            begin = end
    result.queue_layout = 1
    if result.completion_event:
        result.all_event_num_triggers[result.completion_event] = total_workers
    return result


def serialize_worker_queues(cfg: RuntimeConfigC) -> bytes:
    """Validate the typed queue extension before appending it to the dense wire image."""
    records = cfg.worker_queues
    cubes = cfg.num_workers // 2
    if len(records) != cubes + cfg.num_workers:
        raise ValueError("Worker queue record count does not match runtime resources")
    previous = 0
    for i, record in enumerate(records):
        if i == cubes:
            previous = 0
        kind = 0 if i < cubes else 1
        if (record.begin != previous or record.end < record.begin or record.end > cfg.task_index_num[kind]
                or record.ready_depth not in (0, 1, 8) or record.reserved != 0):
            raise ValueError("Invalid worker queue interval or ready policy")
        previous = record.end
        if i in (cubes - 1, len(records) - 1) and previous != cfg.task_index_num[kind]:
            raise ValueError("Worker queues do not cover the complete task-kind queue")
    return bytes(records)

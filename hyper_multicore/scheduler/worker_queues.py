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
"""Component-independent contiguous/strided worker queues and ready contracts."""
from collections import Counter
import ctypes

from .config import RuntimeConfigC

WAIT_HEAD = 0
RESCAN_WINDOW = 1
MAX_READY_LOOKAHEAD = 256
EXPLICIT_QUEUE_LAYOUT = 2


class WorkerQueueC(ctypes.Structure):
    """A task-index view plus selection and waiting, independent of placement."""
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "begin", "end", "ready_depth", "stride", "wait_mode", "poll_interval_us")]

    def positions(self) -> range:
        """Return positions in this worker's task-index array."""
        return range(self.begin, self.end, self.stride)


def split_worker_queues(cfg: RuntimeConfigC) -> tuple[list[WorkerQueueC], list[WorkerQueueC]]:
    """Return the explicit AIC and AIV queue tables in device worker order."""
    records = cfg.worker_queues
    cube_workers = cfg.num_workers // 2
    return records[:cube_workers], records[cube_workers:]


def _attach(cfg, records):
    cfg.worker_queues = (WorkerQueueC * len(records))(*records)
    cfg.queue_layout = EXPLICIT_QUEUE_LAYOUT
    if cfg.completion_event:
        cfg.all_event_num_triggers[cfg.completion_event] = len(records)
    validate_worker_queues(cfg)
    return cfg


def install_strided_worker_queues(cfg: RuntimeConfigC, *, vector_ready_depth: int = 256) -> RuntimeConfigC:
    """Preserve all-worker round-robin placement without rewriting task indices.

    The compatibility wait observes the head with the upstream 5us parameter.
    Cube queues remain FIFO. Selection does not implicitly enable any workers:
    this placement operation explicitly supplies the complete participant set.
    """
    records = []
    for count, workers, depth in ((cfg.task_index_num[0], cfg.num_workers // 2, 0),
                                  (cfg.task_index_num[1], cfg.num_workers, vector_ready_depth)):
        records.extend(WorkerQueueC(min(worker, count), count, depth, workers, WAIT_HEAD, 5)
                       for worker in range(workers))
    return _attach(cfg, records)


def install_worker_queues(cfg: RuntimeConfigC, cubes: list[list[int]], vectors: list[list[int]], *,
                          cube_ready_depth: int = 0, vector_ready_depths: list[int] | None = None) -> RuntimeConfigC:
    """Place every task once; communication can stay FIFO while math selects ready work."""
    if cfg.num_workers % 2 or len(cubes) != cfg.num_workers // 2 or len(vectors) != cfg.num_workers:
        raise ValueError("Worker queues must match the runtime's AIC/AIV resources")
    depths = [0] * len(vectors) if vector_ready_depths is None else vector_ready_depths
    if len(depths) != len(vectors):
        raise ValueError("One ready lookahead is required for each vector worker")
    result = type(cfg)()
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(cfg), ctypes.sizeof(cfg))
    result.__dict__.update(cfg.__dict__)
    records = []
    for kind, queues, indices, per_depths in (
        (0, cubes, result.cube_task_indices, [cube_ready_depth] * len(cubes)),
        (1, vectors, result.vector_task_indices, depths),
    ):
        old = list((cfg.cube_task_indices if kind == 0 else cfg.vector_task_indices)[:cfg.task_index_num[kind]])
        flat = [task for queue in queues for task in queue]
        if Counter(flat) != Counter(old) or len(flat) != len(set(flat)):
            raise ValueError("Worker queues must cover the original work exactly once")
        begin = 0
        for queue, depth in zip(queues, per_depths):
            if not isinstance(depth, int) or isinstance(depth, bool) or not 0 <= depth <= MAX_READY_LOOKAHEAD:
                raise ValueError("Worker ready lookahead must be an integer in [0, 256]")
            end = begin + len(queue)
            for position, task in enumerate(queue, begin):
                indices[position] = task
            records.append(WorkerQueueC(begin, end, depth, 1, RESCAN_WINDOW, 10))
            begin = end
    return _attach(result, records)


def validate_worker_queues(cfg: RuntimeConfigC) -> None:
    """Validate the explicit queue table and its complete, unique index coverage."""
    records = getattr(cfg, "worker_queues", None)
    num_workers = cfg.num_workers
    if records is None or num_workers % 2 or len(records) != num_workers // 2 + num_workers:
        raise ValueError("Worker queue count does not match runtime resources")
    for kind, group in enumerate(split_worker_queues(cfg)):
        count = cfg.task_index_num[kind]
        positions = []
        for record in group:
            if (record.end > count or record.begin > record.end or not record.stride
                    or record.ready_depth > MAX_READY_LOOKAHEAD or record.wait_mode not in (WAIT_HEAD, RESCAN_WINDOW)
                    or (record.wait_mode == RESCAN_WINDOW and not record.poll_interval_us)):
                raise ValueError("Invalid worker queue view or ready/wait policy")
            positions.extend(record.positions())
        if len(positions) != count or set(positions) != set(range(count)):
            raise ValueError("Worker queues must cover every original index exactly once")


def serialize_worker_queues(cfg: RuntimeConfigC) -> bytes:
    """Validate and serialize the explicit queue table."""
    validate_worker_queues(cfg)
    return bytes(cfg.worker_queues)

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Generate the measured two-expert window from the current task graph."""
from typing import Any

from hyper_multicore.shmem._runtime import _udma_qps_per_peer

from .config import TaskType
from .planned_queue import install_worker_queues


def _validate_qp_owners(cfg: Any, spec: Any, vectors: list[list[int]], qps: int) -> None:
    """Reject concurrent producers of one peer/QP without changing task order."""
    owners = {}
    for worker in range(1, 64, 2):
        qp = worker // (64 // qps)
        for tid in vectors[worker]:
            task = cfg.all_tasks[tid]
            if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
                continue
            peer = task.task_index // (task.task_split_num // spec.ep_size)
            if peer == spec.rank_id:
                continue
            key = peer, qp
            owner = owners.setdefault(key, worker)
            if owner != worker:
                raise ValueError(f'QP owner collision: peer={peer}, qp={qp}, workers={owner}/{worker}')


def configure_window_queue(cfg: Any, graph: Any, spec: Any, *, forward: bool) -> None:
    """Preserve tasks, dependencies and QP ownership without shape-specific files."""
    stages = {}
    offset = 0
    for op in graph.topological_sort():
        if op.name != 'terminate' and op.task_num % spec.local_experts:
            raise ValueError(f'Window stage {op.name} cannot partition {op.task_num} tasks across experts')
        stages[op.name] = (offset, op.task_num)
        offset += op.task_num

    def tasks(stage: str, expert: int, lane: int) -> list[int]:
        start, count = stages[stage]
        width = count // spec.local_experts
        return list(range(start + expert * width + lane, start + (expert + 1) * width, 32))

    cube_stages = ('up', 'down') if forward else ('dw2', 'da', 'dw1', 'dx')
    inputs = ('quant_x', 'swiglu_quant') if forward else ('quant_dy', 'swiglu_grad_quant')
    front = 1 if forward else 2
    vectors = [[] for _ in range(64)]
    cubes = [[] for _ in range(32)]
    communication = list(cfg.vector_task_indices[:cfg.task_index_num[3]])
    for lane in range(32):
        for stage in cube_stages[:front]:
            cubes[lane].extend(tasks(stage, 0, lane))
        for expert in range(1, spec.local_experts):
            for stage in cube_stages[:front]:
                cubes[lane].extend(tasks(stage, expert, lane))
            for stage in cube_stages[front:]:
                cubes[lane].extend(tasks(stage, expert - 1, lane))
        for stage in cube_stages[front:]:
            cubes[lane].extend(tasks(stage, spec.local_experts - 1, lane))
        vectors[2 * lane + 1] = communication[lane::32]
        vectors[2 * lane].extend(tasks(inputs[0], 0, lane))
        for expert in range(spec.local_experts):
            if expert + 1 < spec.local_experts:
                vectors[2 * lane].extend(tasks(inputs[0], expert + 1, lane))
            vectors[2 * lane].extend(tasks(inputs[1], expert, lane))
    qps = _udma_qps_per_peer()
    if qps < 32:
        _validate_qp_owners(cfg, spec, vectors, qps)
    install_worker_queues(cfg, cubes, vectors)

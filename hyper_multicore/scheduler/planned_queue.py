"""Load a source/shape-qualified static queue into a no-padding64-AIV CSR layout."""
import hashlib
import json
from pathlib import Path
from typing import Any

from .runtime import serialize_runtime_config

PLANNED_QUEUE_TAG = -1


def configure_planned_queue(cfg: Any, spec: Any, *, forward: bool) -> None:
    phase = 'forward' if forward else 'backward'
    path = Path(spec.mx_schedule_dir) / f'rank{spec.rank_id:02d}-{phase}.queue.json'
    plan = json.loads(path.read_text())
    if (plan['schema'] != 339 or plan['rank'] != spec.rank_id or plan['phase'] != phase
            or plan['runtime_sha256'] != hashlib.sha256(serialize_runtime_config(cfg)).hexdigest()):
        raise ValueError(f'Planned queue does not match the live runtime: {path}')
    install_worker_queues(cfg, plan['cube_workers'], plan['vector_workers'])


def install_worker_queues(cfg: Any, cubes: list[list[int]], vectors: list[list[int]],
                          *, communication_lanes: int = 32) -> None:
    """Install CSR queues while preserving complete work and communication ownership."""
    if len(cubes) != 32 or len(vectors) != 64 or len({len(q) for q in cubes}) != 1:
        raise ValueError('Planned queue requires32 AIC and64 AIV streams')
    if communication_lanes not in (16, 32):
        raise ValueError('Communication lanes must be 16 or 32')
    old_cube = list(cfg.cube_task_indices[:cfg.task_index_num[0]])
    old_vector = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
    split = cfg.task_index_num[3]
    owners = {tid: 2 * (pos % communication_lanes) + 1 for pos, tid in enumerate(old_vector[:split])
              if cfg.all_tasks[tid].task_type in (0, 105, 141)}
    flat_cube = [q[index] for index in range(len(cubes[0])) for q in cubes]
    flat_vector = [tid for q in vectors for tid in q]
    if sorted(flat_cube) != sorted(old_cube) or sorted(flat_vector) != sorted(old_vector):
        raise ValueError('Planned queue must cover every original task exactly once')
    for worker, queue in enumerate(cubes):
        if any(old_cube.index(tid) % 32 != worker for tid in queue):
            raise ValueError('Planned queue cannot move AIC math workers')
    for worker, queue in enumerate(vectors):
        original = (
            [tid for tid in old_vector[:split][worker // 2::communication_lanes]
             if cfg.all_tasks[tid].task_type in (0, 105, 141)]
            if worker % 2 and worker // 2 < communication_lanes else []
        )
        actual = [tid for tid in queue if cfg.all_tasks[tid].task_type in (0, 105, 141)]
        if actual != original or any(owners[tid] != worker for tid in actual):
            raise ValueError('Communication/termination QP owner or order changed')
    for i, tid in enumerate(flat_cube): cfg.cube_task_indices[i] = tid
    for i, tid in enumerate(flat_vector): cfg.vector_task_indices[i] = tid
    offset = 0
    for worker, queue in enumerate(vectors):
        cfg.mix_task_indices[worker] = offset
        offset += len(queue)
    cfg.mix_task_indices[64] = offset
    cfg.task_index_num[3] = PLANNED_QUEUE_TAG

"""Plan BF16 payload transport independently of the expert math precision."""
from typing import Any

from .config import TaskType

TRANSPORT_TYPES = (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM)
COMMUNICATION_TYPES = (TaskType.TASK_TERMINATE, *TRANSPORT_TYPES)


def validate_qp_owners(cfg: Any, spec: Any, vectors: list[list[int]], qps: int) -> None:
    """Keep post, quiet and notification on one owner of each physical peer/QP."""
    owners = {}
    for worker, queue in enumerate(vectors):
        qp = worker // (64 // qps)
        for tid in queue:
            task = cfg.all_tasks[tid]
            if task.task_type not in TRANSPORT_TYPES:
                continue
            if worker % 2 == 0:
                raise ValueError('Transport requires an odd AIV owner')
            peer = task.task_index // (task.task_split_num // spec.ep_size)
            if peer == spec.rank_id:
                continue
            owner = owners.setdefault((peer, qp), worker)
            if owner != worker:
                raise ValueError(f'QP owner collision: peer={peer}, qp={qp}, workers={owner}/{worker}')


def build_transport_queues(cfg: Any, spec: Any, task_ids: list[int], qps: int) -> list[list[int]]:
    """Preserve RATR order while sharing QP1 remote owners and parallel local copies."""
    vectors = [[] for _ in range(64)]
    if qps == 1:
        remote_owners = {
            2 * (((peer - spec.rank_id) % spec.ep_size) % 32) + 1
            for peer in range(spec.ep_size) if peer != spec.rank_id
        }
        self_workers = [worker for worker in range(1, 64, 2) if worker not in remote_owners] or [1]
        for tid in task_ids:
            task = cfg.all_tasks[tid]
            if task.task_type == TaskType.TASK_TERMINATE:
                vectors[1].append(tid)
            elif task.task_type in TRANSPORT_TYPES:
                width = task.task_split_num // spec.ep_size
                peer, local_slot = divmod(task.task_index, width)
                owner = 2 * (((peer - spec.rank_id) % spec.ep_size) % 32) + 1
                if peer == spec.rank_id:
                    owner = self_workers[local_slot % len(self_workers)]
                vectors[owner].append(tid)
    else:
        for lane in range(32):
            vectors[2 * lane + 1] = task_ids[lane::32]
    validate_qp_owners(cfg, spec, vectors, qps)
    return vectors

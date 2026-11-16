"""Precision-independent transfer ownership and backend selection for expert DAGs."""
import os
from functools import lru_cache

from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.tasks.alltoall import set_transfer_engine

TRANSPORT_TASKS = (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM)


@lru_cache(maxsize=1)
def require_queue_backend():
    """Require the native runtime ABI needed to interpret explicit worker queues."""
    # Resolve the selected framework adapter only when the UDMA queue is used.
    # pylint: disable-next=import-outside-toplevel
    import torch
    # pylint: disable-next=import-outside-toplevel
    from hyper_multicore.torch.ops import _load_native
    _load_native()
    if torch.ops.hyper_multicore.worker_queue_layout_version() != 2:
        raise RuntimeError("Explicit worker queues require a matching rebuilt native payload")


def requested_transport() -> str | None:
    """Read an explicit override or defer to the active SHMEM engine."""
    engine = os.environ.get("HYPER_MULTICORE_SHMEM_DATA_ENGINE")
    if engine is None:
        return None
    if engine not in ("mte", "mte_udma"):
        raise ValueError("SHMEM data engine must be mte or mte_udma")
    return "udma" if engine == "mte_udma" else "mte"


def transport_queues(cfg, spec, task_ids):
    """Preserve RATR order on each QP0 owner and distribute independent self copies.

    This is a MoE planning policy, not a SHMEM hardware requirement. Different
    peers remain independent; all posts and quiet calls for one peer share one
    odd AIV owner. Math placement is deliberately outside this function.
    """
    lanes = spec.num_cube_cores
    vectors = [[] for _ in range(2 * lanes)]
    peer_owners = {peer: 2 * (((peer - spec.rank_id) % spec.ep_size) % lanes) + 1
                   for peer in range(spec.ep_size) if peer != spec.rank_id}
    idle = [worker for worker in range(1, 2 * lanes, 2) if worker not in peer_owners.values()]
    self_workers = idle or list(range(1, 2 * lanes, 2))
    for tid in task_ids:
        task = cfg.all_tasks[tid]
        if task.task_type == TaskType.TASK_TERMINATE:
            vectors[1].append(tid)
            continue
        if task.task_type not in TRANSPORT_TASKS:
            raise ValueError("Transport queue contains a math task")
        width = task.task_split_num // spec.ep_size
        tiles = task.task_split_num // spec.num_experts
        if width == 0 or tiles == 0:
            raise ValueError("Invalid MoE transport partition")
        peer, local_slot = divmod(task.task_index, width)
        if peer == spec.rank_id:
            expert, tile = divmod(local_slot, tiles)
            worker = self_workers[(expert + tile) % len(self_workers)]
        else:
            worker = peer_owners[peer]
        set_transfer_engine(task, spec.transport_engine)
        # Preserve the previously effective runtime wait, without borrowing the engine bit.
        task.dependency_poll_interval_us = 1 if spec.transport_engine == 'udma' else 0
        vectors[worker].append(tid)
    validate_transport_owners(cfg, spec, vectors)
    return vectors


def validate_transport_owners(cfg, spec, vectors):
    """Qualify all tasks, including empty fragments, against physical QP ownership."""
    owners = {}
    for worker, queue in enumerate(vectors):
        for tid in queue:
            task = cfg.all_tasks[tid]
            if task.task_type not in TRANSPORT_TASKS:
                continue
            peer = task.task_index // (task.task_split_num // spec.ep_size)
            if peer == spec.rank_id:
                continue
            if owners.setdefault(peer, worker) != worker:
                raise ValueError(f"Multiple AIV owners of peer {peer}/QP0")

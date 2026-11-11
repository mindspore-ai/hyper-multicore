"""Qualify idle-AIV self copies without introducing shared remote QP producers."""
from collections import defaultdict
from dataclasses import replace

import pytest

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.scheduler.config import TaskType
from tests.ut.a5.test_bf16_schedule import queues, specification


@pytest.mark.parametrize('ep,rank', [(4, 0), (4, 3), (8, 5), (16, 9)])
@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_self_tiles_use_spare_odd_aiv_without_delaying_remote_owners(monkeypatch, ep, rank, mode):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    spec = replace(specification(ep, 6, 'window2', rank), dispatch_mode=mode, ready_depth=8)
    _, forward, _, backward = _build_runtime_artifacts(spec)
    remote = {2 * (((peer - rank) % ep) % 32) + 1 for peer in range(ep) if peer != rank}
    spare = [worker for worker in range(1, 64, 2) if worker not in remote]
    for cfg in (forward, backward):
        _, vectors = queues(cfg)
        self_workers = set()
        peer_tasks = defaultdict(list)
        for worker, queue in enumerate(vectors):
            for tid in queue:
                task = cfg.all_tasks[tid]
                if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
                    continue
                width = task.task_split_num // ep
                peer, local_slot = divmod(task.task_index, width)
                if peer == rank:
                    assert worker == spare[local_slot % len(spare)]
                    assert worker not in remote
                    self_workers.add(worker)
                else:
                    assert worker == 2 * (((peer - rank) % ep) % 32) + 1
                    peer_tasks[peer].append((tid, task.task_index))
        assert len(self_workers) > 1
        assert set(peer_tasks) == set(range(ep)) - {rank}
        for tasks in peer_tasks.values():
            assert [tid for tid, _ in tasks] == sorted(tid for tid, _ in tasks)


@pytest.mark.parametrize('ep', [32, 64])
def test_full_remote_owner_pool_does_not_stripe_self_into_remote_workers(monkeypatch, ep):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    _, forward, _, backward = _build_runtime_artifacts(specification(ep, 6, 'window2'))
    for cfg in (forward, backward):
        _, vectors = queues(cfg)
        owners = set()
        for worker, queue in enumerate(vectors):
            for tid in queue:
                task = cfg.all_tasks[tid]
                if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
                    continue
                peer = task.task_index // (task.task_split_num // ep)
                if peer == 0:
                    owners.add(worker)
        assert owners == {1}

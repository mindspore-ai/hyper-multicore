"""Check complete real expert DAGs and QP0 ownership without device execution."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hyper_multicore.modules.mega_moe.backends.bf16.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.transport import TRANSPORT_TASKS, validate_transport_owners
from hyper_multicore.shmem._ordering import SubmissionOrder
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.scheduler.runtime import serialize_runtime_config


@pytest.mark.parametrize('ep,mode', [(4, 'push'), (8, 'pull'), (64, 'push')])
def test_qp0_owners_and_complete_bf16_work(ep, mode):
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * 6, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=0,
                       num_cube_cores=32, dispatch_mode=mode, dispatch_split=512, combine_split=512,
                       transport_engine='udma', ready_depth=8)
    _, forward, _, backward = _build_runtime_artifacts(spec)
    for cfg in (forward, backward):
        assert cfg.queue_layout == 1
        vectors = [list(cfg.vector_task_indices[q.begin:q.end]) for q in cfg.worker_queues[32:]]
        validate_transport_owners(cfg, spec, vectors)
        self_workers = set()
        for worker, queue in enumerate(vectors):
            for tid in queue:
                task = cfg.all_tasks[tid]
                if task.task_type in TRANSPORT_TASKS:
                    peer = task.task_index // (task.task_split_num // ep)
                    assert task.extra_value_2 == 1
                    assert cfg.worker_queues[32 + worker].ready_depth == 0
                    if peer == spec.rank_id:
                        self_workers.add(worker)
        assert len(self_workers) > 1
        assert all(q.ready_depth == 8 for q in cfg.worker_queues[:32])


def test_qp0_submission_order_reuses_completion_event():
    order = SubmissionOrder()
    first = SimpleNamespace(npu_stream=10, wait_event=Mock())
    second = SimpleNamespace(npu_stream=20, wait_event=Mock())
    completion = object()
    order.claim(first)
    with pytest.raises(RuntimeError, match='active submission'):
        order.claim(second)
    order.release(first, completion)
    order.claim(second)
    second.wait_event.assert_called_once_with(completion)
    order.release(second, completion)


@pytest.mark.parametrize('ep,mode', [(4, 'push'), (8, 'pull'), (64, 'push'), (64, 'pull')])
def test_mx_graph_coverage_and_shared_protocol(ep, mode):
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * 6, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=0,
                       num_cube_cores=32, dispatch_mode=mode, dispatch_split=4096, combine_split=4096,
                       transport_engine='udma', ready_depth=8, precision='mxfp8')
    phases = [build_mxfp8_runtime(spec, forward=True),
              build_mxfp8_runtime(spec, forward=False)]
    for plan in phases:
        cfg = plan.config
        assert cfg.queue_layout == 1
        assert serialize_runtime_config(cfg)
        assert len(plan.gmm_descriptors) > 0
        vectors = [list(cfg.vector_task_indices[q.begin:q.end]) for q in cfg.worker_queues[32:]]
        validate_transport_owners(cfg, spec, vectors)
        # Persistent ready/completion lanes have the same positions in F/R/B.
        assert cfg.event_capacity == phases[0].config.event_capacity

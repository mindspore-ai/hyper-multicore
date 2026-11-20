"""Check complete real expert DAGs and QP0 ownership without device execution."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from hyper_multicore.modules.mega_moe.backends.bf16.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.schedule import explain_schedule
from hyper_multicore.modules.mega_moe.transport import TRANSPORT_TASKS, validate_transport_owners
from hyper_multicore.shmem._ordering import SubmissionOrder
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.runtime import serialize_runtime_config


@pytest.mark.parametrize('ep,mode', [(4, 'push'), (8, 'pull'), (64, 'push')])
def test_qp0_owners_and_complete_bf16_work(ep, mode):
    """Keep BF16 communication on its owner and retain every math task."""
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * 6, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=0,
                       num_cube_cores=32, dispatch_mode=mode, dispatch_split=512, combine_split=512,
                       architecture=3510, transport_engine='udma', ready_depth=8)
    _, forward, _, backward = _build_runtime_artifacts(spec)
    for cfg in (forward, backward):
        assert cfg.queue_layout == 2
        vectors = [list(cfg.vector_task_indices[q.begin:q.end]) for q in cfg.worker_queues[32:]]
        validate_transport_owners(cfg, spec, vectors)
        self_workers = set()
        for worker, queue in enumerate(vectors):
            for tid in queue:
                task = cfg.all_tasks[tid]
                if task.task_type in TRANSPORT_TASKS:
                    peer = task.task_index // (task.task_split_num // ep)
                    assert task.dependency_poll_interval_us == 1
                    assert cfg.worker_queues[32 + worker].ready_depth == 0
                    if peer == spec.rank_id:
                        self_workers.add(worker)
        assert len(self_workers) > 1
        assert all(q.ready_depth == 8 for q in cfg.worker_queues[:32])


def test_qp0_submission_order_reuses_completion_event():
    """Keep QP0 submissions ordered with the shared completion event."""
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


def test_default_mte_pull_uses_available_lanes_and_coarser_fragments():
    """Retain each remote credit while reducing the default descriptor count."""
    spec = MegaMoeSpec(4096, 512, 128, 96, 8, None, 32768, 16, None, 0, 32,
                       dispatch_mode='pull', architecture=3510, auto_tokens=True)
    graph, cfg, _, _ = _build_runtime_artifacts(spec)
    assert graph.get_op('dispatch').task_num == 4 * spec.num_experts
    dense, _, _, _ = _build_runtime_artifacts(replace(spec, lowering='dense', dispatch_split=256))
    assert dense.get_op('dispatch').task_num == 16 * spec.num_experts
    by_peer = {}
    for worker, queue in enumerate(cfg.worker_queues[32:]):
        for tid in cfg.vector_task_indices[queue.begin:queue.end]:
            task = cfg.all_tasks[tid]
            if task.task_type != TaskType.TASK_SHMEM_GET_MEM:
                continue
            peer = task.task_index // (task.task_split_num // spec.ep_size)
            by_peer.setdefault(peer, set()).add(worker)
    assert len(by_peer) == spec.ep_size
    assert all(len(owners) == 2 for peer, owners in by_peer.items() if peer != spec.rank_id)


@pytest.mark.parametrize("ep,cubes", [(4, 28), (32, 32), (64, 36)])
def test_bf16_replica_helpers_preserve_math_and_transport_work(ep, cubes):
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * 8, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=0,
                       num_cube_cores=cubes, dispatch_mode="push", dispatch_split=512, combine_split=512,
                       architecture=3510, transport_engine="udma", ready_depth=0)
    ordinary = _build_runtime_artifacts(spec, torch.float32)
    replica = _build_runtime_artifacts(spec, torch.float32, replica_pipeline=True, reserve_merge_workers=True)
    for base, fused in zip(ordinary[1::2], replica[1::2], strict=True):
        base_queues, fused_queues = (explain_schedule(cfg)["queues"] for cfg in (base, fused))
        assert sorted(tid for queue in base_queues for tid in queue["tasks"]) == sorted(
            tid for queue in fused_queues for tid in queue["tasks"])
        assert all(fused_queues[cubes + worker]["tasks"] == [] for worker in range(0, 8, 2))
        assert base.all_event_num_triggers[:] == fused.all_event_num_triggers[:]


@pytest.mark.parametrize('ep,mode', [(4, 'push'), (8, 'pull'), (64, 'push'), (64, 'pull')])
def test_mx_graph_coverage_and_shared_protocol(ep, mode):
    """Verify MX graph task coverage and transport protocol."""
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * 6, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=0,
                       num_cube_cores=32, dispatch_mode=mode, dispatch_split=4096, combine_split=4096,
                       architecture=3510, transport_engine='udma', ready_depth=8, precision='mxfp8')
    phases = [build_mxfp8_runtime(spec, forward=True),
              build_mxfp8_runtime(spec, forward=False)]
    for plan in phases:
        cfg = plan.config
        assert cfg.queue_layout == 2
        assert serialize_runtime_config(cfg)
        assert len(plan.gmm_descriptors) > 0
        vectors = [list(cfg.vector_task_indices[q.begin:q.end]) for q in cfg.worker_queues[32:]]
        validate_transport_owners(cfg, spec, vectors)
        # Persistent ready/completion lanes have the same positions in F/R/B.
        assert cfg.event_capacity == phases[0].config.event_capacity


def test_active_mx_activation_gradient_waits_for_dy_before_dw2():
    """The reordered Cube queue must never wait on the next task's TN fence."""
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=128, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=65536, ep_size=16, ep_group=None, rank_id=0,
                       num_cube_cores=32, dispatch_mode='push', dispatch_split=512, combine_split=512,
                       architecture=3510, transport_engine='udma', ready_depth=0, precision='mxfp8',
                       partial_weight_grad_dtype='float32')
    plan = build_mxfp8_runtime(spec, forward=False, home_experts=6, gradient_helpers=True)
    layout = explain_schedule(plan.config)
    stages = layout['schedule']['stages']
    da, count = stages['da']
    dw2, dw_count = stages['dw_down']
    assert count == dw_count
    for lane, queue in enumerate(layout['queues'][:32]):
        assert queue['tasks'][:2] == [da + lane, dw2 + lane]
    for expert, event in enumerate(plan.gradient_ready_events[0]):
        index = expert * 32
        first, second = [layout['tasks'][index + start] for start in (da, dw2)]
        assert first['dependent_event'] == second['dependent_event'] != event
        assert second['trigger_event'] == event


@pytest.mark.parametrize('tokens', [128, 256, 768, 4096])
@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_bf16_communication_tiles_cover_short_and_tail_plans(tokens, mode):
    """Every peer/expert owns completion credits, including a short final tile."""
    spec = MegaMoeSpec(local_num_tokens=tokens, hidden_size=512, intermediate_size=128,
                       num_experts=24, top_k=8, initial_capacity_factor=2 if mode == 'push' else None,
                       receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
                       num_cube_cores=32, dispatch_mode=mode, dispatch_split=512, combine_split=512,
                       transport_engine='udma', architecture=3510)
    forward_graph, forward, backward_graph, backward = _build_runtime_artifacts(spec)
    fragments = (tokens + 511) // 512
    for graph, config in ((forward_graph, forward), (backward_graph, backward)):
        for name in ('dispatch', 'combine'):
            assert graph.get_op(name).task_num == 24 * fragments
        vectors = [list(config.vector_task_indices[q.begin:q.end]) for q in config.worker_queues[32:]]
        validate_transport_owners(config, spec, vectors)
        assert serialize_runtime_config(config)

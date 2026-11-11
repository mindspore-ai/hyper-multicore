"""Qualify precision-independent QP1 transport ownership and task order."""
from dataclasses import replace

import pytest

from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.scheduler.config import TaskType
from tests.ut.a5.test_bf16_schedule import queues, specification


def projection(cfg, vectors):
    result = []
    for worker, queue in enumerate(vectors):
        values = []
        for tid in queue:
            task = cfg.all_tasks[tid]
            if task.task_type in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
                values.append((int(task.task_type), task.task_index, task.task_split_num, task.task_split_value))
        result.append(values)
    return result


@pytest.mark.parametrize('ep,rank', [(4, 0), (4, 3), (8, 5), (64, 0)])
@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_precisions_share_transport_ownership_and_order(monkeypatch, ep, rank, mode):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    monkeypatch.setenv('HYPER_MULTICORE_MX_GMM_TILE', 't6')
    spec = replace(specification(ep, 6, 'window2', rank), dispatch_mode=mode,
                   mx_vector_policy='window2', ready_depth=8)
    _, forward, _, backward = _build_runtime_artifacts(spec)
    for bf, is_forward in ((forward, True), (backward, False)):
        host = build_mxfp8_runtime(spec, forward=is_forward, ready_depth=8)
        mx = host.config
        monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '32')
        control = build_mxfp8_runtime(spec, forward=is_forward, ready_depth=8)
        monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
        assert host.gmm_descriptors == control.gmm_descriptors
        assert host.quant_descriptors == control.quant_descriptors
        assert bytes(mx.all_tasks) == bytes(control.config.all_tasks)
        assert bytes(mx.all_event_num_triggers) == bytes(control.config.all_event_num_triggers)
        assert bytes(mx.cube_task_indices) == bytes(control.config.cube_task_indices)

        offsets = mx.mix_task_indices
        vectors = [list(mx.vector_task_indices[offsets[w]:offsets[w + 1]]) for w in range(64)]
        assert projection(bf, queues(bf)[1]) == projection(mx, vectors)
        work = list(mx.cube_task_indices[:mx.task_index_num[0]]) + [tid for q in vectors for tid in q]
        assert len(work) == len(set(work))
        assert set(work) == set(range(max(work) + 1))


@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_explicit_dispatch_rejects_conflicting_environment(monkeypatch, mode):
    from hyper_multicore import MegaMoeExperts
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DISPATCH_MODE', 'pull' if mode == 'push' else 'push')
    with pytest.raises(ValueError, match='conflicts'):
        MegaMoeExperts(local_num_tokens=128, hidden_size=128, intermediate_size=128,
                       num_experts=2, top_k=1, create_parameters=False, dispatch_mode=mode)


def test_environment_pull_ignores_downstream_push_reservation(monkeypatch):
    from hyper_multicore import MegaMoeExperts
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DISPATCH_MODE', 'pull')
    model = MegaMoeExperts(local_num_tokens=128, hidden_size=128, intermediate_size=128,
                          num_experts=2, top_k=1, create_parameters=False, initial_capacity_factor=40)
    assert model.dispatch_mode == 'pull'
    assert model.initial_capacity_factor is None
    assert model.capacity_growth_factor is None
    model.close()

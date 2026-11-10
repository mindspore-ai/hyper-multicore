"""Qualify route-independent Pull plans with real host builders and no NPU allocation."""
from collections import OrderedDict
from dataclasses import replace
import hashlib

import pytest

from hyper_multicore.modules.mega_moe.mxfp8_module import _MXFP8Resources
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


def resource(mode: str = 'pull', ep: int = 64, split: int = 4096) -> _MXFP8Resources:
    result = object.__new__(_MXFP8Resources)
    result.spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
        num_experts=ep * 6, top_k=8, initial_capacity_factor=None if mode == 'pull' else 1.25,
        receive_capacity=32768, ep_size=ep, ep_group=None, rank_id=0, num_cube_cores=32,
        dispatch_mode=mode, mx_vector_policy='window2', dispatch_split=split, combine_split=split)
    result._plans = OrderedDict()
    result._expert_window = 0
    result._dispatch_quant = False
    result._empty_credits = False
    result.ready_depth = 1
    return result


def fingerprint(plan: object) -> tuple[str, ...]:
    return tuple(hashlib.sha256(t.numpy().tobytes()).hexdigest()
                 for part in (plan.forward, plan.backward) for t in (part.runtime, part.gmm, part.quant))


@pytest.mark.parametrize('ep,split,qps', [(64, 4096, 1), (8, 512, 16)])
def test_varying_pull_capacity_reuses_immutable_plan(monkeypatch, ep, split, qps):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', str(qps))
    r = resource(ep=ep, split=split)
    first = r.plan_for_capacity(32768, 'cpu')
    before = fingerprint(first)
    for capacity in (1, 31, 32, 63, 64, 32769, 32770, 65536, ep * 32768):
        assert r.plan_for_capacity(capacity, 'cpu') is first
        assert fingerprint(first) == before
    row = r.plan_for_capacity(1, 'cpu', swiglu_row_only=True)
    for capacity in (127, 128, 32771, ep * 32768):
        assert r.plan_for_capacity(capacity, 'cpu', swiglu_row_only=True) is row
    assert first is not row and len(r._plans) == 2
    assert first.spec.receive_capacity == ep * 32768
    assert r.spec.receive_capacity == 32768
    assert all(t.device.type == 'cpu' for p in (first, row) for part in (p.forward, p.backward)
               for t in (part.runtime, part.gmm, part.quant))


@pytest.mark.parametrize('capacity', [0, -1, 64 * 32768 + 1])
def test_pull_plan_rejects_invalid_actual_capacity(monkeypatch, capacity):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    with pytest.raises(ValueError, match='Pull receive capacity'):
        resource().plan_for_capacity(capacity, 'cpu')


def test_push_keeps_distinct_reserved_capacity_plans(monkeypatch):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    r = resource('push')
    first = r.plan_for_capacity(32768, 'cpu')
    second = r.plan_for_capacity(40960, 'cpu')
    assert first is not second
    assert r.plan_for_capacity(32768, 'cpu') is first
    assert first.spec.receive_capacity == 32768 and second.spec.receive_capacity == 40960


def test_logical_bound_preserves_uint32_descriptor_range(monkeypatch):
    from hyper_multicore.modules.mega_moe import mxfp8_module

    r = resource()
    r.spec = replace(r.spec, local_num_tokens=1 << 24)
    # Isolate the cache bound without constructing millions of synthetic tasks.
    monkeypatch.setattr(mxfp8_module, 'build_mxfp8_runtime', lambda *args, **kwargs: None)
    monkeypatch.setattr(mxfp8_module, '_device_runtime', lambda *args: None)
    plan = r.plan_for_capacity(32768, 'cpu')
    assert plan.spec.receive_capacity == (1 << 32) - 1
    assert r.plan_for_capacity(1 << 31, 'cpu') is plan
    with pytest.raises(ValueError, match='Pull receive capacity'):
        r.plan_for_capacity(1 << 32, 'cpu')


def test_large_buffer_byte_bounds_are_not_narrowed():
    from hyper_multicore.scheduler.mxfp8 import MxLaunchDescriptor, serialize_mx_launch_metadata

    sizes = [15 * 1024**3, (1 << 32) + 17]
    data = serialize_mx_launch_metadata(sizes, rank=0, ep=64, experts=384,
        hidden=5120, local_tokens=4096, cube_count=32, runtime_bytes=64, event_bytes=64)
    descriptor = MxLaunchDescriptor.from_buffer_copy(data)
    assert list(descriptor.buffer_bytes)[:2] == sizes

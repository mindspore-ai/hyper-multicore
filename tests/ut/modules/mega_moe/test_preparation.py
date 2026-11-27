# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Current-route validation and owned row/element descriptors during preparation."""
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from hyper_multicore.modules.mega_moe import route
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


def specification(mode="pull", rank=0):
    return MegaMoeSpec(local_num_tokens=3, hidden_size=4, intermediate_size=8,
                       num_experts=4, top_k=2, initial_capacity_factor=None, receive_capacity=16, ep_size=2,
                       ep_group=None, rank_id=rank, num_cube_cores=24,
                       auto_tokens=True, dispatch_mode=mode, architecture=2201)


def packed_metadata(counts, spec):
    """Independent CPU reference for the native packed route layout."""
    p, e = counts.shape
    local, stride = e // p, (e + 31) // 32 * 32
    source = counts.long().cumsum(1) - counts
    dest_counts = counts.reshape(p, p, local).permute(1, 2, 0).contiguous()
    dest = (dest_counts.reshape(p, -1).long().cumsum(1) - dest_counts.reshape(p, -1)).reshape(p, local, p)
    groups = dest[spec.rank_id, :, -1] + dest_counts[spec.rank_id, :, -1]
    fields = (source[spec.rank_id], dest[:, :, spec.rank_id].reshape(-1),
              dest[spec.rank_id].T.reshape(-1), source[:, spec.rank_id * local:(spec.rank_id + 1) * local].reshape(-1))
    lengths = (counts[spec.rank_id], counts[:, spec.rank_id * local:(spec.rank_id + 1) * local].reshape(-1))
    offsets = torch.zeros(2 * (4 * stride + local), dtype=torch.int64)
    sizes = torch.zeros(4 * stride, dtype=torch.int32)
    for unit, multiplier in enumerate((1, spec.hidden_size)):
        for index, values in enumerate(fields):
            start = unit * (4 * stride + local) + index * stride
            offsets[start:start + e] = values * multiplier
        start = unit * (4 * stride + local) + 4 * stride
        offsets[start:start + local] = groups
        for index, values in enumerate(lengths):
            start = unit * 2 * stride + index * stride
            sizes[start:start + e] = values * multiplier
    return offsets, sizes


class DevicePayload:
    device = SimpleNamespace(type="npu", index=0)
    dtype = torch.int32

    def __init__(self, value):
        self.value = value

    def stride(self, dimension):
        return self.value.stride(dimension)


class Readback:
    def __init__(self, state):
        self.state = state

    def copy_(self, tensor, non_blocking):
        assert non_blocking
        self.tensor = tensor.clone()
        self.state.append("copy")

    def tolist(self):
        assert self.state[-1] == "wait", "Host consumed counts before their event completed"
        return self.tensor.tolist()


@pytest.mark.parametrize("mode", ["push", "pull"])
@pytest.mark.parametrize("rank", [0, 1])
def test_preparation_reads_changed_and_empty_routes_and_preserves_saved_views(mode, rank):
    spec = specification(mode, rank)
    retained = []
    state = []
    event = SimpleNamespace(record=lambda _stream: state.append("record"),
                            synchronize=lambda: state.append("wait"))
    for counts, tokens in (([[2, 1, 2, 1], [0, 0, 0, 0]], [3, 0]),
                           ([[0, 0, 3, 3], [0, 0, 0, 0]], [3, 0]),
                           ([[0, 0, 0, 0], [0, 0, 0, 0]], [0, 0])):
        counts = torch.tensor(counts, dtype=torch.int32)
        headers = torch.tensor([[0, n, 15] for n in tokens], dtype=torch.int32)
        payload = torch.cat((headers, counts), 1)
        loads = counts.reshape(2, 2, 2).sum((0, 2))
        summary = torch.cat((headers.long(), counts.sum(1, keepdim=True),
                             counts.amin(1, keepdim=True), counts.amax(1, keepdim=True), loads[:, None]), 1)
        active = counts.sum(0).reshape(2, 2).gt(0).all(1)
        summary = torch.cat((summary, active[:, None]), 1)
        off, size = packed_metadata(counts, spec)
        state.clear()
        with (patch("hyper_multicore._device.device_ub_bytes", return_value=192 * 1024),
              patch.object(route.multicore_ops, "prepare_route", return_value=(off, size, summary)),
              patch.object(route.torch, "empty_like", return_value=Readback(state)),
              patch.object(route.torch.npu, "Event", return_value=event),
              patch.object(route.torch.npu, "current_stream", return_value=object())):
            result, raw, actual = route._read_prepared_device_route(DevicePayload(payload), spec, None)
        assert raw is None
        assert result[0] == max(1, int(loads[rank]))
        expected = route._compute_route_metadata(counts[rank], counts, None, spec, result[0])
        for hidden_size in (1, spec.hidden_size):
            found, reference = actual.for_kernel(hidden_size), expected.for_kernel(hidden_size)
            for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size", "combine_src_off",
                         "combine_target_off", "combine_size", "group_list"):
                torch.testing.assert_close(getattr(found, name), getattr(reference, name))
            assert found.expert_capacity == result[0]
            assert found.local_num_tokens == tokens[rank]
            assert found.plan_tokens == result[1]
            assert found.all_experts_active == bool(active[rank])
        retained.append((actual, actual.group_list.clone()))
    # A later route may not replace an earlier autograd invocation's descriptors.
    for metadata, groups in retained:
        torch.testing.assert_close(metadata.group_list, groups)


@pytest.mark.parametrize("bad", ["counts", "negative", "participation", "weights", "overflow"])
def test_summary_validation_still_rejects_current_invalid_routes(bad):
    spec = specification()
    rows = [[0, 3, 15, 6, 1, 2, 3], [0, 0, 15, 0, 0, 0, 3]]
    if bad == "counts": rows[0][3] = 5
    if bad == "negative": rows[0][4] = -1
    if bad == "participation": rows[1][2] = 7
    if bad == "weights": rows[1][2] |= route._INVALID_WEIGHT_MASK
    if bad == "overflow": rows[0][0] = 1
    with pytest.raises((ValueError, RuntimeError)):
        route._summarize_dynamic_rows(rows, spec)


def test_native_views_bind_same_invocation_geometry():
    counts = torch.tensor([[2, 1, 2, 1], [0, 0, 0, 0]], dtype=torch.int32)
    spec = specification()
    offsets, sizes = packed_metadata(counts, spec)
    metadata = route._unpack_route_metadata(offsets, sizes, spec, 1)
    bound = metadata.with_active_experts(True, 3, 128, capacity=3)
    assert metadata.expert_capacity == 1
    for view in (bound, bound.for_kernel(4)):
        assert (view.expert_capacity, view.local_num_tokens, view.plan_tokens, view.all_experts_active) == (3, 3, 128, True)


@pytest.mark.parametrize('mode', ['push', 'pull'])
@pytest.mark.parametrize('failure', [None, 'counts', 'permutation'])
@pytest.mark.parametrize('fused_metadata', [True, False])
def test_parallel_route_preserves_publication_and_side_stream_lifetimes(mode, failure, fused_metadata):
    """Data work may overlap counts, but publication requires valid current sizes."""
    events = []
    spec = specification(mode)

    class Tensor:
        device = SimpleNamespace(type='npu')
        shape = (3, 4)

        def record_stream(self, stream):
            events.append(('retain', self, stream))

    class Stream:
        def wait_stream(self, stream):
            events.append(('join', self, stream))

    caller, side = Stream(), Stream()
    workspace = SimpleNamespace(route_preparation_stream=None, wait_for_reuse=lambda **_: None,
                                source_buffer=SimpleNamespace(shape=(256, 4)), in_use=False)

    def claim():
        assert not workspace.in_use
        workspace.in_use = True
        events.append('claim')

    def release(**_):
        assert workspace.in_use
        workspace.in_use = False
        events.append('release')

    workspace.claim, workspace.release = claim, release
    hidden, ids, flat, counts = Tensor(), Tensor(), Tensor(), Tensor()
    metadata = SimpleNamespace(record_stream=lambda stream: events.append(('metadata', stream)))
    work = SimpleNamespace(wait=lambda: events.append('collective_wait'))

    @contextmanager
    def stream_context(_stream):
        yield

    def permute(*_):
        events.append('permutation')
        if failure == 'permutation':
            raise RuntimeError('permutation failed')
        return (Tensor(), Tensor())

    def finish(_spec):
        events.append('readback_wait_and_validate')
        if failure == 'counts':
            raise ValueError('invalid current counts')
        return (3, 128, 6), None, metadata

    def prepare(tokens, received):
        assert (tokens, received) == (128, 6)
        assert not workspace.in_use
        events.append('capacity_commit')
        if mode == 'pull':
            claim()

    def direct(*args):
        assert workspace.in_use
        return permute(*args)

    with (patch.object(route.torch.npu, 'current_stream', return_value=caller),
          patch.object(route.torch.npu, 'Stream', return_value=side),
          patch.object(route.torch.npu, 'stream', side_effect=stream_context),
          patch.object(route, '_dynamic_route_payload', return_value=(counts, object(), False)),
          patch.object(route, '_start_dynamic_gather', return_value=(object(), work)),
          patch.object(route, '_permute_topk_input', side_effect=permute),
          patch.object(route, '_start_prepared_device_route', return_value=(SimpleNamespace(
              finish=finish, record_stream=lambda stream: events.append(('metadata', stream)))
              if fused_metadata else None)),
          patch.object(route, '_read_cached_route', side_effect=lambda *_args, **_kwargs: finish(spec)),
          patch.object(route, '_permute_dynamic_pull', side_effect=direct),
          patch.object(route, '_publish_staged_pull', side_effect=AssertionError('unexpected source copy')),
          patch.object(route, '_finalize_dynamic_route', return_value='prepared')):
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                route._prepare_parallel_route(hidden, ids, flat, spec, counts, workspace, 15, prepare)
            assert 'capacity_commit' not in events
            assert not workspace.in_use
        else:
            assert route._prepare_parallel_route(hidden, ids, flat, spec, counts, workspace, 15, prepare) == 'prepared'
            assert events.index('permutation') < events.index('collective_wait')
            assert events.index('readback_wait_and_validate') < events.index('capacity_commit')
            if mode == 'pull':
                assert events.index('claim') < events.index('permutation') < events.index('release')
                assert events.index('release') < events.index('capacity_commit')
                assert workspace.in_use
    for tensor in (hidden, ids, flat, counts):
        assert ('retain', tensor, side) in events
    if failure:
        assert ('join', caller, side) in events
    else:
        assert ('metadata', caller) in events
    assert workspace.route_preparation_stream is side


@pytest.mark.parametrize('initial_source', ['ready', 'absent', 'too_small', 'peer_growth'])
@pytest.mark.parametrize('fail_counts', [False, True])
def test_early_pull_uses_only_the_final_owner_and_never_publishes_invalid_counts(initial_source, fail_counts):
    """Local shape admission is independent of remote growth and current count validation."""
    spec = specification()
    events, writes = [], []

    class Tensor:
        device = SimpleNamespace(type='npu')
        shape = (3, 4)

        def record_stream(self, stream):
            pass

    class Stream:
        def wait_stream(self, stream):
            events.append('join')

    source = None if initial_source == 'absent' else SimpleNamespace(
        shape=(2 if initial_source == 'too_small' else 256, 4))
    workspace = SimpleNamespace(source_buffer=source, in_use=False, route_preparation_stream=None)

    def claim():
        assert not workspace.in_use
        workspace.in_use = True

    def release(**_):
        assert workspace.in_use
        workspace.in_use = False
        events.append('write_submitted')

    workspace.claim, workspace.release = claim, release
    caller, side = Stream(), Stream()
    hidden, ids, flat, counts = Tensor(), Tensor(), Tensor(), Tensor()
    metadata = SimpleNamespace(record_stream=lambda _: None)

    def direct(*_):
        assert workspace.in_use
        writes.append(workspace.source_buffer)
        return workspace.source_buffer, object()

    def finish(_):
        events.append('counts_validated')
        if fail_counts:
            raise ValueError('invalid real counts')
        return (3, 256 if initial_source == 'peer_growth' else 128, 6), None, metadata

    def prepare(tokens, received):
        assert not workspace.in_use
        assert events[-1] == 'counts_validated'
        if initial_source != 'ready':
            workspace.source_buffer = SimpleNamespace(shape=(tokens * spec.top_k, 4))
        events.append('capacity_committed')
        claim()

    def finalize(permuted, *_):
        assert workspace.in_use
        assert permuted[0] is workspace.source_buffer
        assert 'capacity_committed' in events
        return 'prepared'

    @contextmanager
    def stream_context(_):
        yield

    with (patch.object(route.torch.npu, 'current_stream', return_value=caller),
          patch.object(route.torch.npu, 'Stream', return_value=side),
          patch.object(route.torch.npu, 'stream', side_effect=stream_context),
          patch.object(route, '_dynamic_route_payload', return_value=(counts, object(), False)),
          patch.object(route, '_start_dynamic_gather', return_value=(object(), None)),
          patch.object(route, '_start_prepared_device_route', return_value=SimpleNamespace(
              finish=finish, record_stream=lambda _: None)),
          patch.object(route, '_permute_dynamic_pull', side_effect=direct),
          patch.object(route, '_permute_topk_input', side_effect=AssertionError('unexpected ordinary staging')),
          patch.object(route, '_publish_staged_pull', side_effect=AssertionError('unexpected source copy')),
          patch.object(route, '_finalize_dynamic_route', side_effect=finalize)):
        if fail_counts:
            with pytest.raises(ValueError):
                route._prepare_parallel_route(hidden, ids, flat, spec, counts, workspace, 15, prepare)
            assert not workspace.in_use
            assert 'capacity_committed' not in events
            assert len(writes) == int(initial_source in ('ready', 'peer_growth'))
        else:
            assert route._prepare_parallel_route(hidden, ids, flat, spec, counts, workspace, 15, prepare) == 'prepared'
            assert writes[-1] is workspace.source_buffer
            assert len(writes) == (2 if initial_source == 'peer_growth' else 1)
            if initial_source == 'peer_growth':
                assert writes[0] is source and writes[1] is not source

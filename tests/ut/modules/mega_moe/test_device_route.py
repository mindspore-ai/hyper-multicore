# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Budget configuration and the no-readback routing contract."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.modules.mega_moe import route
from hyper_multicore.modules.mega_moe.spec import _device_route_capacity, _device_route_capacity_factor
from tests.ut.modules.mega_moe.test_preparation import specification, packed_metadata


def test_budget_configuration_is_opt_in_and_resolved_once(monkeypatch):
    monkeypatch.delenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', raising=False)
    assert _device_route_capacity_factor() is None
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', '1')
    monkeypatch.delenv('HYPER_MULTICORE_MEGAMOE_RECEIVE_CAPACITY_FACTOR', raising=False)
    module = MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                           tuning_options=MegaMoeTuningOptions(dispatch_mode='pull'))
    assert module.device_route_capacity_factor == 1.05
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_RECEIVE_CAPACITY_FACTOR', '1.2')
    assert module.device_route_capacity_factor == 1.05
    assert _device_route_capacity_factor() == 1.2
    with pytest.raises(ValueError, match='Pull'):
        MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                       tuning_options=MegaMoeTuningOptions(dispatch_mode='push'))


@pytest.mark.parametrize('enabled', [None, '0'])
@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_disabled_budget_does_not_change_existing_configuration(monkeypatch, enabled, mode):
    """A stale experimental budget must not constrain the default exact path."""
    if enabled is None:
        monkeypatch.delenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', raising=False)
    else:
        monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', enabled)
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_RECEIVE_CAPACITY_FACTOR', 'invalid')
    module = MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                           tuning_options=MegaMoeTuningOptions(dispatch_mode=mode))
    assert module.device_route_capacity_factor is None


@pytest.mark.parametrize('value', ['0.9', 'nan', 'inf', '-1', 'invalid'])
def test_invalid_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', '1')
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_RECEIVE_CAPACITY_FACTOR', value)
    with pytest.raises(ValueError, match='finite number'):
        _device_route_capacity_factor()


def test_receive_budget_uses_known_input_not_received_counts():
    spec = replace(specification(), num_experts=384, ep_size=64, top_k=8,
                   hidden_size=5120, device_route_capacity_factor=1.05)
    assert _device_route_capacity(spec, 4096) == 34432
    assert _device_route_capacity(spec, 0) == 128
    with pytest.raises(ValueError, match='int32'):
        _device_route_capacity(spec, 2**30)


@pytest.mark.parametrize('factor', [None, 1.05])
def test_exact_and_budget_routes_keep_separate_selection(factor):
    """Only an explicit budget selects the equal-input experimental route."""
    spec = replace(specification(), auto_tokens=True, device_route_capacity_factor=factor)
    tensor = SimpleNamespace(device=SimpleNamespace(type='npu'))
    workspace = object()
    expected = object()
    with (patch.object(route, '_prepare_parallel_route', return_value=expected) as exact,
          patch.object(route, '_prepare_bounded_route', return_value=expected) as bounded):
        found = route._prepare_dynamic_route(tensor, object(), object(), spec, None, workspace, 15,
                                             prepare_workspace=Mock())
    assert found is expected
    assert exact.called == (factor is None)
    assert bounded.called == (factor is not None)


@pytest.mark.parametrize('fail', [False, True])
def test_bounded_route_joins_device_stream_without_host_readback(fail):
    spec = replace(specification(), device_route_capacity_factor=1.05)
    events = []
    caller = SimpleNamespace(wait_stream=lambda stream: events.append('join'))
    side = SimpleNamespace(wait_stream=lambda stream: None)
    workspace = SimpleNamespace(route_preparation_stream=side)
    tensor = SimpleNamespace(shape=(3, 4), device=SimpleNamespace(type='npu', index=0),
                             record_stream=lambda stream: None)
    counts = torch.tensor([2, 1, 2, 1], dtype=torch.int32)
    packed = packed_metadata(torch.stack((counts, counts)), spec)
    metadata = route._unpack_route_metadata(*packed, spec, 128)
    fields = ('dispatch_src_off', 'dispatch_target_off', 'dispatch_size', 'combine_src_off',
              'combine_target_off', 'combine_size', 'group_list')
    views = [getattr(metadata.for_kernel(unit), name) for unit in (1, spec.hidden_size) for name in fields]
    prepared = object()
    def prepare(*args):
        assert events == ['permute']
        assert args == (128, 128, tensor)
        events.append('prepare')
        return prepared
    def permute(*_):
        events.append('permute')
        return tensor, tensor
    def bind(*_):
        assert events == ['permute', 'prepare', 'join']
        events.append('metadata')
        if fail:
            raise RuntimeError('native launch failed')
        return views
    with (patch.object(route.torch.npu, 'current_stream', return_value=caller),
          patch.object(route.torch.npu, 'stream', return_value=nullcontext()),
          patch.object(route, '_dynamic_route_payload', return_value=(counts, object(), False)),
          patch.object(route, '_start_dynamic_gather', return_value=(counts, Mock())),
          patch.object(route.multicore_ops, 'bounded_route', side_effect=bind),
          patch('hyper_multicore._device.device_ub_bytes', return_value=192*1024),
          patch.object(route, '_permute_dynamic_pull', side_effect=permute),
          patch.object(torch.Tensor, 'record_stream'),
          patch.object(route, '_read_cached_route', side_effect=AssertionError('D2H')),
          patch.object(route, '_read_dynamic_route', side_effect=AssertionError('D2H')),
          patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('D2H'))):
        if fail:
            with pytest.raises(RuntimeError, match='native launch'):
                route._prepare_bounded_route(tensor, tensor, tensor, spec, None, workspace, 15, Mock(), prepare)
        else:
            result = route._prepare_bounded_route(tensor, tensor, tensor, spec, None, workspace, 15, Mock(), prepare)
            assert result.forward_preparation is prepared
            assert result.metadata.expert_capacity == 128
            torch.testing.assert_close(result.metadata.group_list, torch.tensor([4, 6]))
    assert events == ['permute', 'prepare', 'join', 'metadata']

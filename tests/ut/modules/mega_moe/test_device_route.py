# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Device routing configuration and the no-readback routing contract."""
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


def test_device_route_is_opt_in_and_resolved_once(monkeypatch):
    monkeypatch.delenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', raising=False)
    assert _device_route_capacity_factor() is None
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', '1')
    module = MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                           tuning_options=MegaMoeTuningOptions(dispatch_mode='pull'))
    assert _device_route_capacity_factor() == 1.0
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', '0')
    assert _device_route_capacity_factor() is None
    assert module._resource_group.specification['device_route_capacity_factor'] == 1.0
    assert not hasattr(module, 'device_route_capacity_factor')


def test_device_route_rejects_push(monkeypatch):
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', '1')
    with pytest.raises(ValueError, match='requires Pull'):
        MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                       tuning_options=MegaMoeTuningOptions(dispatch_mode='push'))


def test_invalid_device_route_setting_is_rejected(monkeypatch):
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', 'true')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        _device_route_capacity_factor()


def test_receive_bound_uses_known_input_not_received_counts():
    spec = replace(specification(), num_experts=384, ep_size=64, top_k=8,
                   hidden_size=5120, device_route_capacity_factor=1.0)
    assert _device_route_capacity(spec, 4096) == 32768
    assert _device_route_capacity(spec, 0) == 128
    with pytest.raises(ValueError, match='int32'):
        _device_route_capacity(spec, 2**30)


@pytest.mark.parametrize('factor', [None, 1.0])
def test_exact_and_bounded_routes_keep_separate_selection(factor):
    """Only the explicit device routing option selects the bounded route."""
    spec = replace(specification(), auto_tokens=True, device_route_capacity_factor=factor)
    tensor = SimpleNamespace(device=SimpleNamespace(type='npu'))
    workspace = object()
    expected = object()
    with (patch.object(route, '_validate_topk_inputs', return_value=object()),
          patch.object(route, '_prepare_dynamic_route', return_value=expected) as exact,
          patch.object(route, '_prepare_bounded_route', return_value=expected) as bounded):
        found = route.prepare_topk_route(tensor, object(), object(), spec, None, workspace,
                                         prepare_workspace=Mock())
    assert found is expected
    assert exact.called == (factor is None)
    assert bounded.called == (factor is not None)


@pytest.mark.parametrize(('device', 'auto_tokens', 'workspace', 'placement', 'message'), [
    ('cpu', True, object(), None, 'requires automatic NPU'),
    ('npu', False, object(), None, 'requires automatic NPU'),
    ('npu', True, None, None, 'requires automatic NPU'),
    ('npu', True, object(), Mock(), 'does not support expert placement'),
])
def test_explicit_device_route_never_falls_back(device, auto_tokens, workspace, placement, message):
    spec = replace(specification(), auto_tokens=auto_tokens, device_route_capacity_factor=1.0)
    tensor = SimpleNamespace(device=SimpleNamespace(type=device))
    with (patch.object(route, '_validate_topk_inputs', return_value=object()),
          pytest.raises(ValueError, match=message)):
        route.prepare_topk_route(tensor, object(), object(), spec, None, workspace, select_placement=placement)


@pytest.mark.parametrize('growth', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_bounded_route_joins_device_stream_without_host_readback(fail, growth):
    spec = replace(specification(), device_route_capacity_factor=1.0)
    events = []
    caller = SimpleNamespace(wait_stream=lambda stream: events.append('join'))
    side = SimpleNamespace(wait_stream=lambda stream: None,
                           synchronize=lambda: events.append('validate'))
    source = None if growth else SimpleNamespace(shape=(256, 4))
    workspace = SimpleNamespace(route_preparation_stream=side, source_buffer=source)
    tensor = SimpleNamespace(shape=(3, 4), device=SimpleNamespace(type='npu', index=0),
                             record_stream=lambda stream: None)
    counts = torch.tensor([2, 1, 2, 1], dtype=torch.int32)
    packed = packed_metadata(torch.stack((counts, counts)), spec)
    metadata = route._unpack_route_metadata(*packed, spec, 128)
    fields = ('dispatch_src_off', 'dispatch_target_off', 'dispatch_size', 'combine_src_off',
              'combine_target_off', 'combine_size', 'group_list')
    views = [getattr(metadata.for_kernel(unit), name) for unit in (1, spec.hidden_size) for name in fields]
    prepared = object()
    def _prepare(*args):
        assert events[-2:] == ['workspace', 'permute']
        assert args == (128, 128, tensor)
        events.append('prepare')
        return prepared
    def _permute(*_):
        events.append('permute')
        return tensor, tensor
    def _bind(*_):
        assert not events
        events.append('metadata')
        if fail:
            raise RuntimeError('native launch failed')
        return views
    def _prepare_workspace(*_):
        events.append('workspace')
    with (patch.object(route.torch.npu, 'current_stream', return_value=caller),
          patch.object(route.torch.npu, 'stream', return_value=nullcontext()),
          patch.object(route, '_dynamic_route_payload', return_value=(counts, object(), False)),
          patch.object(route, '_start_dynamic_gather', return_value=(counts, Mock())),
          patch.object(route.multicore_ops, 'bounded_route', side_effect=_bind),
          patch('hyper_multicore._device.device_ub_bytes', return_value=192*1024),
          patch.object(route, '_permute_dynamic_pull', side_effect=_permute),
          patch.object(torch.Tensor, 'record_stream'),
          patch.object(route, '_read_cached_route', side_effect=AssertionError('D2H')),
          patch.object(route, '_read_dynamic_route', side_effect=AssertionError('D2H')),
          patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('D2H'))):
        if fail:
            with pytest.raises(RuntimeError, match='native launch'):
                route._prepare_bounded_route(
                    tensor, tensor, tensor, spec, None, workspace, 15, _prepare_workspace, _prepare,
                )
        else:
            result = route._prepare_bounded_route(
                tensor, tensor, tensor, spec, None, workspace, 15, _prepare_workspace, _prepare,
            )
            assert result.forward_preparation is prepared
            assert result.metadata.expert_capacity == 128
            torch.testing.assert_close(result.metadata.group_list, torch.tensor([4, 6]))
    expected = ['metadata']
    if not fail:
        if growth:
            expected.append('validate')
        expected.extend(['workspace', 'permute', 'prepare'])
    assert events == [*expected, 'join']

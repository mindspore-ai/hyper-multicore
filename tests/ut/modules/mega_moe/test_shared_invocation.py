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
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Shared error convergence, source leases and dynamic MX plan bounds."""
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_multicore.modules.mega_moe import invocation
from hyper_multicore.modules.mega_moe.route import _INVALID_WEIGHT_MASK
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend, execution
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_error_converges_in_existing_route_and_releases_source_lease(mode):
    """Use the existing collective to reject weights and release the Pull lease."""
    resources = SimpleNamespace(spec=SimpleNamespace(dispatch_mode=mode, auto_tokens=True, device_route_capacity_factor=None),
                                workspace=Mock(), prepare_auto_workspace=Mock(), placement=None)
    hidden, ids, probs = torch.ones(1, 64), torch.zeros(1, 1), torch.ones(1, 1)
    def _reject(*args, **options):
        del args
        assert options['autograd_mask'] == _INVALID_WEIGHT_MASK
        options['prepare_workspace'](128, 0)
        raise ValueError('collective invalid weight')
    # Invalid roots are deliberately not inspected before the collective.
    with patch.object(invocation, 'prepare_topk_route', side_effect=_reject) as route:
        with pytest.raises(ValueError, match='collective invalid weight'):
            with invocation.prepare_expert_invocation(hidden, ids, probs, None, (None, None),
                                                       resources, weight_error=True):
                pytest.fail('invalid roots must never reach a backend')
    route.assert_called_once()
    assert resources.workspace.release.call_count == int(mode == 'pull')


@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_mx_tail_plans_keep_complete_communication_coverage(mode):
    """Cover every source and destination fragment in a tail plan."""
    spec = MegaMoeSpec(640, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 32,
                       dispatch_mode=mode, dispatch_split=512, combine_split=512,
                       transport_engine='udma', precision='mxfp8')
    for forward in (True, False):
        program = build_mxfp8_runtime(spec, forward=forward)
        for name in ('dispatch', 'combine'):
            op = program.graph.get_op(name)
            assert op.task_num == spec.num_experts * 2
            assert op.split_value == 512
        assert program.config.queue_layout == 2


def test_mx_plan_cache_distinguishes_committed_token_coverage():
    """Check plan coverage and reuse within the committed source capacity."""
    resources = backend.Resources.__new__(backend.Resources)
    resources.spec = MegaMoeSpec(128, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 32,
                                 transport_engine='udma', precision='mxfp8')
    resources._token_capacity = 1024  # pylint: disable=protected-access
    resources._plans = OrderedDict()  # pylint: disable=protected-access
    with patch.object(backend, '_device_runtime', side_effect=lambda host, device, spec: host):
        small = resources.plan_for_capacity(8192, 'cpu', tokens=128)
        large = resources.plan_for_capacity(8192, 'cpu', tokens=640)
        assert resources.plan_for_capacity(8192, 'cpu', tokens=128) is small
        assert small.spec.local_num_tokens == 128
        assert large.spec.local_num_tokens == 640
        assert small.forward.graph.get_op('dispatch').task_num < large.forward.graph.get_op('dispatch').task_num
        with pytest.raises(ValueError, match='committed'):
            resources.plan_for_capacity(8192, 'cpu', tokens=2048)


def test_mx_early_preparation_preserves_pending_invocation_buffers():
    """A shared plan must never reuse activations or event views of an older call."""
    resources = backend.Resources.__new__(backend.Resources)
    resources.spec = SimpleNamespace(dispatch_mode='pull', hidden_size=64,
                                     intermediate_size=128, local_experts=6)
    first_event, second_event = torch.zeros(16), torch.ones(16)
    resources.workspace = SimpleNamespace(prepare_event_counters=Mock(side_effect=[first_event, second_event]))
    plan = object()
    resources.plan_for_capacity = Mock(return_value=plan)
    rows = torch.empty((128, 64), dtype=torch.bfloat16)
    first = resources.prepare_forward(128, 128, rows)
    second = resources.prepare_forward(256, 256, rows)
    assert first[0] is second[0] is plan
    for prepared_invocation, capacity, event in ((first, 128, first_event), (second, 256, second_event)):
        buffers, found_event = prepared_invocation[1]
        assert found_event is event
        assert buffers.dispatch.shape == (capacity, 64)
        assert buffers.up.shape == (capacity, 256)
        assert buffers.down.shape == (capacity, 64)
        assert buffers.x[0].shape == (capacity, 64)
        assert buffers.activation[0].shape == (capacity, 128)
    assert first[1][0].dispatch.data_ptr() != second[1][0].dispatch.data_ptr()
    assert first[1][0].x[2].data_ptr() != second[1][0].x[2].data_ptr()
    assert first[1][0].up.data_ptr() != second[1][0].up.data_ptr()
    assert [call.kwargs for call in resources.plan_for_capacity.call_args_list] == [{'tokens': 128}, {'tokens': 256}]


@pytest.mark.parametrize('early', [False, True])
def test_mx_forward_consumes_current_invocation_preparation(early):
    """Default and budget launches bind this invocation's operands and events."""
    hidden = torch.zeros(2, 64, dtype=torch.bfloat16)
    spec = SimpleNamespace(dispatch_mode='pull', hidden_size=64, intermediate_size=128, local_experts=2)
    workspace = SimpleNamespace(source_buffer=torch.empty(8, 64), routed_buffer=torch.zeros(4, 64),
                                prepare_event_counters=Mock(return_value=torch.zeros(16)))
    plan = SimpleNamespace(forward=SimpleNamespace(gmm=torch.empty(1), quant=torch.empty(1)))
    resources = SimpleNamespace(spec=spec, workspace=workspace, plan_for_capacity=Mock(return_value=plan))
    prepared = execution._prepare_forward_buffers(resources, 128, hidden)
    events = torch.zeros(32)
    metadata = Mock(expert_capacity=128, plan_tokens=2)
    metadata.for_kernel.return_value = metadata
    route = SimpleNamespace(metadata=metadata, routed_tokens=hidden, unpermute_mapping=torch.arange(2),
                            forward_preparation=(plan, (prepared, events)) if early else None)
    weight = SimpleNamespace(col_data=torch.empty(1), col_scale=torch.empty(1))
    adapter = execution.MXFP8Execution(resources)
    adapter._launch = Mock()
    with patch.object(execution.torch_npu, 'npu_moe_token_unpermute', return_value=hidden.clone()):
        adapter.forward(hidden, route, torch.ones(2, 1), weight, weight, True)
    buffers, _, found_events, found_plan = adapter._launch.call_args.args
    assert found_plan is plan
    assert buffers[2] is workspace.source_buffer
    if early:
        assert buffers[0] is prepared.dispatch and found_events is events
        resources.plan_for_capacity.assert_not_called()
        workspace.prepare_event_counters.assert_not_called()
    else:
        resources.plan_for_capacity.assert_called_once_with(128, hidden.device, tokens=2)
        workspace.prepare_event_counters.assert_called_once_with(forward=True)

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

from hyper_multicore.modules.mega_moe import invocation, route
from hyper_multicore.modules.mega_moe.route import _INVALID_WEIGHT_MASK, PreparedTopKRoute
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_error_converges_in_existing_route_and_releases_source_lease(mode):
    """Use the existing collective to reject weights and release the Pull lease."""
    resources = SimpleNamespace(spec=SimpleNamespace(dispatch_mode=mode, auto_tokens=True),
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


@pytest.mark.parametrize("fail_publication", [False, True])
def test_placed_pull_leases_selected_workspace(fail_publication):
    """Publish virtual IDs and release the child lease even if permutation fails."""
    hidden = torch.ones(1, 64)
    ids, virtual_ids, probs = torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1)
    spec = SimpleNamespace(dispatch_mode="pull", auto_tokens=True, precision="bf16", top_k=1)
    table = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]])
    workspace = Mock(in_use=False, source_buffer=torch.empty(8, 64))
    child = SimpleNamespace(
        spec=SimpleNamespace(dispatch_mode="pull", rank_id=0, num_experts=4, top_k=1),
        workspace=workspace,
        prepare_auto_workspace=Mock(side_effect=lambda *args: setattr(workspace, "in_use", True)),
    )
    placement = SimpleNamespace(counts_by_source=table, virtual_ids=virtual_ids, route_metadata=None,
                                receive_capacity=1, received_rows=lambda rank: 1)
    binding = object()
    resources = SimpleNamespace(
        spec=spec, workspace=Mock(), prepare_auto_workspace=Mock(),
        placement=Mock(uses_placement_wire=Mock(return_value=False),
                       select=Mock(return_value=(child, placement, binding))),
    )

    def prepare(*args, **options):
        return route._prepare_placed_route(
            hidden, ids, spec, resources.workspace, None, table[0, :2], torch.zeros(2, 5),
            None, None, options["prepare_workspace"], options["select_placement"],
        )

    def publish(values, selected_ids, output):
        assert selected_ids is virtual_ids and workspace.in_use
        assert output.data_ptr() == workspace.source_buffer.data_ptr()
        if fail_publication:
            raise RuntimeError("source publication failed")
        return output.copy_(values), torch.zeros(1, dtype=torch.int32)

    def exercise():
        with invocation.prepare_expert_invocation(
                hidden, ids, probs, None, (hidden, hidden), resources, planner=object()) as prepared:
            assert prepared.execution_resources is child and prepared.weight_binding is binding
            torch.testing.assert_close(prepared.routed_tokens, hidden)
            workspace.release.assert_not_called()

    with (
        patch.object(invocation, "prepare_topk_route", side_effect=prepare),
        patch.object(route, "_read_cached_route", return_value=(route._RouteSummary(1, 128, 2), None, None)),
        patch.object(route, "_permute_topk_input_out", side_effect=publish),
        patch.object(route, "_finalize_dynamic_route", side_effect=lambda pair, *args: PreparedTopKRoute(
            pair[0], pair[1], table[0], table, None)),
    ):
        if fail_publication:
            with pytest.raises(RuntimeError, match="source publication failed"):
                exercise()
        else:
            exercise()
    child.prepare_auto_workspace.assert_called_once_with(128, 1, hidden)
    resources.prepare_auto_workspace.assert_not_called()
    workspace.release.assert_called_once()
    resources.workspace.release.assert_not_called()


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

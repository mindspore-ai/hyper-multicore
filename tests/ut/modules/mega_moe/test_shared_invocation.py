# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Shared error convergence, source leases and dynamic MX plan bounds."""
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_multicore.modules.mega_moe import invocation
from hyper_multicore.modules.mega_moe.route import _INVALID_WEIGHT_MASK
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


@pytest.mark.parametrize('mode', ['push', 'pull'])
def test_error_converges_in_existing_route_and_releases_source_lease(mode):
    """Use the existing collective to reject weights and release the Pull lease."""
    resources = SimpleNamespace(spec=SimpleNamespace(dispatch_mode=mode, auto_tokens=True),
                                workspace=Mock(), prepare_auto_workspace=Mock())
    hidden, ids, probs = torch.ones(1, 64), torch.zeros(1, 1), torch.ones(1, 1)
    def reject(*args, **options):
        del args
        assert options['autograd_mask'] == _INVALID_WEIGHT_MASK
        options['prepare_workspace'](128, 0)
        raise ValueError('collective invalid weight')
    # Invalid roots are deliberately not inspected before the collective.
    with patch.object(invocation, 'prepare_topk_route', side_effect=reject) as route:
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

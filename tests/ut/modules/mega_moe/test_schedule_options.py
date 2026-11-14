"""Protect schedule selection, alias conflicts and normal/profile queue identity."""
from dataclasses import replace
import json
import os
from unittest.mock import patch

import pytest

from hyper_multicore import MegaMoeExperts, MegaMoeScheduleOptions
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.schedule import (
    describe_schedule_policy, explain_schedule, list_schedule_policies, resolve_schedule_options,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


def test_only_implemented_policy_is_available():
    assert list_schedule_policies() == ('two_expert_pipeline',)
    assert describe_schedule_policy('two_expert_pipeline')['transport_fifo']
    with pytest.raises(ValueError, match='Unknown'):
        resolve_schedule_options(MegaMoeScheduleOptions(policy='half16'))


@pytest.mark.parametrize('name,precision', [
    ('HYPER_MULTICORE_MEGAMOE_QUEUE_READY_LOOKAHEAD', 'bf16'),
    ('HYPER_MULTICORE_MEGAMOE_READY_DEPTH', 'mxfp8'),
    ('HYPER_MULTICORE_MX_READY_DEPTH', 'mxfp8'),
    ('HYPER_MULTICORE_BF16_READY_DEPTH', 'bf16'),
])
def test_existing_aliases_and_explicit_options(name, precision):
    with patch.dict(os.environ, {name: '8'}, clear=True):
        assert resolve_schedule_options(precision=precision).queue_ready_lookahead == 8
        assert resolve_schedule_options(MegaMoeScheduleOptions(queue_ready_lookahead=8),
                                        precision=precision).queue_ready_lookahead == 8
        with pytest.raises(ValueError, match='Conflicting'):
            resolve_schedule_options(MegaMoeScheduleOptions(queue_ready_lookahead=1), precision=precision)


def test_conflicting_aliases_rejected_and_options_frozen_at_construction():
    with patch.dict(os.environ, {'HYPER_MULTICORE_MEGAMOE_QUEUE_READY_LOOKAHEAD': '8',
                                'HYPER_MULTICORE_BF16_READY_DEPTH': '1'}, clear=True):
        with pytest.raises(ValueError, match='Conflicting'):
            resolve_schedule_options()
    with patch.dict(os.environ, {}, clear=True):
        layer = MegaMoeExperts(hidden_size=64, intermediate_size=64, num_experts=4, top_k=2,
                               create_parameters=False, schedule=MegaMoeScheduleOptions(queue_ready_lookahead=8))
        try:
            with patch.dict(os.environ, {'HYPER_MULTICORE_MEGAMOE_READY_DEPTH': '0'}):
                assert layer._resource_group.specification['ready_depth'] == 8
                assert layer._resource_group.specification['schedule_policy'] == 'two_expert_pipeline'
        finally:
            layer.close()


@pytest.mark.parametrize('value', [True, -1, 2, 64, 256, 1.0])
def test_unsupported_lookahead_rejected(value):
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match='0, 1 or 8'):
        resolve_schedule_options(MegaMoeScheduleOptions(queue_ready_lookahead=value))


def test_explain_is_read_only_and_profile_uses_same_queues():
    spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 32,
                       transport_engine='udma', ready_depth=8, precision='mxfp8',
                       dispatch_split=512, combine_split=512)
    plan = build_mxfp8_runtime(spec, forward=True)
    cfg = plan.config
    ordinary = explain_schedule(cfg)
    cfg.diagnostic_mode = 1
    profiled = explain_schedule(cfg)
    assert ordinary['queues'] == profiled['queues']
    assert ordinary['tasks'] == profiled['tasks']
    assert ordinary['plan_sha256'] == profiled['plan_sha256']
    assert ordinary['runtime_sha256'] != profiled['runtime_sha256']
    assert json.loads(json.dumps(profiled))['layout'] == 'explicit'
    # Every odd communication-owner queue remains FIFO, even with math ready selection.
    assert all(queue['queue_ready_lookahead'] == 0 for queue in profiled['queues']
               if queue['kind'] == 'aiv' and queue['worker'] % 2)
    alternative = build_mxfp8_runtime(replace(spec, ready_depth=1), forward=True)
    assert ordinary['queues'] != explain_schedule(alternative.config)['queues']

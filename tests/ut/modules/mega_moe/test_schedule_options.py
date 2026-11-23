"""Protect explicit schedule selection and normal/profile queue identity."""

# The assertions verify internal ownership and the native ABI.
# pylint: disable=protected-access
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import MegaMoeSchedulerOptions, MegaMoeTuningOptions
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.schedule import (
    describe_schedule_policy, explain_schedule, list_schedule_policies, validate_schedule_options,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.backends.bf16.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.backends.bf16 import resources as bf16
from hyper_multicore.scheduler.runtime import serialize_runtime_config


def test_only_implemented_policies_are_available():
    """Expose only the implemented scheduling policies."""
    assert list_schedule_policies() == ('two_expert_pipeline', 'builder_order')
    assert describe_schedule_policy('two_expert_pipeline')['transport_fifo']
    with pytest.raises(ValueError, match='Unknown'):
        validate_schedule_options(MegaMoeSchedulerOptions(policy='half16'))


def test_only_explicit_options_apply(monkeypatch):
    """Check that scheduler choices are explicit and scoped to the instance."""
    monkeypatch.setenv("HYPER_MULTICORE_MEGAMOE_READY_DEPTH", "1")
    assert validate_schedule_options(None).queue_ready_lookahead is None
    options = MegaMoeSchedulerOptions(queue_ready_lookahead=8)
    layer = MegaMoeExperts(hidden_size=64, intermediate_size=64, num_experts=4, top_k=2,
                           tuning_options=MegaMoeTuningOptions(scheduler=options))
    try:
        assert layer._resource_group.specification['ready_depth'] == 8
        assert layer._resource_group.specification['schedule_policy'] == 'auto'
        with pytest.raises(AttributeError):
            layer.precision = 'mxfp8'
    finally:
        layer.close()


@pytest.mark.parametrize('value', [True, -1, 2, 64, 256, 1.0, '8'])
def test_unsupported_lookahead_rejected(value):
    """Reject values outside the allowed lookahead set."""
    with pytest.raises(ValueError, match='0, 1 or 8'):
        validate_schedule_options(MegaMoeSchedulerOptions(queue_ready_lookahead=value))


@pytest.mark.parametrize('options', [MegaMoeSchedulerOptions(lowering='compact'),
                                    MegaMoeSchedulerOptions(batch_experts=True),
                                    MegaMoeSchedulerOptions(nominal_schedule=True)])
def test_mx_rejects_unsupported_lowering_at_construction(options):
    """Reject unsupported MX lowering before device binding."""
    with pytest.raises(ValueError, match='dense per-expert lowering'):
        MegaMoeExperts(hidden_size=64, intermediate_size=64, num_experts=4, top_k=2,
                       precision='mxfp8', tuning_options=MegaMoeTuningOptions(scheduler=options))


def test_arch22_rejects_ineffective_explicit_options():
    """Reject scheduler choices without an implementation on arch22."""
    spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 24)
    with pytest.raises(ValueError, match='Named MegaMoE schedule policies require arch35'):
        _build_runtime_artifacts(replace(spec, schedule_policy='builder_order'))
    with pytest.raises(ValueError, match='compact lowering is unavailable'):
        _build_runtime_artifacts(replace(spec, lowering='compact', dispatch_mode='pull'))


def test_bf16_token_bucket_preserves_bound_lowering_options():
    """Keep BF16 lowering options when selecting a token bucket."""
    resources = bf16.BF16Resources.__new__(bf16.BF16Resources)
    resources.spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 24,
                                 batch_experts=True, nominal_schedule=True)
    resources._token_capacity = 512
    resources._plans = {}
    with patch.object(bf16, 'build_mega_moe_plan', side_effect=lambda spec, _, **kwargs: SimpleNamespace(spec=spec)):
        plan = resources.plan_for_tokens(384, 'cpu')
        assert plan.spec.batch_experts and plan.spec.nominal_schedule
        assert resources.plan_for_tokens(384, 'cpu') is plan
        fused = resources.plan_for_tokens(384, 'cpu', replica_pipeline=True)
        assert not fused.spec.batch_experts and fused.spec.nominal_schedule
        assert resources.plan_for_tokens(384, 'cpu', replica_pipeline=True) is fused
        assert resources.plan_for_tokens(384, 'cpu') is plan


@pytest.mark.parametrize('lookahead', [None, 0, 8])
def test_arch22_compact_batched_plan_preserves_vector_ready_queues(lookahead):
    """Verify that compact batched plans retain arch22 vector queues."""
    spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 24,
                       batch_experts=True, nominal_schedule=True, ready_depth=lookahead)
    _, forward, _, backward = _build_runtime_artifacts(spec)
    for config in (forward, backward):
        assert config.queue_layout == 2
        assert all(q.ready_depth == 0 for q in config.worker_queues[:24])
        assert len(config.worker_queues[24:]) == 48
        assert all(q.ready_depth == (256 if lookahead is None else lookahead)
                   for q in config.worker_queues[24:])
        assert serialize_runtime_config(config)


def test_explain_is_read_only_and_profile_uses_same_queues():
    """Check queue identity in schedule explanation and profiling."""
    spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 32,
                       architecture=3510, transport_engine='udma', ready_depth=8, precision='mxfp8',
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


@pytest.mark.parametrize('precision,mode', [('bf16', 'push'), ('bf16', 'pull'),
                                           ('mxfp8', 'push'), ('mxfp8', 'pull')])
def test_builder_order_preserves_task_owners_events_and_transport(precision, mode):
    """Check that builder ordering preserves ownership and dependencies."""
    spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 32,
                       precision=precision, dispatch_mode=mode, architecture=3510, transport_engine='udma',
                       dispatch_split=512, combine_split=512)

    def configs(selected):
        if precision == 'bf16':
            _, fwd, _, bwd = _build_runtime_artifacts(selected)
            return fwd, bwd
        return tuple(build_mxfp8_runtime(selected, forward=value).config for value in (True, False))

    baseline = configs(spec)
    candidate = configs(replace(spec, schedule_policy='builder_order'))
    changed = False
    for left, right in zip(baseline, candidate):
        a, b = explain_schedule(left), explain_schedule(right)
        assert a['tasks'] == b['tasks']
        assert a['event_triggers'] == b['event_triggers']
        assert a['layout'] == b['layout']
        for first, second in zip(a['queues'], b['queues']):
            assert sorted(first['tasks']) == sorted(second['tasks'])
            if first['kind'] == 'aiv' and first['worker'] % 2:
                assert first == second
            changed |= first['tasks'] != second['tasks']
        assert serialize_runtime_config(right)
    assert changed

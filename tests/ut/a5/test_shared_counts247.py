"""Shared count policy selection preserves existing BF16-only inputs and defaults."""
import importlib.util
from pathlib import Path

import pytest


def preprocess():
    path=Path(__file__).resolve().parents[3]/'hyper_multicore/modules/mega_moe/preprocess.py'
    spec=importlib.util.spec_from_file_location('counts247',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('bf16',[True,False])
def test_shared_policy_is_precision_independent_and_default_unchanged(bf16: bool) -> None:
    m=preprocess()
    assert m.resolve_count_policy({},bf16=bf16)=='baseline'
    for policy in ('baseline','histc'):
        assert m.resolve_count_policy({'HYPER_MULTICORE_MOE_COUNT_POLICY':policy},bf16=bf16)==policy


def test_old_bf16_alias_stays_bf16_only_and_conflicts_are_rejected() -> None:
    m=preprocess()
    assert m.resolve_count_policy({'HYPER_MULTICORE_BF16_COUNT_POLICY':'histc'},bf16=True)=='histc'
    assert m.resolve_count_policy({'HYPER_MULTICORE_BF16_COUNT_POLICY':'baseline'},bf16=False)=='baseline'
    assert m.resolve_count_policy({'HYPER_MULTICORE_BF16_COUNT_POLICY':'histc',
                                   'HYPER_MULTICORE_MOE_COUNT_POLICY':'histc'},bf16=True)=='histc'
    with pytest.raises(ValueError,match='BF16'):
        m.resolve_count_policy({'HYPER_MULTICORE_BF16_COUNT_POLICY':'histc'},bf16=False)
    with pytest.raises(ValueError,match='Conflicting'):
        m.resolve_count_policy({'HYPER_MULTICORE_BF16_COUNT_POLICY':'baseline',
                                'HYPER_MULTICORE_MOE_COUNT_POLICY':'histc'},bf16=True)


def test_existing_geometry_checks_apply_to_resolved_shared_policy() -> None:
    m=preprocess()
    for env,experts,dynamic in (({'HYPER_MULTICORE_MOE_COUNT_POLICY':'bogus'},32,None),
                               ({'HYPER_MULTICORE_MOE_COUNT_POLICY':'histc'},1,None),
                               ({'HYPER_MULTICORE_MOE_COUNT_POLICY':'histc'},32,2048)):
        with pytest.raises(ValueError):
            m.validate_count_policy(m.resolve_count_policy(env,bf16=False),experts,dynamic)

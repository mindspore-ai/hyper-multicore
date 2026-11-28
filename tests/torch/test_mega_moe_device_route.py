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
"""Launch isolated bounded-route device checks."""
import json
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case

_WORKER = str(Path(__file__).with_name('_test_mega_moe_device_route.py'))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_shared_metadata_entries(monkeypatch, tmp_path):
    monkeypatch.setenv('HP_MEGA_MOE_EVIDENCE_DIR', str(tmp_path))
    run_multicore_case(_WORKER, 'test_shared_metadata_entries', num_proc=1, timeout=240)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_bounded_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path))
    run_multicore_case(_WORKER, 'test_bounded_metadata', num_proc=1, timeout=180)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_bounded_training(monkeypatch, tmp_path):
    monkeypatch.setenv("HP_MEGA_MOE_LEVEL1_RESULT", str(tmp_path / "result.json"))
    monkeypatch.setenv('HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE', '1')
    run_multicore_case(_WORKER, 'test_bounded_training', num_proc=2, timeout=360)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize('kind', ['capacity', 'negative', 'tokens', 'weights', 'participation',
                                 'remote_source', 'remote_capacity'])
def test_bounded_rejection(monkeypatch, tmp_path, kind):
    result = tmp_path / 'rejection.json'
    monkeypatch.setenv('HMC_BOUNDED_REJECTION', kind)
    monkeypatch.setenv('HMC_REJECTION_RESULT', str(result))
    run_multicore_case(_WORKER, 'test_bounded_rejection', num_proc=1, timeout=120)
    assert json.loads(result.read_text())['rejected']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_mxfp8_bounded_training(monkeypatch, tmp_path):
    monkeypatch.setenv('HP_MEGA_MOE_LEVEL1_RESULT', str(tmp_path / 'mx-result.json'))
    run_multicore_case(_WORKER, 'test_mxfp8_bounded_training', num_proc=2, timeout=600)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_repeated_metadata_launches():
    run_multicore_case(_WORKER, 'test_repeated_metadata_launches', num_proc=1, timeout=240)

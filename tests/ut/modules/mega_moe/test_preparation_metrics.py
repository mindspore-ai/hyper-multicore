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
"""Regression checks for the corrected timing contract, without device claims."""
import pytest
from tests.common.multicore.preparation_metrics import preparation_windows


def event(name, start, duration, category='kernel'):
    return dict(name=name, ts=start, dur=duration, ph='X', cat=category)


@pytest.mark.parametrize('variant', ['native', 'mega'])
def test_device_endpoints_include_free_and_permute_but_ignore_host_api(variant):
    target = 'hcom_alltoallv_ep_0' if variant == 'native' else 'aclnnMegaMoe_MegaMoe'
    trace = [event(f'hmc_prelude/{variant}/fixed/0', 0, 2000, 'user_annotation'),
             event('HcclAllGather', 5, 10, 'cpu_op'),
             event('hcom_allGather_ep_0', 100, 100, 'communication'),
             event('aclrtFree', 210, 550, 'cpu_op'),
             event('aclnnMoeTokenPermute', 770, 170),
             event(target, 1200, 500)]
    result = preparation_windows(trace, 1)[0]
    assert result['allgather_end_us'] == '200'
    assert result['target_start_us'] == '1200'
    assert result['prepare_ms'] == 1.0


def test_ambiguous_allgathers_are_not_silently_matched():
    trace = [event('hmc_prelude/mega/random/0', 0, 3000, 'user_annotation'),
             event('hcom_allGather_counts', 10, 100, 'communication'),
             event('hcom_allGather_fsdp', 300, 500, 'communication'),
             event('aclnnMegaMoe_MegaMoe', 1200, 500)]
    with pytest.raises(ValueError, match='refuse to guess'):
        preparation_windows(trace, 1)


@pytest.mark.parametrize('missing', ['collective', 'kernel', 'marker'])
def test_missing_device_evidence_is_failure(missing):
    events = dict(marker=event('hmc_prelude/mega/empty/0', 0, 3000, 'user_annotation'),
                  collective=event('hcom_allGather_counts', 10, 100, 'communication'),
                  kernel=event('aclnnMegaMoe_MegaMoe', 1200, 500))
    del events[missing]
    with pytest.raises(ValueError):
        preparation_windows(list(events.values()), 1)


def test_cann_string_timestamps_and_hardware_tracks_preserve_sub_microsecond_precision():
    trace = [dict(ph='M', name='process_name', pid=5, args={'name': 'Ascend Hardware'}),
             event('hmc_prelude/mega/random/0', '1795744765417829.930', 3000, 'cpu_op'),
             event('hcom_allGather_counts', '1795744765417900.125', 100.125, None),
             dict(event('aclnnMegaMoe_MegaMoe', '1795744765418200.375', 100, None), pid=5),
             event('aclnnMegaMoe_MegaMoe', '1795744765418100.1', 10, 'cpu_op')]
    found = preparation_windows(trace, 1)[0]
    assert found['prepare_ms'] == 0.200125
    assert found['allgather_end_us'] == '1795744765418000.250'

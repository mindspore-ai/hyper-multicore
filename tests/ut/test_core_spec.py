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
"""CoreSpec values and Device queries."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, call, patch

import pytest
import torch

from hyper_multicore import _core_spec
from hyper_multicore._core_spec import CoreSpec, get_device_core_spec


@pytest.mark.parametrize("field", ["num_cube_cores", "num_vector_cores"])
@pytest.mark.parametrize("value,error", [(True, TypeError), (1.5, TypeError),
                                        ("24", TypeError), (0, ValueError), (-1, ValueError)])
def test_invalid_core_counts(field, value, error):
    values = {"num_cube_cores": 20, "num_vector_cores": 40}
    values[field] = value
    with pytest.raises(error, match=field):
        CoreSpec(**values)


def test_snapshot_is_immutable_and_does_not_impose_a_launch_ratio():
    spec = CoreSpec(num_cube_cores=20, num_vector_cores=32)
    with pytest.raises(FrozenInstanceError):
        spec.num_cube_cores = 16


def test_query_reads_each_device_limit_without_caching():
    """Observe updated limits on every query for the selected device."""
    query = Mock(side_effect=[
        {"cube_core_num": 24, "vector_core_num": 48},
        {"cube_core_num": 16, "vector_core_num": 32},
    ])
    with patch.object(torch, "npu", Mock(get_device_limit=query), create=True):
        first = get_device_core_spec(3)
        second = get_device_core_spec(3)
    assert first == CoreSpec(24, 48)
    assert second == CoreSpec(16, 32)
    assert query.call_args_list == [call(3), call(3)]


def test_query_does_not_fall_back_when_device_query_fails():
    error = RuntimeError("device query failed")
    with patch.object(torch, "npu", Mock(get_device_limit=Mock(side_effect=error)), create=True):
        with pytest.raises(RuntimeError) as caught:
            get_device_core_spec(0)
    assert caught.value is error


@pytest.mark.parametrize("limits,error", [
    ({"cube_core_num": 24}, KeyError),
    ({"cube_core_num": 24, "vector_core_num": 0}, ValueError),
    ({"cube_core_num": True, "vector_core_num": 48}, TypeError),
])
def test_invalid_device_limits_are_not_replaced_by_defaults(limits, error):
    with patch.object(torch, "npu", Mock(get_device_limit=Mock(return_value=limits)), create=True):
        with pytest.raises(error):
            get_device_core_spec(0)


def test_offline_spec_module_needs_no_torch_import():
    """Load offline defaults without importing either framework in the subprocess."""
    source = Path(_core_spec.__file__)
    # Load the module directly to isolate its dependencies from the package's
    # existing Torch-based diagnostic exports.
    code = '''import builtins, runpy, sys
original_import = builtins.__import__
def offline_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torch_npu'):
        raise AssertionError('offline CoreSpec imported an NPU framework')
    return original_import(name, *args, **kwargs)
builtins.__import__ = offline_import
values = runpy.run_path(sys.argv[1])
spec = values['CoreSpec'](32, 64)
assert spec.num_cube_cores == 32
assert values['DEFAULT_CORE_SPEC'].num_vector_cores == 48
'''
    subprocess.run([sys.executable, "-c", code, str(source)], check=True, capture_output=True, text=True)

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
"""Check device-lease waiting and worker deadlines without an NPU."""

from unittest.mock import MagicMock, call, patch

import pytest

from tests.common.multicore import _case_launcher


@pytest.mark.parametrize(("device_timeout", "worker_timeout"), [(None, 5), (900, 420)])
def test_device_contention_and_worker_deadline(monkeypatch, tmp_path, device_timeout, worker_timeout):
    """A separately bounded lease wait leaves the entire worker deadline available."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("HYPER_MULTICORE_TEST_DEVICE_TYPE", raising=False)
    monkeypatch.setattr(_case_launcher.tempfile, "gettempdir", lambda: str(tmp_path))
    process = MagicMock()
    process.__enter__.return_value = process
    process.wait.return_value = 0

    with (
        patch.object(_case_launcher.time, "monotonic", side_effect=[0, 415, 415, 415]),
        patch.object(_case_launcher.time, "sleep"),
        patch.object(_case_launcher.fcntl, "flock", side_effect=[BlockingIOError, None]),
        patch.object(_case_launcher.subprocess, "Popen", return_value=process),
        patch.object(_case_launcher.os, "killpg"),
    ):
        _case_launcher.run_multicore_case("worker.py", "compact", timeout=420, device_timeout=device_timeout)

    assert process.wait.call_args_list == [call(timeout=worker_timeout), call()]

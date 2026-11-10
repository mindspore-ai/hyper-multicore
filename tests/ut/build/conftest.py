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
"""Keep build tests from reading or cleaning a developer's configured venv storage."""

import pytest


@pytest.fixture(autouse=True)
def isolate_venv_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests explicitly select temporary roots instead of inheriting real storage."""
    monkeypatch.delenv("HYPER_MULTICORE_VENV_PATH", raising=False)

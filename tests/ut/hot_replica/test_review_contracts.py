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
"""Exercise the placement and resource cleanup without starting a device runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hyper_multicore.modules.hot_replica import shmem_transfer


def test_retry_free_skips_an_already_released_symmetric_allocation(monkeypatch):
    pool = SimpleNamespace(weights=("first", "second"), pipeline=None)
    free = Mock(side_effect=[None, RuntimeError("busy"), None])
    monkeypatch.setattr(shmem_transfer.shmem, "free", free)
    with pytest.raises(RuntimeError, match="busy"):
        shmem_transfer.SharedExpertPool._free(pool)  # pylint: disable=protected-access
    assert pool.weights == ("second",)
    shmem_transfer.SharedExpertPool._free(pool)  # pylint: disable=protected-access
    assert [c.args for c in free.call_args_list] == [("first",), ("second",), ("second",)]
    assert not pool.weights

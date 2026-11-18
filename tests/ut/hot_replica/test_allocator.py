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
"""CPU contracts for native HotReplica allocator loading and device UB forwarding."""

from types import ModuleType
from unittest.mock import Mock, patch

import torch

from hyper_multicore.modules.hot_replica import allocator
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_allocator_cache_preserves_native_entry_and_forwards_ub() -> None:
    """Repeated imports must retain the native entry and pass the queried device UB."""
    native = ModuleType("test_native_hot_replica_allocator")
    result = [torch.empty(0)]
    original_allocate = Mock(return_value=result)
    native.allocate = original_allocate
    counts = Mock(spec=torch.Tensor)
    counts.device.index = 3
    counts.device.type = "npu"
    allocator.load_allocator.cache_clear()
    try:
        with (
            patch.object(allocator, "load_native_extension", return_value=native) as imported,
            patch.object(allocator, "_device_ub_bytes", return_value=191 * 1024) as ub_bytes,
        ):
            assert allocator.load_allocator() is native
            assert allocator.load_allocator() is native
            assert allocator.allocate_experts(counts, 128) is result
            assert allocator.allocate_experts(counts, 256) is result
            imported.assert_called_once_with("_a3_allocator")
            ub_bytes.assert_called_with(3)
            assert ub_bytes.call_count == 2
        assert getattr(native, "allocate") is original_allocate
        assert original_allocate.call_args_list[0].args == (counts, 128, 191 * 1024)
        assert original_allocate.call_args_list[1].args == (counts, 256, 191 * 1024)
    finally:
        allocator.load_allocator.cache_clear()

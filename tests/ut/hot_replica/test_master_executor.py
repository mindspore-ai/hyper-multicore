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
"""Master binds the borrowed expert layout from the parent's current weights."""

from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from hyper_multicore.modules.hot_replica.bounded_placement import supports_bounded_placement
from hyper_multicore.modules.hot_replica.pipeline import ReplicaPipeline, supports_pipeline
from hyper_multicore.modules.hot_replica.shmem_transfer import SharedExpertPool


@pytest.mark.parametrize(("first", "second"), [("EIO", "EOI"), ("EOI", "EIO")])
def test_bank_layout_views_follow_the_lease(first, second):

    """Reuse the same symmetric allocation for both supported weight layouts."""
    pool = SharedExpertPool.__new__(SharedExpertPool)
    pool.local_experts, pool.hidden, pool.intermediate = 2, 7, 3
    pool.expert_weight_layout = first
    pool.weights = tuple(torch.zeros(shape) for shape in pool._bank_shapes())  # pylint: disable=protected-access
    pool._alternate_banks = pool.leased_banks = ()  # pylint: disable=protected-access
    pool.access = nullcontext
    pool.lock = RLock()
    pool.closed = pool.in_use = pool.used = False
    pool.completion_event = Mock()
    pool.device = torch.device("cpu")
    stream = Mock()

    with patch.object(torch.npu, "current_stream", return_value=stream):
        for layout in (first, second, first, second):
            pool.claim(layout)
            assert [tuple(w.shape) for w in pool.leased_banks] == list(pool._bank_shapes(layout))  # pylint: disable=protected-access
            assert all(view.data_ptr() == owner.data_ptr() for view, owner in zip(pool.leased_banks, pool.weights))
            pool.release()
            assert not pool.leased_banks


@pytest.mark.parametrize(("layout", "bands"), [("EIO", 41), ("EOI", 42)])
def test_replica_signal_extent_follows_the_physical_gradient_rows(layout, bands):
    """Budget each projection's actual row bands and keep over-width placement valid."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=False, dispatch_mode="push", transport_engine="mte",
        expert_weight_layout=layout, hidden_size=5248, intermediate_size=2688, local_experts=6,
    )
    pool = SimpleNamespace(hidden=spec.hidden_size, intermediate=spec.intermediate_size,
                           expert_weight_layout=layout, ranks=32, blocks=24, local_experts=6)
    assert supports_bounded_placement(spec)
    assert supports_pipeline(spec, 4)
    assert not supports_pipeline(spec, 5)
    expected_words = (3 * pool.ranks + 2 + bands * pool.blocks) * pool.local_experts * 16
    assert ReplicaPipeline(pool).signal_words == expected_words

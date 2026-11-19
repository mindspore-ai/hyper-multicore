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
from hyper_multicore.modules.hot_replica._planner._dispatch import _materialize
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica.bounded_placement import _render_geometry, supports_bounded_placement
from hyper_multicore.modules.hot_replica.pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline
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


def test_fused_backward_invalidates_a_generic_bank_generation_even_with_resident_weights():
    """Retiring the BF16 weight lease also covers FP32 dW writes into the same bank."""
    bank = (torch.empty((4, 4, 8)), torch.empty((4, 8, 2)))
    pool = SimpleNamespace(local_experts=2, ranks=2, rank=0, hidden=8, intermediate=2, blocks=2,
                           expert_weight_layout="EOI", leased_banks=bank, generation=0)
    pipeline = ReplicaPipeline(pool)
    pipeline.storage = torch.empty(pipeline.signal_words, dtype=torch.int32)
    binding = SimpleNamespace(token=object(), edges=(0, (), ()), placement=SimpleNamespace(borrowed_capacity=1))
    weights = pipeline.begin(bank, binding, backward=False)
    assert all(weight.shape[0] == 3 for weight in weights)
    assert pool.generation == 1
    staged = pool.generation
    pipeline.begin(bank, binding, backward=True)
    assert binding.weight_epoch == 1 and binding.gradient_epoch == 2
    assert pool.generation > staged


@pytest.mark.parametrize("layout", ["EIO", "EOI"])
@pytest.mark.parametrize(("used", "batch_experts", "width"), [(4, False, 4), (5, False, 6),
                                                            (6, False, 6), (5, True, 6)])
def test_fusion_accepts_an_empty_tail_in_the_padded_copy_table(layout, used, batch_experts, width):
    """Retain the original cached geometry and only expose actual guests to the fused bank."""
    home, ranks = 8, 2
    config = ExpertReplicaConfig(home * ranks, ranks, 6)
    spec = SimpleNamespace(
        architecture=2201, batch_experts=batch_experts, dispatch_mode="push", transport_engine="mte",
        expert_weight_layout=layout, hidden_size=512, intermediate_size=256, local_experts=home, rank_id=1, top_k=1,
    )
    counts = ((2,) * config.num_experts,) * ranks
    quotas = [{expert: 4 for expert in range(rank * home, (rank + 1) * home)} for rank in range(ranks)]
    for expert in range(home, home + used):
        quotas[0][expert] = quotas[1][expert] = 2
    plan = _materialize(counts, quotas, config)
    plan.validate()
    geometry = _render_geometry(plan, spec, "cpu", config.num_experts * 2)
    assert geometry.width == width
    assert geometry.owners == tuple(row[home:home+width] for row in plan.slot_to_logical)
    expected = torch.tensor(plan.dispatch_counts).reshape(ranks, ranks, config.slots_per_rank)
    expected = expected[:, :, :home+width].reshape(ranks, -1)
    assert torch.equal(geometry.counts, expected)
    # The local rank has no guests. Its published_slots value must not affect
    # the fused extent agreed by the whole EP group.
    placement = SimpleNamespace(published_slots=0, borrowed_capacity=geometry.width, copies_by_rank=geometry.owners)
    slots = fused_replica_slots(spec, placement)
    assert slots == (used if not batch_experts and used <= 5 else 0)
    if slots:
        pipeline = ReplicaPipeline(SimpleNamespace(local_experts=home, rank=1))
        planned, received, owned = pipeline.plan(placement, slots)
        assert planned == len(received) == used
        assert owned == tuple((0, j, j) for j in range(used))

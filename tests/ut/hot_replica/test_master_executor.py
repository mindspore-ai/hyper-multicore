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
from hyper_multicore.modules.hot_replica import mega_moe_adapter
from hyper_multicore.modules.hot_replica._planner._dispatch import _materialize
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica.bounded_placement import _render_geometry, supports_bounded_placement
from hyper_multicore.modules.hot_replica.pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline
from hyper_multicore.modules.hot_replica.shmem_transfer import ExpertWeightBinding, SharedExpertPool
from hyper_multicore.modules.mega_moe.backends.bf16 import resources as bf16


def test_serial_replica_widths_share_one_reservation_with_the_bank():
    """Keep one virtual reservation when the home counters cannot cover all widths."""
    spec = SimpleNamespace(local_experts=6, ep_size=32, top_k=8, initial_capacity_factor=1.25,
                           expert_weight_layout="EIO")
    owner = mega_moe_adapter.PlacementResources.__new__(mega_moe_adapter.PlacementResources)
    owner.parent = SimpleNamespace(spec=spec, heap_manager=SimpleNamespace(heap_limit=None))
    owner.pool, owner.geometries = None, {}
    owner.source_bound, owner.shared, owner.bounded, owner.specification = None, True, object(), {}
    owner.reuses_parent_workspace = False
    owner.pipeline_enabled = True
    hidden = torch.empty(0, dtype=torch.bfloat16)
    layout = Mock()
    with (patch.object(mega_moe_adapter, "_workspace_layout", return_value=layout) as sizing,
          patch.object(mega_moe_adapter, "_get_mega_bank") as bank,
          patch.object(bf16, "BF16Resources") as resource):
        first = owner._geometry(hidden, SimpleNamespace(borrowed_capacity=1, receive_capacity=33280), 4096)
        second = owner._geometry(hidden, SimpleNamespace(borrowed_capacity=2, receive_capacity=34176), 4096)
    assert first is second
    specification, dtype, receive, source = sizing.call_args.args
    assert (specification["num_experts"], dtype, receive, source) == (384, torch.bfloat16, 40960, 4096)
    bank.assert_called_once_with(hidden, owner.parent, owner, pipeline=True, reservations=((specification, layout),))
    resource.assert_called_once()
    assert resource.call_args.kwargs["workspace_layout"] is layout


def test_serial_replica_widths_reuse_home_when_event_capacity_matches():
    """B0 and replica calls can use the same communication buffers with distinct plans."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=False, dispatch_mode="push", transport_engine="mte",
        local_experts=6, num_experts=192, ep_size=32, top_k=8, initial_capacity_factor=1.25,
        expert_weight_layout="EOI", hidden_size=5120, intermediate_size=1792,
    )
    parent = SimpleNamespace(spec=spec, workspace=object(), heap_manager=SimpleNamespace(heap_limit=None))
    planner = mega_moe_adapter.HotReplicaPlanner
    owner = mega_moe_adapter.PlacementResources(
        parent, {"planner_key": (planner.__module__, planner.__qualname__)}, shared=False,
    )
    hidden = torch.empty(0, dtype=torch.bfloat16)
    with patch.object(mega_moe_adapter, "_get_mega_bank") as bank:
        for width in (1, 2, 6):
            assert owner._geometry(hidden, SimpleNamespace(borrowed_capacity=width, receive_capacity=33792), 4096) is parent
    bank.assert_called_once_with(hidden, parent, owner, pipeline=True,
                                 workspace_growth=(parent.workspace, 4096, 33792))
    assert not owner.geometries


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
    binding = SimpleNamespace(token=object(), edges=(0, (), ()), snapshot=None,
                              placement=SimpleNamespace(borrowed_capacity=1))
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


def test_inline_pipeline_fits_between_weight_gradient_and_checkpoint_views():
    """Inline epochs have no extra allocations and clear after a generic write."""
    home, hidden, inner = 6, 256, 128
    bank = torch.ones((2*home, hidden, 2*inner), dtype=torch.bfloat16)
    pool = SimpleNamespace(local_experts=home, rank=0, ranks=2, blocks=24, hidden=hidden, intermediate=inner,
                           expert_weight_layout="EIO", leased_banks=(bank, bank), generation=0)
    pipeline = ReplicaPipeline(pool, inline=True)
    assert pipeline.slot_limit == 3
    binding = SimpleNamespace(edges=(3, (), ()), token=object(), snapshot=None,
                              placement=SimpleNamespace(borrowed_capacity=4))
    with patch("hyper_multicore.modules.hot_replica.pipeline.shmem.barrier") as barrier:
        pipeline.begin((bank, bank), binding, backward=False)
        storage = pipeline.storage
        assert storage.data_ptr() == bank[3].data_ptr()
        dW_offset = (2*home-2*pipeline.slot_limit) * bank[0].numel() * bank.element_size()
        end = storage.data_ptr() + storage.numel() * storage.element_size()
        assert end <= bank.data_ptr() + dW_offset and end <= bank[home].data_ptr()
        assert bool((storage == 0).all())
        storage.fill_(123)
        pipeline.invalidate()
        assert pipeline.resident is None
        pipeline.begin((bank, bank), binding, backward=True)
        assert pipeline.storage.data_ptr() == storage.data_ptr()
        assert bool((pipeline.storage == 0).all())
        assert barrier.call_count == 2


def test_inline_signal_capacity_cannot_admit_a_real_fourth_slot_for_six_home_experts():
    """A full weight and dW bank leaves its occupied fourth slot on the generic path."""
    placement = SimpleNamespace(borrowed_capacity=4, copies_by_rank=((6, 7, 8, 9), (-1,)*4))
    spec = SimpleNamespace(architecture=2201, batch_experts=False, dispatch_mode="push", transport_engine="mte",
                           expert_weight_layout="EOI", hidden_size=256, intermediate_size=128, local_experts=6)
    assert fused_replica_slots(spec, placement) == 4
    assert fused_replica_slots(spec, placement, slot_limit=3) == 0
    placement.copies_by_rank = ((6, 7, 8, -1), (-1,)*4)
    assert fused_replica_slots(spec, placement, slot_limit=3) == 3


def test_fused_checkpoint_restores_locally_only_after_the_binding_is_selected():
    """An agreed replay publishes the restored prefix, without a peer wait."""
    bank = (torch.empty((4, 4, 8)), torch.empty((4, 8, 2)))
    pool = SimpleNamespace(local_experts=2, ranks=2, rank=0, hidden=8, intermediate=2, blocks=2,
                           expert_weight_layout="EOI", leased_banks=bank, generation=0,
                           restore_snapshot=Mock(), statistics={})
    pipeline = ReplicaPipeline(pool)
    pipeline.storage = torch.zeros(pipeline.signal_words, dtype=torch.int32)
    binding = SimpleNamespace(token=object(), snapshot=object(), edges=(1, (1,), ()),
                              placement=SimpleNamespace(borrowed_capacity=2))
    with patch.object(pipeline, "_publish_weights") as publish:
        pipeline.begin(bank, binding, backward=False)
        pipeline.begin(bank, binding, backward=True)
        assert not publish.called
    pool.restore_snapshot.assert_called_once_with(binding.snapshot, 0)
    assert binding.weight_epoch == 1 and binding.gradient_epoch == 2
    assert pool.statistics["fused_snapshot_restores"] == 1


def test_weight_only_pipeline_drains_projection_copies_before_ordered_gradient_return():
    """Keep the native dW destinations for the wide split-weight path."""
    pool = Mock()
    pool.pipeline.plan.return_value = (5, (), ((1, 0, 0),))
    placement = SimpleNamespace(copies=object(), published_slots=5)
    binding = ExpertWeightBinding(pool, placement, "EOI", fused=True, fused_slots=5, return_in_kernel=False)
    weights = (torch.empty(1, dtype=torch.bfloat16), torch.empty(1, dtype=torch.bfloat16))
    gradients = (object(), object())
    ordinary = object()
    assert not binding.owns_returns
    assert binding.runtime(ordinary, gradients) == pool.pipeline.runtime.return_value
    pool.pipeline.runtime.assert_called_once_with(ordinary, binding, None)
    pool.reset_mock()
    assert binding.gradients(gradients, weights) == pool.return_generated_gradients.return_value
    assert [call[0] for call in pool.method_calls] == ["pipeline.complete", "return_generated_gradients"]
    pool.return_generated_gradients.assert_called_once_with(
        gradients, placement.copies, output_dtypes=(torch.bfloat16, torch.bfloat16), published_slots=5,
        prefer_buffered=True)

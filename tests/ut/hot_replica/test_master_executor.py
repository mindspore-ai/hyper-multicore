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
from hyper_multicore.modules.hot_replica.pipeline import (
    ReplicaPipeline, _band_layout, fused_replica_slots, supports_pipeline,
)
from hyper_multicore.modules.hot_replica.shmem_transfer import ExpertWeightBinding, SharedExpertPool
from hyper_multicore.modules.mega_moe.backends.bf16 import resources as bf16
from hyper_multicore.modules.mega_moe.heap_manager import _HeapManager


def test_wide_replica_projection_fits_the_kernel_band_table():
    """Widen dW bands when a projection exceeds the normal 64-band span."""
    assert _band_layout((8192, 3584)) == (292, 29)
    assert _band_layout((8192, 9216)) == (128, 64)
    assert _band_layout((9216, 8192)) == (144, 64)


def test_serial_replica_widths_share_one_reservation_with_the_bank():
    """Keep one virtual reservation when the home counters cannot cover all widths."""
    spec = SimpleNamespace(local_experts=6, ep_size=32, top_k=8, initial_capacity_factor=1.25,
                           partial_weight_grad_dtype=None, batch_experts=False,
                           expert_weight_layout="EIO")
    owner = mega_moe_adapter.PlacementResources.__new__(mega_moe_adapter.PlacementResources)
    owner.parent = SimpleNamespace(spec=spec, heap_manager=object())
    owner.pool, owner.geometries = None, {}
    owner.source_bound, owner.shared, owner.bounded, owner.specification = None, True, object(), {}
    owner.reuses_parent_workspace = False
    owner.pipeline_enabled = True
    hidden = torch.empty(0, dtype=torch.bfloat16)
    layout = Mock()
    with (patch.object(mega_moe_adapter, "_workspace_layout", return_value=layout) as sizing,
          patch.object(mega_moe_adapter, "_get_mega_bank") as bank,
          patch.object(bf16, "BF16Resources") as resource):
        first = owner._geometry(hidden, SimpleNamespace(borrowed_capacity=1, receive_capacity=33280,
                                                        copies_by_rank=((6,), (-1,))), 4096)
        second = owner._geometry(hidden, SimpleNamespace(borrowed_capacity=2, receive_capacity=34176), 4096)
    assert first is second
    specification, dtype, receive, source = sizing.call_args.args
    assert (specification["num_experts"], dtype, receive, source) == (384, torch.bfloat16, 40960, 4096)
    bank.assert_called_once_with(hidden, owner.parent, owner, pipeline=True,
                                 reservations=((specification, layout),))
    resource.assert_called_once()
    assert resource.call_args.kwargs["workspace_layout"] is layout


def test_serial_replica_widths_reuse_home_when_event_capacity_matches():
    """B0 and replica calls can use the same communication buffers with distinct plans."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=False, dispatch_mode="push", transport_engine="mte",
        local_experts=6, num_experts=192, ep_size=32, top_k=8, initial_capacity_factor=1.25,
        expert_weight_layout="EOI", hidden_size=5120, intermediate_size=1792, partial_weight_grad_dtype=None,
    )
    parent = SimpleNamespace(spec=spec, workspace=object(), heap_manager=object())
    planner = mega_moe_adapter.HotReplicaPlanner
    owner = mega_moe_adapter.PlacementResources(
        parent, {"planner_key": (planner.__module__, planner.__qualname__)}, shared=False,
    )
    hidden = torch.empty(0, dtype=torch.bfloat16)
    with patch.object(mega_moe_adapter, "_get_mega_bank") as bank:
        for width in (1, 2, 6):
            placement = SimpleNamespace(borrowed_capacity=width, receive_capacity=33792, copies_by_rank=((6,), (-1,)))
            assert owner._geometry(hidden, placement, 4096) is parent
    bank.assert_called_once_with(hidden, parent, owner, pipeline=True,
                                 workspace_growth=(parent.workspace, 4096, 33792))
    assert not owner.geometries


@pytest.mark.parametrize(("first", "second"), [("EIO", "EOI"), ("EOI", "EIO")])
def test_bank_layout_views_follow_the_lease(first, second):

    """Reuse the same symmetric allocation for both supported weight layouts."""
    pool = SharedExpertPool.__new__(SharedExpertPool)
    pool.local_experts, pool.hidden, pool.intermediate = 2, 7, 3
    pool.bank_experts = 3 * pool.local_experts
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
    assert ReplicaPipeline(pool, shared_layouts=True).signal_words == (
        (3 * pool.ranks + 2 + 42 * pool.blocks) * pool.local_experts * 16
    )


@pytest.mark.parametrize(("home", "layout", "bound", "partial_dtype"), [
    (4, "EOI", None, None), (6, "EIO", 4096, None), (8, "EOI", None, "float32"),
])
def test_fused_bank_reserves_room_for_later_routes_and_separate_flags(home, layout, bound, partial_dtype):
    """Register the replica workspace, weight bank and peer flags in one reserve."""
    spec = SimpleNamespace(
        ep_group=None, local_experts=home, hidden_size=256, intermediate_size=128,
        num_cube_cores=24, expert_weight_layout=layout, partial_weight_grad_dtype=partial_dtype,
    )
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    manager = _HeapManager(None, hidden)
    parent = SimpleNamespace(spec=spec, heap_manager=manager)
    owner = mega_moe_adapter.PlacementResources.__new__(mega_moe_adapter.PlacementResources)
    owner.source_bound, owner.parent = bound, parent
    owner.dedicated_pipeline = bound is None
    virtual = ({}, mega_moe_adapter._WorkspaceLayout(128, (), 32768))
    extent = 3 * home if bound is not None or partial_dtype == "float32" else 2 * home
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=32),
          patch.object(manager, "reserve") as reserve, patch.object(manager, "bind"),
          patch.object(SharedExpertPool, "allocate")):
        pool = mega_moe_adapter._get_mega_bank(hidden, parent, owner, reservations=(virtual,), pipeline=True)
    assert pool.bank_experts == extent
    assert reserve.call_args.args[0][0] is virtual
    bank_layout = reserve.call_args.args[0][1][1]
    bank_bytes = 3 * extent * spec.hidden_size * spec.intermediate_size * hidden.element_size() + 1024
    assert bank_layout.required_bytes == bank_bytes + 4 * pool.pipeline.signal_words + 512
    pool.leased_banks = tuple(torch.empty(shape, device="meta") for shape in pool._bank_shapes())
    assert all(bank.shape[0] == 2 * home for bank in pool._generic_banks())


def test_optional_replica_banks_share_only_matching_partial_formats():
    """Reuse one bank across serial owners with the same physical gradient extent."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=True, dispatch_mode="push", transport_engine="mte",
        ep_group=None, ep_size=4, local_experts=6, num_experts=24, hidden_size=256,
        intermediate_size=128, num_cube_cores=24, expert_weight_layout="EIO",
        partial_weight_grad_dtype=None,
    )
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    manager = _HeapManager(None, hidden)
    planner = mega_moe_adapter.HotReplicaPlanner
    specification = {"planner_key": (planner.__module__, planner.__qualname__)}
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=spec.ep_size),
          patch.object(manager, "reserve"), patch.object(manager, "bind"),
          patch.object(SharedExpertPool, "allocate")):
        owners, pools = [], []
        for layout, partial_dtype in [("EIO", None), ("EOI", None), ("EOI", "float32")]:
            current = SimpleNamespace(**{**vars(spec), "expert_weight_layout": layout,
                                         "partial_weight_grad_dtype": partial_dtype})
            parent = SimpleNamespace(spec=current, heap_manager=manager)
            owners.append(mega_moe_adapter.PlacementResources(parent, specification, False, bf16_partials=True))
            pools.append(mega_moe_adapter._get_mega_bank(hidden, parent, owners[-1], pipeline=True))
    assert pools[0] is pools[1]
    assert [pool.bank_experts for pool in pools] == [12, 12, 18]
    assert pools[0].pipeline.shared_layouts


@pytest.mark.parametrize("home", [4, 6, 8])
@pytest.mark.parametrize("layout", ["EIO", "EOI"])
def test_bf16_partial_return_only_replaces_wide_native_routes(home, layout):
    """Keep narrow FP32 fusion and a full-width path for each partial dtype."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=False, dispatch_mode="push", transport_engine="mte",
        local_experts=home, num_experts=2 * home, ep_size=2, top_k=4, num_cube_cores=24,
        expert_weight_layout=layout, hidden_size=5120, intermediate_size=1792,
        partial_weight_grad_dtype=None,
    )
    parent = SimpleNamespace(spec=spec)
    planner = mega_moe_adapter.HotReplicaPlanner
    configuration = {"planner_key": (planner.__module__, planner.__qualname__)}
    empty = (-1,) * home
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    for dtype, capability, expected in ((None, False, (False, False)),
                                        (None, True, (True, True)),
                                        ("float32", True, (True, False))):
        spec.partial_weight_grad_dtype = dtype
        owner = mega_moe_adapter.PlacementResources(parent, configuration, False, bf16_partials=capability)
        pool = SimpleNamespace(local_experts=home, rank=0, blocks=spec.num_cube_cores, bank_experts=owner.bank_extent)
        pool.pipeline = ReplicaPipeline(pool)
        owner.pool = pool
        with patch.object(owner, "_geometry", return_value=parent):
            for used, mode in ((1, (True, False)), (home, expected)):
                incoming = tuple(range(home, home + used)) + empty[used:]
                placement = SimpleNamespace(borrowed_capacity=home, copies_by_rank=(incoming, empty))
                _, _, binding = owner.bind(hidden, placement, tokens=4096, fused=owner.pipeline_enabled)
                assert binding.fused
                assert (binding.fused_gradients, binding.bf16_partials) == mode


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


@pytest.mark.parametrize(("home", "layout"), [(4, "EOI"), (6, "EIO"), (8, "EOI")])
def test_fused_bank_rebuilds_flags_and_restores_a_retained_binding(home, layout):
    """Restart gives the flags independent storage and the retained graph can republish weights."""
    hidden = torch.empty((0, 16), dtype=torch.bfloat16)
    manager = _HeapManager(None, hidden)
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=2)):
        pool = SharedExpertPool(hidden, None, home, 16, 2, layout, root=manager,
                                access=manager.access, pipeline=True)
    pool.bank_experts = 3 * home
    pool.weights = tuple(torch.full(shape, 2, dtype=hidden.dtype) for shape in pool._bank_shapes())
    pool.pipeline.storage = torch.full((pool.pipeline.signal_words,), 5, dtype=torch.int32)
    placement = SimpleNamespace(
        borrowed_capacity=home, published_slots=home, copies_by_rank=((-1,) * home,) * 2,
    )
    binding = ExpertWeightBinding(pool, placement, layout, fused=True)
    with (patch("torch.npu.Event"), patch("torch.npu.current_stream"), patch.object(torch.Tensor, "record_stream"),
          patch("hyper_multicore.shmem.barrier") as barrier):
        binding.save_snapshot(pool.weights)
        old, specifications = pool._allocations(hidden.dtype)
        assert old[-1] is pool.pipeline.storage
        assert len(old) == len(specifications) == 3
        replacement = tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specifications)
        pool.bind_restarted_allocations(None, hidden.dtype, hidden.device, 0, replacement)
        barrier.assert_called_once_with(blocking=False)
        assert pool.pipeline.epoch == 0
        assert pool.pipeline.storage is replacement[-1] and not torch.count_nonzero(replacement[-1])
        pool.leased_banks = pool.weights
        pool.pipeline.begin(pool.weights, binding, backward=True)
        assert all(torch.equal(tensor[:home], saved) for tensor, saved in zip(pool.weights, binding.snapshot[0]))
        assert pool.pipeline.epoch == 1


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
    assert slots == (used if used <= 5 else 0)
    assert mega_moe_adapter._live_slot_extent(placement) == used
    assert fused_replica_slots(spec, placement, bank_experts=max(2 * home, 3 * used)) == used
    if slots:
        pipeline = ReplicaPipeline(SimpleNamespace(local_experts=home, rank=1))
        planned, received, owned = pipeline.plan(placement, slots)
        assert planned == len(received) == used
        assert owned == tuple((0, j, j) for j in range(used))


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

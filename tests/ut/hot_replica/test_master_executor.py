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
from hyper_multicore.modules.hot_replica.bounded_placement import render_geometry, supports_bounded_placement
from hyper_multicore.modules.hot_replica.mega_moe_adapter import _get_mega_bank
from hyper_multicore.modules.hot_replica.pipeline import (
    ReplicaPipeline, _band_layout, fused_replica_slots, supports_pipeline,
)
from hyper_multicore.modules.hot_replica.shmem_transfer import ExpertWeightBinding, ReplicaExecutionMode, SharedExpertPool
from hyper_multicore.modules.mega_moe.heap_manager import _HeapCapacityError, _HeapManager


def _builtin_placement(parent, *, bf16_partials=False, replica_bank="shared"):
    planner = mega_moe_adapter.HotReplicaPlanner
    options = {"planner_key": (planner.__module__, planner.__qualname__), "replica_bank": replica_bank}
    return mega_moe_adapter.PlacementResources(parent, options, shared=False, bf16_partials=bf16_partials)


def _a3_push_spec(home, layout, hidden, intermediate, **options):
    """Build the A3 MTE shape and leave call-specific capabilities explicit."""
    return SimpleNamespace(
        architecture=2201, dispatch_mode="push", transport_engine="mte", local_experts=home,
        expert_weight_layout=layout, hidden_size=hidden, intermediate_size=intermediate, **options,
    )


def test_wide_replica_projection_fits_the_kernel_band_table():
    """Widen dW bands when a projection exceeds the normal 64-band span."""
    assert _band_layout((8192, 3584)) == (292, 29)
    assert _band_layout((8192, 9216)) == (128, 64)
    assert _band_layout((9216, 8192)) == (144, 64)


@pytest.mark.parametrize("home", (6, 10))
def test_serial_replica_widths_use_the_same_workspace_and_bank(home):
    """Bind all widths to the regular workspace, including wider event graphs."""
    spec = _a3_push_spec(home, "EOI", 5120, 1792, batch_experts=False, num_experts=64 * home,
                         ep_size=64, top_k=8, initial_capacity_factor=1.25, partial_weight_grad_dtype=None)
    parent = SimpleNamespace(spec=spec, workspace=object(), heap_manager=object())
    owner = _builtin_placement(parent)
    hidden = torch.empty(0, dtype=torch.bfloat16)
    with patch.object(mega_moe_adapter, "_get_mega_bank") as bank:
        for width in (1, 2, home):
            table = torch.empty((spec.ep_size, spec.num_experts + spec.ep_size * width))
            placement = SimpleNamespace(
                borrowed_capacity=width, receive_capacity=33792, counts_by_source=table, copies_by_rank=())
            binding = owner.bind(hidden, placement, tokens=4096)
            assert binding.pool is bank.return_value and binding.placement is placement
    bank.assert_called_once_with(hidden, parent, owner, tokens=4096, received=33792,
                                 preferred_receive=None, pipeline=True)


@pytest.mark.parametrize("mandatory", [False, True])
def test_fixed_heap_declines_only_optional_placements(mandatory):
    """Budget rejection may only return to a home route that respects receive limits."""
    spec = _a3_push_spec(4, "EOI", 4096, 1408, ep_size=2, top_k=2,
                         num_experts=8, initial_capacity_factor=1.25, partial_weight_grad_dtype=None)
    parent = SimpleNamespace(spec=spec)
    owner = mega_moe_adapter.PlacementResources(parent, {}, True, bf16_partials=True)
    hidden = torch.empty(1, spec.hidden_size, dtype=torch.bfloat16)
    placement = SimpleNamespace(
        borrowed_capacity=1, receive_capacity=16384, counts_by_source=torch.empty(2, 10), copies_by_rank=())
    with patch.object(mega_moe_adapter, "_get_mega_bank", side_effect=_HeapCapacityError("budget exceeded")) as reserve:
        if mandatory:
            with pytest.raises(_HeapCapacityError, match="budget exceeded"):
                owner.bind(hidden, placement, tokens=8192, mandatory=True)
        else:
            assert owner.bind(hidden, placement, tokens=8192) is None
    reserve.assert_called_once_with(hidden, parent, owner, tokens=8192, received=16384,
                                    preferred_receive=20480, pipeline=True)
    assert owner.pool is None and owner.statistics["placements"] == 0
    assert owner.statistics["placement_candidates"] == owner.statistics["heap_rejections"] == 1


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


@pytest.mark.parametrize(("home", "layout", "partial_dtype", "dedicated"), [
    (4, "EOI", None, True), (6, "EIO", "float32", False), (16, "EOI", "float32", True),
])
def test_fused_bank_reserves_room_for_later_routes_and_separate_flags(home, layout, partial_dtype, dedicated):
    """Register the replica workspace, weight bank and peer flags in one reserve."""
    spec = SimpleNamespace(
        ep_group=None, local_experts=home, hidden_size=256, intermediate_size=128,
        num_cube_cores=24, expert_weight_layout=layout, partial_weight_grad_dtype=partial_dtype, architecture=2201,
    )
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    manager = _HeapManager(None, hidden)
    parent = SimpleNamespace(spec=spec, heap_manager=manager, workspace=object())
    owner = mega_moe_adapter.PlacementResources.__new__(mega_moe_adapter.PlacementResources)
    owner.parent = parent
    owner.dedicated_bank = dedicated
    extent = 3 * home if partial_dtype == "float32" else 2 * home
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=32),
          patch.object(manager, "reserve") as reserve, patch.object(manager, "bind"),
          patch.object(SharedExpertPool, "allocate")):
        pool = _get_mega_bank(hidden, parent, owner, tokens=4096, received=32768, pipeline=True)
    assert pool.bank_experts == extent
    assert reserve.call_args.kwargs['workspace_growth'] == (parent.workspace, 4096, 32768)
    ((_, bank_layout),), = reserve.call_args.args
    bank_bytes = 3 * extent * spec.hidden_size * spec.intermediate_size * hidden.element_size() + 1024
    assert bank_layout.required_bytes == bank_bytes + 4 * pool.pipeline.signal_words + 512
    shapes = pool._bank_shapes()  # pylint: disable=protected-access
    pool.leased_banks = tuple(torch.empty(shape, device="meta") for shape in shapes)
    banks = pool._generic_banks()  # pylint: disable=protected-access
    assert all(bank.shape[0] == 2 * home for bank in banks)


def test_replica_banks_share_only_compatible_owners():
    """Reuse one bank across serial owners with the same physical gradient extent."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=False, dispatch_mode="push", transport_engine="mte",
        ep_group=None, ep_size=4, local_experts=6, num_experts=24, hidden_size=256,
        intermediate_size=128, num_cube_cores=24, expert_weight_layout="EIO",
        partial_weight_grad_dtype=None,
    )
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    manager = _HeapManager(None, hidden)
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=spec.ep_size),
          patch.object(manager, "reserve"), patch.object(manager, "bind"),
          patch.object(SharedExpertPool, "allocate")):
        owners, pools = [], []
        for layout, partial_dtype, policy in [("EIO", None, "shared"), ("EOI", None, "shared"),
                                              ("EOI", "float32", "shared"), ("EOI", None, "dedicated")]:
            current = SimpleNamespace(**{**vars(spec), "expert_weight_layout": layout,
                                         "partial_weight_grad_dtype": partial_dtype})
            parent = SimpleNamespace(spec=current, heap_manager=manager, workspace=object())
            owners.append(_builtin_placement(parent, bf16_partials=True, replica_bank=policy))
            pools.append(_get_mega_bank(hidden, parent, owners[-1], tokens=4096, received=32768, pipeline=True))
    assert pools[0] is pools[1]
    assert [pool.bank_experts for pool in pools] == [12, 12, 18, 12]
    assert all(pools[3] is not pool for pool in pools[:3])
    assert pools[0].pipeline.shared_layouts


@pytest.mark.parametrize("home", range(4, 17))
@pytest.mark.parametrize("layout", ["EIO", "EOI"])
def test_bf16_partial_return_only_replaces_wide_native_routes(home, layout):
    """Choose the fused bands or the staged fallback for each bank width."""
    spec = _a3_push_spec(home, layout, 5120, 1792, batch_experts=False, num_experts=2 * home,
                         ep_size=2, top_k=4, num_cube_cores=24, partial_weight_grad_dtype=None)
    parent = SimpleNamespace(spec=spec)
    empty = (-1,) * home
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    for dtype, capability, expected in ((None, False, ReplicaExecutionMode.STAGED),
                                        (None, True, ReplicaExecutionMode.BF16_BAND),
                                        ("float32", True, ReplicaExecutionMode.FP32_BAND)):
        spec.partial_weight_grad_dtype = dtype
        owner = _builtin_placement(parent, bf16_partials=capability)
        pool = SimpleNamespace(local_experts=home, rank=0, blocks=spec.num_cube_cores, bank_experts=owner.bank_extent)
        pool.pipeline = ReplicaPipeline(pool)
        owner.pool = pool
        for used, mode in ((1, ReplicaExecutionMode.FP32_BAND), (home, expected)):
            incoming = tuple(range(home, home + used)) + empty[used:]
            placement = SimpleNamespace(borrowed_capacity=home, copies_by_rank=(incoming, empty),
                                        counts_by_source=torch.empty((2, 4 * home)))
            if used == home and layout == "EIO" and not capability:
                assert owner.bind(hidden, placement, tokens=4096) is None
                with pytest.raises(RuntimeError, match="wheel with BF16 partial dW"):
                    owner.bind(hidden, placement, tokens=4096, mandatory=True)
                continue
            binding = owner.bind(hidden, placement, tokens=4096)
            assert binding.mode is mode


@pytest.mark.parametrize(("hidden", "intermediate"), [(5120, 1800), (5120, 1793), (5112, 1792), (5112, 1793)])
def test_a3_eio_bf16_bands_preserve_unaligned_matrix_tails(hidden, intermediate):
    """Interior bands use aligned lengths and the last band ends at the true size."""
    spec = _a3_push_spec(6, "EIO", hidden, intermediate, batch_experts=False)
    assert supports_pipeline(spec, 6, bf16_partials=True)
    for rows, columns in ((hidden, 2 * intermediate), (intermediate, hidden)):
        band_rows, bands = _band_layout((rows, columns))
        assert bands == 1 or band_rows * columns % 16 == 0
        assert 0 < rows - (bands - 1) * band_rows <= band_rows
        assert bands <= 64


def test_a3_eio_unaligned_narrow_bank_preserves_bf16_partial_rounding():
    """A formerly separate narrow route retains BF16 partials in the fused plan."""
    spec = _a3_push_spec(5, "EIO", 5113, 1793, batch_experts=False, num_experts=10, ep_size=2,
                         num_cube_cores=24, partial_weight_grad_dtype=None)
    parent = SimpleNamespace(spec=spec)
    owner = _builtin_placement(parent, bf16_partials=True)
    pool = SimpleNamespace(local_experts=5, rank=0, blocks=24, bank_experts=owner.bank_extent)
    pool.pipeline = ReplicaPipeline(pool)
    owner.pool = pool
    placement = SimpleNamespace(borrowed_capacity=1, copies_by_rank=((5,), (-1,)),
                                counts_by_source=torch.empty((2, 12)))
    hidden = torch.empty((0, spec.hidden_size), dtype=torch.bfloat16)
    binding = owner.bind(hidden, placement, tokens=4096)
    assert binding.mode is ReplicaExecutionMode.BF16_BAND


def test_a3_eio_full_fp32_bank_aligns_its_suffix_for_odd_expert_strides():
    """The first FP32 partial begins past the last weight at a valid float address."""
    spec = _a3_push_spec(5, "EIO", 5113, 1793, batch_experts=False, num_experts=10,
                         ep_size=2, partial_weight_grad_dtype="float32")
    owner = _builtin_placement(SimpleNamespace(spec=spec))
    elems = spec.hidden_size * spec.intermediate_size
    assert owner.bank_extent == 16
    assert supports_pipeline(spec, spec.local_experts, bank_experts=owner.bank_extent)
    offset_bytes = elems * (2 * owner.bank_extent - 4 * spec.local_experts)
    assert offset_bytes % 4 == 0
    assert offset_bytes >= 2 * spec.local_experts * elems


def test_fused_backward_invalidates_a_generic_bank_generation_even_with_resident_weights():
    """Retiring the BF16 weight lease also covers FP32 dW writes into the same bank."""
    bank = (torch.empty((4, 4, 8)), torch.empty((4, 8, 2)))
    pool = SimpleNamespace(local_experts=2, ranks=2, rank=0, hidden=8, intermediate=2, blocks=2,
                           expert_weight_layout="EOI", leased_banks=bank, generation=0, statistics={})
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
    shapes = pool._bank_shapes()  # pylint: disable=protected-access
    pool.weights = tuple(torch.full(shape, 2, dtype=hidden.dtype) for shape in shapes)
    pool.pipeline.storage = torch.full((pool.pipeline.signal_words,), 5, dtype=torch.int32)
    placement = SimpleNamespace(
        borrowed_capacity=home, published_slots=home, copies_by_rank=((-1,) * home,) * 2,
    )
    binding = ExpertWeightBinding(pool, placement, layout, execution_mode=ReplicaExecutionMode.FP32_BAND)
    with (patch("torch.npu.Event"), patch("torch.npu.current_stream"), patch.object(torch.Tensor, "record_stream"),
          patch("hyper_multicore.shmem.barrier") as barrier):
        binding.save_snapshot(pool.weights)
        old, specifications = pool._allocations(hidden.dtype)  # pylint: disable=protected-access
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
    spec = _a3_push_spec(home, layout, 512, 256, batch_experts=batch_experts, rank_id=1, top_k=1)
    counts = ((2,) * config.num_experts,) * ranks
    quotas = [{expert: 4 for expert in range(rank * home, (rank + 1) * home)} for rank in range(ranks)]
    for expert in range(home, home + used):
        quotas[0][expert] = quotas[1][expert] = 2
    plan = _materialize(counts, quotas, config)
    plan.validate()
    geometry = render_geometry(plan, spec, "cpu", config.num_experts * 2)
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

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
"""Regression coverage for single-backend placement and original-route fallback."""

from typing import get_type_hints
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_multicore import BasePlanner, ExpertPlacement, MegaMoePlacementOptions, MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner
from hyper_multicore.modules.mega_moe import route, resources as resource_module
from hyper_multicore.modules.mega_moe.backends.bf16.autograd import _plan_events
from hyper_multicore.modules.hot_replica import capability, checkpoint as replay, mx_adapter
from hyper_multicore.modules.hot_replica._planner import aligned
from hyper_multicore.modules.hot_replica.bank_ops import reduce_banks
from hyper_multicore.modules.hot_replica.shmem_transfer import SharedExpertPool
from hyper_multicore.modules.hot_replica.mega_moe_adapter import (
    ExpertWeightBinding, PlacementResources, _bind_custom_copy_rows, _get_mega_bank,
)
from hyper_multicore.modules.hot_replica.mx_gradient import GRADIENT_HELPERS, MXGradientPool
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


@pytest.mark.parametrize("local_experts", (4, 6, 8))
def test_mx_snapshot_reuses_validated_totals_in_the_active_plan(local_experts):
    """The validated snapshot takes the same exact placement as the raw count path."""
    ep, tokens, top_k = 32, 4096, 8
    spec = MegaMoeSpec(local_num_tokens=tokens, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * local_experts, top_k=top_k, initial_capacity_factor=4,
                       receive_capacity=ep * tokens * top_k, ep_size=ep, ep_group=None,
                       rank_id=0, num_cube_cores=32, auto_tokens=True, architecture=3510, precision="mxfp8")
    resource = mx_adapter.MXPlacementResources(SimpleNamespace(spec=spec, workspace=object()), {}, True)
    resource.nodes = tuple(rank // 8 for rank in range(ep))
    resource._previous_active = True
    wire = torch.zeros((ep, spec.num_experts + 3), dtype=torch.int32)
    wire[:, 1] = tokens
    wire[:, 2] = mx_adapter._TRAINING_MASK
    wire[:, 3:3 + top_k] = tokens
    hidden = SimpleNamespace(shape=(tokens, spec.hidden_size), device=wire.device)
    source_rows = []
    summary, _, _, snapshot = resource._read_route(wire, hidden, None, source_rows, use_snapshot=True)

    def _geometry(plan, *_):
        return SimpleNamespace(plan=plan, storage_bytes=lambda: 0)

    with patch.object(mx_adapter, "_render_geometry", side_effect=_geometry):
        baseline = resource._select(hidden, wire, summary, wire.tolist(), source_rows,
                                    mx_adapter._TRAINING_MASK, None)
        with patch.object(aligned, "_counts_matrix", side_effect=AssertionError("counts already validated")):
            actual = resource._select(hidden, wire, summary, None, source_rows, mx_adapter._TRAINING_MASK, None,
                                      snapshot=snapshot, use_snapshot=True)
            repeated = resource._select(hidden, wire, summary, None, source_rows, mx_adapter._TRAINING_MASK, None,
                                        snapshot=snapshot, use_snapshot=True)
    assert baseline.plan.transfers
    baseline.plan.validate()
    assert actual.plan == baseline.plan
    assert repeated is actual

    # A negative expert count is invalid even when the source sum is correct.
    wire[0, 3 + top_k] = -1
    wire[0, 4 + top_k] = 1
    with pytest.raises(RuntimeError, match="actual distinct Top-K"):
        resource._read_route(wire, hidden, None, [], use_snapshot=True)


def test_mx_gradient_bank_registers_with_the_virtual_workspace():
    """Reserve the FP32 guest inbox and both projections in the route transaction."""
    hidden = torch.empty((0, 128), dtype=torch.bfloat16)
    spec = SimpleNamespace(local_experts=6, hidden_size=128, intermediate_size=64, num_cube_cores=24,
                           ep_group=object(), expert_weight_layout="EIO", architecture=3510)
    manager = Mock()
    manager.access.side_effect = nullcontext
    owner = Mock(bounded=None)
    virtual_reservation = ({"kind": "virtual-route"}, object())
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=16),
          patch.object(MXGradientPool, "allocate") as allocate):
        pool = _get_mega_bank(hidden, SimpleNamespace(spec=spec, heap_manager=manager), owner,
                              pool_type=MXGradientPool, reservations=(virtual_reservation,))
    reservations, = manager.reserve.call_args.args
    assert reservations[0] is virtual_reservation and len(reservations) == 2
    key, layout = reservations[1]
    assert key["kind"] == "hot_replica_mx_gradients"
    fp32_guests = 2 * 3 * spec.hidden_size * spec.intermediate_size * 4
    flags = (3 * pool.flag_words + GRADIENT_HELPERS * 16) * 4
    scratch = GRADIENT_HELPERS * 2 * pool.chunk_elements * 4
    assert layout.required_bytes == fp32_guests + flags + scratch + 1024
    assert owner in pool.owners and pool.pipeline is None
    manager.bind.assert_called_once_with(pool, spec, key)
    allocate.assert_called_once_with()


@pytest.mark.parametrize(
    "precision,mode,value,error",
    [
        ("bf16", "push", True, TypeError),
        ("mxfp8", "push", HotReplicaPlanner(), ValueError),
        ("mxfp8", "pull", HotReplicaPlanner(), ValueError),
    ],
)
def test_unsupported_configuration_fails_before_binding(precision, mode, value, error):
    """Reject unsupported placement options before device resource binding."""
    with pytest.raises(error):
        MegaMoeExperts(
            hidden_size=128,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
            precision=precision,
            tuning_options=MegaMoeTuningOptions(dispatch_mode=mode),
            expert_placement=MegaMoePlacementOptions(strategy="adaptive", planner=value),
        )


@pytest.mark.parametrize(("options", "error"), [
    (True, TypeError),
    (MegaMoePlacementOptions(strategy="unknown"), ValueError),
    (MegaMoePlacementOptions(strategy="adaptive"), TypeError),
    (MegaMoePlacementOptions(strategy="none", planner=HotReplicaPlanner()), ValueError),
])
def test_invalid_placement_strategy_fails_before_binding(options, error):
    with pytest.raises(error):
        MegaMoeExperts(hidden_size=128, intermediate_size=128, num_experts=4, top_k=2, expert_placement=options)


def test_placement_requires_multiple_ranks():
    with pytest.raises(ValueError, match="EP group"):
        MegaMoeExperts(
            hidden_size=128,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
            expert_placement=MegaMoePlacementOptions(strategy="adaptive", planner=HotReplicaPlanner()),
        )


def test_bf16_pull_accepts_adaptive_placement():
    """Bind the existing BF16 placement configuration to Pull."""
    from hyper_multicore.modules.mega_moe import module

    with patch.object(module, "_resolve_ep_size", return_value=2):
        expert = MegaMoeExperts(
            hidden_size=128, intermediate_size=128, num_experts=4, top_k=2,
            tuning_options=MegaMoeTuningOptions(dispatch_mode="pull"),
            expert_placement=MegaMoePlacementOptions("adaptive", HotReplicaPlanner()),
        )
    try:
        assert expert._resource_group.specification["dispatch_mode"] == "pull"  # pylint: disable=protected-access
        assert expert._resource_group.specification["hot_replica"]  # pylint: disable=protected-access
    finally:
        expert.close()


def test_a3_bf16_pull_keeps_replica_capability_guard():
    """Reject an unqualified replica transport before allocating its banks."""
    with (
        patch.object(capability._collective, "exchange", side_effect=lambda value, **_: [value, value]),
        pytest.raises(RuntimeError, match="BF16 HotReplica Pull requires Ascend950"),
    ):
        capability.require_capability(None, None, architecture=2201, dispatch_mode="pull")


@pytest.mark.parametrize("options", [None, MegaMoePlacementOptions(strategy="none")])
def test_default_keeps_placement_disabled(options):
    expert = MegaMoeExperts(hidden_size=128, intermediate_size=128, num_experts=4, top_k=2, expert_placement=options)
    try:
        assert expert._resource_group.specification["hot_replica"] is False  # pylint: disable=protected-access
    finally:
        expert.close()


@pytest.mark.parametrize("name,arch", [("Ascend910B2", 2201), ("Ascend950", 3510)])
def test_architecture_alone_does_not_enable_hot_replica(name, arch):
    """Require the actual A3 device identity in addition to the architecture."""
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=5), dtype=torch.bfloat16)
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=arch),
        patch.object(torch.npu, "get_device_name", return_value=name) as device_name,
        patch.object(capability, "validate_native_compatibility") as manifest,
        pytest.raises(ValueError, match="Atlas A3"),
    ):
        capability._require_a3(tensor)  # pylint: disable=protected-access
    device_name.assert_called_once_with(5)
    manifest.assert_not_called()


def test_supported_device_requires_selected_adapter_artifacts():
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=3), dtype=torch.bfloat16)
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=2201),
        patch.object(torch.npu, "get_device_name", return_value="Ascend910_9391"),
        patch.object(capability, "validate_native_compatibility", return_value={"variant": {}}),
        pytest.raises(RuntimeError, match="no Atlas A3"),
    ):
        capability._require_a3(tensor)  # pylint: disable=protected-access


def test_a5_bf16_requires_its_own_replica_bank(tmp_path):
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), dtype=torch.bfloat16)
    artifacts = {"soc": "ascend950", "bridge": "bridge.so", "kernels": "kernels.so"}
    selected = {"root": tmp_path, "variant": {"hot_replica_bf16_arch35": artifacts}}
    (tmp_path / artifacts["bridge"]).touch()
    with patch.object(capability, "validate_native_compatibility", return_value=selected):
        with pytest.raises(RuntimeError, match="Ascend950 BF16 HotReplica artifact is missing: kernels"):
            capability.require_capability(tensor, None, architecture=3510)
        (tmp_path / artifacts["kernels"]).touch()
        capability.require_capability(tensor, None, architecture=3510)


def test_mx_capability_checks_cube_geometry_before_loading_replica_artifacts() -> None:
    """Converge a device limitation during binding, before a hotspot can allocate guest resources."""
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=3), dtype=torch.bfloat16)
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=3510),
        patch.object(capability, "validate_native_compatibility") as manifest,
        pytest.raises(RuntimeError, match="HotReplica capability failed.*requires 32 Cube cores, got 24"),
    ):
        capability.require_capability(tensor, None, mx_cube_cores=24)
    manifest.assert_not_called()


def test_no_move_preserves_original_prepared_route_and_capacity():
    """Preserve validated routing and capacity when placement is declined."""
    hidden = torch.empty(2, 8)
    ids = torch.tensor([[0], [1]])
    spec, workspace, metadata = SimpleNamespace(dispatch_mode="push"), object(), object()
    counts = torch.tensor([1, 1])
    gathered = torch.tensor([[0, 2, 1, 1, 1]])
    permutation = (hidden, torch.tensor([0, 1]))
    original = object()
    prepare, select, work = Mock(), Mock(return_value=None), Mock()
    summary = route._RouteSummary(2, 128, 2)
    with (
        patch.object(route, "_read_cached_route", return_value=(summary, [], metadata)) as read,
        patch.object(route, "_finalize_dynamic_route", return_value=original) as finish,
    ):
        result = route._prepare_placed_route(  # pylint: disable=protected-access
            hidden, ids, spec, workspace, permutation, counts, gathered, work, None, prepare, select
        )
    assert result is original
    read.assert_called_once()
    work.wait.assert_called_once()
    prepare.assert_called_once_with(128, 2)
    assert finish.call_args.args[0] is permutation
    assert finish.call_args.args[4] is metadata
    assert finish.call_args.args[7] is spec
    assert finish.call_args.args[8] is workspace


def test_failed_weight_publication_releases_bank():
    pool = Mock()
    pool.prefetch.side_effect = RuntimeError("launch failed")
    binding = ExpertWeightBinding(pool, SimpleNamespace(copies=object(), borrowed_capacity=1), "EOI")
    with pytest.raises(RuntimeError, match="launch failed"):
        binding.acquire((object(), object()))
    pool.release.assert_called_once()


def test_failed_split_publication_drains_streams_before_bank_release():
    pool = Mock()
    pool.pipeline.begin.side_effect = RuntimeError("peer copy failed")
    binding = ExpertWeightBinding(pool, Mock(), "EIO", fused=True)
    with patch("hyper_multicore.modules.hot_replica.shmem_transfer._ordered_transfers",
               return_value=nullcontext()), pytest.raises(RuntimeError, match="peer copy failed"):
        binding.acquire((object(), object()))
    assert [name for name, _, _ in pool.method_calls][-2:] == ["pipeline.complete", "release"]
    assert pool.pipeline.resident is None


def test_child_resources_prevent_parent_cleanup():
    parent = PlacementResources(Mock(), {}, False)
    parent.pool = Mock()
    parent.pool.can_close.return_value = False
    assert not parent.can_close()
    parent.pool.can_close.return_value = True
    child = Mock()
    child.can_close.return_value = False
    parent.geometries[1] = child
    assert not parent.can_close()


def test_fp32_owner_gradient_reaches_native_reduction_without_rounding():
    gradient = torch.tensor([[[1.003]], [[2.003]]], dtype=torch.float32)
    banks = (torch.zeros(4, 1, 1, dtype=torch.bfloat16),)
    copies = torch.empty(1, 0, dtype=torch.int32)
    bridge = Mock()
    with patch("hyper_multicore.modules.hot_replica.bank_ops.load_bridge", return_value=bridge):
        actual = reduce_banks((gradient,), banks, copies, 0, 1)[0]
    torch.testing.assert_close(actual, gradient, rtol=0, atol=0)
    torch.testing.assert_close(bridge.reduce.call_args.args[0], gradient, rtol=0, atol=0)


@pytest.mark.parametrize(("partial_dtype", "new_bridge", "native"), [
    (torch.bfloat16, True, True), (torch.bfloat16, False, False), (torch.float32, True, False),
])
def test_native_home_gradient_keeps_fp32_guest_payload_and_legacy_fallback(partial_dtype, new_bridge, native):
    """Keep original local precision and the ordered FP32 wire in the buffered reducer."""
    gradient = torch.arange(12, dtype=partial_dtype).view(3, 2, 2)
    bank = torch.empty(4, 2, 2, dtype=torch.bfloat16)
    copies = torch.tensor([[2], [0]], dtype=torch.int32)
    bridge = SimpleNamespace(reduce=Mock())
    if new_bridge:
        bridge.reduce_buffered = Mock()
        bridge.buffered_plan_fits = Mock(return_value=True)
    with patch("hyper_multicore.modules.hot_replica.bank_ops.load_bridge", return_value=bridge):
        result, = reduce_banks((gradient,), (bank,), copies, 0, 1, (torch.bfloat16,), prefer_buffered=True)
    called = bridge.reduce_buffered if native else bridge.reduce
    called.assert_called_once()
    source, output, slots, *_ = called.call_args.args
    assert source.dtype == (partial_dtype if native else torch.float32)
    assert output is result and result.dtype == torch.bfloat16
    assert result.untyped_storage().data_ptr() != gradient.untyped_storage().data_ptr()
    torch.testing.assert_close(slots, gradient[2:].float(), rtol=0, atol=0)


def test_placement_reuses_validated_source_totals_and_autograd_mask():
    """Reuse existing source counts and gradient participation without rereading."""
    hidden = torch.empty(2, 8)
    ids = torch.tensor([[0], [1]])
    counts = torch.tensor([1, 1])
    gathered = torch.tensor([[0, 2, 1, 1, 1]])
    selection = Mock(return_value=None)

    def read(
        payload: object,
        spec: object,
        workspace: object,
        device: object,
        source_rows: list[int],
        **kwargs: object
    ) -> tuple:
        """Expose source totals from the existing validation readback."""
        del payload, spec, workspace, device, kwargs
        source_rows.extend((2, 6))
        return route._RouteSummary(2, 128, 6), None, None

    with patch.object(route, "_read_cached_route", side_effect=read), patch.object(route, "_finalize_dynamic_route"):
        route._prepare_placed_route(  # pylint: disable=protected-access
            hidden, ids, SimpleNamespace(dispatch_mode="push"), object(), (hidden, ids), counts, gathered,
            None, None, Mock(), selection, 1
        )
    assert selection.call_args.args[4] == [2, 6]
    assert selection.call_args.args[6] == 1


def test_configuration_mismatch_fails_before_optional_capability_collective():
    """Reject inconsistent rank settings before optional capability collectives."""
    cls = resource_module._MegaMoeExecutionResources  # pylint: disable=protected-access
    specification = {"hot_replica": True, "ep_group": object()}
    spec = SimpleNamespace(local_num_tokens=128)
    with (
        patch.object(resource_module, "bind_mega_moe_spec", return_value=spec),
        patch.object(resource_module.dist, "is_initialized", return_value=True),
        patch.object(cls, "_validate_distributed_configuration", side_effect=ValueError("configurations differ")),
        patch.object(capability, "require_capability") as gate,
        patch.object(resource_module, "_get_heap_manager") as heap,
        pytest.raises(ValueError, match="configurations differ"),
    ):
        cls(
            specification,
            torch.empty(1, 1),
            expert_weight_layout="EIO",
            shared=False,
            active_specifications=(specification,),
        )
    gate.assert_not_called()
    heap.assert_not_called()


def test_rejected_transfer_cost_reuses_only_identical_count_snapshots():
    """Reuse negative cost decisions without caching routes or current weights."""
    spec = SimpleNamespace(
        architecture=0, ep_size=2, local_experts=4, top_k=2, hidden_size=8,
        intermediate_size=4, num_cube_cores=24, rank_id=0, apply_probs_before_fc2=True, dispatch_mode="push",
    )
    owner = PlacementResources(SimpleNamespace(spec=spec), {}, False)
    hidden, ids, gathered = torch.empty(2, 8), torch.zeros(2, 2), torch.zeros(2, 11)
    planner = HotReplicaPlanner()
    candidate = SimpleNamespace(receive_capacity=2, transfer_experts=2)
    module = "hyper_multicore.modules.hot_replica.planner"
    with (
        patch(module + ".placement_is_profitable", side_effect=[True, False, True, False]) as cost,
        patch(module + ".build_placement", return_value=candidate) as build,
    ):
        arguments = (hidden, ids, gathered, route._RouteSummary(8, 128, 8), [4, 4], torch.zeros(4), 1)
        assert owner.select(*arguments, planner=planner, host_rows=[[1, 2, 1, 4]]) is None
        assert owner.select(*arguments, planner=planner, host_rows=[[1, 2, 1, 4]]) is None
        assert build.call_count == 1 and cost.call_count == 2
        assert owner.select(*arguments, planner=planner, host_rows=[[1, 2, 1, 3]]) is None
        assert build.call_count == 2 and cost.call_count == 4
    assert owner.pool is None and not owner.geometries


@pytest.mark.parametrize("budget", [2, 6])
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_a5_bf16_bounded_quota_uses_the_real_guest_budget(budget, dispatch_mode):
    spec = SimpleNamespace(
        architecture=3510, dispatch_mode=dispatch_mode, transport_engine="udma", ep_size=2, num_experts=12,
        local_experts=6, top_k=6, hidden_size=5120,
        intermediate_size=1792, ep_group=object(), rank_id=0, num_cube_cores=32,
        apply_probs_before_fc2=False, partial_weight_grad_dtype="float32", batch_experts=False,
    )
    planner = HotReplicaPlanner(replica_slots_per_rank=budget)
    settings = {"planner_key": (HotReplicaPlanner.__module__, HotReplicaPlanner.__qualname__,
                                planner.configuration_key()),
                "replica_slots_per_rank": budget}
    with patch("hyper_multicore.modules.hot_replica.mega_moe_adapter.has_a5_bf16_pipeline", return_value=False):
        placement = PlacementResources(SimpleNamespace(spec=spec), settings, False)
    placement.rank_nodes = (0, 1)
    hidden = torch.empty((1, 5120), dtype=torch.bfloat16).expand(4096, 5120)
    ids = torch.arange(6, dtype=torch.int32).repeat(4096, 1)
    gathered = torch.tensor([[0] * 3 + [4096] * 6 + [0] * 6] * 2, dtype=torch.int32)
    inverse = torch.argsort(torch.argsort(ids.flatten(), stable=True))
    with (
        patch.object(planner, "plan", side_effect=AssertionError("A3 transfer estimate used on A5")),
        patch.object(placement, "bind", side_effect=lambda _hidden, plan, **_: plan),
    ):
        selected = placement.select(
            hidden, ids, gathered, route._RouteSummary(4096, 4096, 49152), [24576, 24576], inverse,
            13, planner=planner, host_rows=gathered.tolist(),
        )
    assert 1 <= selected.borrowed_capacity <= budget
    assert sum(expert >= 0 for expert in selected.copies_by_rank[1]) == min(budget, 3)


def test_a5_bf16_mte_pull_uses_the_generic_planner():
    """Keep the MTE path on its existing admission and separate bank protocol."""
    spec = SimpleNamespace(architecture=3510, dispatch_mode="pull", transport_engine="mte", num_experts=12, ep_size=2)
    planner = HotReplicaPlanner()
    settings = {"planner_key": (type(planner).__module__, type(planner).__qualname__, planner.configuration_key())}
    placement = PlacementResources(SimpleNamespace(spec=spec), settings, False)
    assert not placement.uses_a5_quotas
    assert not placement.pipeline_enabled


def test_checkpoint_agrees_on_version_and_activation(monkeypatch):
    """Reject stale parameters across ranks, even if the token route matches."""
    monkeypatch.setattr(torch.npu, "current_stream", lambda _device: "s")
    monkeypatch.setattr(replay.dist, "all_gather_into_tensor", lambda out, x, **_: out.copy_(x.expand_as(out)))
    spec = SimpleNamespace(ep_size=2, ep_group=object())
    hidden, probabilities = torch.ones((1, 4)), torch.ones((1, 2))
    ids = torch.tensor([[0, 1]])
    owner = object()
    version = [0]
    binding = Mock()

    def _invocation():
        record, recomputing = replay.route_record(owner)
        if recomputing:
            route_matches = replay.agree_replay(
                record, hidden, ids, probabilities, None, spec, 15, weight_error=False,
            )
            return route_matches, record.allow_activation
        record.ids, record.stream, record.route = ids.clone(), "s", object()
        record.weight_binding = binding
        record.inputs = hidden.detach(), probabilities.detach()
        record.input_versions = (0, 0)
        record.activation = object()
        return None

    wrapped = replay.wrap_checkpoint(_invocation, lambda: version[0], prepare_replay=True)
    with torch.no_grad():
        wrapped()
    binding.prepare_replay.assert_not_called()
    assert wrapped() == (True, True)
    binding.prepare_replay.assert_called_once_with()
    version[0] += 1
    assert wrapped() == (False, False)
    binding.prepare_replay.assert_called_once_with()


def test_checkpoint_with_supplied_counts_requires_the_original_order_and_histogram(monkeypatch):
    """A replay hit requires the actual Router inputs on every EP rank."""
    monkeypatch.setattr(torch.npu, "current_stream", lambda _device: "s")
    peer_matches = [True]

    def gather(out, local, **_):
        out[:] = local
        if not peer_matches[0]:
            out[1] = 0

    monkeypatch.setattr(replay.dist, "all_gather_into_tensor", gather)
    spec = SimpleNamespace(ep_size=2, ep_group=object())
    hidden, probabilities = torch.ones((2, 4)), torch.ones((2, 1))
    ids = torch.arange(4, dtype=torch.int64)[::2].view(2, 1)
    counts = torch.tensor([1, 9, 0, 9, 1, 9, 0, 9], dtype=torch.int64)[::2]
    owner = object()

    def invocation(current_ids, current_counts):
        record, recomputing = replay.route_record(owner)
        if recomputing:
            return replay.agree_replay(
                record, hidden, current_ids, probabilities, current_counts, spec, 15, weight_error=False,
            )
        record.stream = "s"
        with record.save_supplied_route(current_ids, current_counts):
            record.route = object()
        return None

    wrapped = replay.wrap_checkpoint(invocation, lambda: 0)
    with torch.no_grad():
        wrapped(ids, counts)
    assert wrapped(ids.clone(), counts.clone())
    assert not wrapped(ids.flip(0), counts)
    assert not wrapped(ids, counts.roll(1))
    assert not wrapped(ids, None)
    peer_matches[0] = False
    assert not wrapped(ids, counts)


def test_early_weight_restore_uses_bank_generation(monkeypatch):
    """A copy can be consumed after its stream event or recovered if the bank changed."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)
    copy_stream = SimpleNamespace(device="npu:0", wait_event=Mock())
    compute_stream = SimpleNamespace(device="npu:0", wait_event=Mock())
    active = [copy_stream]
    monkeypatch.setattr(torch.npu, "current_stream", lambda _: active[0])
    monkeypatch.setattr(torch.npu, "Stream", lambda **_: copy_stream)
    monkeypatch.setattr(torch.npu, "Event", Mock)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    pool = SharedExpertPool(torch.empty((0, 4)), object(), 1, 2, 1, "EIO", root=Mock(), access=nullcontext)
    pool.weights = (torch.zeros(2, 4, 4), torch.zeros(2, 2, 4))
    pool.completion_event = Mock()
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    placement = SimpleNamespace(copies=torch.empty((0,), dtype=torch.int32), borrowed_capacity=1)
    binding = ExpertWeightBinding(pool, placement, "EIO")
    tails = tuple(torch.full_like(bank[1:], 2) for bank in pool.weights)
    ready = object()
    binding.snapshot = tails, ready
    binding.prepare_replay()
    copy_stream.wait_event.assert_called_once_with(ready)
    pool.completion_event.record.assert_called_once_with(copy_stream)

    active[0] = compute_stream
    homes = tuple(torch.full_like(bank[:1], 3) for bank in pool.weights)
    weights = binding.acquire(homes)
    compute_stream.wait_event.assert_called_once_with(pool.completion_event)
    assert pool.statistics["early_restore_hits"] == 1
    for weight, home, tail in zip(weights, homes, tails):
        torch.testing.assert_close(weight, torch.cat((home, tail)))
    binding.release()

    active[0] = copy_stream
    binding.prepare_replay()
    # A later lease allowed another graph to overwrite the bank.
    pool.generation += 1
    for bank in pool.weights:
        bank.zero_()
    active[0] = compute_stream
    weights = binding.acquire(homes)
    for weight, home, tail in zip(weights, homes, tails):
        torch.testing.assert_close(weight, torch.cat((home, tail)))
    assert pool.statistics["early_restores"] == 2
    assert pool.statistics["early_restore_hits"] == 1
    binding.release()


def test_virtual_event_views_keep_the_ready_generation_at_the_physical_tail():
    """Virtual plans of different expert widths must meet at the same ready address."""
    physical = 8192
    events = torch.empty(physical + 1024, dtype=torch.uint8)
    workspace = SimpleNamespace(event_counter_bytes=physical, prepare_event_counters=Mock(return_value=events))
    for logical in (4096, 6144, physical):
        view = _plan_events(workspace, SimpleNamespace(event_counter_bytes=logical), forward=False)
        assert view[logical].data_ptr() == events[physical].data_ptr()


def test_forward_uses_injected_planner_after_count_exchange():
    """Use the injected instance on the normal forward path."""
    from hyper_multicore.modules.mega_moe import invocation, module

    class RecordingPlanner(BasePlanner):
        def plan(self, request):
            self.seen = request
            return None

    planner = RecordingPlanner()
    with patch.object(module, "_resolve_ep_size", return_value=2):
        options = MegaMoePlacementOptions(strategy="adaptive", planner=planner)
        expert = MegaMoeExperts(hidden_size=8, intermediate_size=4, num_experts=4, top_k=1, expert_placement=options)
    hidden, ids = torch.empty(2, 8), torch.zeros(2, 1, dtype=torch.int32)
    probs, roots = torch.ones(2, 1), (torch.empty(2, 8, 8), torch.empty(2, 4, 8))
    spec = SimpleNamespace(architecture=0, auto_tokens=True, dispatch_mode="push", precision="bf16",
                           intermediate_size=4, rank_id=0,
                           num_cube_cores=24, apply_probs_before_fc2=True)
    owner = SimpleNamespace(spec=spec, workspace=Mock(), prepare_auto_workspace=Mock())
    owner.placement = PlacementResources(owner, expert._resource_group.specification, False)
    counts = torch.tensor([[0, 2, 1, 2, 0, 0, 0], [0, 2, 1, 0, 0, 0, 2]])
    ordinary = object()

    def exchanged(*args, **kwargs):
        assert args[0] is hidden
        return route._prepare_placed_route(
            hidden, ids, spec, owner.workspace, (hidden, ids), counts[0, 3:], counts, None, None,
            kwargs["prepare_workspace"], kwargs["select_placement"], kwargs["autograd_mask"],
        )

    summary = route._RouteSummary(2, 128, 2)
    try:
        with (
            patch.object(expert, "_prepare_forward_inputs", return_value=(hidden, False)),
            patch.object(expert, "_get_execution_resources", return_value=owner),
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(replay, "route_record", return_value=(None, False)),
            patch.object(invocation, "prepare_topk_route", side_effect=exchanged) as exchange,
            patch.object(route, "_read_cached_route", return_value=(summary, None, None)) as read,
            patch.object(route, "_finalize_dynamic_route", return_value=ordinary),
            patch.object(expert._backend, "forward", return_value=hidden) as forward,
        ):
            assert expert(hidden, ids, probs, expert_weights=roots) is hidden
        exchange.assert_called_once()
        read.assert_called_once()
        torch.testing.assert_close(planner.seen.counts_by_source, counts[:, 3:])
        assert forward.call_args.args[-1] is ordinary
    finally:
        expert.close()


def test_public_placement_hints_and_omitted_local_capacity():
    """Resolve public types and sum destination columns across all source ranks."""
    assert get_type_hints(MegaMoePlacementOptions)["planner"] == BasePlanner | None
    placement = ExpertPlacement(torch.tensor([[0]]), torch.tensor([[2, 0, 1, 3], [1, 0, 2, 1]]),
                                torch.tensor([[1], [0]], dtype=torch.int32), 1, 7)
    assert [placement.received_rows(rank) for rank in range(2)] == [3, 7]


@pytest.mark.parametrize("guest_count", [0, 3])
def test_custom_placement_derives_fused_rows_from_counts(guest_count):
    """Require a producer for every published guest on both EP ranks."""
    copies = torch.tensor([[2, -1], [-1, -1]], dtype=torch.int32)
    counts = torch.tensor([[3, 2, guest_count, 0, 4, 5, 0, 0], [1, 0, 0, 0, 2, 1, 0, 0]])
    placement = ExpertPlacement(torch.tensor([[2]]), counts, copies, 2, 12)
    bound = [
        _bind_custom_copy_rows(SimpleNamespace(local_experts=2, ep_size=2, rank_id=rank), placement)
        for rank in range(2)
    ]
    if guest_count:
        assert [result.copies_by_rank for result in bound] == [((2, -1), (-1, -1))] * 2
        assert [(result.owner_rows, result.published_slots) for result in bound] == [((4, 2), 1), ((6, 6), 0)]
    else:
        assert all(not result.copies_by_rank for result in bound)


def test_a3_eio_custom_placement_keeps_active_local_slots_and_ignores_idle_guests():
    """Only real guest producers contribute to the fused return schedule."""
    copies = torch.tensor([[0, 3], [-1, -1]], dtype=torch.int32)
    counts = torch.tensor([[3, 2, 3, 0, 4, 5, 0, 0], [1, 0, 0, 0, 2, 1, 0, 0]])
    spec = SimpleNamespace(local_experts=2, ep_size=2, rank_id=0)
    placement = ExpertPlacement(torch.tensor([[2]]), counts, copies, 2, 12)
    bound = _bind_custom_copy_rows(spec, placement, fused_return=True)
    assert bound.copies_by_rank == ((0, -1), (-1, -1))
    assert bound.published_slots == 1
    assert bound.owner_rows == (4, 2)
    counts[0, 2] = 0
    assert _bind_custom_copy_rows(spec, placement, fused_return=True) is None


def test_checkpoint_request_is_training_with_automatic_capacity():
    """Request a training plan even during the no-grad checkpoint forward."""
    spec = SimpleNamespace(architecture=0, rank_id=0, intermediate_size=4, num_cube_cores=24,
                           apply_probs_before_fc2=True, dispatch_mode="push")
    owner = SimpleNamespace(spec=spec)
    resources = PlacementResources(owner, {}, False)
    ids = torch.tensor([[0]])
    with patch("hyper_multicore.modules.hot_replica.mega_moe_adapter.is_checkpointing", return_value=True):
        request = resources.request(torch.zeros(1, 4), ids, torch.zeros(2, 5), route._RouteSummary(1, 128, 1),
                                    [1, 1], ids, 0, host_rows=None)
    assert request.training and request.checkpointing
    assert request.ordinary_receive_capacity is None


@pytest.mark.parametrize("home", [4, 5, 8, 12, 16])
@pytest.mark.parametrize("batch_experts", [False, True])
def test_a3_saved_route_uses_shared_quota_geometry(home, batch_experts):
    """Use the same quotas for the policy, checkpoint and batch lowering."""
    spec = SimpleNamespace(
        architecture=2201, batch_experts=batch_experts, dispatch_mode="push", transport_engine="mte",
        ep_size=2, rank_id=0, num_experts=2 * home, local_experts=home, hidden_size=32, intermediate_size=64,
        top_k=2, num_cube_cores=24, apply_probs_before_fc2=True, expert_weight_layout="EOI",
        partial_weight_grad_dtype=None,
    )
    owner = SimpleNamespace(spec=spec)
    planner = HotReplicaPlanner()
    resources = PlacementResources(owner, {
        "planner_key": (HotReplicaPlanner.__module__, HotReplicaPlanner.__qualname__),
    }, False, bf16_partials=True)
    resources.bounded.nodes = (0, 0)
    ids = torch.arange(2, dtype=torch.int32).expand(2048, 2).contiguous()
    source_counts = [1, 2048, 12, 2048, 2048] + [0] * (2 * home - 2)
    wire = torch.tensor([source_counts] * 2, dtype=torch.int32)
    hidden = torch.empty(2048, 32, dtype=torch.bfloat16)
    summary = route._RouteSummary(8192, 2048, 8192)
    inverse = ids.flatten().argsort(stable=True).argsort().view_as(ids)
    rows = wire.tolist()
    with (
        patch("hyper_multicore.modules.hot_replica.planner.placement_receive_limit", return_value=4096),
        patch("hyper_multicore.modules.hot_replica.planner.placement_is_profitable", return_value=True),
        patch.object(resources, "bind") as bind,
        patch("hyper_multicore.modules.hot_replica.mega_moe_adapter.is_checkpointing", return_value=True),
    ):
        request = resources.request(hidden, ids, wire, summary, [4096, 4096], inverse, 12, host_rows=rows)
        expected = planner.plan(request)
        with patch.object(planner, "plan", side_effect=AssertionError("generic planner called")):
            resources.select(hidden, ids, wire, summary, [4096, 4096], inverse, 12, planner=planner,
                             checkpointing=not batch_experts, host_rows=rows)
    actual = bind.call_args.args[1]
    assert actual is not None and expected is not None
    torch.testing.assert_close(actual.virtual_ids, expected.virtual_ids)
    torch.testing.assert_close(actual.counts_by_source, expected.counts_by_source)
    torch.testing.assert_close(actual.copies, expected.copies)
    assert (actual.receive_capacity, actual.transfer_experts) == (
        expected.receive_capacity, expected.transfer_experts,
    )
    assert resources.bounded.geometry is None
    assert bind.call_args.kwargs["fused"]


def test_manifest_checks_custom_policy_settings():
    """Different settings on two ranks fail in the resource configuration exchange."""
    cls = resource_module._MegaMoeExecutionResources
    owner = SimpleNamespace(spec=SimpleNamespace(ep_size=2, ep_group=object(), expert_weight_layout="EIO",
                                                 architecture="ascend910_93"))
    def manifest(key):
        with patch.object(resource_module.dist, "all_gather_object") as gather:
            cls._validate_distributed_configuration(owner, ({"planner_key": ("example", "Skew", key)},))
        return gather.call_args.args[1]

    remote = manifest("1.5")
    def mismatch(results, local, group):
        assert group is owner.spec.ep_group
        results[:] = [local, remote]

    with patch.object(resource_module.dist, "all_gather_object", side_effect=mismatch):
        with pytest.raises(ValueError, match="configurations differ"):
            cls._validate_distributed_configuration(owner, ({"planner_key": ("example", "Skew", "2.0")},))

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

from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hyper_multicore.experimental import BasePlanner, ExpertPlacement, PlacementRequest
from hyper_multicore.modules.hot_replica import bounded_placement
from hyper_multicore.modules.hot_replica.mega_moe_adapter import PlacementResources
from hyper_multicore.modules.hot_replica.shmem_transfer import ExpertWeightBinding, ReplicaExecutionMode
from hyper_multicore.modules.mega_moe import invocation, route
from hyper_multicore.modules.mega_moe.backends.bf16 import execution
from hyper_multicore.modules.mega_moe.heap_manager import _HeapCapacityError
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec

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


class _FixedPlanner(BasePlanner):
    def __init__(self, result):
        self.result = result

    def plan(self, request):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _exercise_quota_agreement(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        spec = SimpleNamespace(rank_id=rank, ep_size=2, ep_group=dist.group.WORLD, local_experts=1,
                               architecture=2201, num_experts=2, num_cube_cores=24, batch_experts=False,
                               hidden_size=64, intermediate_size=32, expert_weight_layout="EOI",
                               transport_engine="mte", dispatch_mode="push", partial_weight_grad_dtype="native")
        resources = PlacementResources(SimpleNamespace(spec=spec), {}, False)
        request = PlacementRequest(((1, 1), (1, 1)), 2, 4, rank, 64, 32, 24, True)
        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        hidden = torch.zeros((1, 64), dtype=torch.bfloat16)
        good = ExpertPlacement(((1,), (0,)), ((1, 1, 0, 0), (0, 0, 1, 1)))
        mismatched = replace(good, dispatch_counts=tuple(reversed(good.dispatch_counts)))
        wrong_expert = replace(good, dispatch_counts=((2, 0, 0, 0), (0, 0, 1, 1)))
        cases = (
            (mismatched, "differs across EP ranks"),
            (None, "differs across EP ranks"),
            (RuntimeError("planner failed"), "rank=1.*planner failed"),
            (wrong_expert, "rank=1.*conserve every source/logical-expert count"),
        )
        for rejected, match in cases:
            planner = _FixedPlanner(good if rank == 0 else rejected)
            with pytest.raises(ValueError, match=match):
                resources._select_custom(planner, request, hidden, ids, None, None, mandatory=False)
            assert resources.pool is None

        lower = bounded_placement._Geometry.lower

        def wrong_position(geometry, *args):
            result = lower(geometry, *args)
            return replace(result, virtual_ids=result.virtual_ids.flip(-1)) if rank == 0 else result

        planner = _FixedPlanner(good)
        with (
            patch.object(bounded_placement._Geometry, "lower", wrong_position),
            pytest.raises(ValueError, match="rank=0.*logical expert at a source Top-K position"),
        ):
            resources._select_custom(planner, request, hidden, ids, None, None, mandatory=False)
        result = resources._select_custom(planner, request, hidden, ids, None, None, mandatory=False)
        assert result.virtual_ids.tolist() == ([[0, 1]] if rank == 0 else [[3, 2]])
        assert result.counts_by_source.tolist() == [[1, 1, 0, 0], [0, 0, 1, 1]]
        assert result.copies.tolist() == [[1], [0]]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU/Gloo")
def test_custom_planner_converges_quota_and_token_errors(tmp_path):
    """Reject asymmetric dispatch and wrong token identities before binding a bank."""
    mp.start_processes(_exercise_quota_agreement, args=((tmp_path / "rendezvous").as_uri(),),
                       nprocs=2, start_method="fork")


@pytest.mark.parametrize("mandatory", [False, True])
def test_existing_bank_growth_respects_hard_receive_limit(mandatory):
    """A failed placed growth may retry the original route only when allowed."""
    hidden = torch.empty((1, 4))
    ids = torch.zeros((1, 1), dtype=torch.int32)
    spec = SimpleNamespace(dispatch_mode="push", auto_tokens=True)
    binding = SimpleNamespace(placement=SimpleNamespace(receive_capacity=4), mandatory=mandatory)
    prepare = Mock(side_effect=[_HeapCapacityError("fixed heap"), None])
    original = route._RouteSummary(2, 128, 2)
    owner = SimpleNamespace(
        spec=spec, workspace=object(), prepare_auto_workspace=prepare,
        placement=SimpleNamespace(select=Mock(return_value=binding), uses_placement_wire=lambda: False,
                                  statistics={"heap_rejections": 0, "placements": 0}),
    )

    def exchanged(*_args, **options):
        return route._prepare_placed_route(
            hidden, ids, spec, None, None, None, None, None, None,
            options["prepare_workspace"], options["select_placement"],
        )

    with (
        patch.object(invocation, "prepare_topk_route", side_effect=exchanged),
        patch.object(route, "_read_cached_route", return_value=(original, None, None)),
        patch.object(route, "_materialize_route_permutation", return_value=(hidden, ids)),
        patch.object(route, "_finalize_dynamic_route", return_value="home") as finalize,
    ):
        call = invocation.prepare_expert_invocation(hidden, ids, hidden, None, (hidden, hidden), owner, planner=Mock())
        if mandatory:
            with pytest.raises(_HeapCapacityError, match="fixed heap"), call:
                pytest.fail("required placement exceeded the heap")
            prepare.assert_called_once_with(128, 4, hidden)
            finalize.assert_not_called()
        else:
            with call as prepared:
                assert prepared == "home"
            assert [entry.args for entry in prepare.call_args_list] == [(128, 4, hidden), (128, 2, hidden)]
    assert owner.placement.statistics == {"heap_rejections": 1, "placements": 0}


@pytest.mark.parametrize("placement_wire", [False, True])
def test_mismatched_dump_mode_rejects_cached_counts_before_placement(placement_wire):
    """Both route readers revalidate control flags even when counts repeat."""
    spec = MegaMoeSpec(128, 8, 4, 4, 1, 1.25, 256, 2, None, 0, 24)
    spec = replace(spec, auto_tokens=True)
    workspace = SimpleNamespace()
    hidden, ids = torch.empty(2, 8), torch.zeros(2, 1, dtype=torch.int32)
    gathered = torch.tensor([[0, 2, 1, 2, 0, 0, 0], [0, 2, 1, 0, 0, 0, 2]])
    if placement_wire:
        bounded_placement.read_placement_wire(gathered, spec, workspace, None)
    else:
        summary = route._summarize_host_route(gathered.tolist(), spec)
        route._cache_host_route(workspace, spec, gathered.tolist(), summary, None, hidden.device)
    gathered[1, 2] |= route._TENSOR_DUMP_MASK
    prepare, select = Mock(), Mock()
    with pytest.raises(RuntimeError, match="Tensor Dump mode differs across EP ranks"):
        route._prepare_placed_route(
            hidden, ids, spec, workspace, None, gathered[0, 3:], gathered, None,
            None, prepare, select, reuse_placement_geometry=placement_wire,
        )
    prepare.assert_not_called()
    select.assert_not_called()


def test_tensor_dump_preserves_the_original_replay_binding():
    """Materialize diagnostic operands without replacing the saved band protocol."""
    pool, placement = Mock(), Mock()
    snapshot = object()
    binding = ExpertWeightBinding(pool, placement, "EOI", snapshot=snapshot,
                                  execution_mode=ReplicaExecutionMode.FP32_BAND)
    diagnostic = binding.for_tensor_dump()
    assert diagnostic.mode is ReplicaExecutionMode.STAGED and not diagnostic.fused
    assert diagnostic.snapshot is snapshot and diagnostic.placement is placement
    assert diagnostic.token is not binding.token
    assert binding.mode is ReplicaExecutionMode.FP32_BAND and binding.fused_gradients


def test_submission_counts_the_executed_protocol_and_skips_activation_reuse():
    """A staged diagnostic launch and a fused launch each count once."""
    spec = SimpleNamespace(dispatch_mode="push", apply_probs_before_fc2=True, num_experts=2, architecture=0)
    resources = SimpleNamespace(spec=spec, workspace=Mock(), plan_for_tokens=Mock())
    resources.placement = PlacementResources(resources, {}, False)
    hidden, ids = torch.empty(2, 8), torch.zeros(2, 1, dtype=torch.int32)
    placement = SimpleNamespace(counts_by_source=torch.ones((2, 4)))
    pool = Mock()
    pool.pipeline.plan.return_value = (1, (), (1,))
    binding = ExpertWeightBinding(pool, placement, "EOI", execution_mode=ReplicaExecutionMode.FP32_BAND)
    prepared = route.PreparedTopKRoute(hidden, ids, ids, ids, SimpleNamespace(plan_tokens=128),
                                      weight_binding=binding)

    def forward(current):
        return execution.forward(None, hidden, hidden, ids, hidden, (hidden, hidden), resources, current)

    with (
        patch.object(execution, "execute_mega_moe_with_permutation", return_value=hidden) as launch,
        patch.object(execution, "_CheckpointedMegaMoeFunction") as retained,
    ):
        forward(replace(prepared, tensor_dump=True))
        assert launch.call_args.kwargs["weight_binding"].mode is ReplicaExecutionMode.STAGED
        forward(prepared)
        assert launch.call_args.kwargs["weight_binding"] is binding
        forward(replace(prepared, checkpoint_record=SimpleNamespace(allow_activation=True, activation=())))
        retained.apply.assert_called_once()
        launch.side_effect = RuntimeError("launch failed")
        with pytest.raises(RuntimeError, match="launch failed"):
            forward(prepared)
    stats = resources.placement.statistics
    assert stats["placements"] == 2
    assert stats["staged_bindings"] == stats["fp32_band_bindings"] == stats["skipped_recomputes"] == 1


def test_successful_fused_weight_acquire_records_prefetch_and_returns_operands():
    """Exercise the successful projection-ready path as well as failure cleanup."""
    pool, placement = Mock(), Mock()
    pool.statistics = {}
    binding = ExpertWeightBinding(pool, placement, "EOI", execution_mode=ReplicaExecutionMode.FP32_BAND)
    weights = (torch.empty(1), torch.empty(1))
    with patch.object(shmem_transfer, "_ordered_transfers", return_value=nullcontext()):
        assert binding.acquire(weights) is pool.pipeline.begin.return_value
        binding.release()
    pool.pipeline.begin.assert_called_once_with(weights, binding, False)
    assert pool.statistics["fused_prefetches"] == 1
    pool.release.assert_called_once_with()

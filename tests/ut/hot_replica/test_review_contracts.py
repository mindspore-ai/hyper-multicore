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

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
"""Validate placement transfer cost accounting."""

from dataclasses import replace

import pytest

from hyper_multicore.modules.base_planner import PlacementRequest
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import ReplicaCost, bf16_cost, select_replica_plan
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner, as_placement


def test_cost_charges_the_busy_owner_and_guest_in_each_stage():
    """Evaluate distinct endpoint loads using the production BF16 cost model."""
    training = bf16_cost(5120, 1792, 24, True)
    inference = bf16_cost(5120, 1792, 24, False)
    assert training.transfer(2, 1) > training.transfer(1, 2)
    assert inference.transfer(2, 1) == inference.transfer(1, 2)
    assert training.transfer(2, 1) > training.transfer(1, 1)
    assert training.setup > bf16_cost(6144, 2048, 24, True).setup


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_cross_node_cost_requires_a_finite_positive_multiplier(value):
    """Reject invalid or ambiguous values before EP resource binding."""
    with pytest.raises(ValueError, match="finite positive multiplier"):
        HotReplicaPlanner(cross_node_cost=value)


def test_cross_node_cost_is_part_of_ep_planner_identity():
    """Two EP members may only share a planner configuration with the same charge."""
    assert HotReplicaPlanner(cross_node_cost=1).configuration_key() == "cross_node_cost=1.0"
    assert HotReplicaPlanner(cross_node_cost=0.75).configuration_key() != HotReplicaPlanner().configuration_key()


def test_cross_node_cost_does_not_veto_mandatory_receive_capacity():
    """A required BF16 capacity plan still has to satisfy the actual row limit."""
    counts = [[1024, 1024, 0, 0]] * 4
    plan = select_replica_plan(counts, ExpertReplicaConfig(4, 4, 1), receive_capacity_percent=100,
                               nodes=(0, 0, 1, 1), prices=ReplicaCost(1e4, ((100., 100.),)),
                               cross_node_cost=2.0)
    assert max(plan.destination_loads) <= 2048
    plan.validate()


@pytest.mark.parametrize(("home_count", "receive_percent", "uses_replica"), [
    (10000, "auto", False),
    (10000, 100, True),
    (15500, "auto", True),
])
def test_builtin_planner_scores_independently_of_committed_receive_rows(home_count, receive_percent, uses_replica):
    """Keep the same quotas across growth and retain explicit capacity requirements."""
    other_count = 16000 - home_count
    counts = ((home_count, home_count, other_count, other_count),) * 2
    request = PlacementRequest(counts, 4 * home_count, 64000, 0, 5120, 1792, 24, True, rank_nodes=(0, 0))
    planner = HotReplicaPlanner(replica_slots_per_rank=1, receive_capacity_percent=receive_percent)
    expected = planner.plan(request)
    assert (expected is not None) == uses_replica
    for capacity in (16000, 32000, 40064, 64000):
        assert planner.plan(replace(request, ordinary_receive_capacity=capacity)) == expected


def test_equal_replica_cost_prefers_fewer_cross_node_token_rows():
    """Use source quotas for a tie even when every whole-matrix edge costs 1.0."""
    replicas = 6
    counts = []
    for rank in range(16):
        row = [0] * 96
        for position in range(4096 * 8):
            expert = (position + 13 * rank) % 24 + 48 * (rank // 8)
            row[expert] += 1
        counts.append(row)
    config = ExpertReplicaConfig(96, 16, replicas)
    options = {"nodes": (0,) * 8 + (1,) * 8, "prices": ReplicaCost(phases=((896., 896.),))}
    neutral = select_replica_plan(counts, config, **options, cross_node_cost=1.0)
    expensive = select_replica_plan(counts, config, **options, cross_node_cost=2.0)
    assert neutral.transfers and neutral == expensive
    assert all(edge.owner_rank // 8 == edge.target_rank // 8 for edge in neutral.transfers)
    neutral.validate()


def test_automatic_budget_can_choose_a_separate_single_slot_trajectory():
    """A full-budget trajectory can miss a cheaper physical width at the same exposed cost."""
    counts = (
        (2, 20, 9, 3, 17, 0, 20, 2, 3, 0, 20, 20),
        (0, 0, 17, 17, 20, 0, 3, 20, 0, 5, 3, 5),
        (3, 5, 1, 0, 0, 5, 0, 0, 1, 0, 0, 3),
        (0, 17, 17, 1, 20, 1, 0, 17, 5, 20, 20, 0),
    )
    budget = ExpertReplicaConfig(12, 4, 3)
    prices = ReplicaCost(setup=1., phases=((3., 3.),))
    full = select_replica_plan(counts, budget, prices=prices)
    one = select_replica_plan(counts, replace(budget, replica_slots_per_rank=1), prices=prices)
    auto = select_replica_plan(counts, budget, prices=replace(prices, search_single_slot=True))
    assert full.compact_borrowed_width == 2
    assert as_placement(auto) == as_placement(one)
    auto.validate()

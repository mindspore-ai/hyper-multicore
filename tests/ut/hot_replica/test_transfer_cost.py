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

from hyper_multicore.modules.hot_replica._planner.aligned import materialize_aligned_moves
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import (
    ReplicaCapacityError, ReplicaCost, SourceQuota, bf16_cost, select_replica_plan,
)
from hyper_multicore.modules.hot_replica.planner import as_placement
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner


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
    options = dict(nodes=(0,) * 8 + (1,) * 8, prices=ReplicaCost(phases=((896., 896.),)))
    neutral = select_replica_plan(counts, config, **options, cross_node_cost=1.0)
    expensive = select_replica_plan(counts, config, **options, cross_node_cost=2.0)
    assert neutral.transfers and neutral == expensive
    assert all(edge.owner_rank // 8 == edge.target_rank // 8 for edge in neutral.transfers)
    neutral.validate()


@pytest.mark.parametrize(("arch", "precision", "engine", "charge_whole_copies", "minimum_rows"), [
    (2201, "bf16", "mte", False, 1),
])
def test_training_auto_uses_the_backend_coefficients(arch, precision, engine, charge_whole_copies, minimum_rows):
    """Select the backend coefficients in the common score and preserve the explicit limit."""
    planner = HotReplicaPlanner()
    prices, limit, _ = planner._policy(5120, 1792, 32 if arch == 3510 else 24, True, engine,
                                       60000, 32768 * 16, 16, None, architecture=arch, precision=precision)
    assert (prices.matrix_copy > 0) == charge_whole_copies
    assert prices.minimum_replica_rows == minimum_rows
    assert limit is None
    hard = HotReplicaPlanner(receive_capacity_percent=102)
    hard_prices, limit, _ = hard._policy(5120, 1792, 32 if arch == 3510 else 24, True, engine,
                                       60000, 32768 * 16, 16, None, architecture=arch, precision=precision)
    assert hard_prices.matrix_copy == 0
    assert limit == 33424


def test_matrix_copy_charge_can_reject_an_optional_plan_but_respects_a_hard_limit():
    """Charge every additional matrix without discarding a required placement."""
    counts = ((4096, 0, 0, 0),) * 4
    config = ExpertReplicaConfig(4, 4, 1)
    assert select_replica_plan(counts, config, prices=ReplicaCost()) is not None
    prices = ReplicaCost(matrix_copy=7000.)
    assert select_replica_plan(counts, config, prices=prices) is None
    required = select_replica_plan(counts, config, prices=prices, limit=4096)
    required.validate()
    assert max(required.destination_loads) == 4096


def test_full_budget_can_select_the_same_physical_layout_as_one_slot():
    """Keep a one-slot layout that the full-budget greedy path cannot produce."""
    counts = (
        (0, 0, 0, 0, 0, 3, 0, 20, 1, 5, 20, 17),
        (9, 0, 20, 1, 0, 0, 0, 20, 20, 0, 0, 9),
        (0, 2, 0, 0, 0, 20, 0, 0, 17, 9, 9, 0),
        (0, 0, 0, 9, 20, 3, 0, 0, 5, 0, 3, 0),
    )
    cap = ExpertReplicaConfig(12, 4, 3)
    prices = ReplicaCost(setup=1., phases=((3., 3.),), matrix_copy=4.)
    full = select_replica_plan(counts, cap, prices=prices)
    one = select_replica_plan(counts, replace(cap, replica_slots_per_rank=1), prices=prices)
    auto = select_replica_plan(counts, cap, prices=replace(prices, search_single_slot=True))
    assert full.compact_borrowed_width == 2
    assert auto.compact_borrowed_width == 1
    assert as_placement(auto) == as_placement(one)
    auto.validate()


def test_multi_slot_search_preserves_a_required_wide_layout():
    """Failure of the one-slot budget cannot veto a feasible wider placement."""
    counts = ((10, 10, 10, 0, 0, 0),) * 2
    prices = ReplicaCost(setup=1e4, matrix_copy=1e4, search_single_slot=True)
    with pytest.raises(ReplicaCapacityError):
        select_replica_plan(counts, ExpertReplicaConfig(6, 2, 1), prices=prices, limit=30)
    plan = select_replica_plan(counts, ExpertReplicaConfig(6, 2, 3), prices=prices, limit=30)
    assert plan.compact_borrowed_width == 2
    assert plan.destination_loads == (30, 30)
    plan.validate()


def test_materialized_admission_keeps_hard_capacity_requests():
    """Apply measured admission only after materializing an optional placement."""
    counts = ((4096, 0, 0, 0),) * 4
    config = ExpertReplicaConfig(4, 4, 1)
    prices = ReplicaCost(admission=ReplicaCost(setup=14000.))
    assert select_replica_plan(counts, config, prices=prices) is None
    assert max(select_replica_plan(counts, config, prices=prices, limit=4096).destination_loads) == 4096


def test_aligned_plan_intersects_original_source_rows_and_preserves_the_tail():
    """Lower K64 cuts using global expert row order even across source ranks."""
    plan = select_replica_plan(
        ((150, 0), (43, 0)), ExpertReplicaConfig(2, 2, 1), quantum=64, limit=129,
        source_quota=SourceQuota.RANK_ORDER, materialize=materialize_aligned_moves,
    )
    assert plan.intervals == ((0, 128), (0, 0), (0, 0), (128, 193))
    assert plan.dispatch_counts == ((128, 0, 0, 22), (0, 0, 0, 43))
    plan.validate()


def test_search_and_capacity_repair_respect_peer_edges():
    """An unreachable edge cannot satisfy an explicit receive limit."""
    counts = ((64, 0), (64, 0))
    config = ExpertReplicaConfig(2, 2, 1)
    edges = ((True, False), (False, True))
    assert select_replica_plan(counts, config, peer_edges=edges) is None
    with pytest.raises(ReplicaCapacityError):
        select_replica_plan(counts, config, peer_edges=edges, limit=64)

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

import pytest

from hyper_multicore.modules.hot_replica._planner.planner import build_expert_replica_plan
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner
from hyper_multicore.modules.hot_replica.placement import placement_is_profitable


def test_charge_replay_weight_restore():
    """Include backward weight restoration in the placement cost estimate."""
    common = {
        "maximum_received": 300000,
        "target_load": 131072,
        "hidden_size": 2560,
        "intermediate_size": 768,
        "cube_cores": 24,
        "ep_size": 8,
        "local_experts": 64,
        "weight_bytes": 64 * 3 * 2560 * 768 * 2,
        "gradient_bytes": 64 * 3 * 2560 * 768 * 4,
    }
    assert placement_is_profitable(**common, projection_flops=18, backward_prefetch_weight_bytes=common["weight_bytes"])
    assert not placement_is_profitable(
        **common, projection_flops=18, backward_prefetch_weight_bytes=8 * common["weight_bytes"]
    )


def test_actual_copy_count_and_backward_fetch_change_profitability(monkeypatch):
    """Charge every borrowed expert and both forward/backward peer fetches."""
    from hyper_multicore.modules.hot_replica import placement  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(placement, "_A3_FLOPS_PER_CUBE", 1.0)
    monkeypatch.setattr(placement, "_A3_STAGING_BYTES_PER_SECOND", 1e12)
    monkeypatch.setattr(placement, "_A3_PEER_BYTES_PER_SECOND", 250.0)
    monkeypatch.setattr(placement, "_A3_PLACEMENT_SECONDS", 0.0)
    geometry = {
        "maximum_received": 1, "target_load": 0, "hidden_size": 1, "intermediate_size": 1,
        "cube_cores": 1, "weight_bytes": 400, "gradient_bytes": 800, "local_experts": 4,
        "ep_size": 2, "projection_flops": 3,
    }
    assert placement_is_profitable(**geometry, transfer_experts=1, backward_prefetch_weight_bytes=400)
    assert not placement_is_profitable(**geometry, transfer_experts=2, backward_prefetch_weight_bytes=400)
    assert placement_is_profitable(**geometry, transfer_experts=2, backward_prefetch_weight_bytes=0)


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
    plan = build_expert_replica_plan(counts, 1, minimum_load_reduction=0.25, minimum_replica_rows=256,
                                     capacity_limit=2048, receive_capacity=2048, rank_nodes=(0, 0, 1, 1),
                                     transfer_row_credit=1024, cross_node_cost=2.0)
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
    options = dict(minimum_replica_rows=1024, minimum_load_reduction=0.25, capacity_limit=393216,
                   receive_capacity=393216, rank_nodes=(0,) * 8 + (1,) * 8, transfer_row_credit=896)
    neutral = build_expert_replica_plan(counts, replicas, **options, cross_node_cost=1.0)
    expensive = build_expert_replica_plan(counts, replicas, **options, cross_node_cost=2.0)
    assert neutral.transfers and neutral == expensive
    assert all(edge.owner_rank // 8 == edge.target_rank // 8 for edge in neutral.transfers)
    neutral.validate()

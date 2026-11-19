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
"""CPU contracts for planner count formats and optional execution dependencies."""

import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.hot_replica import placement
from hyper_multicore.modules.hot_replica._planner.aligned import build_aligned_expert_plan
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.mx_peak_planner import build_mx_local_peak_plan


class TestHotReplicaPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

    def test_mx_skips_whole_search_when_largest_expert_cannot_fit(self) -> None:
        config = ExpertReplicaConfig(4, 4, 1)
        matrix = ((16384, 0, 0, 0),) + ((0, 0, 0, 0),) * 3
        with patch("hyper_multicore.modules.hot_replica._planner.whole._select_whole_targets") as whole:
            plan = build_aligned_expert_plan(
                matrix, config, capacity_limit=16384, minimum_replica_rows=64, minimum_load_reduction=.25,
            )
        whole.assert_not_called()
        plan.validate()
        self.assertEqual(max(plan.destination_loads), 8192)

    def test_mx_local_peak_refines_tied_ranks_without_changing_node_traffic(self) -> None:
        config = ExpertReplicaConfig(8, 8, 1)
        nodes = (0, 0, 0, 0, 1, 1, 1, 1)
        matrix = ((8192, 0, 0, 0, 8192, 0, 0, 0),) * 8
        baseline = build_aligned_expert_plan(
            matrix, config, capacity_limit=131072, minimum_replica_rows=8192, minimum_load_reduction=.25,
            rank_nodes=nodes,
        )
        options = dict(capacity=131072, minimum_rows=8192, minimum_reduction=.25, row_credit=0, nodes=nodes)
        plan = build_mx_local_peak_plan(matrix, config, transfer_rows=3178, **options)
        plan.validate()
        self.assertEqual(max(baseline.destination_loads), 32768)
        self.assertEqual(max(plan.destination_loads), 16384)
        self.assertTrue(all(nodes[edge.owner_rank] == nodes[edge.target_rank] for edge in plan.transfers))
        for before, after in zip(baseline.dispatch_counts, plan.dispatch_counts):
            # The total is conserved for each source rank and destination node.
            self.assertEqual(sum(before[:8]), sum(after[:8]))
        self.assertEqual(build_mx_local_peak_plan(matrix, config, transfer_rows=32768, **options), baseline)

    def test_mx_local_peak_does_not_admit_an_inactive_short_route(self) -> None:
        config = ExpertReplicaConfig(4, 4, 1)
        matrix = ((1024, 0, 0, 0),) * 4
        plan = build_mx_local_peak_plan(
            matrix, config, capacity=4096, minimum_rows=8192, minimum_reduction=.25,
            row_credit=896, nodes=(0, 0, 1, 1), transfer_rows=3178,
        )
        plan.validate()
        self.assertFalse(plan.transfers)

    def test_receive_limit_matches_full_and_short_pr19_routes(self) -> None:
        limit = placement.placement_receive_limit
        self.assertEqual(limit(131072, 8, 64, 8, 2560, 2), 170496)
        self.assertEqual(limit(32768, 8, 64, 8, 2560, 2), 42624)
        self.assertEqual(limit(1024, 8, 64, 8, 2560, 2), 8192)
        self.assertEqual(limit(1024, 8, 64, 8, 65536, 2), 1408)
        self.assertEqual(limit(0, 8, 64, 8, 2560, 2), 0)

    def test_cost_charges_gradient_wire_bytes_and_peer_fanout(self) -> None:
        """Identical arithmetic must not ignore larger gradients or more peers."""
        geometry = (11000, 4000, 5120, 1792, 24, 3 * 3 * 5120 * 1792 * 2)
        common = {"projection_flops": 12, "local_experts": 3}
        bf16 = geometry[-1]
        self.assertTrue(placement.placement_is_profitable(*geometry, ep_size=2, gradient_bytes=bf16, **common))
        self.assertFalse(placement.placement_is_profitable(*geometry, ep_size=8, gradient_bytes=2 * bf16, **common))

    def test_cost_scales_with_expert_parameter_size_not_imbalance_ratio(self) -> None:
        """The same rank imbalance can justify small experts but not large experts."""
        base = {
            "maximum_received": 300000,
            "target_load": 131000,
            "hidden_size": 2560,
            "intermediate_size": 768,
            "cube_cores": 24,
            "projection_flops": 12,
            "ep_size": 8,
        }
        one = 3 * 2560 * 768 * 2
        self.assertTrue(
            placement.placement_is_profitable(**base, local_experts=64, weight_bytes=64 * one, gradient_bytes=128 * one)
        )
        self.assertFalse(
            placement.placement_is_profitable(
                **base, local_experts=64, weight_bytes=256 * one, gradient_bytes=512 * one
            )
        )

    def test_cost_rejects_invalid_transfer_extents(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient traffic"):
            placement.placement_is_profitable(100, 20, 64, 32, 24, 1024, gradient_bytes=-1)
        with self.assertRaisesRegex(ValueError, "cost geometry"):
            placement.placement_is_profitable(100, 20, 64, 32, 24, 1024, ep_size=0)

    def test_ungated_cost_accounts_for_two_projections(self) -> None:
        """A marginal gated saving must not enable borrowing for a cheaper ungated FFN."""
        geometry = (300000, 0, 512, 128, 20, 0)
        self.assertTrue(placement.placement_is_profitable(*geometry, projection_flops=6))
        self.assertFalse(placement.placement_is_profitable(*geometry, projection_flops=4))

    def test_capacity_placement_preserves_source_intervals(self) -> None:
        """Split the same logical expert in source-rank order for both wire dtypes."""
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([[4, 0], [2, 0]], dtype=dtype)
                result = placement.build_placement(
                    torch.empty(4, 4), torch.zeros(4, 1, dtype=torch.int32),
                    counts, 3, 0, source_inverse=torch.arange(4, dtype=torch.int32),
                )
                self.assertIsNotNone(result)
                torch.testing.assert_close(result.virtual_ids, torch.tensor([[0], [0], [0], [3]], dtype=torch.int32))
                torch.testing.assert_close(result.counts_by_source,
                                           torch.tensor([[3, 0, 0, 1], [0, 0, 0, 2]], dtype=torch.int32))
                torch.testing.assert_close(result.copies, torch.tensor([[-1], [0]], dtype=torch.int32))
                self.assertEqual((result.receive_capacity, result.published_slots), (3, 0))

    def test_wide_counts_reject_overflow_before_allocator(self) -> None:
        """A large INT64 wire count must never wrap to a negative planner value."""
        counts = torch.tensor([[2**31, 0], [0, 0]], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "INT32 planner limit"):
            placement.build_placement(torch.empty(0, 4), torch.empty(0, 1, dtype=torch.int32), counts, 1, 0)

    def test_invalid_topologies_fail_before_allocator(self) -> None:
        """An invalid rank or expert extent cannot launch an asymmetric plan."""
        for counts, rank in (
            (torch.zeros(2, 3, dtype=torch.int32), 0),
            (torch.zeros(2, 4, dtype=torch.int32), 2),
            (torch.zeros(4, dtype=torch.int32), 0),
        ):
            with self.subTest(shape=counts.shape, rank=rank):
                with (
                    patch("hyper_multicore.modules.hot_replica.placement.build_capacity_plan") as allocate,
                    self.assertRaises(ValueError),
                ):
                    placement.build_placement(
                        torch.empty(0, 4),
                        torch.empty(0, 1, dtype=torch.int32),
                        counts,
                        1,
                        rank,
                    )
                allocate.assert_not_called()

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
from hyper_multicore.modules.hot_replica._planner.receive import receive_limit


class TestHotReplicaPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

    def test_receive_limit_is_exact_and_auto_has_no_limit(self) -> None:
        self.assertIsNone(receive_limit("auto", 525, 4))
        self.assertEqual(receive_limit(102, 525, 4), 134)
        self.assertEqual(receive_limit(100, 525, 4), 132)

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

    def test_capacity_placement_honors_an_explicit_guest_budget(self) -> None:
        counts = torch.tensor([[4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0],
                               [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        ids = torch.arange(6, dtype=torch.int32).repeat_interleave(4).view(-1, 1)
        hidden = torch.empty(24, 4)
        inverse = torch.arange(24, dtype=torch.int32)
        ordinary = placement.build_placement(hidden, ids, counts, 24, 1, source_inverse=inverse)
        capped = placement.build_placement(
            hidden, ids, counts, 24, 1, source_inverse=inverse, replica_slots_per_rank=2,
        )
        self.assertEqual((ordinary.published_slots, ordinary.receive_capacity), (3, 24))
        self.assertEqual((capped.borrowed_capacity, capped.published_slots, capped.receive_capacity), (2, 2, 32))
        self.assertEqual(capped.counts_by_source.sum().item(), ids.numel() * 2)

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
                    patch("hyper_multicore.modules.hot_replica.placement.search_ordered_plan") as allocate,
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

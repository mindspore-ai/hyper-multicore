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

import sys
from dataclasses import replace
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.hot_replica import placement


class TestHotReplicaPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

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

    def test_native_and_legacy_counts_share_the_int32_planner(self) -> None:
        """Normalize equivalent wire types without changing the ownership decision."""
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([[512, 0], [0, 0]], dtype=dtype)
                copies = torch.full((2, 1), -1, dtype=torch.int32)
                with (
                    patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
                    patch(
                        "hyper_multicore.modules.hot_replica.placement.allocate_experts",
                        return_value=(torch.zeros(2, 2, dtype=torch.int32), None, None, copies),
                    ) as allocate,
                ):
                    result = placement.build_placement(
                        torch.empty(512, 4),
                        torch.zeros(512, 1, dtype=torch.int32),
                        counts,
                        256,
                        0,
                    )
                self.assertIsNone(result)
                actual_counts, actual_capacity = allocate.call_args.args
                self.assertEqual(actual_capacity, 256)
                self.assertEqual(actual_counts.dtype, torch.int32)
                torch.testing.assert_close(actual_counts, counts.to(torch.int32))

    def test_expert_ids_reuse_the_planners_host_copy(self) -> None:
        """Both rank-local and full diagnostic maps use the placement's original Host snapshot."""
        counts = torch.tensor([[4, 0], [0, 0]], dtype=torch.int32)
        copies = torch.tensor([[-1], [0]], dtype=torch.int32)
        ids = torch.tensor([[0], [0], [0], [0]], dtype=torch.int32)
        allocation = (torch.tensor([[2, 4], [0, 0]]), None, None, copies)
        routes = (ids.flatten(), torch.zeros(2, 4, dtype=torch.int32))
        with (
            patch.object(placement, "allocate_experts", return_value=allocation),
            patch.object(placement, "render_expert_major", return_value=routes),
        ):
            result = placement.build_placement(torch.empty(4, 2), ids, counts, 2, 1, source_inverse=torch.arange(4))
        self.assertEqual(replace(result, host_copies=None).expert_ids(), (0, None, 1, 0))
        copies.fill_(-1)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected placement readback")):
            self.assertEqual(result.expert_ids(), (0, None, 1, 0))
            self.assertEqual(result.expert_ids(1), (1, 0))

    def test_wide_counts_reject_overflow_before_allocator(self) -> None:
        """A large INT64 wire count must never wrap to a negative planner value."""
        counts = torch.tensor([[2**31, 0], [0, 0]], dtype=torch.int64)
        with (
            patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
            patch("hyper_multicore.modules.hot_replica.placement.allocate_experts") as allocate,
            self.assertRaisesRegex(ValueError, "INT32 planner limit"),
        ):
            placement.build_placement(torch.empty(0, 4), torch.empty(0, 1, dtype=torch.int32), counts, 1, 0)
        allocate.assert_not_called()

    def test_invalid_topologies_fail_before_allocator(self) -> None:
        """An invalid rank or expert extent cannot launch an asymmetric plan."""
        for counts, rank in (
            (torch.zeros(2, 3, dtype=torch.int32), 0),
            (torch.zeros(2, 4, dtype=torch.int32), 2),
            (torch.zeros(4, dtype=torch.int32), 0),
        ):
            with self.subTest(shape=counts.shape, rank=rank):
                with (
                    patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
                    patch("hyper_multicore.modules.hot_replica.placement.allocate_experts") as allocate,
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

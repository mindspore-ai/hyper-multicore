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
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_multicore.experimental import HotReplicaPlanner, PlacementRequest
from hyper_multicore.modules.hot_replica.bounded_placement import render_geometry
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import SourceQuota, receive_limit, select_replica_plan


class TestHotReplicaPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

    def test_receive_limit_is_exact_and_auto_has_no_limit(self) -> None:
        self.assertIsNone(receive_limit("auto", 525, 4))
        self.assertEqual(receive_limit(102, 525, 4), 134)
        self.assertEqual(receive_limit(100, 525, 4), 132)

    def test_capacity_placement_preserves_source_intervals(self) -> None:
        """Split the same logical expert in source-rank order for both wire dtypes."""
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([[4, 0], [2, 0]], dtype=dtype)
                plan = select_replica_plan(counts.numpy(), ExpertReplicaConfig(2, 2, 1), soft_target=3,
                                           source_quota=SourceQuota.RANK_ORDER)
                ids = torch.zeros(4, 1, dtype=torch.int32)
                geometry = render_geometry(plan, SimpleNamespace(rank_id=0), ids.device, 4, prepare_metadata=False)
                result = geometry.lower(torch.empty(4, 4), ids, torch.arange(4, dtype=torch.int32))
                torch.testing.assert_close(result.virtual_ids, torch.tensor([[0], [0], [0], [3]], dtype=torch.int32))
                torch.testing.assert_close(result.counts_by_source,
                                           torch.tensor([[3, 0, 0, 1], [0, 0, 0, 2]], dtype=torch.int32))
                torch.testing.assert_close(result.copies, torch.tensor([[-1], [0]], dtype=torch.int32))
                self.assertEqual((result.receive_capacity, result.published_slots), (3, 0))

    def test_capacity_placement_honors_an_explicit_guest_budget(self) -> None:
        counts = ((4,) * 6 + (0,) * 6,) * 2
        ordinary = select_replica_plan(counts, ExpertReplicaConfig(12, 2, 6), soft_target=24)
        capped = select_replica_plan(counts, ExpertReplicaConfig(12, 2, 2), soft_target=24)
        self.assertEqual((ordinary.replica_slots_used[1], max(ordinary.destination_loads)), (3, 24))
        self.assertEqual((capped.compact_borrowed_width, capped.replica_slots_used[1],
                          max(capped.destination_loads)), (2, 2, 32))
        self.assertEqual(sum(capped.destination_loads), 48)

    def test_wide_counts_reject_overflow_before_allocator(self) -> None:
        """A large INT64 wire count must never wrap to a negative planner value."""
        request = PlacementRequest(((2**31, 0), (0, 0)), 2**31, 2**31, 0, 4, 4, 24, False)
        with self.assertRaisesRegex(ValueError, "INT32 planner limit"):
            HotReplicaPlanner().plan(request)

    def test_invalid_topologies_fail_before_allocator(self) -> None:
        """An invalid rank or expert extent cannot launch an asymmetric plan."""
        request = PlacementRequest(((0, 0), (0, 0)), 0, 0, 0, 4, 4, 24, False)
        for changes in ({"counts_by_source": ((0, 0, 0),) * 2}, {"rank": 2}, {"counts_by_source": ()}):
            with self.subTest(changes=changes):
                with (
                    patch("hyper_multicore.modules.hot_replica.planner.select_replica_plan") as allocate,
                    self.assertRaises(ValueError),
                ):
                    HotReplicaPlanner().plan(replace(request, **changes))
                allocate.assert_not_called()

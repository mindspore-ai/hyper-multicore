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
"""Keep directional graph events and persistent workspace generations aligned."""

import itertools
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from hyper_multicore.scheduler.config import ATOMIC_ADD_VALUE_LEN, READY_CACHE_LINE_BYTES, mega_moe_event_capacity


class TestFusionEventCapacity(unittest.TestCase):
    """Exercise a topology whose endpoint events cross a counter alignment boundary."""

    def _assert_graph_workspace_capacity(self, spec, expected):
        """Both native graphs must place ready generations after the allocated prefix."""
        _, forward, _, backward = _build_runtime_artifacts(spec)
        workspace = MegaMoeWorkspace(shared=False)
        with patch("hyper_multicore.modules.mega_moe.workspace.shmem.empty",
                   side_effect=lambda shape, **kw: torch.empty(shape, dtype=kw["dtype"])), \
                patch("hyper_multicore.modules.mega_moe.workspace.torch.npu", create=True):
            workspace.ensure(spec, torch.bfloat16, torch.device("cpu"))
        self.assertEqual(workspace.event_counter_bytes, expected * 4)
        persistent_bytes = 2 * (spec.ep_size + 1) * READY_CACHE_LINE_BYTES
        for cfg, storage in ((forward, workspace.forward_event_counters),
                             (backward, workspace.backward_event_counters)):
            self.assertEqual(cfg.event_capacity, expected)
            self.assertEqual(storage.numel(), cfg.event_capacity * 4 + persistent_bytes)
            self.assertLessEqual(cfg.ready_event + 2 * ATOMIC_ADD_VALUE_LEN, cfg.event_capacity)
            for task in cfg.all_tasks[:cfg.task_num]:
                self.assertLessEqual(task.trigger_event + ATOMIC_ADD_VALUE_LEN, cfg.event_capacity)

    def test_route_capacity_preserves_legacy_layout(self):
        """E510/EP3 needs two extra lanes only when either route direction is fused."""
        self.assertEqual(mega_moe_event_capacity(510, 3), 1040)
        self.assertEqual(mega_moe_event_capacity(510, 3, route_fusion=True), 1056)
        for mode, (forward, backward) in itertools.product(
                ("push", "pull"), (("legacy", "legacy"), ("both", "legacy"), ("legacy", "both"))):
            with self.subTest(mode=mode, forward=forward, backward=backward):
                spec = MegaMoeSpec(
                    128, 128, 128, 510, 2, None, 256, 3, None, 0, 20,
                    dispatch_mode=mode, auto_tokens=True,
                    forward_fusion_mode=forward, backward_fusion_mode=backward,
                )
                self._assert_graph_workspace_capacity(spec, 1040 if forward == backward == "legacy" else 1056)

    def test_mixed_shared_directions_use_one_counter_prefix(self):
        """An external shared direction retains the same ready-tail base as its fused peer."""
        for mode, (shared_forward, shared_backward) in itertools.product(
                ("push", "pull"), ((True, False), (False, True))):
            with self.subTest(mode=mode, shared_forward=shared_forward, shared_backward=shared_backward):
                spec = MegaMoeSpec(
                    128, 128, 128, 510, 2, None, 256, 3, None, 0, 20,
                    dispatch_mode=mode, auto_tokens=True,
                    forward_fusion_mode="both", backward_fusion_mode="legacy",
                    shared_intermediate_size=128, forward_shared_expert=shared_forward,
                    backward_shared_expert=shared_backward,
                )
                self._assert_graph_workspace_capacity(spec, 1088)

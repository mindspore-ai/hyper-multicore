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
"""Host regression for independent routing and shared-expert switches."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.fusion import _prepare_backward_route
from hyper_multicore.modules.mega_moe.diagnostics import bind_shared_diagnostic_ports
from hyper_multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.shared import shared_backward_buffers, shared_tiling_bytes
from hyper_multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType
from hyper_multicore.scheduler.graph import OpType


class TestSharedExpertSchedule(unittest.TestCase):
    """Check shared dependencies without launching an accelerator kernel."""

    def test_ragged_push_backward_uses_receive_capacity(self):
        """A source can send more routed rows than it receives from the group."""
        spec = MegaMoeSpec(
            257, 8, 4, 4, 2, None, 384, 2, None, 1, 20,
            schedule_tokens=512, auto_tokens=True,
            forward_fusion_mode="both", backward_fusion_mode="both",
        )
        workspace = SimpleNamespace(expert_buffer=torch.empty(384, 8), routed_buffer=torch.empty(1024, 8))
        saved = SimpleNamespace(dispatch=torch.empty(257, 8), expert_output=torch.empty(514, 8),
                                permute_src_index=torch.arange(514, dtype=torch.int32),
                                topk_weights=torch.ones(257, 2), expert_output_reusable=False)
        context = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace)
        route = _prepare_backward_route(context, torch.empty(257, 8), saved)
        self.assertEqual(tuple(route.dispatch.shape), (257, 8))
        self.assertEqual(route.dispatch.data_ptr(), workspace.expert_buffer.data_ptr())
        self.assertEqual(tuple(route.kernel_grad_output.shape), (257, 8))
        self.assertEqual(route.permute_src_index.numel(), 514)

    def test_empty_shared_weight_gradients_do_not_alias_live_route_scratch(self):
        """A rank with no source tokens produces zero shared gradients independently."""
        x, w1, w2 = torch.empty(0, 8), torch.empty(6, 8), torch.empty(8, 3)
        saved = (x, w1, w2, torch.empty(0, 6), torch.empty(0, 3))
        scratch = (torch.full((128,), float("nan")), torch.full((128,), float("nan")))
        with patch("hyper_multicore.modules.mega_moe.shared.pointer_table", return_value=torch.empty(0)):
            values, _, reuse = shared_backward_buffers(x, saved, scratch=scratch, integrated_output=True)
        self.assertFalse(reuse)
        for index in (2, 10):
            self.assertEqual(torch.count_nonzero(values[index]).item(), 0)
            self.assertNotEqual(values[index].data_ptr(), scratch[1].data_ptr())
        self.assertTrue(all(torch.isnan(value).all() for value in scratch))

    def _check_endpoint_rows(self, op, tasks, tokens):
        """Endpoint descriptors retain real rows and their completion dependency."""
        self.assertEqual(op.outputs[0].shape[0], tokens)
        self.assertEqual(len(tasks), max(1, (tokens + op.split_value - 1) // op.split_value))
        self.assertTrue(all(task.outputs[0].dim[0] == tokens for task in tasks))
        self.assertTrue(all(task.dependent_event != EVENT_INVALID_ID for task in tasks))

    def _check_shared_vector_rows(self, op, tasks, tokens, tiling):
        """Vector tails keep 128-row slot offsets and exact valid-row extents."""
        self.assertEqual(len(tasks), max(1, (tokens + 127) // 128))
        for index, task in enumerate(tasks):
            rows = max(0, min(128, tokens - index * 128))
            self.assertEqual(task.task_split_value, rows)
            for descriptors, tensors in ((task.inputs, op.inputs), (task.outputs, op.outputs)):
                for descriptor, tensor in zip(descriptors, tensors):
                    width = tensor.shape[1]
                    self.assertEqual(descriptor.base_ptr_offset, index * 128 * width)
                    self.assertLessEqual(descriptor.base_ptr_offset + rows * width, tokens * width)
            if op.op_type in (OpType.SWIGLU, OpType.SWIGLU_GRAD):
                expected_offset = op.fill_config.tiling_offset + (80 if 0 < rows < 128 else 0)
                self.assertEqual(task.tiling_data_offset, expected_offset)
                self.assertLessEqual(task.tiling_data_offset + 80, len(tiling))

    def _check_dynamic_graph_rows(self, spec, graph, cfg, backward):
        """Check all local endpoints and shared stages against one invocation shape."""
        tokens = spec.local_num_tokens
        tiling = shared_tiling_bytes(spec, backward=backward)
        self.assertEqual(len(tiling), 8 + 2016 * 20 * (4 if backward else 2)
                         + 80 * (1 + int(bool(tokens % 128))))
        cursor = 0
        for op in graph.topological_sort():
            tasks = cfg.all_tasks[cursor:cursor + op.task_num]
            cursor += op.task_num
            if op.name in {"weighted_unpermute", "unpermute_grad", "topk_weight_grad"}:
                self._check_endpoint_rows(op, tasks, tokens)
            elif op.name.startswith("shared_"):
                if op.op_type == OpType.GMM:
                    self.assertTrue(all(task.task_split_value == tokens for task in tasks))
                else:
                    self._check_shared_vector_rows(op, tasks, tokens, tiling)
                    if tokens == 0:
                        self.assertGreaterEqual(cfg.all_event_num_triggers[tasks[0].trigger_event], op.task_num)

    def test_dynamic_endpoint_and_shared_tail_bounds(self):
        """Local endpoints use actual rows while communication retains a larger bucket."""
        for tokens in (0, 1, 127, 128, 129, 257):
            with self.subTest(tokens=tokens):
                spec = MegaMoeSpec(
                    tokens, 5120, 128, 4, 2, None, 1024, 2, None, 0, 20,
                    schedule_tokens=512, auto_tokens=True,
                    forward_fusion_mode="both", backward_fusion_mode="both",
                    shared_intermediate_size=256, forward_shared_expert=True, backward_shared_expert=True,
                )
                fg, fc, bg, bc = _build_runtime_artifacts(spec)
                for graph, cfg, backward in ((fg, fc, False), (bg, bc, True)):
                    self._check_dynamic_graph_rows(spec, graph, cfg, backward)

    def test_shared_scratch_ownership_and_capacity(self):
        """Serial shared scratch stays disjoint and returned dX owns compact storage."""
        x, w1, w2 = torch.empty(4, 8), torch.empty(6, 8), torch.empty(8, 3)
        saved = (x, w1, w2, torch.empty(4, 6), torch.empty(4, 3))
        for capacity in (1, 128):
            with self.subTest(capacity=capacity), patch(
                    "hyper_multicore.modules.mega_moe.shared.pointer_table",
                    return_value=torch.empty(0)):
                act, up = torch.empty(capacity), torch.empty(capacity)
                values, _, reuse = shared_backward_buffers(x, saved, scratch=(act, up), integrated_output=True)
                self.assertEqual(reuse, capacity == 128)
                self.assertEqual(values[11].untyped_storage().nbytes(), x.numel() * x.element_size())
                self.assertNotEqual(values[11].data_ptr(), values[8].data_ptr())
                if reuse:
                    self.assertEqual(values[4].data_ptr(), act.data_ptr())
                    self.assertEqual(values[6].data_ptr(), up.data_ptr())
                    self.assertEqual(values[8].storage_offset(), saved[4].numel())
                    self.assertEqual(values[2].storage_offset(), saved[3].numel())
                    self.assertEqual(values[10].storage_offset(), saved[3].numel() + w2.numel())
                else:
                    self.assertNotEqual(values[4].data_ptr(), act.data_ptr())
                    self.assertNotEqual(values[6].data_ptr(), up.data_ptr())

    def _check_shared_endpoint(self, graph, cfg, names, forward):
        """The route endpoint writes the shared result and waits for all readers."""
        endpoint_name = "weighted_unpermute" if forward else "unpermute_grad"
        if not graph.has_op(endpoint_name):
            return
        endpoint = graph.get_op(endpoint_name)
        self.assertEqual(names[-1], endpoint_name if forward else "shared_dx_add")
        slot = 5 if forward else 11
        self.assertEqual(endpoint.fill_config.shared_output_slot, slot)
        task_type = TaskType.TASK_WEIGHTED_UNPERMUTE if forward else TaskType.TASK_UNPERMUTE_REDUCE_GRAD
        endpoint_tasks = [task for task in cfg.all_tasks[:cfg.task_num] if task.task_type == task_type]
        self.assertTrue(endpoint_tasks)
        for task in endpoint_tasks:
            self.assertEqual(task.outputs[0].tensor_type, 2)
            self.assertEqual(task.outputs[0].dynamic_dim, slot)
            self.assertEqual(task.extra_value_1 & 2, 2 if forward else 0)
        if forward:
            self.assertTrue(all(op.name.startswith("shared_") for op in endpoint.predecessors))
        else:
            self.assertIn(endpoint, graph.get_op("shared_act_grad").predecessors)

    def _check_shared_diagnostic_ports(self, graph, cfg, forward):
        """Diagnostic sidecars bind shared tensors without mutating ordinary graphs."""
        diagnostic_graph = bind_shared_diagnostic_ports(graph, cfg, forward=forward)
        shared_ops = [op for op in diagnostic_graph.topological_sort() if op.name.startswith("shared_")]
        self.assertTrue(all(op.diagnostic_host_ports for op in shared_ops))
        self.assertTrue(all("shared_group_list" not in {tensor.name for tensor in op.inputs} for op in shared_ops))
        sidecar_base = 29 if forward else 35
        self.assertTrue(all(position >= sidecar_base for op in shared_ops for position in op.param_positions))
        self.assertFalse(any(op.diagnostic_host_ports for op in graph.topological_sort()
                             if op.name.startswith("shared_")))
        parameters = 35 if forward else (47 if graph.has_op("unpermute_grad") else 46)
        validate_mega_kernel_diagnostic_contract(diagnostic_graph, cfg, parameters)

    def _check_shared_completion(self, graph, cfg, forward):
        """Every shared stage completes before the graph's final termination."""
        ordered = graph.topological_sort()
        names = [op.name for op in ordered]
        first_shared = next(i for i, name in enumerate(names) if name.startswith("shared_"))
        endpoint_name = "weighted_unpermute" if forward else "unpermute_grad"
        self.assertTrue(all(name.startswith("shared_") or name == endpoint_name for name in names[first_shared:]))
        self._check_shared_endpoint(graph, cfg, names, forward)
        task_count = sum(op.task_num for op in ordered)
        terminal = cfg.all_tasks[task_count]
        self.assertEqual(terminal.task_type, TaskType.TASK_TERMINATE)
        producers = sum(cfg.all_tasks[i].trigger_event == terminal.dependent_event for i in range(task_count))
        self.assertEqual(cfg.all_event_num_triggers[terminal.dependent_event], producers)
        self.assertGreater(producers, 0)
        self._check_shared_diagnostic_ports(graph, cfg, forward)

    def test_independent_direction_and_route_modes(self):
        """All mode combinations retain route semantics and complete every shared output."""
        modes = ("legacy", "permute", "unpermute", "both")
        for forward, backward, shared_forward, shared_backward in itertools.product(
                modes, modes, (False, True), (False, True)):
            with self.subTest(forward=forward, backward=backward,
                              shared_forward=shared_forward, shared_backward=shared_backward):
                spec = MegaMoeSpec(
                    128, 5120, 128, 4, 2, None, 512, 2, None, 0, 20,
                    forward_fusion_mode=forward, backward_fusion_mode=backward,
                    shared_intermediate_size=256, forward_shared_expert=shared_forward,
                    backward_shared_expert=shared_backward,
                )
                fg, fc, bg, bc = _build_runtime_artifacts(spec)
                self.assertTrue(can_reuse_backward_dispatch(bc, 2, 20))
                for graph, cfg, enabled, terminal_stage in (
                        (fg, fc, shared_forward, "shared_down_proj"),
                        (bg, bc, shared_backward, "shared_w1_grad")):
                    self.assertEqual(graph.has_op(terminal_stage), enabled)
                    if enabled:
                        self._check_shared_completion(graph, cfg, graph is fg)

    def test_reuse_rejects_changed_routed_swiglu_ports(self):
        """Shared tasks must not hide an unknown direct reader or broken route chain."""
        spec = MegaMoeSpec(
            128, 5120, 128, 4, 2, None, 512, 2, None, 0, 20,
            forward_fusion_mode="both", backward_fusion_mode="both",
            shared_intermediate_size=256, forward_shared_expert=True, backward_shared_expert=True,
        )
        _, _, _, cfg = _build_runtime_artifacts(spec)
        self.assertTrue(can_reuse_backward_dispatch(cfg, 2, 20))
        task = next(task for task in cfg.all_tasks[:cfg.task_num]
                    if task.task_type == TaskType.TASK_SWI_GLU_GRAD and task.inputs[0].tensor_type != 2)
        task.inputs[0].input_position = 0
        self.assertFalse(can_reuse_backward_dispatch(cfg, 2, 20))
        task.inputs[0].input_position = 8
        shared = next(task for task in cfg.all_tasks[:cfg.task_num]
                      if task.task_type == TaskType.TASK_SWI_GLU_GRAD and task.inputs[0].tensor_type == 2)
        shared.inputs[0].input_position = 35
        self.assertFalse(can_reuse_backward_dispatch(cfg, 2, 20))
        shared.inputs[0].input_position = 36
        cfg.all_event_num_triggers[task.trigger_event] += 1
        self.assertFalse(can_reuse_backward_dispatch(cfg, 2, 20))

    def test_representative_full_fusion_storage_and_fences(self):
        """The performance shape joins all routed work before residual reduction."""
        spec = MegaMoeSpec(
            4096, 5120, 1792, 32, 8, None, 49152, 4, None, 0, 20,
            forward_fusion_mode="both", backward_fusion_mode="both",
            shared_intermediate_size=1792, forward_shared_expert=True, backward_shared_expert=True,
        )
        fg, fc, bg, bc = _build_runtime_artifacts(spec)
        self.assertTrue(can_reuse_backward_dispatch(bc, 8, 20))
        for graph, cfg, endpoint, slot in ((fg, fc, "weighted_unpermute", 5),
                                           (bg, bc, "unpermute_grad", 11)):
            self.assertEqual(graph.topological_sort()[-1].name, endpoint if graph is fg else "shared_dx_add")
            self.assertEqual(graph.get_op(endpoint).fill_config.shared_output_slot, slot)
            shared_first = graph.get_op("shared_up_proj" if graph is fg else "shared_act_grad")
            event = shared_first.fill_config.dependency
            if graph is fg:
                self.assertEqual(event, EVENT_INVALID_ID)
                route_event = graph.get_op(endpoint).fill_config.dependent_event
                producers = [task for task in cfg.all_tasks[:cfg.task_num]
                             if task.trigger_event == route_event]
                self.assertEqual(cfg.all_event_num_triggers[route_event], len(producers))
                self.assertTrue(any(task.task_type == TaskType.TASK_GROUPED_MATMUL
                                    and task.dependent_event != EVENT_INVALID_ID
                                    for task in producers))
            else:
                producers = [task for task in cfg.all_tasks[:cfg.task_num] if task.trigger_event == event]
                self.assertEqual(cfg.all_event_num_triggers[event], len(producers))
                self.assertTrue(producers)
        residual = bg.get_op("shared_dx_add")
        gate = bg.get_op("shared_gate_grad")
        self.assertEqual(residual.predecessors, [gate])
        self.assertEqual(residual.fill_config.dependency, gate.fill_config.trigger)
        self.assertNotEqual(gate.fill_config.trigger, bg.get_op("shared_w1_grad").fill_config.trigger)

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
"""CPU storage-lifetime tests for the MegaMoe autograd bridge."""

import csv
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_multicore.diagnostics import MegaKernelDiagnostics, dumper, profiler
from hyper_multicore.diagnostics.dumper import _task as task_dump
from hyper_multicore.diagnostics.profiler import cycle_trace
from hyper_multicore.modules.base_planner import ExpertPlacement
from hyper_multicore.modules.hot_replica.shmem_transfer import ExpertWeightBinding
from hyper_multicore.modules.mega_moe.backends.bf16 import autograd as function_module
from hyper_multicore.modules.mega_moe.backends.bf16.autograd import _MegaMoeFunction
from hyper_multicore.modules.mega_moe.backends.bf16.plan import _build_runtime_artifacts, _prepare_diagnostics
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


def _capture_task_ports(buffer: torch.Tensor, parameters: Any) -> None:
    """Emulate native copies using the source, length and destination in each record."""
    raw = buffer.numpy()
    magic, _, _, count = task_dump.TASK_DUMP_HEADER.unpack_from(raw)[:4]
    if magic != task_dump.TASK_DUMP_MAGIC:
        return
    for i in range(count):
        offset = task_dump.TASK_DUMP_RECORDS_OFFSET + i * task_dump.TASK_DUMP_RECORD.size
        record = task_dump.TASK_DUMP_RECORD.unpack_from(raw, offset)
        position, source, _, dest, size = record[2:7]
        buffer[dest:dest + size].copy_(parameters[position].view(torch.uint8).flatten()[source:source + size])
        struct.pack_into("<I", raw, offset + 7 * 4, task_dump.TASK_DUMP_STATUS_CAPTURED)


# The private autograd bridge and saved execution buffers are the contracts under test.
class TestMegaMoeFunction(unittest.TestCase):
    """Exercise real autograd contexts with mocked communication and kernels."""

    # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dump_labels_activation_rows_with_global_experts(self) -> None:
        """Follow real BF16 launch metadata into the same CSV and Tensor read by a developer."""
        spec = MegaMoeSpec(256, 1024, 512, 8, 2, 1.25, 640, 2, None, 1, 20)
        forward, _ = _prepare_diagnostics(spec, *_build_runtime_artifacts(spec), "cpu")
        plan = SimpleNamespace(spec=spec, fwd_diagnostics=forward, up_proj_tiling=None,
                               swiglu_tiling=None, down_proj_tiling=None)
        placement = ExpertPlacement(
            virtual_ids=torch.tensor([4, 6, 6]),
            counts_by_source=torch.tensor([[0, 0, 0, 0, 1, 0, 2, 0], [0, 0, 0, 0, 1, 0, 1, 0]]),
            copies=torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32),
            borrowed_capacity=2, receive_capacity=5,
        )
        binding = ExpertWeightBinding(SimpleNamespace(local_experts=2), placement, "EIO")
        group_list = placement.counts_by_source[:, 4:].sum(0).cumsum(0)
        offsets = torch.zeros(6, dtype=torch.int64)
        route = function_module.RouteMetadata(*(offsets,) * 6, group_list=group_list, expert_capacity=6)
        rows = torch.tensor([1., 3., 10., 20., 30., 1000.], dtype=torch.bfloat16)
        activation = rows[:, None].repeat(1, spec.intermediate_size)
        intermediates = function_module._ForwardIntermediates(activation, activation, activation)
        execution = function_module._ForwardExecution(
            activation, activation, activation, None, torch.empty(0), intermediates,
        )

        def launch(expert: int) -> None:
            """Replay the same plan with new placement and distinguishable output rows."""
            placement.copies[1, 0] = expert
            binding.placement = replace(placement, host_copies=placement.copies.numpy().copy())
            activation.copy_((rows + expert)[:, None])
            function_module._launch_forward_kernel(
                plan, route, activation, activation, activation, execution, weight_binding=binding,
            )

        with tempfile.TemporaryDirectory() as directory, patch.object(function_module, "record_plan_stream"):
            root = Path(directory)
            config = dumper.DumpConfig(root / "kernel", stages="SwiGLU", io="output", outputs=("tensor", "csv"))
            with dumper.dump(config) as session, patch.object(function_module.multicore_ops,
                                                             "mega_moe_with_diagnostic_buffer"):
                launch(0)
                launch(1)
            manifest, = session.last_manifest_paths
            with manifest.with_name("summary.csv").open(encoding="utf-8", newline="") as stream:
                summaries = list(csv.DictReader(stream))
            self.assertEqual(len(summaries), 2)
            for expert, row in enumerate(summaries):
                self.assertEqual(json.loads(row["expert_rows"]), [[2, 0, 2], [3, 2, 2], [expert, 2, 5]])
                self.assertEqual(int(row["elements"]), 5 * spec.intermediate_size)
                self.assertAlmostEqual(float(row["mean"]), 12.8 + expert)
                port, = map(tuple, json.loads(row["ports"]))
                payload = dumper.load_tensor(manifest, invocation_id=int(row["invocation"]), port=port)
                expected_rows = (rows[2:5] + expert)[:, None].expand(3, spec.intermediate_size)
                torch.testing.assert_close(payload[2:5], expected_rows)
            self._check_placed_task(root, launch, rows[2:5, None].expand(3, spec.intermediate_size))
            self._check_placed_trace(root, launch, forward, placement.copies)

    def _check_placed_task(self, root, launch, borrowed_rows):
        """Emulate device byte copies and verify the exact selected activation in the manifest."""
        for selected in (0, 1):
            config = dumper.DumpConfig(root / f"task{selected}", level="task", stages="SwiGLU", io="output",
                                       owner_ids=selected)
            with dumper.dump(config) as session, patch.object(
                function_module.multicore_ops, "mega_moe_with_diagnostic_buffer",
                side_effect=lambda *args: _capture_task_ports(args[-6], args),
            ):
                launch(0)
                launch(1)
            manifest, = session.last_manifest_paths
            invocation, = json.loads(manifest.read_text())["invocations"]
            tasks = invocation["tasks"]
            self.assertEqual({task["owner"]["id"] for task in tasks}, {selected})
            active, = (task for task in tasks if task["tensors"][0]["storedBytes"])
            payload = dumper.load_tensor(manifest, invocation_id=invocation["invocationId"],
                                         port=("SwiGLU", "output", 0), task_id=active["taskId"])
            torch.testing.assert_close(payload, borrowed_rows + selected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gmm_weight_summary_uses_assigned_slots(self) -> None:
        """An empty borrowed slot retains stale bytes; an assigned zero-token expert still has a weight."""
        owners = (0, 1, None, None, 2, 3, None, 0)
        for batched in (False, True):
            with self.subTest(batched=batched), tempfile.TemporaryDirectory() as directory:
                spec = MegaMoeSpec(256, 1024, 512, 8, 2, 1.25, 640, 2, None, 1, 20, batch_experts=batched)
                forward, _ = _prepare_diagnostics(spec, *_build_runtime_artifacts(spec), "cpu")
                runtime = forward._runtime
                weight = runtime.graph.get_op("up_proj").inputs[1]
                bank = torch.empty(weight.shape, dtype=torch.bfloat16)
                for slot, value in enumerate((1, 2, 99, 4)):
                    bank[slot].fill_(value)
                operands = [None] * runtime.parameter_count
                operands[weight.param_position] = bank
                group_position = runtime.diagnostic_runtime_config.dynamic_data.dynamic_input_position
                operands[group_position] = torch.tensor([2, 2, 2, 5])
                tasks = {}
                for task, stage in forward.metadata.task_stage_names.items():
                    if stage == "GMM1":
                        tasks.setdefault(forward.metadata.task_owner_ids.get(task), task)
                stride = bank[0].numel()
                config = dumper.DumpConfig(directory, level="task", task_ids=tuple(tasks.values()), io="input",
                                           tensor_names=weight.name, outputs=("tensor", "csv"),
                                           max_bytes_per_tensor=(3 * stride + 1) * bank.element_size())
                with dumper.dump(config) as session, forward.launch(
                    operands, direction="forward", event_counters=torch.zeros(16), resolve_owner_ids=lambda: owners,
                ) as call:
                    _capture_task_ports(call.diagnostic_buffer, operands)
                manifest, = session.last_manifest_paths
                with manifest.with_name("summary.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if not batched:
                    self.assertEqual({(int(r["owner"]), float(r["mean"])) for r in rows}, {(2, 1), (3, 2), (0, 4)})
                    continue
                row, = rows
                ranges = [[0, 2 * stride], [3 * stride, 3 * stride + 1]]
                self.assertEqual(json.loads(row["valid_ranges"]), ranges)
                self.assertEqual((row["range"], int(row["elements"]), float(row["max"])),
                                 ("valid_elements", 2 * stride + 1, 4))
                self.assertAlmostEqual(float(row["mean"]), (3 * stride + 4) / (2 * stride + 1))
                task, = json.loads(manifest.read_text())["invocations"][0]["tasks"]
                self.assertEqual(task["tensors"][0]["validRanges"], ranges)
                payload = dumper.load_tensor(manifest, invocation_id=0, tensor_name=weight.name)
                self.assertEqual(payload[2 * stride].item(), 99)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_batched_gmm_activation_requires_ready_rows(self) -> None:
        """Capture a complete activation only when every selected row is ready at task entry."""
        for experts, stage, name in ((8, "GMM2", "swiglu_out"), (2, "GMM2", "swiglu_out"),
                                    (8, "W1Grad", "swiglu_out"), (8, "W2Grad", "w2_grad_x1")):
            with self.subTest(experts=experts, stage=stage), tempfile.TemporaryDirectory() as directory:
                spec = MegaMoeSpec(256, 1024, 512, experts, 2, 1.25, 640, 2, None, 1, 20, batch_experts=True)
                forward, backward = _prepare_diagnostics(spec, *_build_runtime_artifacts(spec), "cpu")
                direction = "forward" if stage == "GMM2" else "backward"
                handle = forward if direction == "forward" else backward
                runtime = handle._runtime
                task = next(i for i, label in handle.metadata.task_stage_names.items() if label == stage)
                op = next(op for op in runtime.graph.topological_sort() if op.diagnostic_stage_name == stage)
                activation = next(t for t in op.inputs if t.name == name)
                rows = torch.arange(2 * activation.shape[1]).reshape(2, -1).to(torch.bfloat16)
                operands = [None] * runtime.parameter_count
                operands[activation.param_position] = rows
                pos = runtime.diagnostic_runtime_config.dynamic_data.dynamic_input_position
                operands[pos] = torch.full((spec.local_experts,), 2)
                config = dumper.DumpConfig(directory, level="task", task_ids=task, io="input", tensor_names=name,
                                           directions=direction)
                if stage == "GMM2" and spec.local_experts > 1:
                    with self.assertLogs("hyper_multicore.diagnostics.dumper.api", "WARNING") as logs:
                        with dumper.dump(config) as session, handle.launch(
                            operands, direction=direction, event_counters=torch.zeros(16),
                        ) as call:
                            self.assertIs(call.runtime_config_tensor, handle.runtime_config_tensor)
                    self.assertIn("unsupported_port", logs.output[0])
                    self.assertIn("SwiGLU.output[0]", logs.output[0])
                    self.assertFalse(session.last_manifest_paths)
                    continue
                with dumper.dump(config) as session, handle.launch(
                    operands, direction=direction, event_counters=torch.zeros(16),
                ) as call:
                    _capture_task_ports(call.diagnostic_buffer, operands)
                manifest, = session.last_manifest_paths
                payload = dumper.load_tensor(manifest, invocation_id=0, task_id=task, tensor_name=name)
                torch.testing.assert_close(payload, rows)

    def _check_placed_trace(self, root, launch, forward, copies):
        """Decode the same virtual owner twice after the second launch changed the placement."""
        def copy_to_host(tensor: torch.Tensor) -> torch.Tensor:
            """Emulate independent D2H storage and reject another placement readback."""
            self.assertNotEqual(tensor.data_ptr(), copies.data_ptr(), "profiler reread the placement")
            return tensor.clone()

        metadata = forward.metadata
        task_id = next(i for i, owner in metadata.task_owner_ids.items()
                       if owner == 6 and metadata.task_stage_names[i] == "SwiGLU")
        task_index = forward._runtime.diagnostic_runtime_config.all_tasks[task_id].task_index
        record = {
            "desc_id": cycle_trace.GRAPH_STAGE_DESC_BASE + metadata.task_stage_ids[task_id],
            "task_id": task_id, "owner_id": 6, "stage_task_index": task_index, "core_type": 2,
            "block_id": 1, "entry_cycle": 1, "start_cycle": 2, "end_cycle": 3,
        }
        with (
            patch.object(function_module.multicore_ops, "mega_moe_with_diagnostic_buffer"),
            patch.object(torch, "npu", create=True),
            patch.object(torch.Tensor, "cpu", copy_to_host),
            patch.object(cycle_trace, "_get_soc_name", return_value="ascend910b"),
            patch.object(cycle_trace, "_decode_cycle_records", side_effect=lambda *_: ([record.copy()], [1], 0)),
            profiler.profile(detailed_task_names=True) as session,
        ):
            launch(0)
            launch(1)
        trace = session.export_chrome_trace(root / "trace.json")
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        self.assertEqual([(e["args"]["invocation_id"], e["args"]["owner_id"]) for e in events], [(0, 0), (1, 1)])
        for expert, event in enumerate(events):
            self.assertEqual(event["args"]["task_stage"], "SwiGLU")
            self.assertEqual(event["name"], f"Expert{expert}_SwiGLU_task{task_index + 1}")

    def test_plan_stream_identity_preserves_registration(self) -> None:
        """Register plan storage without requesting a queue-draining stream handle."""
        device = Mock(type="npu")
        forward, backward, tiling, diagnostic = [Mock(spec=torch.Tensor) for _ in range(4)]
        for tensor in (forward, backward, tiling, diagnostic):
            tensor.device = device
        plan = function_module.MegaMoePlan(
            spec=None, forward_graph=None,
            fwd_diagnostics=SimpleNamespace(runtime_config_tensor=forward),
            up_proj_tiling=tiling, swiglu_tiling=None, down_proj_tiling=None,
            backward_graph=None, bwd_diagnostics=SimpleNamespace(runtime_config_tensor=backward),
            act_grad_tiling=None, gate_grad_tiling=None, w1_grad_tiling=None,
            w2_grad_tiling=None, swiglu_grad_tiling=None,
        )
        first = SimpleNamespace(stream_id=10)
        same = SimpleNamespace(stream_id=10)
        other = SimpleNamespace(stream_id=20)
        plan.record_stream(forward, stream=first)
        plan.record_stream(backward, stream=same)
        plan.record_stream(diagnostic, stream=same)
        plan.record_stream(forward, stream=other)
        for tensor in (forward, backward, tiling):
            self.assertEqual([entry.args[0] for entry in tensor.record_stream.call_args_list], [first, other])
        diagnostic.record_stream.assert_called_once_with(same)


    def test_ragged_pull_uses_reserved_source_prefix(self) -> None:
        """Forward and backward retain SHMEM addresses while exposing only actual rows."""
        buffer = torch.full((256, 4), float("nan"))
        workspace = SimpleNamespace(source_buffer=buffer)
        spec = SimpleNamespace(apply_probs_before_fc2=False, dispatch_mode="pull", hidden_size=4)
        rows = torch.arange(12).reshape(3, 4).float()
        dispatch, source = function_module._dispatch_and_source(spec, workspace, rows, 7)
        self.assertEqual(tuple(dispatch.shape), (7, 4))
        self.assertEqual(source.data_ptr(), buffer.data_ptr())
        torch.testing.assert_close(source, rows)
        self.assertTrue(torch.isnan(buffer[3:]).all())
        context = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace,
                                  source_rows=3, has_unpermute=False)
        staged, probabilities = function_module._stage_backward_source(context, rows + 1, ())
        torch.testing.assert_close(staged, rows + 1)
        self.assertIsNone(probabilities)
        context.source_rows = 0
        context.has_unpermute = True
        context.needs_input_grad = (False,) * 7 + (True,)
        context.topk_dtype = torch.float32
        with patch.object(function_module.multicore_ops, "mega_moe_unpermute_grad_out") as unpermute:
            empty, gradient = function_module._stage_backward_source(
                context, rows[:0], (torch.empty(0, dtype=torch.int32), rows[:0], torch.empty(0, 2)))
        unpermute.assert_not_called()
        self.assertEqual(tuple(empty.shape), (0, 4))
        self.assertEqual(tuple(gradient.shape), (0, 2))
        self.assertEqual(function_module._native_source(empty).data_ptr(), buffer.data_ptr())

    def test_empty_native_source_preserves_symmetric_storage(self) -> None:
        """Empty pull ranks must retain the symmetric base used to address remote ranks."""
        storage = torch.arange(32).reshape(8, 4)
        empty_source = storage[:0]
        native_source = function_module._native_source(empty_source)
        self.assertEqual(native_source.data_ptr(), storage.data_ptr())
        torch.testing.assert_close(native_source, storage[:1])
        self.assertEqual(tuple(empty_source.shape), (0, 4))
        ordinary_empty = torch.empty(0, 4)
        placeholder = function_module._native_source(ordinary_empty)
        self.assertEqual(tuple(placeholder.shape), (1, 4))
        self.assertNotEqual(placeholder.data_ptr(), 0)

    def test_eoi_weight_operands_and_gradients_keep_contiguous_layout(self) -> None:
        """Native EOI operands and gradients already match the parameter storage layout."""
        spec = SimpleNamespace(
            expert_weight_layout="EOI",
            local_experts=2,
            hidden_size=4,
            intermediate_size=3,
        )
        plan = SimpleNamespace(spec=spec)
        gate_up = torch.empty(2, 6, 4)
        down = torch.empty(2, 4, 3)
        self.assertIs(function_module._native_weight_operand(gate_up, plan), gate_up)
        self.assertIs(function_module._native_weight_operand(down, plan), down)

        gradients = function_module._allocate_backward_intermediates(
            spec,
            8,
            torch.empty(8, 4),
            gate_up,
            down,
        )
        for weight, gradient in ((gate_up, gradients.grad_weight1), (down, gradients.grad_weight2)):
            self.assertEqual(gradient.shape, weight.shape,
                             msg=f"Gradient shape: expected={weight.shape}, actual={gradient.shape}")
            self.assertEqual(gradient.stride(), weight.stride(),
                             msg=f"Gradient stride: expected={weight.stride()}, actual={gradient.stride()}")
            self.assertEqual(torch.count_nonzero(gradient).item(), 0,
                             msg=f"Empty-expert gradients: expected=0, actual={gradient}")


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fp32_weight_gradients_preserve_activation_dtype(self) -> None:
        """Keep partial dW unquantized without promoting activation scratch."""
        spec = SimpleNamespace(hidden_size=4, intermediate_size=3)
        for shapes in (((2, 4, 6), (2, 3, 4)), ((2, 6, 4), (2, 4, 3))):
            with self.subTest(shapes=shapes):
                weights = tuple(torch.empty(shape, dtype=torch.bfloat16) for shape in shapes)
                gradients = function_module._allocate_backward_intermediates(
                    spec, 8, torch.empty(8, 4, dtype=torch.bfloat16), *weights,
                    weight_grad_dtype=torch.float32)
                for weight, gradient in zip(weights, (gradients.grad_weight1, gradients.grad_weight2)):
                    self.assertEqual(gradient.shape, weight.shape)
                    self.assertEqual(gradient.dtype, torch.float32)
                    self.assertEqual(torch.count_nonzero(gradient).item(), 0)
                for value in (gradients.act_grad, gradients.swiglu_grad, gradients.gate_dx):
                    self.assertEqual(value.dtype, torch.bfloat16)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_unpacks_saved_tensors_once(self) -> None:
        """Feature: MegaKernel diagnostics.
        Description: Checkpoint unpack hooks must only be consumed once per backward call.
        Expectation: Backward unpacks saved tensors once.
        """
        tensors = tuple(torch.ones(1) for _ in range(13))
        context = Mock()
        context.weight_binding = None
        context.needs_input_grad = (False,) * 9
        gradient = torch.ones(1)
        context.plan.spec.dispatch_mode = "push"
        context.plan.spec.apply_probs_before_fc2 = False
        execution = function_module._BackwardExecution(
            dispatch=torch.ones(1), grad_x=torch.ones(1), event_counters=Mock(),
            gmm_workspace=None, swiglu_workspace=None,
            intermediates=function_module._BackwardIntermediates(*(torch.ones(1) for _ in range(5))),
        )
        with (
            patch.object(type(context), "saved_tensors", create=True,
                         new_callable=PropertyMock, side_effect=[tensors, RuntimeError("duplicate unpack")]) as unpack,
            patch.object(function_module, "_prepare_backward_execution", return_value=execution),
            patch.object(function_module, "_launch_backward_kernel"),
            patch.object(function_module, "_restore_input_gradient", return_value=gradient) as restore,
            patch.object(function_module, "_release_completed_graph"),
        ):
            result = _MegaMoeFunction.backward(context, gradient)
        unpack.assert_called_once_with()
        restore.assert_called_once_with(context, execution.grad_x, tensors[12:])
        self.assertIs(result[0], gradient)
        self.assertIs(result[1], execution.intermediates.grad_weight1)
        self.assertIs(result[2], execution.intermediates.grad_weight2)
        context.workspace.release.assert_called_once_with(stream=context.workspace.lease_stream)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_deferred_backward_keeps_local_owned_storage_after_workspace_reuse(self) -> None:
        """Feature: Deferred backward storage.
        Description: Reuse the workspace between deferred forward and backward.
        Expectation: Backward retains its own local tensors.
        """
        self._check_deferred_backward(permuted=False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_permutation_gradient_consumes_workspace_before_release(self) -> None:
        """Feature: Deferred backward storage.
        Description: Run the permutation gradient while tracking workspace release.
        Expectation: Consume the receive buffer before releasing its lease.
        """
        self._check_deferred_backward(permuted=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_frozen_tokens_do_not_launch_a_permutation_gradient(self) -> None:
        """Feature: Deferred backward storage.
        Description: Backpropagate with frozen token activations.
        Expectation: Skip the token permutation gradient.
        """
        self._check_deferred_backward(permuted=True, input_grad=False)

    def test_dynamic_tokens_restore_each_deferred_forward_length(self) -> None:
        """Use per-call token counts despite later forwards reusing a larger source buffer."""
        self._check_deferred_backward(permuted=True, vary_tokens=True)

    def test_non_reentrant_checkpoint_unpacks_saved_tensors_once(self) -> None:
        """Checkpoint recomputation must not fail when backward reads its permutation state."""
        # CPU-only inputs must not select an installed accelerator for autocast replay.
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self._check_deferred_backward(permuted=False, replay=True)

    def _check_deferred_forward(
        self, source: Any, weight1: Any, weight2: Any, route: Any, plan: Any, workspace: Any,
        down_pointers: list[int],
        *, permuted: bool, vary_tokens: bool, replay: bool,
    ) -> Any:
        """Run one forward and check that its autograd storage survives workspace reuse."""
        tokens = source.shape[0]
        if permuted:
            ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int32)
            prepared = SimpleNamespace(
                routed_tokens=source.detach().repeat_interleave(2, dim=0),
                metadata=route,
                unpermute_mapping=(torch.arange(tokens * 2, dtype=torch.int32) if vary_tokens
                                   else torch.tensor([0, 2, 1, 3], dtype=torch.int32)),
            )
            output = function_module.execute_mega_moe_with_permutation(
                source, ids, weight1, weight2, prepared, plan, workspace
            )
        elif replay:
            operation = partial(function_module.execute_mega_moe, route=route, plan=plan, workspace=workspace)
            output = checkpoint(operation, source, weight1, weight2, use_reentrant=False)
        else:
            output = function_module.execute_mega_moe(source, weight1, weight2, route, plan, workspace)
        if not replay:
            saved = output.grad_fn.saved_tensors
            self.assertEqual(len(saved), 13 if permuted and source.requires_grad else 12)
            self.assertEqual(saved[0].data_ptr(), down_pointers[-1])
            self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
            self.assertEqual(saved[0].untyped_storage().nbytes(), route.expert_capacity * 4 * source.element_size())
        self.assertEqual(tuple(output.shape), (tokens * 2 if permuted else 4, 4))
        return output

    def _check_deferred_backward(self, *, permuted: bool, input_grad: bool = True,
                                vary_tokens: bool = False, replay: bool = False) -> None:
        """Exercise delayed backward with an optional input permutation boundary."""
        spec = SimpleNamespace(apply_probs_before_fc2=False, hidden_size=4, intermediate_size=2, rank_id=0,
                               ep_size=2, num_experts=4, local_num_tokens=2, top_k=2,
                               dispatch_mode="push", expert_weight_layout="EIO", auto_tokens=False)
        plan = SimpleNamespace(
            spec=spec, reuse_backward_dispatch=False, weight_grad_dtype=None, event_counter_bytes=None,
        )
        for name in ("up_proj", "swiglu", "down_proj", "act_grad", "gate_grad",
                     "w1_grad", "w2_grad", "swiglu_grad"):
            setattr(plan, f"{name}_tiling", None)
        runtime = SimpleNamespace(runtime_config_tensor=None, kernel_name="mega_moe", rank=0)
        plan.fwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.bwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.fwd_runtime_config_tensor = plan.bwd_runtime_config_tensor = None
        workspace = Mock(
            expert_capacity=128,
            expert_buffer=torch.empty(128, 4),
            routed_buffer=torch.empty(8 if vary_tokens else 4, 4),
            forward_event_counters=torch.empty(16),
            backward_event_counters=torch.empty(16),
            gmm_workspace=torch.empty(1),
            swiglu_grad_workspace=torch.empty(1),
        )
        weight1 = torch.ones(2, 4, 4, requires_grad=True)
        weight2 = torch.ones(2, 2, 4, requires_grad=True)
        down_pointers = []
        backward_capacities = []

        def release_workspace(*, stream: Any = None) -> None:
            """Make accidental reads after the lease immediately observable."""
            self.assertIs(stream, workspace.lease_stream)
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))

        workspace.release.side_effect = release_workspace

        def permutation_gradient(grad: Any, mapping: Any, token_rows: int, top_k: int) -> Any:
            """Require direct consumption of shared rows and return owned token rows."""
            self.assertEqual(grad.data_ptr(), workspace.routed_buffer.data_ptr())
            expected_rows = (1, 3, 2, 4)[int(grad[0, 0]) - 1] if vary_tokens else 2
            self.assertEqual((token_rows, top_k), (expected_rows, 2))
            self.assertEqual(grad.shape[0], token_rows * top_k)
            self.assertEqual(mapping.tolist(), list(range(token_rows * 2)) if vary_tokens else [0, 2, 1, 3])
            self.assertTrue(torch.isfinite(grad).all())
            return grad.reshape(token_rows, top_k, 4).sum(dim=1)

        def forward_kernel(*args: Any) -> None:
            """Fill every output with a call-specific sentinel."""
            dispatch, source = args[0], args[2]
            up_proj, activation, down_proj, combine = args[7], args[8], args[11], args[12]
            tag = source[0, 0].item()
            capacity = up_proj.shape[0]
            self.assertEqual(tuple(down_proj.shape), (capacity, 4))
            self.assertEqual(tuple(activation.shape), (capacity, 2))
            self.assertEqual(tuple(dispatch.shape), (128, 4))
            dispatch.fill_(tag)
            up_proj.fill_(tag + 10)
            activation.fill_(tag + 20)
            down_proj.fill_(tag + 30)
            combine.fill_(tag + 40)
            down_pointers.append(down_proj.data_ptr())

        def backward_kernel(*args: Any) -> None:
            """Check saved values before emulating the in-place gradient outputs."""
            saved_dispatch = args[17]
            capacity = saved_dispatch.shape[0]
            backward_capacities.append(capacity)
            tag = saved_dispatch[0, 0].item()
            self.assertTrue(torch.equal(saved_dispatch, torch.full_like(saved_dispatch, tag)))
            self.assertTrue(torch.equal(args[9], torch.full_like(args[9], tag + 10)))
            self.assertTrue(torch.equal(args[5], torch.full_like(args[5], tag + 20)))
            for tensor, width in ((args[8], 2), (args[10], 4), (args[12], 4)):
                self.assertEqual(tuple(tensor.shape), (capacity, width))
            self.assertEqual(tuple(args[13].shape), (8 if vary_tokens else 4, 4))
            self.assertEqual(torch.count_nonzero(args[6][0]).item(), 0)
            self.assertEqual(torch.count_nonzero(args[18][0]).item(), 0)
            args[13].fill_(tag)
            args[6].fill_(tag)
            args[18].fill_(tag)

        def zero_cold(first: torch.Tensor, second: torch.Tensor, groups: torch.Tensor) -> None:
            """Clear expert gradient slices whose cumulative token counts are unchanged."""
            previous = 0
            for expert, end in enumerate(groups.tolist()):
                if end == previous:
                    first[expert].zero_()
                    second[expert].zero_()
                previous = end

        capacities = (7, 2, 1, 5)
        outputs = []
        sources = []
        with (
            patch.object(function_module.multicore_ops, "zero_empty_expert_gradients", side_effect=zero_cold),
            patch.object(MegaKernelDiagnostics, "launch",
                         side_effect=AssertionError("ordinary path must bypass launch")),
            patch.object(
                function_module.multicore_ops,
                "mega_moe",
                side_effect=forward_kernel,
            ),
            patch.object(
                function_module.multicore_ops,
                "mega_moe_grad",
                side_effect=backward_kernel,
            ),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=permutation_gradient) as mock_permutation,
        ):
            for tag, capacity in enumerate(capacities, start=1):
                tokens = (1, 3, 2, 4)[tag - 1] if vary_tokens else 2
                source = torch.full((tokens if permuted else 4, 4), float(tag), requires_grad=input_grad)
                route = function_module.RouteMetadata(
                    *(torch.zeros(4, dtype=torch.int64) for _ in range(6)),
                    group_list=torch.tensor([0, capacity]), expert_capacity=capacity)
                output = self._check_deferred_forward(
                    source, weight1, weight2, route, plan, workspace, down_pointers,
                    permuted=permuted, vary_tokens=vary_tokens, replay=replay,
                )
                outputs.append(output)
                sources.append(source)
            for tag in range(len(outputs), 0, -1):
                self.assertTrue(torch.equal(outputs[tag - 1], torch.full_like(outputs[tag - 1], float(tag + 40))))
                outputs[tag - 1].sum().backward()
                source = sources[tag - 1]
                expected = torch.full_like(source, float(tag * (2 if permuted else 1)))
                if input_grad:
                    self.assertTrue(torch.equal(source.grad, expected))
                else:
                    self.assertIsNone(source.grad)
        self.assertTrue(torch.equal(weight1.grad, torch.full_like(weight1, 10.0)))
        self.assertTrue(torch.equal(weight2.grad, torch.full_like(weight2, 10.0)))
        self.assertEqual(mock_permutation.call_count, len(outputs) if permuted and input_grad else 0)
        self.assertEqual(backward_capacities, list(reversed(capacities)))
        self.assertEqual(workspace.claim.call_count, 12 if replay else 8)
        self.assertEqual(workspace.release.call_count, 12 if replay else 8)


if __name__ == "__main__":
    unittest.main()

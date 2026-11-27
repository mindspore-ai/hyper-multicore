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
"""Unit tests for task-level Torch MegaKernel Tensor dump protocol."""

import json
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import struct
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from torch import (
    bfloat16 as torch_bfloat16,
    float32 as torch_float32,
    frombuffer as torch_frombuffer,
    ones as torch_ones,
    tensor as torch_tensor,
    uint8 as torch_uint8,
    zeros as torch_zeros,
)
from torch.testing import assert_close

from hyper_multicore import prepare_mega_kernel_diagnostics
from hyper_multicore.diagnostics import schedule
from hyper_multicore.diagnostics.common.call import LaunchContext
from hyper_multicore.diagnostics.dumper import load_tensor
from hyper_multicore.diagnostics.dumper import api as dumper_api
from hyper_multicore.diagnostics.dumper import _task as tensor_dump_task
from hyper_multicore.diagnostics.common.session import _reset_active_session_for_test
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskAiCoreType, TaskType
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.scheduler.worker_queues import install_strided_worker_queues, install_worker_queues
from tests.common.mark_utils import arg_mark


def _captured_task_dump(values, dtype=torch_float32):
    """Build one Device-like captured Task record for CPU-only tests."""
    payload = torch_tensor(values, dtype=dtype)
    payload_bytes = payload.numel() * payload.element_size()
    payload_offset = tensor_dump_task._align_up(
        tensor_dump_task.TASK_DUMP_RECORDS_OFFSET
        + tensor_dump_task.TASK_DUMP_MAX_RECORDS * tensor_dump_task.TASK_DUMP_RECORD.size
    )
    record = tensor_dump_task._TaskDumpRecordPlan(
        task_id=7,
        task_index=2,
        stage="SwiGLU",
        owner_id=3,
        owner_label="Expert",
        core_type="AIV",
        core_id=1,
        io="output",
        index=0,
        phase="after",
        parameter_position=8,
        tensor_name="swiglu_out",
        dtype=dtype,
        source_offset_bytes=32,
        actual_shape=tuple(payload.shape),
        payload_offset=payload_offset,
        payload_bytes=payload_bytes,
        full_bytes=payload_bytes,
        truncated=False,
        note="task-local test tile",
        record_offset=tensor_dump_task.TASK_DUMP_RECORDS_OFFSET,
    )
    total_bytes = tensor_dump_task._align_up(payload_offset + payload_bytes)
    raw_buffer = tensor_dump_task._pack_task_buffer((record,), total_bytes=total_bytes)
    raw_buffer[payload_offset:payload_offset + payload_bytes] = bytes(payload.view(torch_uint8).tolist())
    struct.pack_into(
        "<I",
        raw_buffer,
        record.record_offset + 7 * 4,
        tensor_dump_task.TASK_DUMP_STATUS_CAPTURED,
    )
    prepared = tensor_dump_task._PreparedTaskDump(
        runtime_config_tensor=torch_ones(1, dtype=torch_uint8),
        buffer=torch_frombuffer(raw_buffer, dtype=torch_uint8).clone(),
        records=(record,),
        total_bytes=len(raw_buffer),
    )
    return prepared


class TestTaskDumpWorkerPlacement(unittest.TestCase):
    """Check device worker IDs for each supported scheduler layout."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_round_robin_worker_ids(self):
        """Feature: Task dump worker selection.
        Description: Map AIC, paired AIV, and all AIV schedules to device IDs.
        Expectation: IDs follow the device slot mapping and wrap at the active worker count.
        """
        config = allocate_runtime_config(8)
        config.num_workers = 4
        config.task_index_num[:2] = (3, 5)
        config.cube_task_indices[:3] = (2, 0, 1)
        config.vector_task_indices[:5] = (4, 5, 3, 6, 7)
        for stride, expected in ((2, (1, 3, 1, 3, 1)), (1, (0, 1, 2, 3, 0))):
            with self.subTest(stride=stride):
                locations = {
                    location.task_id: location
                    for location in tensor_dump_task._iter_task_locations(config, vector_worker_stride=stride)
                }
                self.assertEqual(
                    [(locations[task_id].core_type, locations[task_id].core_id) for task_id in (2, 0, 1)],
                    [("AIC", 0), ("AIC", 1), ("AIC", 0)],
                )
                self.assertEqual(
                    [(locations[task_id].core_type, locations[task_id].core_id) for task_id in (4, 5, 3, 6, 7)],
                    [("AIV", worker_id) for worker_id in expected],
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_worker_ids(self):
        """Feature: Task dump worker selection.
        Description: Place tasks using contiguous and strided device queue views.
        Expectation: Each range maps to its owning AIC or AIV, including even AIVs.
        """
        config = allocate_runtime_config(6)
        config.num_workers = 4
        config.task_index_num[:2] = (3, 3)
        config.cube_task_indices[:3] = (0, 1, 2)
        config.vector_task_indices[:3] = (3, 4, 5)
        contiguous = install_worker_queues(config, [[], [2, 0, 1]], [[3, 4], [], [5], []])
        strided = install_strided_worker_queues(config)
        for layout, expected in ((contiguous, (1, 1, 1, 0, 0, 2)), (strided, (0, 1, 0, 0, 1, 2))):
            with self.subTest(worker_ids=expected):
                locations = tensor_dump_task._iter_task_locations(layout)
                self.assertEqual({location.task_id: location.core_id for location in locations},
                                 dict(enumerate(expected)))


class TestTaskDumpProtocol(unittest.TestCase):
    """Validate the fixed Task record ABI and Torch Tensor decoding."""


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_plans_records_from_graph_sized_runtime(self):
        """Feature: MegaKernel diagnostics.
        Description: Retain dynamic RuntimeConfig arrays while planning task-local views.
        Expectation: Plans records from graph sized runtime.
        """
        runtime_config = allocate_runtime_config(1)
        runtime_config.task_num = 1
        runtime_config.num_workers = 2
        runtime_config.task_index_num[1] = 1
        runtime_config.vector_task_indices[0] = 0
        task = runtime_config.all_tasks[0]
        task.task_type = TaskType.TASK_SWI_GLU
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.num_inputs = 1
        task.num_outputs = 1
        task.task_index = 0
        task.task_split_num = 1
        task.task_split_value = 2
        task.profile_owner_id = 999  # Deliberately unrelated to the common owner metadata.
        for parameter_position, tensor_desc in enumerate((task.inputs[0], task.outputs[0])):
            tensor_desc.num_dims = 1
            tensor_desc.dim[0] = 2
            tensor_desc.data_type = 4
            tensor_desc.input_position = parameter_position

        graph_input = TensorSpec("input", [2], param_position=0, dtype_size=4)
        graph_output = TensorSpec("output", [2], param_position=1, dtype_size=4)
        operator = OperatorNode(
            "swiglu",
            OpType.SWIGLU,
            [graph_input],
            [graph_output],
            [0, 1],
            2,
            SplitSpec([(0, 0)], lambda _: 1, [0]),
            -1,
            None,
            diagnostic_stage_name="SwiGLU",
        )
        operator.task_num = 1
        graph = ComputeGraph().add_op(operator)
        diagnostics = prepare_mega_kernel_diagnostics(
            graph, runtime_config, kernel_name="swiglu", parameter_count=2, device="cpu",
            owner_label="Expert", owner_resolver=lambda *_: 4,
        )
        runtime = diagnostics._runtime
        config = dumper_api.DumpConfig("dump", level="task", stages="SwiGLU", core_types="AIV", core_ids=1)

        prepared = tensor_dump_task._prepare_task_dump(
            runtime,
            (torch_ones(2), torch_zeros(2)),
            config,
        )

        self.assertEqual([(record.io, record.phase) for record in prepared.records], [
            ("input", "before"),
            ("output", "after"),
        ])
        self.assertEqual({record.core_type for record in prepared.records}, {"AIV"})
        self.assertEqual({record.core_id for record in prepared.records}, {1})
        self.assertEqual({record.owner_id for record in prepared.records}, {4})
        total_bytes = tensor_dump_task._align_up(max(
            record.payload_offset + record.payload_bytes for record in prepared.records
        ))
        header = tensor_dump_task.TASK_DUMP_HEADER.unpack_from(prepared.buffer.numpy())
        self.assertEqual(prepared.total_bytes, total_bytes)
        self.assertEqual(prepared.buffer.numel(), total_bytes)
        self.assertEqual(header[-1], total_bytes)
        self.assertEqual(
            prepared.runtime_config_tensor[
                RuntimeConfigC.diagnostic_mode.offset
            ].item(),
            tensor_dump_task.DeviceDiagnosticMode.TASK_TENSOR_DUMP,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_launch_config_matches_remapped_record_identity(self):
        """Feature: MegaKernel diagnostics.
        Description: Reindex sparse queues without missing a record or sampling another task.
        Expectation: Uploaded task descriptors match every selected record and preserve queue order.
        """
        config = allocate_runtime_config(4)
        config.task_num = 3
        config.num_workers = 2
        config.task_index_num[1] = 3
        config.vector_task_indices[:3] = (2, 1, 3)
        config.all_tasks[3].task_type = TaskType.TASK_TERMINATE
        source = TensorSpec("source", [2], param_position=0, dtype_size=4)
        output = TensorSpec("output", [2], param_position=1, dtype_size=4)
        operator = OperatorNode("copy", OpType.SWIGLU, [source], [output], [0, 1], 2,
                                SplitSpec(None, lambda _: 1, [-1]), -1, None)
        operator.task_num = 3
        graph = ComputeGraph().add_op(operator)
        for task_id in range(3):
            task = config.all_tasks[task_id]
            task.task_type = TaskType.TASK_SWI_GLU
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.task_index = task_id
            task.num_inputs = task.num_outputs = 1
            for position, descriptor in enumerate((task.inputs[0], task.outputs[0])):
                descriptor.input_position = position
                descriptor.data_type = 4
                descriptor.base_ptr_offset = task_id * 2

        def adapt_graph(original_graph: ComputeGraph, adapted_config: RuntimeConfigC) -> ComputeGraph:
            """Remove an unscheduled task while retaining the original execution order."""
            adapted_graph = deepcopy(original_graph)
            adapted_graph.topological_sort()[0].task_num = 2
            for destination, original in enumerate((1, 2, 3)):
                adapted_config.all_tasks[destination] = config.all_tasks[original]
            adapted_config.vector_task_indices[:3] = (1, 0, 2)
            adapted_config.task_num = 2
            return adapted_graph

        diagnostics = prepare_mega_kernel_diagnostics(
            graph, config, kernel_name="copy", parameter_count=2, device="cpu",
            graph_adapter=adapt_graph, owner_resolver=lambda _op, task, _context: task.task_index,
        )
        runtime = diagnostics._runtime
        original_config = bytes(config)
        standard_config = runtime.runtime_config_tensor.clone()
        parameters = (torch_tensor([10., 11., 20., 21., 30., 31.]), torch_zeros(6))
        for selected_ids in (None, (1,)):
            with self.subTest(task_ids=selected_ids):
                selection = dumper_api.DumpConfig(
                    "dump", level="task", stages="Copy", io="input", task_ids=selected_ids,
                )
                prepared = tensor_dump_task._prepare_task_dump(runtime, parameters, selection)
                uploaded = type(config).from_buffer_copy(bytes(prepared.runtime_config_tensor.tolist()))
                queue = list(uploaded.vector_task_indices[:uploaded.task_index_num[1]])
                self.assertEqual([record.task_id for record in prepared.records],
                                 [0, 1] if selected_ids is None else [1])
                for record in prepared.records:
                    self.assertIn(record.task_id, queue)
                    executed = uploaded.all_tasks[record.task_id]
                    self.assertEqual(executed.task_index, record.task_index)
                    self.assertEqual(record.owner_id, executed.task_index)
                    self.assertEqual(record.source_offset_bytes, executed.inputs[0].base_ptr_offset * 4)
                    self.assertEqual(bytes(executed), bytes(config.all_tasks[record.task_index]))
                self.assertEqual(queue, [1, 0, 2])
                self.assertEqual(uploaded.all_tasks[queue[-1]].task_type, TaskType.TASK_TERMINATE)
                self.assertEqual(uploaded.diagnostic_mode, tensor_dump_task.DeviceDiagnosticMode.TASK_TENSOR_DUMP)
        self.assertEqual(bytes(config), original_config)
        assert_close(runtime.runtime_config_tensor, standard_config)


class TestTaskDumpExport(unittest.TestCase):
    """Validate Task complete launch arguments and task-centric export layout."""

    def tearDown(self) -> None:
        """Reset the process-local diagnostic session after each test."""
        _reset_active_session_for_test()

    @staticmethod
    def _launch_context() -> LaunchContext:
        """Supply the ordinary arguments for a simulated Task dump launch."""
        runtime = SimpleNamespace(
            graph="graph", runtime_config_tensor=torch_ones(1, dtype=torch_uint8),
            rank=0, kernel_name="test_kernel", metadata=None,
        )
        return LaunchContext(runtime, ("parameter",), "forward", torch_zeros(16))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_uncaptured_record_propagates_even_if_stage_contains_oom_text(self):
        """Keep a missing Device record distinct from an allocation error."""
        prepared = _captured_task_dump([4.0, 6.0])
        record = replace(prepared.records[0], stage="DefaultCPUAllocator: not enough memory: stage")
        status_offset = record.record_offset + 7 * 4
        prepared.buffer[status_offset:status_offset + 4] = 0
        prepared = replace(prepared, records=(record,))
        with tempfile.TemporaryDirectory() as output_dir:
            config = dumper_api.DumpConfig(output_dir, level="task", stages="SwiGLU")
            with (
                mock.patch.object(dumper_api, "_prepare_task_dump", return_value=prepared),
                dumper_api.dump(config) as dumper,
                self.assertRaisesRegex(RuntimeError, "Task dump record was not captured"),
            ):
                dumper.prepare_call(self._launch_context()).complete()
            self.assertFalse(dumper.last_manifest_paths)
            self.assertFalse(list(Path(output_dir).iterdir()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exports_task_files_and_metadata(self):
        """Feature: MegaKernel diagnostics.
        Description: Exports task files and metadata.
        Expectation: Exports task files and metadata.
        """
        expected = torch_tensor([4.0, 6.0], dtype=torch_bfloat16)
        prepared = _captured_task_dump([4.0, 6.0], dtype=expected.dtype)
        with tempfile.TemporaryDirectory() as output_dir, redirect_stdout(StringIO()) as log:
            config = dumper_api.DumpConfig(
                output_dir,
                level="task",
                stages="SwiGLU",
                directions="forward",
                outputs=("tensor", "log"),
            )
            with mock.patch.object(
                dumper_api,
                "_prepare_task_dump",
                return_value=prepared,
            ):
                with dumper_api.dump(config) as dumper:
                    context = self._launch_context()
                    call = dumper.prepare_call(context)
                    call.complete()
                    call.complete()  # Completion must be idempotent for a published invocation.
                    self.assertFalse(dumper.last_manifest_paths)

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            task = invocation["tasks"][0]
            tensor = task["tensors"][0]
            self.assertEqual(manifest["level"], "task")
            self.assertEqual(manifest["schemaVersion"], 6)
            self.assertEqual(manifest["tensorSerialization"], "torch.save")
            self.assertEqual(manifest["tensorFileLayout"], "stage_only")
            self.assertEqual(manifest["capturePhaseSemantics"], "task_boundary")
            self.assertNotIn("dumpPhaseSemantics", manifest)
            self.assertEqual(manifest["stageViewStorage"], "per_task_files")
            self.assertEqual(manifest["invocationCount"], 1)
            self.assertEqual(invocation["taskCount"], 1)
            self.assertEqual(task["taskId"], 7)
            self.assertEqual((task["coreType"], task["coreId"]), ("AIV", 1))
            self.assertEqual(task["stage"], "SwiGLU")
            self.assertEqual(task["owner"], {"label": "Expert", "id": 3})
            self.assertNotIn("metadataFile", task)
            self.assertFalse(list(manifest_path.parent.rglob("task.json")))
            self.assertIn("SwiGLU.output[0] task=7 owner=3 worker=AIV/1", log.getvalue())
            self.assertIn("phase=after range=task_view", log.getvalue())
            self.assertEqual(tensor["capturePhase"], "after")
            self.assertNotIn("dumpPhase", tensor)
            self.assertEqual(tensor["sourceOffsetBytes"], 32)

            legacy = deepcopy(manifest)
            legacy["schemaVersion"] = 5
            for old_invocation in legacy["invocations"]:
                for old_task in old_invocation["tasks"]:
                    sidecar = Path(old_task["tensors"][0]["file"]).with_name("task.json")
                    (manifest_path.parent / sidecar).write_text(json.dumps(old_task), encoding="utf-8")
                    old_task["metadataFile"] = sidecar.as_posix()
            legacy_path = manifest_path.with_name("legacy_manifest.json")
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            for path in (manifest_path, legacy_path):
                payload = load_tensor(
                    path, invocation_id=0, port=("SwiGLU", "output", 0), task_id=7, worker=("AIV", 1),
                )
                assert_close(payload, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_final_skipped_window_clears_previous_manifest_paths(self):
        """Finalize skipped windows and retain results across unused steps."""
        context = self._launch_context()
        prepared = _captured_task_dump([4.0, 6.0])
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=2)
        for tail in ("exit", "unused", "resource_skip"):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as output_dir:
                config = dumper_api.DumpConfig(output_dir, level="task", stages="SwiGLU")
                on_dump_ready = mock.Mock()
                skipped = MemoryError("test allocation failed") if tail == "resource_skip" else None
                with (
                    mock.patch.object(dumper_api, "_prepare_task_dump", side_effect=(prepared, skipped, skipped)),
                    dumper_api.dump(config, schedule=capture_schedule, on_dump_ready=on_dump_ready) as dumper,
                ):
                    dumper.prepare_call(context).complete()
                    if tail == "resource_skip":
                        dumper.prepare_call(context).complete()
                    dumper.step()
                    previous_paths = dumper.last_manifest_paths
                    self.assertEqual(len(previous_paths), 1)
                    if tail != "unused":
                        dumper.prepare_call(context).complete()
                self.assertEqual(dumper.last_manifest_paths, previous_paths if tail == "unused" else ())
                self.assertTrue(previous_paths[0].is_file())
                self.assertEqual(tuple(previous_paths[0].parents[1].glob("*/manifest.json")), previous_paths)
                on_dump_ready.assert_called_once_with(dumper)

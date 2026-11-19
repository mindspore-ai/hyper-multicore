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
from copy import deepcopy
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from torch import (
    bfloat16 as torch_bfloat16,
    float32 as torch_float32,
    frombuffer as torch_frombuffer,
    load as torch_load,
    ones as torch_ones,
    tensor as torch_tensor,
    uint8 as torch_uint8,
    zeros as torch_zeros,
)
from torch.testing import assert_close

from hyper_multicore import prepare_mega_kernel_diagnostics
from hyper_multicore.diagnostics.dumper import api as tensor_dump
from hyper_multicore.diagnostics.dumper import _task as tensor_dump_task
from hyper_multicore.diagnostics.common.session import _reset_active_session_for_test
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskAiCoreType, TaskType
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.scheduler.runtime import allocate_runtime_config
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
    raw_buffer = tensor_dump_task._pack_task_buffer((record,))
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


class TestTaskDumpProtocol(unittest.TestCase):
    """Validate the fixed Task record ABI and Torch Tensor decoding."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_decodes_captured_record(self):
        """Feature: MegaKernel diagnostics.
        Description: Decodes captured record.
        Expectation: Decodes captured record.
        """
        prepared = _captured_task_dump([1.5, -2.0])

        snapshots = tensor_dump_task._decode_task_dump(prepared)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].record.task_id, 7)
        self.assertEqual(snapshots[0].stored_shape, (2,))
        assert_close(snapshots[0].data, torch_tensor([1.5, -2.0]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_decodes_bfloat16_without_conversion(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep the original bfloat16 dtype when decoding Device payload bytes.
        Expectation: Decodes bfloat16 without conversion.
        """
        prepared = _captured_task_dump([1.5, -2.0], dtype=torch_bfloat16)

        snapshot = tensor_dump_task._decode_task_dump(prepared)[0]

        self.assertEqual(snapshot.original_dtype, "bfloat16")
        self.assertEqual(snapshot.stored_dtype, "bfloat16")
        self.assertEqual(snapshot.data.dtype, torch_bfloat16)
        assert_close(snapshot.data, torch_tensor([1.5, -2.0], dtype=torch_bfloat16))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_record_not_captured_by_device(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects record not captured by device.
        Expectation: Rejects record not captured by device.
        """
        prepared = _captured_task_dump([1.0])
        prepared.buffer[
            tensor_dump_task.TASK_DUMP_RECORDS_OFFSET + 7 * 4
        ] = tensor_dump_task.TASK_DUMP_STATUS_PENDING

        with self.assertRaisesRegex(RuntimeError, "was not captured"):
            tensor_dump_task._decode_task_dump(prepared)

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
        config = tensor_dump.TensorDumpConfig("dump", level="task", stages="SwiGLU")

        prepared = tensor_dump_task._prepare_task_dump(
            runtime,
            graph,
            (torch_ones(2), torch_zeros(2)),
            config,
            rank=0,
            device="cpu",
        )

        self.assertEqual([(record.io, record.phase) for record in prepared.records], [
            ("input", "before"),
            ("output", "after"),
        ])
        self.assertEqual({record.core_type for record in prepared.records}, {"AIV"})
        self.assertEqual({record.owner_id for record in prepared.records}, {4})
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
                selection = tensor_dump.TensorDumpConfig(
                    "dump", level="task", stages="Copy", io="input", task_ids=selected_ids,
                )
                prepared = tensor_dump_task._prepare_task_dump(
                    runtime, runtime.graph, parameters, selection, rank=0, device="cpu",
                )
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exports_task_files_and_metadata(self):
        """Feature: MegaKernel diagnostics.
        Description: Exports task files and metadata.
        Expectation: Exports task files and metadata.
        """
        prepared = _captured_task_dump([4.0, 6.0])
        with tempfile.TemporaryDirectory() as output_dir:
            config = tensor_dump.TensorDumpConfig(
                output_dir,
                level="task",
                stages="SwiGLU",
                directions="forward",
            )
            with mock.patch.object(
                tensor_dump,
                "_prepare_task_dump",
                return_value=prepared,
            ):
                with tensor_dump.tensor_dump(config) as dumper:
                    events = torch_zeros(16)
                    call = dumper._acquire_call(
                        "graph", ("parameter",), kernel_name="test_kernel", direction="forward",
                        rank=0, device="cpu", runtime=SimpleNamespace(runtime_config_tensor="normal"),
                        event_counters=events,
                    )
                    self.assertIs(call.runtime_config_tensor, prepared.runtime_config_tensor)
                    self.assertIs(call.diagnostic_buffer, prepared.buffer)
                    self.assertIs(call.event_counters, events)
                    call.complete()

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            task = invocation["tasks"][0]
            tensor = task["tensors"][0]
            task_path = manifest_path.parent / task["metadataFile"]
            task_document = json.loads(task_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["level"], "task")
            self.assertEqual(manifest["schemaVersion"], 5)
            self.assertEqual(manifest["tensorSerialization"], "torch.save")
            self.assertEqual(manifest["tensorFileLayout"], "stage_only")
            self.assertEqual(manifest["capturePhaseSemantics"], "task_boundary")
            self.assertNotIn("dumpPhaseSemantics", manifest)
            self.assertEqual(manifest["stageViewStorage"], "per_task_files")
            self.assertEqual(invocation["taskCount"], 1)
            self.assertEqual(task["taskId"], 7)
            self.assertEqual(task["stage"], "SwiGLU")
            self.assertEqual(task_document["owner"], {"label": "Expert", "id": 3})
            self.assertEqual(tensor["capturePhase"], "after")
            self.assertNotIn("dumpPhase", tensor)
            self.assertEqual(tensor["sourceOffsetBytes"], 32)
            assert_close(
                torch_load(
                    manifest_path.parent / tensor["file"],
                    map_location="cpu",
                    weights_only=True,
                ),
                torch_tensor([4.0, 6.0]),
            )

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
"""MegaMoe-owned diagnostic names and expert resolution tests."""

import unittest

import torch

from hyper_multicore.diagnostics.common.graph import DiagnosticSpec, build_graph_metadata
from hyper_multicore.diagnostics.profiler.cycle_trace import (
    CORE_HEADER, NO_TASK_ID, PROFILE_RECORD, _CycleTraceConfig,
    _calculate_event_driven_profile_layout, _parse_cycle_buffer, _profile_slot_layouts,
)
from hyper_multicore.modules.mega_moe.diagnostics import REPLICA_TRACE_STAGES, _mega_moe_owner_options
from hyper_multicore.modules.mega_moe.backends.bf16.autograd import _diagnostic_runtime
from hyper_multicore.modules.mega_moe.backends.bf16.forward.graph import build_forward_graph
from hyper_multicore.scheduler.config import TaskDescC, TaskSplitValue, TaskType
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from tests.common.mark_utils import arg_mark


class TestMegaMoeDiagnostics(unittest.TestCase):
    """Keep business-specific diagnostic rules outside generic tool tests."""

    def test_profiler_retains_replica_launch_suffix(self):
        """Preserve both runtime images and the device's extension offset."""
        ordinary = torch.tensor([1, 0, 2], dtype=torch.uint8)
        profiler = torch.tensor([1, 1, 2], dtype=torch.uint8)
        extended = torch.tensor([1, 0, 2, 0, 3, 4], dtype=torch.uint8)
        self.assertEqual(_diagnostic_runtime(profiler, ordinary, extended).tolist(), [1, 1, 2, 0, 3, 4])
        self.assertEqual(extended.tolist(), [1, 0, 2, 0, 3, 4])
        self.assertIs(_diagnostic_runtime(profiler, ordinary, None), profiler)
        with self.assertRaisesRegex(RuntimeError, "runtime extension offset"):
            _diagnostic_runtime(profiler[:2], ordinary, extended)

    def test_replica_intervals_keep_slot_identity_and_have_record_budget(self):
        """Decode a guest wait and an owner band wait without inventing graph tasks."""
        config = allocate_runtime_config(16)
        config.num_workers = 48
        config.task_index_num[0] = 1
        layout = _calculate_event_driven_profile_layout(
            config, extra_aic_records_per_task=1, extra_aiv_records_per_worker=2 * (4 + 8),
        )
        self.assertEqual(layout.aiv_required_records, 24)
        self.assertEqual(layout.aic_required_records,
                         _calculate_event_driven_profile_layout(config).aic_required_records + 1)
        raw = bytearray(layout.buffer_size)
        for core, worker, offset, capacity in _profile_slot_layouts(
            layout.aic_record_capacity, layout.aiv_record_capacity, layout.cube_slot_count,
        ):
            if worker != 0:
                continue
            CORE_HEADER.pack_into(raw, offset, 1000, 1, 0, core, worker, capacity, 0)
            descriptor, index, physical_expert = (0x10100, 1, 15) if core == 1 else (0x10105, 7, 9)
            PROFILE_RECORD.pack_into(raw, offset + CORE_HEADER.size, 1050, 1100,
                                     descriptor, NO_TASK_ID, index, physical_expert)
        trace = _parse_cycle_buffer(
            raw, config=_CycleTraceConfig(
                rank=1, device_id=1, cycle_frequency_mhz=50.0, detailed_task_names=True,
                kernel_name="mega_moe", owner_label="Expert", stage_names=REPLICA_TRACE_STAGES,
                soc_name="ascend910_93", cube_slot_count=layout.cube_slot_count,
            ),
            aic_record_capacity=layout.aic_record_capacity, aiv_record_capacity=layout.aiv_record_capacity,
        )
        durations = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        self.assertEqual([event["name"] for event in durations],
                         ["Expert15_ReplicaW13WeightWait", "Expert9_ReplicaDW2LongestBandReadyWait"])
        self.assertEqual([event["args"]["stage_task_index"] for event in durations], [1, 7])
        self.assertEqual([event["dur"] for event in durations], [1.0, 1.0])
        self.assertEqual(trace["megaKernelCycleTrace"]["droppedRecordCount"], 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mega_moe_uses_graph_names_and_expert_owners(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep MegaMoe display semantics without a second task-matching table.
        Expectation: Mega moe uses graph names and expert owners.
        """
        topology = TaskSplitValue()
        topology.rank_id = 1
        graph = build_forward_graph(topology)
        for operator in graph.topological_sort():
            operator.task_num = 1
        runtime_config = allocate_runtime_config(16)
        runtime_config.task_num = 5

        dispatch = TaskDescC()
        dispatch.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        dispatch.task_split_num = 64
        dispatch.task_index = 4
        runtime_config.all_tasks[0] = dispatch

        gmm1 = TaskDescC()
        gmm1.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm1.task_index = 48
        runtime_config.all_tasks[1] = gmm1

        swiglu = TaskDescC()
        swiglu.task_type = TaskType.TASK_SWI_GLU
        swiglu.task_split_value = 128
        swiglu.task_index = 128
        runtime_config.all_tasks[2] = swiglu

        gmm2 = TaskDescC()
        gmm2.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm2.task_index = 72
        runtime_config.all_tasks[3] = gmm2

        combine = TaskDescC()
        combine.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        combine.task_split_num = 64
        combine.task_index = 4
        runtime_config.all_tasks[4] = combine

        options = _mega_moe_owner_options(topology, num_cube_cores=24)
        original = bytes(runtime_config)
        metadata = build_graph_metadata(graph, runtime_config, DiagnosticSpec("mega_moe", **options))
        self.assertEqual(bytes(runtime_config), original)
        self.assertEqual(
            metadata.task_stage_names,
            {0: "Dispatch", 1: "GMM1", 2: "SwiGLU", 3: "GMM2", 4: "Combine"},
        )
        self.assertEqual(metadata.task_owner_ids[0], 2)
        self.assertEqual(metadata.task_owner_ids[1], 10)
        self.assertEqual(metadata.task_owner_ids[2], 10)
        self.assertEqual(metadata.task_owner_ids[3], 11)
        self.assertEqual(metadata.task_owner_ids[4], 10)

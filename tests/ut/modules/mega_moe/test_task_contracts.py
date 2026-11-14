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
"""Malformed task rejection and graph-declared probability effects."""
from dataclasses import replace
import unittest

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts, _build_task_values
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.schedule_policy import MegaMoeSchedulePolicy, pipeline_cube_queue
from hyper_multicore.modules.mega_moe.task_options import validate_task_options
from hyper_multicore.scheduler.config import TaskDescC, TaskType
from hyper_multicore.scheduler.runtime import copy_runtime_config, serialize_runtime_config


class TestTaskContracts(unittest.TestCase):
    """Exercise the production lowering and serialization boundaries."""

    @staticmethod
    def spec(mode: str = "push", weighted: bool = True) -> MegaMoeSpec:
        """Build one small validated topology for the production plan compiler."""
        return MegaMoeSpec(256, 128, 512, 12, 2, 1.25 if mode == "push" else None,
                           640, 4, None, 0, 20, dispatch_mode=mode,
                           apply_probs_before_fc2=weighted)

    def test_probability_effects_exist_before_diagnostics(self):
        """Same graph operand drives execution, effects and diagnostic views."""
        for mode in ("push", "pull"):
            spec = self.spec(mode)
            fwd, fc, bwd, bc = _build_runtime_artifacts(spec)
            for graph, config in ((fwd, fc), (bwd, bc)):
                for op in graph.topological_sort():
                    if not op.tensor_effects:
                        continue
                    effect, = op.tensor_effects
                    self.assertIs(effect.operand, op.outputs[-1])
                    self.assertEqual(effect.completion, "token_and_scalar")
                    self.assertEqual(effect.operand.shape, [4, 2048])
                dispatch, = graph.get_op("dispatch").tensor_effects
                if mode == "pull":
                    self.assertEqual(dispatch.remote_reads, ("source_p",))
                    self.assertEqual(dispatch.writes, ("received_p",))
                else:
                    self.assertEqual(dispatch.reads, ("source_p",))
                    self.assertEqual(dispatch.remote_writes, ("received_p",))
                self.assertTrue(serialize_runtime_config(config))
            activation, = bwd.get_op("swiglu_grad").tensor_effects
            self.assertEqual(activation.writes, ("received_dp",))
            combine, = bwd.get_op("combine").tensor_effects
            self.assertEqual(combine.reads, activation.writes)
            self.assertEqual(combine.remote_writes, ("returned_dp",))

    def test_final_serializer_rejects_invalid_options(self):
        """Unknown bits and a communication-only bit cannot enter a SwiGLU task."""
        _, config, _, _ = _build_runtime_artifacts(self.spec(weighted=False))
        index = next(i for i, task in enumerate(config.all_tasks) if task.task_type == TaskType.TASK_SWI_GLU)
        for flags in (8, 12, 0x80000000):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                config.all_tasks[index].extra_value_1 = flags
                serialize_runtime_config(config)

    def test_probability_operand_and_payload_rejections(self):
        """Reject missing, wrong-unit and wrong-direction operands and mixed payloads."""
        _, config, _, _ = _build_runtime_artifacts(self.spec())
        original = next(t for t in config.all_tasks if t.task_type == TaskType.TASK_SWI_GLU)
        for field, value in (("data_type", 2), ("input_position", 25), ("base_ptr_offset", 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                task = TaskDescC.from_buffer_copy(bytes(original))
                setattr(task.outputs[1], field, value)
                validate_task_options(task)
        for count in (0, 1, 5):
            with self.subTest(outputs=count), self.assertRaises(ValueError):
                task = TaskDescC.from_buffer_copy(bytes(original))
                task.num_outputs = count
                validate_task_options(task)
        task = TaskDescC.from_buffer_copy(bytes(original))
        task.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        task.extra_value_1 = 12
        with self.assertRaises(ValueError):
            validate_task_options(task)
        task.task_type = TaskType.TASK_SHMEM_GET_MEM
        task.extra_value_1 = 8
        with self.assertRaises(ValueError):
            validate_task_options(task)

    def test_scalar_lowering_preserves_token_queues_and_events(self):
        """No extra launches, task IDs or producer credits arise from scalar effects."""
        for mode in ("push", "pull"):
            for tokens in (128, 131072):
                spec = replace(self.spec(mode), local_num_tokens=tokens)
                _, forward, _, backward = _build_runtime_artifacts(spec)
                _, base_forward, _, base_backward = _build_runtime_artifacts(
                    replace(spec, apply_probs_before_fc2=False))
                for current, base in ((forward, base_forward), (backward, base_backward)):
                    self.assertEqual(list(current.task_index_num), list(base.task_index_num))
                    self.assertEqual(list(current.all_event_num_triggers), list(base.all_event_num_triggers))
                    for queue in ("cube_task_indices", "vector_task_indices"):
                        self.assertEqual(list(getattr(current, queue)), list(getattr(base, queue)))

    def test_diagnostic_copy_retains_validation(self):
        """Changing diagnostic mode cannot drop the host task ABI validator."""
        _, config, _, _ = _build_runtime_artifacts(self.spec(weighted=False))
        copied = copy_runtime_config(config)
        copied.diagnostic_mode = 1
        task = next(t for t in copied.all_tasks if t.task_type == TaskType.TASK_SWI_GLU)
        task.extra_value_1 = 8
        with self.assertRaises(ValueError):
            serialize_runtime_config(copied)
        self.assertTrue(serialize_runtime_config(config))

    def test_cube_pipeline_uses_stage_identity_and_independent_ready_policy(self):
        """Queue order follows logical stages even if physical parameter slots move."""
        spec = replace(self.spec(), local_num_tokens=131072)
        graph, config, _, _ = _build_runtime_artifacts(spec)
        moved = copy_runtime_config(config)
        # Deliberately alias every physical output slot. Stage identity still
        # distinguishes Up/Down and must reproduce the original queue exactly.
        for task in moved.all_tasks:
            task.outputs[0].input_position = 0
        topology = _build_task_values(spec)
        pipeline_cube_queue(moved, graph, topology, spec.num_cube_cores)
        self.assertEqual(list(moved.cube_task_indices), list(config.cube_task_indices))
        policy = MegaMoeSchedulePolicy(vector_ready_queue=False, pipeline_vector=True, pipeline_cube=True)
        policy.apply(moved, graph, topology, spec.num_cube_cores)
        self.assertEqual(moved.vector_ready_queue, 0)
        self.assertEqual(list(moved.cube_task_indices), list(config.cube_task_indices))

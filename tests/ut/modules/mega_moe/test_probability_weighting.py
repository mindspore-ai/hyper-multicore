# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Pre-FC2 probability placement, storage and schedule contracts."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe import function as function_module

from hyper_multicore.modules.mega_moe.function import _stage_probabilities, _probability_planes, _kernel_workspace
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.workspace import MegaMoeWorkspace, _workspace_allocation_specs


from hyper_multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_multicore.diagnostics.dump.l1 import L1TensorViewContext, L1TensorView
from hyper_multicore.modules.mega_moe.diagnostics import _adapt_mega_moe_diagnostic_graph
from hyper_multicore.modules.mega_moe.function import _diagnostic_parameters
from hyper_multicore.modules.mega_moe.tensor_dump import resolve_mega_moe_l1_tensor_view
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.graph import TensorSpec


class TestProbabilityWeighting(unittest.TestCase):
    def setUp(self) -> None:
        def scatter_nd_update(target, indices, updates):
            self.assertEqual(indices.dtype, torch.int32)
            self.assertEqual(target.numel(), updates.numel())
            target[indices[:, 0].long()] = updates
            return target

        loader = patch.object(function_module, "load_torch_npu", return_value=SimpleNamespace(
            npu_scatter_nd_update_=scatter_nd_update))
        loader.start()
        self.addCleanup(loader.stop)

    def test_probability_planes_cover_lossless_routing_for_both_transports(self) -> None:
        """Expert movement cannot overflow scalar storage or require a second growth."""
        for mode in ("push", "pull"):
            for ep, topk in ((2, 2), (4, 4), (8, 8)):
                with self.subTest(mode=mode, ep=ep, topk=topk):
                    common = dict(dispatch_mode=mode, routed_slots=256 * topk,
                                  hidden_size=1024, ep_size=ep, num_experts=ep * 8,
                                  dtype=torch.bfloat16)
                    expected = None
                    for received in (128, ep * 256 * topk):
                        allocations = dict(_workspace_allocation_specs(
                            **common, receive_capacity=received, apply_probs_before_fc2=True))
                        plane = allocations.pop("probability_workspace")
                        self.assertEqual(plane.shape, (4, ep * 256 * topk))
                        self.assertEqual(plane.dtype, torch.float32)
                        self.assertEqual(plane.shape, expected or plane.shape)
                        expected = plane.shape
                        baseline = dict(_workspace_allocation_specs(**common, receive_capacity=received))
                        self.assertEqual(allocations, baseline)

    def test_pre_fc2_uses_original_completion_events(self) -> None:
        """Probability traffic adds no task, collective or completion count."""
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                spec = MegaMoeSpec(256, 1024, 512, 12, 2,
                                   1.25 if mode == "push" else None,
                                   640 if mode == "push" else 512, 4, None, 0, 20,
                                   dispatch_mode=mode)
                _, fwd_base, _, bwd_base = _build_runtime_artifacts(spec)
                _, fwd, _, bwd = _build_runtime_artifacts(replace(spec, apply_probs_before_fc2=True))
                for baseline, current, position in ((fwd_base, fwd, 16), (bwd_base, bwd, 25)):
                    self.assertEqual(list(current.task_index_num), list(baseline.task_index_num))
                    self.assertEqual(list(current.all_event_num_triggers), list(baseline.all_event_num_triggers))
                    ids = list(current.vector_task_indices[:current.task_index_num[1]])
                    weighted = [current.all_tasks[i] for i in ids if current.all_tasks[i].extra_value_1 & 12]
                    self.assertTrue(weighted)
                    for task in weighted:
                        aux = task.outputs[task.num_outputs - 1]
                        self.assertEqual(aux.input_position, position)
                        self.assertEqual(aux.dim[1], spec.ep_size * spec.routed_slots)
                        self.assertEqual(aux.base_ptr_offset, 0)

    def test_probability_staging_preserves_selected_zeros_and_slot_order(self) -> None:
        """The inverse permutation, not nonzero scores, determines source rows."""
        mapping = torch.tensor([4, 0, 6, 2, 1, 7, 5, 3], dtype=torch.int32)
        probs = torch.tensor([[0., .125], [.25, .375], [.5, .625], [.75, .875]])
        workspace = MegaMoeWorkspace(shared=False)
        workspace.probability_workspace = torch.full((4, 16), -1.)
        _stage_probabilities(workspace, mapping, probs, SimpleNamespace(ep_size=1, plan_tokens=8, top_k=2))
        torch.testing.assert_close(workspace.probability_workspace[0, :8][mapping.long()], probs.flatten())
        self.assertEqual(workspace.probability_workspace[0, 4].item(), 0.)
        self.assertTrue(torch.equal(workspace.probability_workspace[0, 8:], torch.full((8,), -1.)))
        self.assertTrue(torch.equal(workspace.probability_workspace[1:], torch.full((3, 16), -1.)))

    def test_plan_strides_survive_schedule_buckets_and_heap_growth(self) -> None:
        """Older retained plans keep the same scalar offsets after physical growth."""
        workspace = MegaMoeWorkspace(shared=False)
        for capacity in (256, 1024):
            workspace.probability_workspace = torch.zeros((4, 2 * capacity * 2))
            for tokens in (128, 256):
                spec = MegaMoeSpec(128, 1024, 512, 6, 2, 1.25, 640, 2, None, 0, 20,
                                   schedule_tokens=tokens, apply_probs_before_fc2=True)
                planes = _probability_planes(workspace, spec)
                stride = 2 * tokens * 2
                self.assertEqual(tuple(planes.shape), (4, stride))
                self.assertIs(planes, _probability_planes(workspace, spec))
                self.assertEqual(planes.data_ptr(), workspace.probability_workspace.data_ptr())
                _, forward, _, backward = _build_runtime_artifacts(spec)
                for config in (forward, backward):
                    for i in config.vector_task_indices[:config.task_index_num[1]]:
                        task = config.all_tasks[i]
                        if task.extra_value_1 & 12:
                            self.assertEqual(task.outputs[task.num_outputs - 1].dim[1], stride)

    def test_probability_aliases_retire_with_the_allocation(self) -> None:
        """Cached metadata cannot retain a released symmetric buffer."""
        workspace = MegaMoeWorkspace(shared=False)
        spec = SimpleNamespace(ep_size=2, plan_tokens=128, top_k=2, apply_probs_before_fc2=True)
        workspace.probability_workspace = torch.zeros(4, 512)
        planes = _probability_planes(workspace, spec)
        raw = _kernel_workspace(workspace, spec)
        self.assertIs(raw, _kernel_workspace(workspace, spec))
        workspace.probability_workspace = torch.zeros(4, 1024)
        self.assertIsNot(planes, _probability_planes(workspace, spec))
        self.assertEqual(_kernel_workspace(workspace, spec).data_ptr(), workspace.probability_workspace.data_ptr())
        with patch("hyper_multicore.modules.mega_moe.workspace.shmem.free"):
            workspace._free_symmetric_tensors()
        self.assertIsNone(workspace.probability_views)
        self.assertIsNone(workspace.probability_workspace)

    def test_inference_stages_probabilities_with_the_current_plan(self) -> None:
        """Checkpoint's initial no-grad forward obeys the same weighting layout."""
        workspace = MegaMoeWorkspace(shared=False)
        workspace.probability_workspace = torch.zeros(4, 512)
        workspace.in_use = True
        spec = SimpleNamespace(apply_probs_before_fc2=True, ep_size=2, plan_tokens=128,
                               top_k=2, expert_weight_layout="EIO")
        mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        probs = torch.tensor([[0., .25], [.5, .75]])
        rows = torch.arange(8.).reshape(4, 2)
        route = SimpleNamespace(routed_tokens=rows, unpermute_mapping=mapping,
                                metadata=SimpleNamespace(expert_capacity=4, for_kernel=lambda width: None))
        spec.hidden_size = 2
        execution = SimpleNamespace(source=rows, combine=rows)
        with torch.no_grad(), patch.object(function_module, "_prepare_forward_execution", return_value=execution), \
                patch.object(function_module, "_launch_forward_kernel"), \
                patch.object(function_module.multicore_ops, "moe_token_permute_grad", return_value=rows[:2]) as restore:
            output = function_module._execute_mega_moe_without_autograd(
                probs, rows, rows, route, SimpleNamespace(spec=spec), workspace, workspace_claimed=True)
        torch.testing.assert_close(workspace.probability_workspace[0, mapping.long()], probs.flatten())
        torch.testing.assert_close(output, rows[:2])
        restore.assert_called_once()


    def test_diagnostic_graph_preserves_the_scheduled_work(self) -> None:
        """Profiling remaps IDs without changing work, order or completion events."""
        for mode in ("push", "pull"):
            for weighted in (False, True):
                for batched in (False, True):
                    spec = MegaMoeSpec(
                        256, 1024, 512, 12, 2, 1.25 if mode == "push" else None,
                        640 if mode == "push" else 512, 4, None, 0, 20,
                        dispatch_mode=mode, apply_probs_before_fc2=weighted, batch_experts=batched,
                    )
                    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(spec)
                    cases = (
                        (forward_graph, forward_config, 17 if weighted else 16),
                        (backward_graph, backward_config, 26 if weighted else 20),
                    )
                    for graph, config, count in cases:
                        original = bytes(config)
                        copied = type(config).from_buffer_copy(original)
                        adapted = _adapt_mega_moe_diagnostic_graph(graph, copied)
                        validate_mega_kernel_diagnostic_contract(adapted, copied, count)
                        self.assertEqual(bytes(config), original)
                        self.assertEqual(list(copied.all_event_num_triggers), list(config.all_event_num_triggers))
                        queues = (
                            (config.cube_task_indices, copied.cube_task_indices, config.task_index_num[0]),
                            (config.vector_task_indices, copied.vector_task_indices, config.task_index_num[1]),
                        )
                        for before, after, length in queues:
                            self.assertEqual(
                                [bytes(config.all_tasks[i]) for i in before[:length]],
                                [bytes(copied.all_tasks[i]) for i in after[:length]],
                            )

    def test_scratch_bindings_preserve_logical_plan_strides(self) -> None:
        """Retained plans expose logical scratch bounds after physical heap growth."""
        spec = SimpleNamespace(apply_probs_before_fc2=True, ep_size=4, plan_tokens=128, top_k=2)
        plan = SimpleNamespace(spec=spec)
        buffer = torch.arange(4 * 4096, dtype=torch.float32).view(torch.uint8)
        for size, tilings in ((16, ()), (20, (None,) * 5)):
            result = _diagnostic_parameters(plan, (None,) * size, buffer, tilings)
            self.assertEqual(len(result), size + len(tilings) + 1)
            self.assertEqual(result[-1].shape, (4, 1024))
            self.assertEqual(result[-1].data_ptr(), buffer.data_ptr())
            self.assertEqual(result[-1].dtype, torch.float32)
        spec.apply_probs_before_fc2 = False
        parameters = (None,) * 16
        self.assertIs(_diagnostic_parameters(plan, parameters, buffer), parameters)

    def test_probability_snapshots_use_local_scalar_slices(self) -> None:
        """L1 dumps snapshot initialized local rows, never unused remote scratch."""
        cases = (
            (TaskType.TASK_SWI_GLU, 4, 1), (TaskType.TASK_SWI_GLU_GRAD, 4, 2),
            (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, 4, 0),
            (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, 8, 2), (TaskType.TASK_SHMEM_GET_MEM, 4, 1),
        )
        for kind, flag, plane in cases:
            parameters = [torch.zeros(1)] * 17
            parameters[0] = torch.tensor([0, 129 * 4])
            parameters[2] = torch.tensor([0, 129 * 4])
            parameters[3] = torch.tensor([0, 131 * 4])
            parameters[6] = torch.tensor([129, 260])
            parameters[16] = torch.zeros(4, 512)
            inputs = [SimpleNamespace(param_position=n, shape=[2, 4]) for n in range(4)]
            operator = SimpleNamespace(inputs=inputs, outputs=[SimpleNamespace(shape=[260, 4])])
            task = SimpleNamespace(
                task_type=kind, task_split_num=4, task_split_value=128, task_index=3, extra_value_1=flag,
            )
            config = SimpleNamespace(dynamic_data=SimpleNamespace(dynamic_group_size=2, dynamic_input_position=6))
            context = L1TensorViewContext(
                operator, 0, 1, task, SimpleNamespace(data_type=4),
                TensorSpec("probability_workspace", [4, 512], 16, dtype_size=4),
                "output", 1, parameters, config, 0, "scalar", L1TensorView(0, (4, 512)), {},
                owner_context=SimpleNamespace(topology=SimpleNamespace(rank_id=0)),
            )
            view = resolve_mega_moe_l1_tensor_view(context)
            self.assertEqual(view.shape, (3,))
            self.assertEqual(view.source_offset_bytes, (plane * 512 + 257) * 4)
            self.assertLessEqual(view.source_offset_bytes + 3 * 4, parameters[16].numel() * 4)


    def test_local_scalar_prefixes_reuse_aliases_across_dynamic_rows_and_growth(self) -> None:
        """Bound actual rows independently from bucket stride and physical capacity."""
        workspace = MegaMoeWorkspace(shared=False)
        spec = SimpleNamespace(ep_size=2, plan_tokens=16, top_k=2)
        for capacity in (64, 128):
            workspace.probability_workspace = torch.arange(4 * capacity, dtype=torch.float32).view(4, capacity)
            previous_rows, previous_source, previous_gradient = None, None, None
            for rows in (8, 8, 0, 0, 3, 3, 16, 16):
                source = function_module._local_probability_view(workspace, spec, rows)
                gradient = function_module._local_probability_view(workspace, spec, rows, gradient=True)
                planes = _probability_planes(workspace, spec)
                torch.testing.assert_close(source, planes[0, :rows])
                torch.testing.assert_close(gradient, planes[3, :rows])
                self.assertIs(source, function_module._local_probability_view(workspace, spec, rows))
                self.assertIs(gradient, function_module._local_probability_view(
                    workspace, spec, rows, gradient=True))
                if rows == previous_rows:
                    self.assertIs(source, previous_source)
                    self.assertIs(gradient, previous_gradient)
                previous_rows, previous_source, previous_gradient = rows, source, gradient


    def test_backward_stages_saved_mapping_without_expert_ids(self) -> None:
        """Both transports consume the same retained mapping, including selected zero P."""
        mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        probs = torch.tensor([[0., .25], [.5, .75]])
        grad = torch.arange(6., dtype=torch.float32).reshape(2, 3)
        for mode in ("push", "pull"):
            workspace = MegaMoeWorkspace(shared=False)
            workspace.probability_workspace = torch.full((4, 16), -1.)
            workspace.source_buffer = torch.empty(4, 3)
            spec = SimpleNamespace(apply_probs_before_fc2=True, dispatch_mode=mode,
                                   ep_size=2, plan_tokens=4, top_k=2, hidden_size=3)
            ctx = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace, source_rows=4)

            def permute(tokens, inverse, probabilities, output, output_probabilities):
                self.assertIs(inverse, mapping)
                output[inverse.long()] = tokens.repeat_interleave(2, dim=0)
                output_probabilities[inverse.long()] = probabilities.flatten()

            with patch.object(function_module.multicore_ops, "moe_token_permute_from_mapping_out", permute), \
                    patch.object(function_module.multicore_ops, "moe_token_permute_out") as sort:
                result, grad_probs = function_module._stage_backward_source(ctx, grad, (mapping, probs))
            sort.assert_not_called()
            self.assertIsNone(grad_probs)
            torch.testing.assert_close(result[mapping.long()], grad.repeat_interleave(2, dim=0))
            torch.testing.assert_close(workspace.probability_workspace[0, mapping.long()], probs.flatten())
            self.assertTrue(torch.equal(workspace.probability_workspace[0, 4:], torch.full((12,), -1.)))

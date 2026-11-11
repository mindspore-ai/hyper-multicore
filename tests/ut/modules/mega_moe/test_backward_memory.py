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
"""CPU ownership checks for consuming routed SwiGLU inputs in final backward."""

import itertools
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_multicore.modules.mega_moe.fusion import (
    _allocate_backward_intermediates, _can_reuse_saved_tensors, _claim_saved_tensor, _load_forward_state,
    _prepare_backward_buffers, _prepare_backward_route, _reusable_swiglu_input,
)


_KEEP_GRAPH = "hyper_multicore.modules.mega_moe.fusion.torch._C._autograd._get_current_graph_task_keep_graph"


class _ConsumeSavedInput(torch.autograd.Function):
    """Exercise the real engine's retained and final backward state on CPU."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, records: list, slot: int) -> torch.Tensor:
        """Keep an owned internal tensor with no alias to the caller's input."""
        ctx.saved_tensors_reusable = _can_reuse_saved_tensors()
        ctx.save_for_backward(*([None] * slot), values.detach().clone())
        ctx.slot = slot
        ctx.records = records
        return values * 2

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Overwrite the saved input only if the production selector permits it."""
        up = ctx.saved_tensors[ctx.slot]
        can_reuse = _claim_saved_tensor(ctx, ctx.slot)
        ctx.maybe_clear_saved_tensors()
        spec = SimpleNamespace(intermediate_size=up.shape[1] // 2)
        reusable = (_reusable_swiglu_input(up, spec, up.shape[0], grad_output)
                    if can_reuse else None)
        ctx.records.append((reusable is up, up.clone()))
        if reusable is not None:
            reusable.zero_()
        return grad_output * 2, None, None


class TestBackwardMemory(unittest.TestCase):
    """Preserve retained graphs while removing one owned backward allocation."""

    def test_final_backward_reuses_exact_receive_shape(self):
        """A zero-receive rank's one-row sentinel uses the same safe selection."""
        spec = SimpleNamespace(intermediate_size=4, hidden_size=8)
        with torch.no_grad(), patch(_KEEP_GRAPH, return_value=False):
            for capacity in (1, 3, 257):
                with self.subTest(capacity=capacity):
                    up = torch.empty(capacity, 8, dtype=torch.bfloat16)
                    grad_output = torch.empty(0, 8, dtype=torch.bfloat16)
                    selected = _reusable_swiglu_input(up, spec, capacity, grad_output)
                    self.assertIs(selected, up)
                    weight = torch.empty(8, 8, dtype=torch.bfloat16)
                    fresh = _allocate_backward_intermediates(torch, spec, capacity, grad_output, weight, weight)
                    reused = _allocate_backward_intermediates(
                        torch, spec, capacity, grad_output, weight, weight, reusable_up=selected,
                    )
                    self.assertNotEqual(fresh[3].data_ptr(), up.data_ptr())
                    self.assertIs(reused[3], up)
                    self.assertNotEqual(reused[2].data_ptr(), up.data_ptr())
                    self.assertEqual(up.numel() * up.element_size(), capacity * 2 * spec.intermediate_size * 2)

    def test_unknown_retained_or_higher_order_engine_falls_back(self):
        """Only an explicit final engine task may consume a saved tensor."""
        spec = SimpleNamespace(intermediate_size=4)
        up = torch.empty(3, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            for state in (True, None, 0):
                with self.subTest(state=state), patch(_KEEP_GRAPH, return_value=state):
                    self.assertIsNone(_reusable_swiglu_input(up, spec, 3, up))
            with patch(_KEEP_GRAPH, None):
                self.assertIsNone(_reusable_swiglu_input(up, spec, 3, up))
            with patch(_KEEP_GRAPH, side_effect=RuntimeError("engine query unavailable")):
                self.assertIsNone(_reusable_swiglu_input(up, spec, 3, up))
        with torch.enable_grad(), patch(_KEEP_GRAPH, return_value=False):
            self.assertIsNone(_reusable_swiglu_input(up, spec, 3, up))

    def test_incompatible_storage_falls_back(self):
        """In-place use requires exact rows, width, dtype, device and contiguity."""
        spec = SimpleNamespace(intermediate_size=4)
        grad_output = torch.empty(3, 8, dtype=torch.bfloat16)
        candidates = (
            torch.empty(2, 8, dtype=torch.bfloat16),
            torch.empty(4, 8, dtype=torch.bfloat16),
            torch.empty(3, 7, dtype=torch.bfloat16),
            torch.empty(24, dtype=torch.bfloat16),
            torch.empty(8, 3, dtype=torch.bfloat16).t(),
            torch.empty(3, 8, dtype=torch.float16),
            torch.empty(3, 8, dtype=torch.float32),
            torch.empty(3, 8, dtype=torch.bfloat16, device="meta"),
        )
        with torch.no_grad(), patch(_KEEP_GRAPH, return_value=False):
            for up in candidates:
                with self.subTest(shape=up.shape, dtype=up.dtype, device=up.device):
                    self.assertIsNone(_reusable_swiglu_input(up, spec, 3, grad_output))
            self.assertIsNone(_reusable_swiglu_input(grad_output, spec, 3, grad_output.float()))
            self.assertIsNone(_reusable_swiglu_input(grad_output[:0], spec, 0, grad_output))

    def test_retained_graph_survives_until_final_consume(self):
        """Two retained passes preserve inputs before the final pass consumes them."""
        for slot in (1, 16):
            with self.subTest(slot=slot):
                source = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8).requires_grad_()
                records = []
                output = _ConsumeSavedInput.apply(source, records, slot)
                for retain in (True, True, False):
                    gradient, = torch.autograd.grad(output, source, torch.ones_like(output), retain_graph=retain)
                    self.assertTrue(torch.equal(gradient, torch.full_like(source, 2)))
                self.assertEqual([reused for reused, _ in records], [False, False, True])
                self.assertTrue(all(torch.equal(value, source) for _, value in records))

    def test_saved_hooks_preserve_cached_and_user_visible_storage(self):
        """Hooks may return their cache or a user tensor even after their context exits."""
        for slot, unpack_alias in itertools.product((1, 16), (False, True)):
            with self.subTest(slot=slot, unpack_alias=unpack_alias):
                source = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8).requires_grad_()
                original = source.detach().clone()
                records, packed = [], []

                def pack(value: torch.Tensor) -> torch.Tensor:
                    """Keep a detached cache sharing the originally saved storage."""
                    packed.append(value.detach())
                    return packed[-1]

                def unpack(value: torch.Tensor) -> torch.Tensor:
                    """Optionally substitute equivalent user-visible tensor storage."""
                    return source.detach() if unpack_alias else value

                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    self.assertFalse(_can_reuse_saved_tensors())
                    output = _ConsumeSavedInput.apply(source, records, slot)
                self.assertTrue(_can_reuse_saved_tensors())
                gradient, = torch.autograd.grad(output, source, torch.ones_like(output))
                self.assertFalse(records[0][0])
                self.assertTrue(torch.equal(gradient, torch.full_like(source, 2)))
                self.assertTrue(torch.equal(source, original))
                self.assertTrue(all(torch.equal(value, original) for value in packed))

    def test_hook_probe_restores_disabled_state_and_falls_back_if_unknown(self):
        """An empty public probe preserves outer hook settings and has a safe fallback."""
        self.assertTrue(_can_reuse_saved_tensors())
        with torch.autograd.graph.disable_saved_tensors_hooks("outer hooks disabled"):
            self.assertTrue(_can_reuse_saved_tensors())
            with self.assertRaisesRegex(RuntimeError, "outer hooks disabled"):
                with torch.autograd.graph.saved_tensors_hooks(lambda value: value, lambda value: value):
                    pass
        self.assertTrue(_can_reuse_saved_tensors())
        with patch("hyper_multicore.modules.mega_moe.fusion.torch.autograd.graph", SimpleNamespace()):
            self.assertFalse(_can_reuse_saved_tensors())
        with patch("hyper_multicore.modules.mega_moe.fusion.torch.autograd.graph.disable_saved_tensors_hooks",
                   side_effect=RuntimeError("unsupported hooks probe")):
            self.assertFalse(_can_reuse_saved_tensors())

    def test_late_saved_hooks_preserve_cached_and_user_visible_storage(self):
        """Hooks registered after forward must also prevent saved-storage consumption."""
        for slot, unpack_alias in itertools.product((1, 16), (False, True)):
            with self.subTest(slot=slot, unpack_alias=unpack_alias):
                source = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8).requires_grad_()
                original = source.detach().clone()
                records, packed = [], []
                output = _ConsumeSavedInput.apply(source, records, slot)

                def pack(value: torch.Tensor) -> torch.Tensor:
                    """Keep a detached cache before the final backward starts."""
                    packed.append(value.detach())
                    return packed[-1]

                def unpack(value: torch.Tensor) -> torch.Tensor:
                    """Return cached or equivalent caller-visible tensor storage."""
                    return source.detach() if unpack_alias else value

                output.grad_fn._raw_saved_tensors[slot].register_hooks(pack, unpack)
                gradient, = torch.autograd.grad(output, source, torch.ones_like(output))
                self.assertFalse(records[0][0])
                self.assertEqual(len(packed), 1)
                self.assertTrue(torch.equal(gradient, torch.full_like(source, 2)))
                self.assertTrue(torch.equal(source, original))
                self.assertTrue(torch.equal(packed[0], original))

    def test_saved_version_check_runs_before_final_claim(self):
        """Registering the temporary guard must not suppress normal version errors."""
        for slot in (1, 16):
            with self.subTest(slot=slot):
                source = torch.ones(3, 8, dtype=torch.bfloat16, requires_grad=True)
                records = []
                output = _ConsumeSavedInput.apply(source, records, slot)
                with torch.no_grad():
                    output.grad_fn.saved_tensors[slot].add_(1)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                    torch.autograd.grad(output, source, torch.ones_like(output))
                self.assertFalse(records)

    def test_saved_ownership_metadata_defaults_to_no_reuse(self):
        """Old contexts and unknown metadata cannot opt into consuming saved storage."""
        spec = SimpleNamespace(forward_shared_expert=True, backward_permute_fused=False,
                               backward_unpermute_fused=False, backward_shared_expert=False)
        ctx = SimpleNamespace(plan=SimpleNamespace(spec=spec), saved_tensors=(None,) * 22)
        self.assertFalse(_load_forward_state(ctx).up_proj_reusable)
        ctx.saved_tensors_reusable = True
        with torch.no_grad(), patch(_KEEP_GRAPH, return_value=False):
            self.assertFalse(_load_forward_state(ctx).up_proj_reusable)
            ctx._raw_saved_tensors = (None, SimpleNamespace(register_hooks=lambda *_args: None))
            self.assertTrue(_load_forward_state(ctx).up_proj_reusable)

    def test_snapshot_claim_only_for_writable_endpoint(self):
        """Only backward permute plus unpermute without shared writes saved slot16."""
        for permute, unpermute, shared in itertools.product((False, True), repeat=3):
            with self.subTest(permute=permute, unpermute=unpermute, shared=shared):
                spec = SimpleNamespace(forward_shared_expert=True, backward_shared_expert=shared,
                                       backward_permute_fused=permute, backward_unpermute_fused=unpermute)
                slots = tuple(SimpleNamespace(register_hooks=Mock()) for _ in range(17))
                ctx = SimpleNamespace(plan=SimpleNamespace(spec=spec), saved_tensors=(None,) * 22,
                                      saved_tensors_reusable=True, _raw_saved_tensors=slots)
                with torch.no_grad(), patch(_KEEP_GRAPH, return_value=False):
                    saved = _load_forward_state(ctx)
                writable = permute and unpermute and not shared
                self.assertTrue(saved.up_proj_reusable)
                self.assertEqual(saved.expert_output_reusable, writable)
                self.assertEqual(slots[16].register_hooks.call_count, int(writable))

    def test_snapshot_route_copies_unclaimed_storage(self):
        """An unclaimed expert-output snapshot cannot become a writable dX buffer."""
        spec = SimpleNamespace(forward_permute_fused=True, backward_permute_fused=True,
                               backward_unpermute_fused=True, backward_shared_expert=False,
                               dispatch_mode="push", local_num_tokens=3, hidden_size=8)
        workspace = SimpleNamespace(expert_buffer=torch.empty(6, 8), routed_buffer=torch.empty(6, 8))
        ctx = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace)
        for reusable in (False, True):
            with self.subTest(reusable=reusable):
                snapshot = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8)
                original = snapshot.clone()
                saved = SimpleNamespace(dispatch=torch.empty(6, 8), expert_output=snapshot,
                                        permute_src_index=torch.arange(6), topk_weights=torch.ones(3, 2),
                                        expert_output_reusable=reusable)
                route = _prepare_backward_route(ctx, torch.ones(3, 8, dtype=torch.bfloat16), saved)
                self.assertEqual(route.grad_x.data_ptr() == snapshot.data_ptr(), reusable)
                route.grad_x.zero_()
                self.assertTrue(torch.equal(snapshot, torch.zeros_like(snapshot) if reusable else original))

    def test_shared_scratch_preserves_the_owned_input_alias(self):
        """Shared modes continue borrowing only the existing routed scratch slots."""
        def empty(*shape: int) -> torch.Tensor:
            """Allocate a BF16 test operand on CPU."""
            return torch.empty(*shape, dtype=torch.bfloat16)

        for shared, integrated in itertools.product((False, True), repeat=2):
            with self.subTest(shared=shared, integrated=integrated):
                spec = SimpleNamespace(intermediate_size=4, hidden_size=8,
                                       backward_shared_expert=shared, backward_unpermute_fused=integrated)
                saved = SimpleNamespace(
                    dispatch=empty(64, 8), up_proj=empty(64, 8), weight1=empty(2, 8, 8), weight2=empty(2, 8, 4),
                    shared_input=empty(2, 8), shared_gate_up_weight=empty(6, 8), shared_down_weight=empty(8, 3),
                    shared_up_proj=empty(2, 6), shared_activation=empty(2, 3), up_proj_reusable=True,
                )
                plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=True)
                workspace = SimpleNamespace(gmm_workspace=empty(1), swiglu_grad_workspace=empty(1),
                                            prepare_event_counters=lambda **_kwargs: empty(1))
                route = SimpleNamespace(dispatch=empty(64, 8), kernel_grad_output=empty(2, 8))
                with torch.no_grad(), patch(_KEEP_GRAPH, return_value=False), patch(
                        "hyper_multicore.modules.mega_moe.shared.pointer_table", return_value=empty(1)):
                    buffers = _prepare_backward_buffers(plan, workspace, saved, route, route.kernel_grad_output)
                self.assertIs(buffers.swiglu_grad, saved.up_proj)
                self.assertEqual(buffers.gate_dx.data_ptr(), route.dispatch.data_ptr())
                self.assertNotEqual(buffers.act_grad.data_ptr(), saved.up_proj.data_ptr())
                self.assertNotEqual(saved.shared_up_proj.data_ptr(), saved.up_proj.data_ptr())
                if shared:
                    self.assertTrue(buffers.shared_gradients_in_scratch)
                    self.assertEqual(buffers.shared_tensors[6].data_ptr(), saved.up_proj.data_ptr())

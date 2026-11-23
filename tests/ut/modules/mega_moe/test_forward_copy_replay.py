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
"""CPU regressions for the A3 Push non-reentrant checkpoint copy policy."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_multicore.modules.hot_replica import checkpoint as replay
from hyper_multicore.modules.mega_moe import route as route_module
from hyper_multicore.modules.mega_moe.backends.bf16 import autograd, execution
from hyper_multicore.modules.mega_moe.route import PreparedTopKRoute, RouteMetadata
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


class _WorkspaceCopies(TorchDispatchMode):
    """Observe reads of reusable buffers without replacing Tensor operations."""

    def __init__(self, workspace):
        super().__init__()
        self.dispatch = workspace.expert_buffer.untyped_storage().data_ptr()
        self.combine = workspace.routed_buffer.untyped_storage().data_ptr()
        self.phase = None
        self.copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default and args[1].untyped_storage().data_ptr() == self.dispatch:
            self.copies.append((self.phase, "dispatch"))
        if func is torch.ops.aten.clone.default and args[0].untyped_storage().data_ptr() == self.combine:
            self.copies.append((self.phase, "combine"))
        return func(*args, **(kwargs or {}))


class _CpuExecution:
    """Use real BF16 launch orchestration with a simple CPU expert function."""

    def __init__(self, monkeypatch, tokens):
        self.inputs = (
            (torch.arange(tokens * 4, dtype=torch.bfloat16).view(tokens, 4) / 4).requires_grad_(),
            torch.ones((2, 4, 4), dtype=torch.bfloat16, requires_grad=True),
            torch.ones((2, 4, 2), dtype=torch.bfloat16, requires_grad=True),
            torch.tensor([[.25, .75], [.5, .25]], dtype=torch.bfloat16)[:tokens].requires_grad_(),
        )
        self.ids = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)[:tokens]
        self.mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)[:tokens * 2]
        self.rows = tokens * 2
        self.capacity = max(self.rows, 1)
        self.owner = object()
        self.calls = []
        self.offsets = torch.zeros(4, dtype=torch.int32)
        self.workspace = Mock(
            expert_buffer=torch.empty((8, 4), dtype=torch.bfloat16),
            routed_buffer=torch.empty((8, 4), dtype=torch.bfloat16),
            gmm_workspace=torch.empty(1),
            swiglu_grad_workspace=torch.empty(1),
            prepare_event_counters=Mock(return_value=torch.empty(16)),
            in_use=False,
        )
        self.workspace.claim.side_effect = self.claim
        self.workspace.release.side_effect = self.release
        self.observed = _WorkspaceCopies(self.workspace)
        self.spec = MegaMoeSpec(
            max(tokens, 1), 4, 2, 4, 2, 1.0, 8, 2, None, 0, 20, expert_weight_layout="EOI",
        )
        plan = SimpleNamespace(
            spec=self.spec, event_counter_bytes=None, reuse_backward_dispatch=False, weight_grad_dtype=None,
        )
        self.resources = SimpleNamespace(
            spec=self.spec,
            plan_for_tokens=Mock(return_value=plan),
            workspace=self.workspace,
            placement=SimpleNamespace(statistics={}),
        )
        backend = SimpleNamespace(npu_moe_token_unpermute=self.unpermute)
        monkeypatch.setattr(route_module, "load_torch_npu", lambda: backend)
        monkeypatch.setattr(autograd, "load_torch_npu", lambda: backend)
        monkeypatch.setattr(autograd, "_launch_forward_kernel", self.forward_kernel)
        monkeypatch.setattr(autograd, "_launch_backward_kernel", self.backward_kernel)
        monkeypatch.setattr(autograd.multicore_ops, "mega_moe_unpermute_grad_out", self.unpermute_grad)
        monkeypatch.setattr(autograd.multicore_ops, "moe_token_permute_grad", self.permute_grad)
        monkeypatch.setattr(autograd.multicore_ops, "zero_empty_expert_gradients", Mock())

    def claim(self):
        """Reject nested leases."""
        assert not self.workspace.in_use
        self.workspace.in_use = True

    def release(self, *, stream):
        """Poison shared rows as soon as their lease expires."""
        assert self.workspace.in_use and stream is self.workspace.lease_stream
        self.workspace.in_use = False
        self.workspace.expert_buffer.fill_(float("nan"))
        self.workspace.routed_buffer.fill_(float("nan"))

    def unpermute(self, output, mapping, *, probs):
        """Restore the sorted rows using a differentiable CPU reference."""
        assert torch.isfinite(output).all()
        if output.untyped_storage() is self.workspace.routed_buffer.untyped_storage():
            assert self.workspace.in_use
        ordered = output[mapping.long()].view(*probs.shape, 4)
        return (ordered.float() * probs[..., None]).sum(dim=1).to(output.dtype)

    def unpermute_grad(self, output, grad, mapping, probs, grad_permuted, grad_probs):
        """Match the public unpermute-gradient ABI and validate replay storage."""
        assert self.workspace.in_use
        assert output.is_contiguous() and torch.isfinite(output).all()
        expanded = grad[:, None, :].float()
        grad_permuted.index_copy_(
            0, mapping.long(), (expanded * probs[..., None]).to(output.dtype).view(-1, 4),
        )
        grad_probs.copy_((output[mapping.long()].view(*probs.shape, 4).float() * expanded).sum(dim=-1))

    @staticmethod
    def permute_grad(grad, mapping, tokens, top_k):
        """Transpose the token expansion."""
        return grad[mapping.long()].view(tokens, top_k, 4).sum(dim=1)

    def forward_kernel(self, _plan, _metadata, source, w1, w2, buffers, **_):
        """Write all live workspace rows and identify original versus replay launches."""
        assert self.workspace.in_use
        self.observed.phase = "R" if self.calls else "F"
        self.calls.append(self.observed.phase)
        buffers.dispatch[:self.capacity].zero_()
        buffers.dispatch[:self.rows].copy_(source)
        buffers.intermediates.up_proj.zero_()
        buffers.intermediates.activation.zero_()
        buffers.intermediates.down_proj.fill_(-100)
        scale = torch.arange(1, self.rows + 1, dtype=source.dtype)[:, None] + w1[0, 0, 0] + w2[0, 0, 0]
        buffers.combine[:self.rows].copy_(source * scale)

    def backward_kernel(self, _plan, saved, source, buffers, **_):
        """Read replay dispatch state and produce all expert gradients."""
        assert self.workspace.in_use
        assert torch.isfinite(saved.dispatch).all()
        assert torch.equal(saved.dispatch[:self.rows], self.sorted_rows(self.inputs[0]))
        assert torch.count_nonzero(saved.up_proj) == 0 and torch.count_nonzero(saved.activation) == 0
        self.calls.append("B")
        self.observed.phase = "B"
        scale = torch.arange(1, self.rows + 1, dtype=source.dtype)[:, None] + 2
        buffers.grad_x[:self.rows].copy_(source * scale)
        dw = (source.float() * saved.dispatch[:self.rows].float()).sum()
        for weight in (buffers.intermediates.grad_weight1, buffers.intermediates.grad_weight2):
            weight.zero_()
            weight[0, 0, 0] = dw

    def sorted_rows(self, hidden):
        """Expand input rows and scatter each one to its expert-major slot."""
        expanded = hidden.detach().repeat_interleave(2, dim=0)
        return torch.empty_like(expanded).index_copy_(0, self.mapping.long(), expanded)

    def invoke(self, hidden, w1, w2, probs, *, record_mode):
        """Enter normal BF16 execution with a fresh route for fallback cases."""
        record = None
        if record_mode == "both" or (record_mode == "original" and not self.calls):
            record, _ = replay.route_record(self.owner)
        route = PreparedTopKRoute(
            self.sorted_rows(hidden), self.mapping, self.offsets, self.offsets,
            RouteMetadata(*(self.offsets,) * 6, group_list=torch.tensor([0, self.rows]),
                          expert_capacity=self.capacity, plan_tokens=max(hidden.shape[0], 1)),
            checkpoint_record=record,
        )
        return execution.forward(None, hidden, hidden, self.ids, probs, (w1, w2), self.resources, route)


@pytest.mark.parametrize(("tokens", "record_mode"), [(2, "both"), (2, "original"), (2, "none"), (0, "none")])
def test_checkpoint_defers_only_original_copies(monkeypatch, tokens, record_mode):
    """Recompute both saved tensors and match ordinary forward and all gradients."""
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")

    def run(defer):
        toy = _CpuExecution(monkeypatch, tokens)
        tape = replay.ReplayCheckpoint(lambda: 0, defer_forward_copies=defer)

        def operation(*inputs):
            return toy.invoke(*inputs, record_mode=record_mode)

        with toy.observed:
            output = checkpoint(operation, *toy.inputs, use_reentrant=False,
                                context_fn=lambda: (tape.scope(False), tape.scope(True)))
            assert toy.calls == ["F"]
            assert output.untyped_storage() is not toy.workspace.routed_buffer.untyped_storage()
            gradients = torch.autograd.grad(output.sum(), toy.inputs)
        assert toy.calls == ["F", "R", "B"]
        assert toy.workspace.claim.call_count == toy.workspace.release.call_count == 3
        for field in ("original_dispatch_copies_skipped", "original_combine_copies_skipped"):
            assert toy.resources.placement.statistics.get(field, 0) == int(defer)
        forward_copies = [copy for copy in toy.observed.copies if copy[0] != "B"]
        return output, gradients, forward_copies

    expected, expected_gradients, baseline = run(False)
    actual, actual_gradients, candidate = run(True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    expected_dprobs = torch.tensor([[7.5, 4.5], [33.0, 22.0]], dtype=torch.bfloat16)[:tokens]
    torch.testing.assert_close(actual_gradients[-1], expected_dprobs, atol=0, rtol=0)
    assert ("F", "dispatch") in baseline and ("F", "combine") in baseline
    assert sorted(candidate) == [("R", "combine"), ("R", "dispatch")]


@pytest.mark.parametrize(("overrides", "allowed"), [
    ({}, True), ({"architecture": 3510}, False), ({"expert_weight_layout": "EIO"}, False),
    ({"transport_engine": "sdma"}, False),
    ({"dispatch_mode": "pull"}, False), ({"apply_probs_before_fc2": True}, False),
])
def test_copy_policy_eligibility(monkeypatch, overrides, allowed):
    """Restrict the new copy policy to the supported BF16 Push contract."""
    toy = _CpuExecution(monkeypatch, 2)
    toy.resources.spec = replace(toy.spec, **overrides)
    toy.resources.plan_for_tokens.return_value.spec = toy.resources.spec
    bridge = Mock(return_value=torch.empty_like(toy.inputs[0]))
    monkeypatch.setattr(execution, "execute_mega_moe_with_permutation", bridge)
    monkeypatch.setattr(execution, "restore_topk_output", lambda output, *_: output)
    tape = replay.ReplayCheckpoint(lambda: 0, defer_forward_copies=True)
    for scope, expected in ((nullcontext(), None), (tape.scope(False), False), (tape.scope(True), True)):
        with scope:
            toy.invoke(*toy.inputs, record_mode="none")
        assert bridge.call_args.kwargs.get("checkpoint_replay") is (expected if allowed else None)


@pytest.mark.parametrize("invalid", ["dispatch", "combine"])
def test_original_placeholder_rejects_backward_without_replay(monkeypatch, invalid):
    """Reject each unmaterialized tensor before launching an expert kernel."""
    toy = _CpuExecution(monkeypatch, 2)
    placeholder = autograd._checkpoint_placeholder  # pylint: disable=protected-access
    buffers = iter(("dispatch", "combine"))
    monkeypatch.setattr(
        autograd, "_checkpoint_placeholder",
        lambda tensor: placeholder(tensor) if next(buffers) == invalid else torch.empty_like(tensor),
    )
    tape = replay.ReplayCheckpoint(lambda: 0, defer_forward_copies=True)
    with tape.scope(False), toy.observed:
        output = toy.invoke(*toy.inputs, record_mode="none")
    assert not toy.observed.copies
    with pytest.raises(RuntimeError, match="(?i)(placeholder|recomput)"):
        output.sum().backward()
    assert toy.calls == ["F"]
    assert toy.workspace.claim.call_count == toy.workspace.release.call_count == 1


def test_copy_opt_in_rejects_no_grad_checkpoint_forward():
    """A reentrant original forward cannot use the new saved-tensor policy."""
    tape = replay.ReplayCheckpoint(lambda: 0, defer_forward_copies=True)
    with torch.no_grad(), pytest.raises(RuntimeError, match="non-reentrant"), tape.scope(False):
        pytest.fail("Reentrant copy elision must fail before entering the scope")

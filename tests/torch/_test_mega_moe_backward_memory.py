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
"""Prove ordinary up-projection reuse and safe retained fallback through public backward."""

from contextlib import ExitStack, contextmanager
import math
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts
from hyper_multicore.modules.mega_moe import fusion
from tests.common.multicore._mega_moe_utils import write_evidence
from tests.torch._test_mega_moe_shared import _check_empty_experts, _forward, _route_ids


_HIDDEN, _INTERMEDIATE, _SHARED = 5120, 128, 256
_NAMES = ("output", "dx", "dprob", "routed_dw1", "routed_dw2", "shared_dw1", "shared_dw2")
_CASES = (
    ("ragged", (129, 257), "balanced"),
    ("zero_input", (0, 257), "balanced"),
    ("zero_receiver", (129, 257), "zero_experts"),
    ("zero_all", (0, 0), "balanced"),
)


def _batches(rank, device):
    """Keep off/on inputs identical while source and receive lengths differ."""
    batches = []
    for index, (name, lengths, pattern) in enumerate(_CASES):
        torch.manual_seed(92600 + 2 * index + rank)
        tokens = lengths[rank]
        source = torch.randn(tokens, _HIDDEN, device=device, dtype=torch.bfloat16)
        probabilities = torch.rand(tokens, 2, device=device)
        probabilities /= probabilities.sum(-1, keepdim=True)
        ids = _route_ids(tokens, 4, 2, device, pattern)
        counts = torch.bincount(ids.flatten().long(), minlength=4).int()
        dist.all_reduce(counts)
        upstream = torch.randn_like(source) / math.sqrt(_HIDDEN)
        batches.append((name, source, ids, probabilities, upstream, counts.chunk(2)[rank].cpu()))
    return batches


def _weights(rank, device):
    """Reuse identical seeded parameters in the allocation-off and automatic runs."""
    torch.manual_seed(92730 + rank)
    shapes = ((2, _HIDDEN, 2 * _INTERMEDIATE), (2, _INTERMEDIATE, _HIDDEN),
              (2 * _SHARED, _HIDDEN), (_HIDDEN, _SHARED))
    scales = (_HIDDEN, _INTERMEDIATE, _HIDDEN, _SHARED)
    return tuple(torch.randn(shape, device=device, dtype=torch.bfloat16) / math.sqrt(scale)
                 for shape, scale in zip(shapes, scales))


def _expected_owned(record, slot):
    """Ownership requires final backward and no global or slot-specific hooks."""
    return (not record["keep_graph"] and not record["saved_hooks"]
            and not (record["late_hook"] and record["late_slot"] == slot))


def _assert_observed_ownership(record):
    """Check both independent saved-storage claims at the actual native boundary."""
    assert not record["grad_enabled"], record
    reusable = _expected_owned(record, 1)
    assert record["up_proj_reusable"] == reusable, record
    assert record["alias"] == bool(record["enabled"] and reusable and record["elements"]), record
    snapshot_owned = record["snapshot_written"] and _expected_owned(record, 16)
    assert record["snapshot_reusable"] == snapshot_owned, record
    if record["snapshot_written"]:
        assert record["snapshot_alias"] == snapshot_owned, record


@contextmanager
def _observe_backward(records, phase, enabled):
    """Record scalar ownership evidence without retaining any intermediate Tensor."""
    prepare = fusion._prepare_backward_buffers

    def _observe(plan, workspace, saved, route_state, grad_output):
        """Inspect actual engine state and allocated buffers at the native boundary."""
        buffers = prepare(plan, workspace, saved, route_state, grad_output)
        up, gradient = saved.up_proj, buffers.swiglu_grad
        keep_graph = bool(torch._C._autograd._get_current_graph_task_keep_graph())
        reusable = saved.up_proj_reusable is True
        alias = bool(up.numel() and up.data_ptr() == gradient.data_ptr())
        snapshot_written = (plan.spec.backward_permute_fused and plan.spec.backward_unpermute_fused
                            and not plan.spec.backward_shared_expert)
        snapshot_alias = bool(saved.expert_output.numel()
                              and saved.expert_output.data_ptr() == route_state.grad_x.data_ptr())
        records.append({
            "phase": phase["name"], "enabled": enabled, "keep_graph": keep_graph,
            "saved_hooks": phase["hooks"], "late_hook": phase["late_hook"], "late_slot": phase["late_slot"],
            "up_proj_reusable": reusable, "snapshot_written": snapshot_written,
            "snapshot_reusable": saved.expert_output_reusable, "snapshot_alias": snapshot_alias,
            "grad_enabled": torch.is_grad_enabled(), "alias": alias,
            "tokens": plan.spec.local_num_tokens, "receive_rows": saved.dispatch.shape[0],
            "up_pointer": int(up.data_ptr()), "scratch_pointer": int(gradient.data_ptr()),
            "up_storage_pointer": int(up.untyped_storage().data_ptr()),
            "scratch_storage_pointer": int(gradient.untyped_storage().data_ptr()),
            "up_offset": up.storage_offset(), "scratch_offset": gradient.storage_offset(),
            "elements": up.numel(), "bytes": up.numel() * up.element_size(),
        })
        assert tuple(up.shape) == tuple(gradient.shape)
        _assert_observed_ownership(records[-1])
        return buffers

    with ExitStack() as stack:
        if not enabled:
            stack.enter_context(patch.object(fusion, "_reusable_swiglu_input", return_value=None))
        stack.enter_context(patch.object(fusion, "_prepare_backward_buffers", side_effect=_observe))
        yield


def _invocation(layer, batch, weights):
    """Construct a fresh graph, leaving automatic token metadata to the public API."""
    _, source, ids, probabilities, _, _ = batch
    inputs = (source.detach().clone().requires_grad_(), ids,
              probabilities.detach().clone().requires_grad_(), None, weights)
    return inputs, _forward(layer, *inputs)


def _backward(invocation, batch, phase, label, retained=False, *, hooked=False, late_hook=False, late_slot=1):
    """Use ordinary backward and copy every public result away from reused NPU storage."""
    inputs, output = invocation
    variables = (inputs[0], inputs[2], *inputs[4])
    for value in variables:
        value.grad = None
    phase["name"] = label
    phase["hooks"] = hooked
    phase["late_hook"] = late_hook
    phase["late_slot"] = late_slot
    output.backward(batch[4], retain_graph=retained)
    torch.npu.synchronize()
    assert all(value.grad is not None for value in variables), f"missing gradient: {label}"
    actual = tuple(value.detach().cpu().clone() for value in (output, *(item.grad for item in variables)))
    assert all(torch.isfinite(value).all() for value in actual), f"non-finite result: {label}"
    _check_empty_experts(actual, batch[5])
    return actual


def _assert_exact(actual, expected, label):
    """Storage reuse cannot change any output or gradient value."""
    for name, observed, reference in zip(_NAMES, actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0,
                                   msg=lambda message: f"{label} {name}: {message}")


def _check_lifecycle(layer, batches, weights, phase, expected):
    """Switch a retained graph to final backward and consume two pending graphs in reverse."""
    batch = batches[0]
    invocation = _invocation(layer, batch, weights)
    for index, retained in enumerate((True, True, False)):
        actual = _backward(invocation, batch, phase, f"retained_{index}", retained)
        _assert_exact(actual, expected[batch[0]], f"retained_{index}")
    pending = [(batch, _invocation(layer, batch, weights)) for batch in (batches[0], batches[2])]
    snapshots = [invocation[1].detach().cpu().clone() for _, invocation in pending]
    for index in (1, 0):
        batch, invocation = pending[index]
        actual = _backward(invocation, batch, phase, f"pending_{index}")
        _assert_exact(actual, expected[batch[0]], f"pending_{index}")
        for (_, old), snapshot in zip(pending, snapshots):
            torch.testing.assert_close(old[1].detach().cpu(), snapshot, rtol=0, atol=0)


def _mega_moe_node(output):
    """Find the actual custom Function below the public output reshape."""
    pending, seen = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        if type(node).__name__ == "_MegaMoeFunctionBackward":
            return node
        seen.add(node)
        pending.extend(child for child, _ in node.next_functions)
    raise AssertionError("public MegaMoe output has no saved-state autograd node")


def _check_saved_hooks(layer, batch, weights, phase, expected, *, late=False, slot=1):
    """Protect externally cached tensors from hooks installed during or after forward."""
    cache = []

    def _pack(tensor):
        """Cache actual tensors and independent values to detect writes through saved aliases."""
        cache.append((tensor, tensor.detach().cpu().clone()))
        return tensor

    if late:
        invocation = _invocation(layer, batch, weights)
        _mega_moe_node(invocation[1])._raw_saved_tensors[slot].register_hooks(_pack, lambda tensor: tensor)
    else:
        with torch.autograd.graph.saved_tensors_hooks(_pack, lambda tensor: tensor):
            invocation = _invocation(layer, batch, weights)
    inputs, _ = invocation
    leaves = (inputs[0], inputs[2], *inputs[4])
    snapshots = tuple(value.detach().cpu().clone() for value in leaves)
    label = "late_hook" if late else "saved_hooks"
    actual = _backward(invocation, batch, phase, label, hooked=not late, late_hook=late, late_slot=slot)
    _assert_exact(actual, expected, label)
    assert cache, "saved_tensors_hooks did not observe any forward state"
    for tensor, snapshot in (*cache, *zip(leaves, snapshots)):
        torch.testing.assert_close(tensor.detach().cpu(), snapshot, rtol=0, atol=0)


def _exercise(mode, batches, parameters, enabled, records, expected=None):
    """Change only the private reuse policy; schedules, tensors and kernels stay identical."""
    layer = MegaMoeExperts(
        hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE, num_experts=4, top_k=2,
        ep_size=2, ep_group=dist.group.WORLD, dispatch_mode=mode, create_parameters=False,
        shared_intermediate_size=_SHARED, fusion_mode="full",
    )
    weights = tuple(value.detach().clone().requires_grad_() for value in parameters)
    phase, results = {}, {}
    try:
        with _observe_backward(records, phase, enabled):
            for batch in batches:
                actual = _backward(_invocation(layer, batch, weights), batch, phase, batch[0])
                results[batch[0]] = actual
                if expected is not None:
                    _assert_exact(actual, expected[batch[0]], batch[0])
            _check_saved_hooks(layer, batches[0], weights, phase, results[batches[0][0]])
            _check_saved_hooks(layer, batches[0], weights, phase, results[batches[0][0]], late=True)
            if enabled:
                _check_lifecycle(layer, batches, weights, phase, expected)
        return results
    finally:
        layer.close()


def _snapshot_summary(records):
    """Require two slot16 reuse hits and four protected snapshot fallbacks."""
    snapshots = [record for record in records if record["snapshot_written"]]
    hits = sum(record["snapshot_alias"] for record in snapshots)
    assert len(snapshots) == 6 and hits == 2, snapshots
    return {"snapshot_reuse_hits": hits, "snapshot_fallbacks": len(snapshots) - hits}


def _reuse_summary(records):
    """Require actual ordinary hits and separate retained/global/late-hook fallbacks."""
    result = _snapshot_summary(records)
    records = [record for record in records if not record["snapshot_written"]]
    hits = sum(record["alias"] for record in records)
    retained = [record for record in records if record["keep_graph"]]
    assert hits > 0 and len(retained) == 2 and not any(record["alias"] for record in retained)
    result.update({"ordinary_reuse_hits": hits, "retained_fallbacks": len(retained)})
    for key, field in (("saved_hooks", "saved_hook_fallbacks"), ("late_hook", "late_hook_fallbacks")):
        hooked = [record for record in records if record[key]]
        assert len(hooked) == 2 and all(not record["keep_graph"] and not record["alias"] for record in hooked)
        result[field] = len(hooked)
    return result


def _exercise_snapshot_hooks(mode, batch, parameters, records):
    """Protect routed slot16 when mixed fusion writes dX into its owned snapshot."""
    layer = MegaMoeExperts(
        hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE, num_experts=4, top_k=2,
        ep_size=2, ep_group=dist.group.WORLD, dispatch_mode=mode, create_parameters=False,
        shared_intermediate_size=_SHARED, forward_fusion_mode="both", backward_fusion_mode="both",
        forward_shared_expert=True, backward_shared_expert=False,
    )
    weights = tuple(value.detach().clone().requires_grad_() for value in parameters)
    phase = {}
    try:
        with _observe_backward(records, phase, True):
            expected = _backward(_invocation(layer, batch, weights), batch, phase, "snapshot_ordinary")
            invocation = _invocation(layer, batch, weights)
            for index, retained in enumerate((True, True, False)):
                actual = _backward(invocation, batch, phase, f"snapshot_retained_{index}", retained)
                _assert_exact(actual, expected, f"snapshot_retained_{index}")
            _check_saved_hooks(layer, batch, weights, phase, expected)
            _check_saved_hooks(layer, batch, weights, phase, expected, late=True, slot=16)
    finally:
        layer.close()


def test_mega_moe_backward_memory():
    """Compare ordinary reuse with the same-source fallback and prove retained safety."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise ValueError("backward memory regression requires two ranks")
    device = torch.device("npu", rank)
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    records = []
    try:
        batches, parameters = _batches(rank, device), _weights(rank, device)
        expected = _exercise(mode, batches, parameters, False, records)
        _exercise(mode, batches, parameters, True, records, expected)
        _exercise_snapshot_hooks(mode, batches[0], parameters, records)
        write_evidence({"rank": dist.get_rank(), "passed": True, "dispatch_mode": mode,
                        **_reuse_summary(records),
                        "cases": _CASES, "launches": records})
    finally:
        dist.destroy_process_group()

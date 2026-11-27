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
"""Device regression for local-capacity buffers with outstanding forwards."""

import gc
import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_multicore import MegaMoeExperts, shmem
from hyper_multicore.diagnostics import profiler as multicore_profiler, schedule
from hyper_multicore.torch import ops
from tests.torch import _test_mega_moe as baseline
from tests.common.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


def _check_strided_layer(layer: MegaMoeExperts, common: torch.nn.Module, shape: baseline.MoeShape) -> None:
    """Compare strided inputs and external-weight views, including original-parameter updates."""
    source, upstream = baseline.make_data(shape)
    ids, probs, counts = baseline.make_balanced_route(shape)
    gate_up_base = layer.gate_up_weight.detach().transpose(-1, -2).contiguous().requires_grad_(True)
    down_base = torch.zeros((shape.local_experts, shape.intermediate_size, shape.hidden_size * 2),
                            dtype=source.dtype, device=source.device)
    down_base[:, :, ::2].copy_(layer.down_weight.detach())
    down_base.requires_grad_(True)
    weight1, weight2 = gate_up_base.transpose(-1, -2), down_base[:, :, ::2]
    assert not weight1.is_contiguous() and not weight2.is_contiguous(), "external weights must exercise views"
    expected = baseline.run_layer(common, source, ids, probs, counts, upstream)
    storage = torch.empty((shape.local_num_tokens, shape.hidden_size * 2), dtype=source.dtype, device=source.device)
    storage[:, ::2].copy_(source)
    hidden = storage[:, ::2].detach().requires_grad_(True)
    route_weights = probs.detach().requires_grad_(True)
    actual = layer(hidden, ids, route_weights, tokens_per_expert=counts, expert_weights=(weight1, weight2))
    actual.backward(upstream)
    torch.optim.SGD((gate_up_base, down_base), lr=1e-3).step()
    torch.npu.synchronize()
    grad1, grad2 = gate_up_base.grad.transpose(-1, -2), down_base.grad[:, :, ::2]
    assert layer.gate_up_weight.grad is None and layer.down_weight.grad is None, "owned weights must remain unused"
    assert torch.count_nonzero(down_base.grad[:, :, 1::2]).item() == 0, "unused slice entries received gradients"
    assert torch.count_nonzero(down_base[:, :, 1::2]).item() == 0, "unused slice entries changed during SGD"
    pairs = (
        ("output", actual, expected.output),
        ("input gradient", hidden.grad, expected.input_grad),
        ("route gradient", route_weights.grad, expected.route_weight_grad),
        ("gate/up gradient", grad1, expected.gate_up_weight_grad),
        ("down gradient", grad2, expected.down_weight_grad),
        ("gate/up update", weight1, expected.gate_up_weight),
        ("down update", weight2, expected.down_weight),
    )
    for name, observed, reference in pairs:
        baseline.assert_close(f"strided {name}", observed, reference)


def test_mega_moe_model_replacement_and_layouts() -> None:
    """Feature: Model replacement and caller tensor layouts.

    Description: Replace a collected model with a larger push or pull model and use strided activations and weights.
    Expectation: Retained heaps grow and all outputs, original-parameter gradients and updates match common EP.
    """
    os.environ.pop("HYPER_MULTICORE_SHMEM_HEAP_SIZE", None)
    records = []
    for mode in ("push", "pull"):
        start_shmem_lifetime()
        shape = baseline.MoeShape()
        small, common = baseline.new_layers(shape, dispatch_mode=mode)
        source, upstream = baseline.make_data(shape)
        ids, probs, counts = baseline.make_balanced_route(shape)
        baseline.run_layer(small, source, ids, probs, counts, upstream)
        # Manager identity is an internal ownership invariant across model replacement.
        old_resources = small._resource_group.resources  # pylint: disable=protected-access
        old_bytes = shmem.debug_state()["config"]["heap_size_bytes"]
        del small, common
        gc.collect()
        larger_shape = replace(shape, local_num_tokens=512, hidden_size=1024)
        larger, reference = baseline.new_layers(larger_shape, dispatch_mode=mode)
        try:
            _check_strided_layer(larger, reference, larger_shape)
            resources = larger._resource_group.resources  # pylint: disable=protected-access
            state = shmem.debug_state()
            assert resources.heap_manager is old_resources.heap_manager, f"rank={baseline.RANK}: manager replaced"
            assert resources.heap_manager.epoch == 1, f"rank={baseline.RANK}: epoch={resources.heap_manager.epoch}"
            assert state["config"]["heap_size_bytes"] > old_bytes, f"rank={baseline.RANK}: heap did not grow"
            records.append({"mode": mode, "old_heap_bytes": old_bytes,
                            "heap_bytes": state["config"]["heap_size_bytes"], "epoch": 1,
                            "strided_activations_match": True, "strided_weights_match": True,
                            "original_parameter_gradients_and_updates_match": True,
                            "unused_slice_entries_unchanged": True})
        finally:
            larger.close()
        assert shmem.debug_state()["state"] == "Uninitialized", f"rank={baseline.RANK}: runtime stayed active"
    write_evidence({"model_replacement_and_layouts": records})


def _route(shape: baseline.MoeShape, pattern: str) -> tuple:
    """Produce exact counts with odd tails or a completely empty destination."""
    positions = torch.arange(shape.local_num_tokens * shape.top_k).reshape(-1, shape.top_k)
    ids = (positions + baseline.RANK * shape.top_k).remainder(shape.num_experts)
    if pattern == "tail" and baseline.RANK == 0:
        ids[0, 0] = shape.local_experts
    elif pattern.startswith("destination"):
        destination = int(pattern[-1])
        ids = positions.remainder(shape.local_experts) + destination * shape.local_experts
    counts = torch.bincount(ids.reshape(-1), minlength=shape.num_experts).to(torch.int32)
    weights = torch.full(ids.shape, 1.0 / shape.top_k, dtype=torch.float32, device=baseline.DEVICE)
    return ids.to(device=baseline.DEVICE, dtype=torch.int32), weights, counts.to(baseline.DEVICE)


def _deferred_forwards(layers: list, hidden: list, routes: list) -> list:
    """Submit both forwards, optionally with checkpoint replay on alternating streams."""
    outputs = []
    replay = os.getenv("HP_MEGA_MOE_GROWTH_CASE") == "checkpoint" and isinstance(layers[0], MegaMoeExperts)
    streams = [torch.npu.Stream() for _ in range(2)] if replay else [torch.npu.current_stream()] * 2
    default_stream = torch.npu.current_stream()
    for member, tensor, (ids, weights, counts), stream in zip((layers * 2)[:2], hidden, routes, streams):
        with torch.npu.stream(stream):
            stream.wait_stream(default_stream)
            if replay:
                output = checkpoint(baseline.forward_layer, member, tensor, ids, weights,
                                    tokens_per_expert=counts, use_reentrant=False)
            else:
                output = baseline.forward_layer(member, tensor, ids, weights, tokens_per_expert=counts)
        default_stream.wait_stream(stream)
        outputs.append(output)
    return outputs


def _deferred_pair(layer: torch.nn.Module, shape: baseline.MoeShape, patterns: tuple) -> list:
    """Run two routes before reverse backward without changing the weights."""
    layers = layer if isinstance(layer, list) else [layer]
    for member in layers:
        member.zero_grad(set_to_none=True)
    source, upstream = baseline.make_data(shape)
    hidden = [(source * (1.0 + index / 8.0)).detach().requires_grad_(True) for index in range(2)]
    routes = [_route(shape, pattern) for pattern in patterns]
    route_weights = [weights.requires_grad_(True) for _, weights, _ in routes]
    outputs = _deferred_forwards(layers, hidden, routes)
    for output in reversed(outputs):
        output.backward(upstream, retain_graph=True)
        output.backward(upstream)
    torch.npu.synchronize()
    compared = outputs + [tensor.grad for tensor in hidden + route_weights]
    for member in layers:
        compared.extend(baseline.expert_weight_gradients(member))
    for index, tensor in enumerate(compared):
        assert tensor is not None, f"rank={baseline.RANK}: missing memory regression tensor {index}."
        baseline.assert_finite(f"memory regression tensor {index}", tensor)
    return [tensor.detach().cpu() for tensor in compared]


def test_mega_moe_local_capacity_lifetime() -> None:
    """Feature: Transport storage lifetime and conservative receive-buffer reuse.

    Description: Compare retained backward, hotspot/tail routes and profiling with the common backend.
    Expectation: All gradients match with reuse on/off, and pull fits a source-sized heap.
    """
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=1024)
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    results = []
    factors = (None, 1.5) if mode == "push" else (None,)
    for factor in factors:
        start_shmem_lifetime()
        mega, common = baseline.new_layers(shape, initial_capacity_factor=factor)
        resources = mega._get_execution_resources(baseline.make_data(shape)[0])  # pylint: disable=protected-access
        select_plan = resources.plan_for_tokens
        original = ops.mega_moe_grad_with_diagnostic_buffer
        pairs = [("balanced", "tail"), ("tail", "balanced")]
        if factor is None:
            pairs.extend([("destination0", "destination1"), ("destination1", "tail")])
        try:
            for reuse in (False, True):
                def _select_plan(tokens, device, expected_reuse=reuse, original_plan=select_plan):
                    """Exercise both reuse policies with the actual cached automatic plan."""
                    plan = original_plan(tokens, device)
                    assert plan.reuse_backward_dispatch
                    return replace(plan, reuse_backward_dispatch=expected_reuse)

                def _backward(*args, expected_reuse=reuse, kernel=original):
                    """Check receive-buffer ownership at the native boundary."""
                    assert (args[0].data_ptr() == args[12].data_ptr()) == expected_reuse
                    assert args[17].data_ptr() != args[0].data_ptr()
                    return kernel(*args)

                with patch.object(resources, "plan_for_tokens", side_effect=_select_plan), patch.object(
                    ops, "mega_moe_grad_with_diagnostic_buffer", side_effect=_backward,
                ), patch.object(
                    ops, "mega_moe_grad", side_effect=partial(_backward, kernel=ops.mega_moe_grad),
                ):
                    for patterns in pairs:
                        expected = _deferred_pair(common, shape, patterns)
                        capture = factor is None and patterns == pairs[0]
                        profile_context = multicore_profiler.profile(
                            schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
                        ) if capture else nullcontext()
                        with profile_context as profiler:
                            actual = _deferred_pair(mega, shape, patterns)
                            if profiler is not None:
                                profiler.step()
                        if profiler is not None:
                            trace = profiler.export_chrome_trace(
                                Path(os.environ["HP_MEGA_MOE_LEVEL1_RESULT"]).with_name(
                                    f"{mode}_reuse{reuse}_rank{baseline.RANK}_trace.json"))
                            metadata = trace["megaKernelCycleTrace"]
                            assert metadata["invocationCount"] == 6, (
                                f"rank={baseline.RANK}: expected 6 invocations, got {metadata}"
                            )
                            assert metadata["droppedRecordCount"] == 0, (
                                f"rank={baseline.RANK}: expected no dropped records, got {metadata}"
                            )
                        for index, (reference, observed) in enumerate(zip(expected, actual)):
                            baseline.assert_close(f"capacity patterns={patterns} tensor={index}", observed, reference)
                        results.append({"factor": factor, "reuse": reuse, "patterns": patterns,
                                        "profiled": capture, "match": True})
            if mode == "pull":
                data_bytes = shape.local_num_tokens * shape.top_k * shape.hidden_size * 2
                assert 2 * data_bytes < shmem.debug_state()["config"]["heap_size_bytes"] < 3 * data_bytes
                assert resources.workspace.expert_buffer is None
        finally:
            mega.close()
    write_evidence({"mode": mode, "local_capacity_lifetime": results})


def _check_unknown_owner(mega: MegaMoeExperts, shape: baseline.MoeShape) -> None:
    """Converge one rank's unmanaged reference and preserve every old buffer."""
    # Force the private lazy bind so the failure probe has live transport buffers.
    hidden = baseline.make_data(shape)[0]
    # pylint: disable-next=protected-access
    resources = mega._get_execution_resources(hidden)
    resources.prepare_auto_workspace(shape.local_num_tokens, shape.local_num_tokens * shape.top_k, hidden)
    before = resources.workspace.expert_buffer
    if baseline.RANK == 0:
        shmem.acquire()
    try:
        resources.ensure_capacity(shape.local_num_tokens * shape.top_k + 1)
    except RuntimeError as error:
        assert "one healthy Runtime user" in str(error), error
    else:
        raise AssertionError("Unknown owner did not prevent growth")
    finally:
        if baseline.RANK == 0:
            shmem.release()
    assert resources.workspace.expert_buffer is before
    assert before.untyped_storage().nbytes() > 0



def test_mega_moe_heap_growth() -> None:
    """Feature: Online push heap growth.

    Description: Grow between two forwards, then reverse and repeat both backwards.
    Expectation: All outputs and gradients match common EP; stale storage is invalidated.
    """
    os.environ.pop("HYPER_MULTICORE_SHMEM_HEAP_SIZE", None)
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=1024)
    mega, common = baseline.new_layers(shape, initial_capacity_factor=1.0, capacity_growth_factor=2.0)
    variant = os.getenv("HP_MEGA_MOE_GROWTH_CASE", "single")
    managed, references = [mega], [common]
    if variant == "checkpoint":
        second, reference = baseline.new_layers(shape, initial_capacity_factor=1.0, capacity_growth_factor=2.0)
        managed.append(second)
        references.append(reference)
        MegaMoeExperts.share_execution_resources(managed)
    observations = []
    old_buffers = []

    def observe(module: torch.nn.Module, unused_args: tuple, unused_output: torch.Tensor) -> None:
        """Record heap epochs and retain old storage to check invalidation.

        Args:
            module: Layer invoking the hook.
            unused_args: Forward positional inputs.
            unused_output: Completed forward output.
        """
        del unused_args, unused_output
        resources = module._resource_group.resources  # pylint: disable=protected-access
        observations.append((resources.heap_manager.epoch, resources.workspace.expert_capacity,
                             shmem.debug_state()["config"]["heap_size_bytes"]))
        old_buffers.append(resources.workspace.expert_buffer)

    handles = [member.register_forward_hook(observe) for member in managed]
    try:
        if variant == "single":
            _check_unknown_owner(mega, shape)
        for patterns in (("balanced", "destination0"), ("destination1", "balanced")):
            expected = _deferred_pair(references, shape, patterns)
            actual = _deferred_pair(managed, shape, patterns)
            for index, (reference, observed) in enumerate(zip(expected, actual)):
                baseline.assert_close(f"grow patterns={patterns} tensor={index}", observed, reference)
        epochs = [item[0] for item in observations]
        initial_epoch = epochs[0]
        assert epochs[:2] == [initial_epoch, initial_epoch + 1], observations
        assert all(epoch == initial_epoch + 1 for epoch in epochs[2:]), observations
        assert observations[1][2] > observations[0][2], observations
        assert old_buffers[0].untyped_storage().nbytes() == 0
        # The process Heap Manager owns the sole SHMEM Runtime reference.
        reference_count = shmem.debug_state()["reference_count"]
        assert reference_count == 1, reference_count
        resources = mega._resource_group.resources  # pylint: disable=protected-access
        write_evidence({"variant": variant, "observations": observations,
                        "growth": resources.heap_manager.growth_records, "reference_count": reference_count})
    finally:
        for handle in handles:
            handle.remove()
        for member in managed:
            member.close()
    assert shmem.debug_state()["state"] == "Uninitialized"

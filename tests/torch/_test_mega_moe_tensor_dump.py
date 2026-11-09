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
"""Two-rank Torch worker for one MegaMoe forward L0 Tensor dump."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts
from hyper_multicore.diagnostics import schedule
from hyper_multicore.diagnostics import dump as tensor_dump
from tests.torch import _test_mega_moe as common
from tests.torch._test_mega_moe_shared import _route_ids


_DUMP_TENSORS = (
    "target",
    "src",
    "up_proj_weight",
    "up_proj_glist",
    "up_proj_y",
    "swiglu_out",
    "down_proj_weight",
    "down_proj_glist",
    "down_proj_y",
    "combine_out",
)


def _build_layer(shape: common.MoeShape) -> MegaMoeExperts:
    """Construct the small representative two-rank MegaMoe layer."""
    return MegaMoeExperts(
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=common.DEVICE, dtype=torch.bfloat16)


def test_mega_moe_forward_tensor_dump_worker() -> None:
    """Feature: MegaKernel diagnostics.
    Description: Warm up, synchronize ranks and dump one small forward invocation.
    Expectation: Mega moe forward tensor dump worker.
    """
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    layer = _build_layer(shape)
    result_dir = os.getenv("HP_MEGA_MOE_DUMP_RESULT_DIR")
    if not result_dir:
        raise ValueError("HP_MEGA_MOE_DUMP_RESULT_DIR must name the dump output directory")
    config = tensor_dump.TensorDumpConfig(
        result_dir, kernel_names="mega_moe", tensor_names=_DUMP_TENSORS, directions="forward", ranks=0,
    )
    capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
    try:
        with torch.no_grad():
            layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.npu.synchronize()
            dist.barrier()
            with tensor_dump.tensor_dump(config, schedule=capture_schedule) as dumper:
                layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                dumper.step()
        if common.RANK == 0:
            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tensors = manifest["invocations"][0]["tensors"]
            names = {item["name"] for item in tensors}
            assert names == set(_DUMP_TENSORS), (
                f"unexpected dump tensors: expected={set(_DUMP_TENSORS)}, got={names}"
            )
            assert manifest["schemaVersion"] == 4, (
                f"unexpected manifest schema: expected=4, got={manifest['schemaVersion']}"
            )
            assert manifest["tensorSerialization"] == "torch.save", (
                f"unexpected Tensor serialization: {manifest['tensorSerialization']}"
            )
            assert manifest["tensorFileLayout"] == "stage_only", (
                f"unexpected Tensor file layout: {manifest['tensorFileLayout']}"
            )
            assert manifest["invocations"][0]["kernelName"] == "mega_moe", (
                f"unexpected kernel name: {manifest['invocations'][0]['kernelName']}"
            )
            swiglu_ports = [
                (port["stage"], port["io"], port["index"])
                for item in tensors
                if item["name"] == "swiglu_out"
                for port in item["ports"]
            ]
            expected_swiglu_ports = [("SwiGLU", "output", 0), ("GMM2", "input", 0)]
            assert swiglu_ports == expected_swiglu_ports, (
                f"unexpected swiglu ports: expected={expected_swiglu_ports}, got={swiglu_ports}"
            )
            invocation_dir = manifest_path.parent / "invocation_000000_forward"
            assert not (invocation_dir / "before").exists(), f"unexpected before directory: {invocation_dir}"
            assert not (invocation_dir / "after").exists(), f"unexpected after directory: {invocation_dir}"
            for metadata in tensors:
                primary_path = manifest_path.parent / metadata["file"]
                assert metadata["file"] == metadata["ports"][0]["file"], (
                    f"primary file does not reference the first stage port: metadata={metadata}"
                )
                for port in metadata["ports"]:
                    stage_path = manifest_path.parent / port["file"]
                    assert stage_path.is_file(), f"missing stage file: path={stage_path}"
                    assert primary_path.samefile(stage_path), (
                        f"stage ports do not share storage: primary={primary_path}, stage={stage_path}"
                    )
                payload = torch.load(primary_path, map_location="cpu", weights_only=True)
                assert list(payload.shape) == metadata["actualShape"], (
                    f"shape mismatch for {metadata['name']}: "
                    f"expected={metadata['actualShape']}, got={list(payload.shape)}"
                )
                assert torch.isfinite(payload).all().item(), (
                    f"non-finite dump for {metadata['name']}: payload={payload}"
                )
                assert str(payload.dtype).removeprefix("torch.") == metadata["storedDtype"], (
                    f"dtype mismatch for {metadata['name']}: "
                    f"expected={metadata['storedDtype']}, got={payload.dtype}"
                )
            print(f"MEGA_KERNEL_TENSOR_DUMP={manifest_path}")
    finally:
        layer.close()
    dist.barrier()


def test_mega_moe_forward_l1_tensor_dump_worker() -> None:
    """Feature: MegaKernel diagnostics.
    Description: Dump task-local SwiGLU inputs and outputs from one forward invocation.
    Expectation: Mega moe forward l1 tensor dump worker.
    """
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    layer = _build_layer(shape)
    result_dir = os.getenv("HP_MEGA_MOE_DUMP_RESULT_DIR")
    if not result_dir:
        raise ValueError("HP_MEGA_MOE_DUMP_RESULT_DIR must name the dump output directory")
    config = tensor_dump.TensorDumpConfig(
        result_dir,
        level="L1",
        kernel_names="mega_moe",
        stages="SwiGLU",
        directions="forward",
        ranks=0,
        io=("input", "output"),
        device_memory_budget_mb=32,
    )
    capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
    try:
        with torch.no_grad():
            layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.npu.synchronize()
            dist.barrier()
            with tensor_dump.tensor_dump(config, schedule=capture_schedule) as dumper:
                layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                dumper.step()
        if common.RANK == 0:
            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            tasks = invocation["tasks"]
            assert manifest["level"] == "L1", f"unexpected dump level: {manifest['level']}"
            assert manifest["schemaVersion"] == 4, (
                f"unexpected manifest schema: expected=4, got={manifest['schemaVersion']}"
            )
            assert manifest["tensorSerialization"] == "torch.save", (
                f"unexpected Tensor serialization: {manifest['tensorSerialization']}"
            )
            assert manifest["tensorFileLayout"] == "stage_only", (
                f"unexpected Tensor file layout: {manifest['tensorFileLayout']}"
            )
            assert manifest["capturePhaseSemantics"] == "task_boundary", (
                f"unexpected capture semantics: {manifest['capturePhaseSemantics']}"
            )
            assert tasks, "L1 dump did not export any SwiGLU tasks"
            assert len({task["taskId"] for task in tasks}) == len(tasks), (
                f"L1 task IDs are not unique: tasks={tasks}"
            )
            for task in tasks:
                assert task["stage"] == "SwiGLU", f"unexpected L1 stage: task={task}"
                assert task["coreType"] == "AIV", f"unexpected L1 core type: task={task}"
                assert (manifest_path.parent / task["metadataFile"]).is_file(), (
                    f"missing task metadata: task={task}"
                )
                tensors = {(item["io"], item["index"]): item for item in task["tensors"]}
                assert set(tensors) == {("input", 0), ("output", 0)}, (
                    f"unexpected SwiGLU task tensors: tensors={tensors}"
                )
                task_values = {}
                for (io, _), metadata in tensors.items():
                    payload = torch.load(
                        manifest_path.parent / metadata["file"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    task_values[io] = payload
                    assert list(payload.shape) == metadata["storedShape"], (
                        f"stored shape mismatch: metadata={metadata}, shape={payload.shape}"
                    )
                    assert torch.isfinite(payload).all().item(), (
                        f"non-finite L1 payload: metadata={metadata}, payload={payload}"
                    )
                    assert str(payload.dtype).removeprefix("torch.") == metadata["storedDtype"], (
                        f"stored dtype mismatch: metadata={metadata}, dtype={payload.dtype}"
                    )
                    expected_width = shape.intermediate_size * (2 if io == "input" else 1)
                    assert payload.shape[1] == expected_width, (
                        f"unexpected SwiGLU {io} width: expected={expected_width}, got={payload.shape[1]}"
                    )
                gate, up = task_values["input"].float().chunk(2, dim=-1)
                torch.testing.assert_close(task_values["output"].float(), torch.nn.functional.silu(gate) * up,
                                           rtol=2e-2, atol=5e-3,
                                           msg=f"rank={common.RANK}: incorrect L1 SwiGLU task snapshot: {task}")
            print(f"MEGA_KERNEL_L1_TENSOR_DUMP={manifest_path}")
    finally:
        layer.close()
    dist.barrier()


def _read_shared_tail_dump(path: Path, tokens: int) -> dict:
    """Require exact one-row tail records in both directions and load their values."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["level"] == "L1"
    assert {value["direction"] for value in manifest["invocations"]} == {"forward", "backward"}
    result = {}
    for invocation in manifest["invocations"]:
        for task in invocation["tasks"]:
            assert task["stageTaskIndex"] == (tokens - 1) // 128
            values = {}
            for item in task["tensors"]:
                assert item["actualShape"][0] == item["storedShape"][0] == 1, item
                assert item["sourceOffsetBytes"] == (tokens - 1) * item["actualShape"][1] * 2, item
                assert not item["truncated"], item
                tensor = torch.load(path.parent / item["file"], map_location="cpu", weights_only=True)
                assert torch.isfinite(tensor).all(), item
                values[item["io"], item["index"]] = tensor
            result[task["stage"]] = values
    assert set(result) == {"shared_swiglu", "shared_swiglu_grad", "shared_dx_add"}, result.keys()
    return result


def _check_shared_tail_values(values: dict, expected_up: torch.Tensor, expected_dx: torch.Tensor) -> None:
    """Check captured intermediates and the exact final shared residual gradient."""
    forward = values["shared_swiglu"]
    torch.testing.assert_close(forward["input", 0], expected_up, rtol=2e-2, atol=2e-3)
    gate, up = forward["input", 0].chunk(2, dim=-1)
    torch.testing.assert_close(forward["output", 0], torch.nn.functional.silu(gate) * up,
                               rtol=2e-2, atol=2e-3)
    backward = values["shared_swiglu_grad"]
    torch.testing.assert_close(backward["input", 1], forward["input", 0], rtol=0, atol=0)
    packed = backward["input", 1].detach().requires_grad_()
    gate, up = packed.chunk(2, dim=-1)
    expected = torch.autograd.grad(torch.nn.functional.silu(gate) * up, packed, backward["input", 0])[0]
    torch.testing.assert_close(backward["output", 0], expected, rtol=5e-2, atol=1e-2)
    residual = values["shared_dx_add"]
    summed = (residual["input", 0].float() + residual["input", 1].float()).bfloat16()
    torch.testing.assert_close(residual["output", 0], summed, rtol=0, atol=0)
    torch.testing.assert_close(residual["output", 0], expected_dx, rtol=0, atol=0)


def test_mega_moe_shared_l1_tail_worker() -> None:
    """Capture full-fusion shared F+B tails for unequal rank lengths 129/257."""
    tokens = (129, 257)[common.RANK]
    torch.manual_seed(92700 + common.RANK)
    source = torch.randn(tokens, 5120, device=common.DEVICE, dtype=torch.bfloat16).requires_grad_()
    probabilities = torch.full((tokens, 2), 0.5, device=common.DEVICE, requires_grad=True)
    ids = _route_ids(tokens, 4, 2, common.DEVICE, "balanced")
    layer = MegaMoeExperts(
        hidden_size=5120, intermediate_size=128, num_experts=4, top_k=2,
        shared_intermediate_size=256, ep_size=2, ep_group=dist.group.WORLD, fusion_mode="full",
        dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
    ).to(device=common.DEVICE, dtype=torch.bfloat16)
    upstream = torch.randn_like(source) / (5120 ** 0.5)
    variables = (source, probabilities, layer.gate_up_weight, layer.down_weight,
                 layer.shared_gate_up_weight, layer.shared_down_weight)
    selection = tensor_dump.TensorDumpConfig(
        os.environ["HP_MEGA_MOE_DUMP_RESULT_DIR"], level="L1", kernel_names="mega_moe",
        stages=("shared_swiglu", "shared_swiglu_grad", "shared_dx_add"),
        stage_task_indices=(tokens - 1) // 128, device_memory_budget_mb=32,
    )
    try:
        output = layer(source, ids, probabilities)
        gradients = torch.autograd.grad(output, variables, upstream)
        expected = tuple(value.detach().cpu().clone() for value in (output, *gradients))
        with torch.no_grad():
            expected_up = torch.mm(source, layer.shared_gate_up_weight.t())[-1:].cpu()
        torch.npu.synchronize()
        dist.barrier()
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
        with tensor_dump.tensor_dump(selection, schedule=capture_schedule) as dumper:
            observed = layer(source, ids, probabilities)
            actual_gradients = torch.autograd.grad(observed, variables, upstream)
            torch.npu.synchronize()
            dumper.step()
        for actual, reference in zip((observed, *actual_gradients), expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        manifest = dumper.last_manifest_paths[0]
        values = _read_shared_tail_dump(manifest, tokens)
        _check_shared_tail_values(values, expected_up, expected[1][-1:])
        print(f"MEGA_MOE_SHARED_L1_PASS rank={common.RANK} tokens={tokens} manifest={manifest}", flush=True)
    finally:
        layer.close()
    dist.barrier()

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
"""Two-rank Torch workers for MegaMoe Kernel and Task dumps."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts
from hyper_multicore.diagnostics import profiler as multicore_profiler, schedule
from hyper_multicore.diagnostics.dumper import DumpConfig, dump, load_tensor
from hyper_multicore.experimental import MegaMoeSchedulerOptions, MegaMoeTuningOptions
from tests.common.multicore.owned_mega_moe import OwnedMegaMoeExperts as _MegaMoeExperts
from tests.common.multicore._dump_faults import fail_dump_output_payload_write
from tests.torch import _test_mega_moe as common


def _build_layer(shape: common.MoeShape, *, batch_experts: bool = False) -> MegaMoeExperts:
    """Construct the small representative two-rank MegaMoe layer."""
    return _MegaMoeExperts(
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        tuning_options=MegaMoeTuningOptions(
            dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
            scheduler=MegaMoeSchedulerOptions(batch_experts=batch_experts),
        ),
        ep_group=dist.group.WORLD,
    ).to(device=common.DEVICE, dtype=torch.bfloat16)


def test_mega_moe_forward_tensor_dump_worker() -> None:
    """Validate the numeric result and an expert-row summary on two ranks."""
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    output_dir = Path(os.environ["HP_MEGA_MOE_DUMP_RESULT_DIR"])
    config = DumpConfig(output_dir, stages="SwiGLU", io="output", ranks=0, outputs=("tensor", "csv"))
    layer = _build_layer(shape)
    try:
        with torch.no_grad():
            expected = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            with dump(config) as dumper:
                actual = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
        if common.RANK == 0:
            path, = dumper.last_manifest_paths
            payload = load_tensor(path, invocation_id=0, port=("SwiGLU", "output", 0))
            with path.with_name("summary.csv").open(encoding="utf-8", newline="") as stream:
                row, = csv.DictReader(stream)
            valid_rows = json.loads(row["expert_rows"])[-1][2]
            assert row["range"] == "expert_rows"
            assert 0 < valid_rows <= payload.shape[0]
            assert payload.dtype == torch.bfloat16 and payload.shape[1] == shape.intermediate_size
            assert float(row["max"]) == payload[:valid_rows].max().item()
        else:
            assert not dumper.last_manifest_paths
    finally:
        layer.close()
    dist.barrier()


def _assert_swiglu_task_dump(manifest_path: Path, intermediate_size: int) -> list[dict]:
    """Read native task payloads and check SwiGLU arithmetic."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invocation, = manifest["invocations"]
    tasks = invocation["tasks"]
    assert tasks, "Task dump did not export any SwiGLU tasks"
    for task in tasks:
        assert task["stage"] == "SwiGLU" and task["coreType"] == "AIV", task
        operands, output = (
            load_tensor(
                manifest_path, invocation_id=invocation["invocationId"], port=("SwiGLU", io, 0),
                task_id=task["taskId"], worker=("AIV", task["coreId"]),
            )
            for io in ("input", "output")
        )
        assert operands.shape[1] == intermediate_size * 2
        assert output.shape[1] == intermediate_size
        gate, up = operands.float().chunk(2, dim=-1)
        torch.testing.assert_close(
            output.float(), torch.nn.functional.silu(gate) * up, rtol=2e-2, atol=5e-3,
            msg=f"rank={common.RANK}: incorrect Task SwiGLU snapshot: {task}",
        )
    return tasks


def test_mega_moe_batched_task_tensor_dump_worker(caplog) -> None:
    """Check remapped Task payloads and numeric parity on a batched-expert plan."""
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    layer = _build_layer(shape, batch_experts=True)
    result_dir = os.getenv("HP_MEGA_MOE_DUMP_RESULT_DIR")
    if not result_dir:
        raise ValueError("HP_MEGA_MOE_DUMP_RESULT_DIR must name the dump output directory")
    config = DumpConfig(
        result_dir,
        level="task",
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
            expected = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.npu.synchronize()
            dist.barrier()
            with dump(DumpConfig(result_dir, level="task", stages="GMM2", io="input")) as skipped:
                actual = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
            assert not skipped.last_manifest_paths
            assert any("reason=unsupported_port" in message and "SwiGLU.output[0]" in message
                       for message in caplog.messages)
            with dump(config, schedule=capture_schedule) as dumper:
                actual = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                dumper.step()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
        if common.RANK == 0:
            manifest_path = dumper.last_manifest_paths[0]
            _assert_swiglu_task_dump(manifest_path, shape.intermediate_size)
            print(f"MEGA_KERNEL_TASK_TENSOR_DUMP={manifest_path}")
    finally:
        layer.close()
    dist.barrier()


def _profiled_swiglu_locations(trace: dict) -> set[tuple[int, str, int]]:
    """Read device worker IDs for SwiGLU computation from a complete trace."""
    metadata = trace["megaKernelCycleTrace"]
    assert metadata["invocationCount"] == 1
    assert metadata["droppedRecordCount"] == 0
    locations = set()
    for event in trace["traceEvents"]:
        if event.get("cat") != "MegaKernelInternal" or event["name"] != "SwiGLU":
            continue
        args = event["args"]
        if args["core_type"] == "AIV":
            locations.add((args["task_id"], args["core_type"], args["block_id"]))
    assert locations, f"rank={common.RANK}: no SwiGLU computation in profiler trace"
    return locations


def test_mega_moe_task_dump_core_selection_worker(caplog) -> None:
    """Feature: Task dump worker selection.
    Description: Use profiler worker IDs for selected and empty Task dumps on two ranks.
    Expectation: Matching snapshots use trace IDs and both ranks complete the empty selection.
    """
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    output_dir = Path(os.environ["HP_MEGA_MOE_DUMP_RESULT_DIR"])
    capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
    layer = _build_layer(shape)
    try:
        with torch.no_grad():
            expected = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.npu.synchronize()
            dist.barrier()

            with multicore_profiler.profile(
                schedule=capture_schedule, detailed_task_names=False,
            ) as profiler:
                profiled = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                profiler.step()
            torch.testing.assert_close(profiled, expected, rtol=2e-2, atol=5e-3)
            trace = profiler.export_chrome_trace(output_dir / f"rank{common.RANK}_trace.json")
            observed = _profiled_swiglu_locations(trace)
            worker_ids = {worker_id for _, _, worker_id in observed}
            assert all(worker_id % 2 == 1 for worker_id in worker_ids), worker_ids
            selected_worker = torch.tensor([min(worker_ids)], dtype=torch.int32, device=common.DEVICE)
            dist.broadcast(selected_worker, src=0)
            selected_id = selected_worker.item()

            for name, worker_id in (("matched", selected_id), ("empty", 0)):
                dump_dir = output_dir / name
                config = DumpConfig(
                    dump_dir, level="task", kernel_names="mega_moe", stages="SwiGLU",
                    directions="forward", ranks=0, core_types="AIV", core_ids=worker_id,
                    io=("input", "output"), device_memory_budget_mb=32,
                )
                caplog.clear()
                with dump(config, schedule=capture_schedule) as dumper:
                    actual = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                    dumper.step()
                torch.npu.synchronize()
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
                warnings = [
                    record.getMessage() for record in caplog.records
                    if record.name == "hyper_multicore.diagnostics.dumper.api"
                    and "no Tensor snapshots to export for this call" in record.getMessage()
                ]
                if common.RANK != 0:
                    assert not warnings
                    assert not dumper.last_manifest_paths
                    continue
                if name == "empty":
                    assert not dumper.last_manifest_paths
                    assert len(warnings) == 1
                    assert "kernel=mega_moe rank=0 direction=forward" in warnings[0]
                    assert not list(dump_dir.rglob("manifest.json"))
                    assert not list(dump_dir.rglob("*.pt"))
                    continue
                assert not warnings
                assert len(dumper.last_manifest_paths) == 1
                manifest_path = dumper.last_manifest_paths[0]
                tasks = _assert_swiglu_task_dump(manifest_path, shape.intermediate_size)
                assert {(task["taskId"], task["coreType"], task["coreId"]) for task in tasks} == {
                    entry for entry in observed if entry[2] == worker_id
                }
                print(f"MEGA_KERNEL_TASK_WORKER={worker_id} MANIFEST={manifest_path}", flush=True)
    finally:
        layer.close()
    dist.barrier()
    print(f"MEGA_KERNEL_TASK_EMPTY_SELECTION_COMPLETE rank={common.RANK}", flush=True)


def test_mega_moe_task_dump_resource_failure_worker(caplog) -> None:
    """Keep both ranks computing through a local buffer limit and disk failure."""
    shape = common.MoeShape(local_num_tokens=512, intermediate_size=512)
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    output_dir = Path(os.environ["HP_MEGA_MOE_DUMP_RESULT_DIR"])
    layer = _build_layer(shape)
    try:
        with torch.no_grad():
            expected = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            for reason in ("device_buffer_budget_exceeded", "write_failed"):
                dump_dir = output_dir / reason
                rank_zero = common.RANK == 0
                write_case = reason == "write_failed"
                config = DumpConfig(
                    dump_dir, level="task", stages="SwiGLU",
                    device_memory_budget_mb=1 if rank_zero and not write_case else 32,
                    outputs=("tensor", "csv") if write_case else ("tensor",),
                )
                failure = fail_dump_output_payload_write() if rank_zero and write_case else nullcontext()
                caplog.clear()
                with failure, dump(config) as dumper:
                    actual = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                following = layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
                torch.testing.assert_close(following, expected, rtol=2e-2, atol=5e-3)
                warnings = [
                    record.getMessage() for record in caplog.records if "Task dump skipped" in record.getMessage()
                ]
                if rank_zero:
                    warning, = warnings
                    assert f"rank=0 direction=forward step=0 reason={reason}" in warning
                    assert not dumper.last_manifest_paths
                else:
                    assert not warnings
                    manifest_path, = dumper.last_manifest_paths
                    assert manifest_path.is_file()
                if write_case:
                    window = dump_dir / f"rank_{common.RANK}" / "window_0000"
                    assert (window / "summary.csv").is_file()
                dist.barrier()
    finally:
        layer.close()
    dist.barrier()

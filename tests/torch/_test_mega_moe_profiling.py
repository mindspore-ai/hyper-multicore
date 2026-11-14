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
"""Two-rank Torch worker for a steady-state MegaMoe forward profiling capture."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore.diagnostics import profiler as multicore_profiler, schedule as diagnostic_schedule
from hyper_multicore import MegaMoeExperts
from hyper_multicore._core_spec import get_device_core_spec
from tests.common.multicore.owned_mega_moe import OwnedMegaMoeExperts
from tests.torch import _test_mega_moe as common


_PROFILE_SHAPE = common.MoeShape(
    # The legacy TP=2 test partitions global seq_size=1024 into 512 tokens per rank.
    local_num_tokens=512,
    hidden_size=5120,
    intermediate_size=2048,
    num_experts=16,
    top_k=8,
    ep_size=2,
)
_EXPECTED_AIV_RECORD_COUNT = 451


def _build_profile_layer() -> MegaMoeExperts:
    """Construct the representative two-rank MegaMoe layer."""
    return OwnedMegaMoeExperts(
        hidden_size=_PROFILE_SHAPE.hidden_size,
        intermediate_size=_PROFILE_SHAPE.intermediate_size,
        num_experts=_PROFILE_SHAPE.num_experts,
        top_k=_PROFILE_SHAPE.top_k,
        ep_group=dist.group.WORLD,
    ).to(device=common.DEVICE, dtype=torch.bfloat16)


def _assert_complete_forward_trace(trace: dict) -> None:
    """Reject incomplete or ambiguously named device records."""
    metadata = trace["megaKernelCycleTrace"]
    # Two GMM stages per local expert, each recording wait, compute and trigger.
    cube_slots = get_device_core_spec(common.DEVICE.index).num_cube_cores
    cube_records = cube_slots * (_PROFILE_SHAPE.num_experts // _PROFILE_SHAPE.ep_size) * 2 * 3
    expected_core_records = {"AIC": cube_records, "AIV": _EXPECTED_AIV_RECORD_COUNT}
    expected_record_count = sum(expected_core_records.values())
    assert metadata["invocations"][0]["workerCapacities"]["cubeSlots"] == cube_slots
    internal_events = [event for event in trace["traceEvents"] if event.get("cat") == "MegaKernelInternal"]
    core_record_counts = Counter(event["args"]["core_type"] for event in internal_events)

    assert metadata["invocationCount"] == 1, f"rank={common.RANK}: expected one forward invocation, got {metadata}."
    assert (
        metadata["recordCount"] == expected_record_count
    ), f"rank={common.RANK}: incomplete record count: {metadata}."
    assert metadata["droppedRecordCount"] == 0, f"rank={common.RANK}: profiling dropped records: {metadata}."
    assert (
        core_record_counts == expected_core_records
    ), f"rank={common.RANK}: unexpected AIC/AIV records: {core_record_counts}."
    assert all(
        0 < event["args"]["start_cycle"] <= event["args"]["end_cycle"]
        for event in internal_events
    ), f"rank={common.RANK}: profiling exported an invalid cycle record."
    assert {event["args"]["direction"] for event in internal_events} == {"forward"}
    event_names = {event["name"] for event in internal_events}
    assert any("GMM1_WaitDependency" in name for name in event_names)
    assert any("GMM1_TriggerEvent" in name for name in event_names)
    assert any("GMM2_WaitDependency" in name for name in event_names)
    assert any("GMM2_TriggerEvent" in name for name in event_names)


def test_mega_moe_forward_profiling() -> None:
    """Feature: MegaKernel diagnostics.
    Description: Warm up, synchronize ranks, and export one representative forward trace.
    Expectation: Mega moe forward profiling.
    """
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(_PROFILE_SHAPE)
    hidden_states, _ = common.make_data(_PROFILE_SHAPE)
    layer = _build_profile_layer()
    schedule = diagnostic_schedule(wait=0, warmup=0, active=1, repeat=1)
    try:
        with torch.no_grad():
            layer(
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert=tokens_per_expert,
            )
            torch.npu.synchronize()
            dist.barrier()

            with multicore_profiler.mega_kernel_profile(
                schedule=schedule,
                detailed_task_names=True,
            ) as profiler:
                layer(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    tokens_per_expert=tokens_per_expert,
                )
                profiler.step()

        result_dir = os.getenv("HP_MEGA_MOE_PROFILE_RESULT_DIR")
        if not result_dir:
            raise ValueError("HP_MEGA_MOE_PROFILE_RESULT_DIR must name the trace output directory.")
        output_path = Path(result_dir) / f"rank{common.RANK}_mega_kernel_trace.json"
        trace = profiler.export_chrome_trace(output_path)
        _assert_complete_forward_trace(trace)
        print(f"MEGA_KERNEL_PROFILE_TRACE={output_path}")
    finally:
        layer.close()
    dist.barrier()

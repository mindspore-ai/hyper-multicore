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
"""Lightweight launcher for the representative Torch MegaMoe profiling ST."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.multicore._evidence_helpers import capture_worker_logs
from tests.common.port_utils import allocate_port
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe_profiling.py")
_WORLD_SIZE = 2
_EXPECTED_AIV_RECORD_COUNT = 451
_LOCAL_EXPERTS = 8


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for Torch."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with bash build.sh")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_standalone_profiler(monkeypatch, tmp_path: Path) -> None:
    """Feature: MegaKernel diagnostics.
    Description: Export complete steady-state forward traces for both ranks.
    Expectation: Mega moe standalone profiler.
    """
    _prepare_torch_multicore_test_environment()
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path))) / "profiling"
    monkeypatch.delenv("HYPER_MULTICORE_SHMEM_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PROFILE_RESULT_DIR", str(result_dir))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment(), capture_worker_logs(
        _WORKER, "test_mega_moe_forward_profiling", result_dir / "worker_logs",
    ):
        run_multicore_case(_WORKER, "test_mega_moe_forward_profiling", num_proc=_WORLD_SIZE)

    for rank in range(_WORLD_SIZE):
        trace_path = result_dir / f"rank{rank}_mega_kernel_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        metadata = trace["megaKernelCycleTrace"]
        assert metadata["invocationCount"] == 1
        cube_slots = metadata["invocations"][0]["workerCapacities"]["cubeSlots"]
        expected_records = cube_slots * _LOCAL_EXPERTS * 2 * 3 + _EXPECTED_AIV_RECORD_COUNT
        assert metadata["recordCount"] == expected_records
        assert metadata["droppedRecordCount"] == 0

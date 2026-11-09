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
"""Launch exact backward-storage comparisons without importing device frameworks."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._evidence_helpers import capture_worker_logs, run_acceptance_worker
from tests.common.multicore._test_env import (
    multicore_adapter_is_available, prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_backward_memory(monkeypatch, tmp_path: Path, dispatch_mode: str) -> None:
    """Require exact gradients, ordinary reuse and retained fallback on both transports."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("Backward memory ST requires the rebuilt multicore Torch payload")
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    worker = str(Path(__file__).with_name("_test_mega_moe_backward_memory.py"))

    def launch(result_dir: Path) -> None:
        """Preserve rank logs from the isolated two-rank correctness worker."""
        with without_inherited_rank_environment(), capture_worker_logs(
                worker, "test_mega_moe_backward_memory", result_dir / "worker_logs"):
            run_multicore_case(worker, "test_mega_moe_backward_memory", num_proc=2)

    result = run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_backward_memory", 2, launch,
                                   heap_bytes=256 * 1024**2)
    for rank in result["ranks"]:
        assert rank["passed"] and rank["ordinary_reuse_hits"] > 0, f"missing reuse evidence: {rank}"
        assert rank["retained_fallbacks"] == 2 and rank["saved_hook_fallbacks"] == 2, (
            f"incomplete backward lifecycle evidence: {rank}"
        )
        assert rank["late_hook_fallbacks"] == 2 and len(rank["launches"]) == 23, (
            f"incomplete late-hook and backward launch evidence: {rank}"
        )
        assert rank["snapshot_reuse_hits"] == 2 and rank["snapshot_fallbacks"] == 4, (
            f"incomplete routed snapshot ownership evidence: {rank}"
        )

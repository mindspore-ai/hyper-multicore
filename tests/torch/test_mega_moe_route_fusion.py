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
"""Framework-free launcher for indexed pull route-fusion regression."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._evidence_helpers import capture_worker_logs, run_acceptance_worker
from tests.common.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_indexed_pull_fusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Indexed pull route fusion.
    Description: Launch the two-rank short-dispatch and retained-backward regression.
    Expectation: Both ranks complete all 32 routing and fusion combinations.
    """
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("indexed pull ST requires the rebuilt multicore Torch payload")
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", "pull")
    worker = str(Path(__file__).with_name("_test_mega_moe_route_fusion.py"))

    def launch(result_dir: Path) -> None:
        """Keep framework imports inside the spawned worker process."""
        with without_inherited_rank_environment(), capture_worker_logs(
            worker, "test_indexed_pull_fusion", result_dir / "worker_logs"):
            run_multicore_case(worker, "test_indexed_pull_fusion", num_proc=2)

    result = run_acceptance_worker(monkeypatch, tmp_path, "test_indexed_pull_fusion", 2, launch)
    for rank in result["ranks"]:
        assert rank["passed"] and len(rank["cases"]) == 32, (
            f"expected 32 successful cases on rank {rank['rank']}, got {rank}"
        )

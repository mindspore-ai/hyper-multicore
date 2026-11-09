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
"""Launch shared-expert correctness cases in fresh distributed processes."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._evidence_helpers import capture_worker_logs, run_acceptance_worker
from tests.common.multicore._test_env import (
    multicore_adapter_is_available, prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
@pytest.mark.parametrize("pattern,representative,full_first,matrix", [
    ("random", False, False, False),
    ("random", False, True, False),
    ("zero_experts", False, False, True),
    ("skewed", False, False, False),
    ("random", False, False, True),
    ("balanced", True, False, False),
])
def test_shared_expert_numerics(monkeypatch, pattern, representative, full_first, matrix):
    """Check external precision, empty experts and repeated backward without warmup."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("Shared-expert ST requires a payload built with bash build.sh")
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    monkeypatch.delenv("HYPER_MULTICORE_SHMEM_HEAP_SIZE", raising=False)
    for name, value in {
        "HP_MEGA_MOE_SHARED_ROUTE_PATTERN": pattern,
        "HP_MEGA_MOE_SHARED_REPRESENTATIVE": int(representative),
        "HP_MEGA_MOE_TEST_FULL_FIRST": int(full_first),
        "HP_MEGA_MOE_TEST_FULL_ONLY": int(not matrix),
        "HP_MEGA_MOE_TEST_ROUTE_MATRIX": int(matrix),
    }.items():
        monkeypatch.setenv(name, str(value))
    worker = str(Path(__file__).with_name("_test_mega_moe_shared.py"))
    with without_inherited_rank_environment():
        run_multicore_case(worker, "test_shared_direction_switches", num_proc=4 if representative else 2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_dynamic_shared_fusion(monkeypatch, tmp_path, dispatch_mode):
    """Check real token shapes, growth, empty ranks and retained full-fusion graphs."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("Dynamic shared fusion requires the rebuilt multicore Torch payload")
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    worker = str(Path(__file__).with_name("_test_mega_moe_shared_dynamic.py"))

    def launch(result_dir: Path) -> None:
        """Keep framework imports and device setup in the worker process."""
        with without_inherited_rank_environment(), capture_worker_logs(
                worker, "test_dynamic_shared_fusion", result_dir / "worker_logs"):
            run_multicore_case(worker, "test_dynamic_shared_fusion", num_proc=2)

    result = run_acceptance_worker(monkeypatch, tmp_path, "test_dynamic_shared_fusion", 2, launch,
                                   heap_bytes=256 * 1024**2)
    for rank in result["ranks"]:
        assert rank["passed"] and len(rank["cases"]) == len(rank["switches"]) == 8, (
            f"incomplete dynamic fusion evidence: {rank}"
        )

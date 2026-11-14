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

"""Lightweight launcher for the source-tree MoE baseline."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._evidence_helpers import capture_worker_logs
from tests.common.multicore._test_env import without_inherited_rank_environment

_WORKER = str(Path(__file__).with_name("_test_moe_baseline.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_moe_baseline_ep_permutation(tmp_path: Path) -> None:
    """Compare fused EP routing, score/weight gradients and empty destinations."""
    if not (Path(__file__).resolve().parents[2] / "examples" / "_native.py").is_file():
        pytest.skip("the example baseline is available only in the source tree")
    case = "test_moe_baseline_ep_permutation"
    with without_inherited_rank_environment(), capture_worker_logs(_WORKER, case, tmp_path / "worker_logs"):
        run_multicore_case(_WORKER, case, num_proc=2, timeout=420)

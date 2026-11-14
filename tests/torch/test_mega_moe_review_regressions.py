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

"""Lightweight launchers for the PR17 device review regressions."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._evidence_helpers import capture_worker_logs
from tests.common.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)
from tests.common.port_utils import allocate_port

_WORKER = str(Path(__file__).with_name("_test_mega_moe_review_regressions.py"))


def _run_review_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """Activate the matching native payload and isolate one device worker."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("MegaMoe review ST requires a native payload built with bash build.sh")
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", "1")
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment(), capture_worker_logs(_WORKER, case, tmp_path / "worker_logs"):
        run_multicore_case(_WORKER, case, num_proc=1, timeout=420)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_compact_header_uses_normal_ready_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run the normal compact-ready-queue header and backward regression."""
    _run_review_worker(monkeypatch, tmp_path, "test_compact_header_uses_normal_ready_queue")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_fp32_native_weight_gradients_before_autograd_cast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Run the FP32 native partial-weight-gradient regression in its own worker."""
    _run_review_worker(monkeypatch, tmp_path, "test_fp32_native_weight_gradients_before_autograd_cast")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_reentrant_no_grad_then_grad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run the no-grad then grad-enabled invocation regression."""
    _run_review_worker(monkeypatch, tmp_path, "test_reentrant_no_grad_then_grad")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_fp32_native_dw_matches_cpu_fp32_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run the native FP32 weight-gradient reference regression."""
    _run_review_worker(monkeypatch, tmp_path, "test_fp32_native_dw_matches_cpu_fp32_reference")

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
"""Lightweight launchers for representative Torch MegaMoe Tensor dump STs."""

from __future__ import annotations

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe_tensor_dump.py")
_WORLD_SIZE = 2


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for Torch."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with bash build.sh")


def _run_dump_case(monkeypatch, output_dir: Path, case_name: str) -> None:
    """Run one two-rank dump worker with an isolated output directory."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_DUMP_RESULT_DIR", str(output_dir))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(_WORKER, case_name, num_proc=_WORLD_SIZE)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_forward_tensor_dump(monkeypatch, tmp_path: Path) -> None:
    """Feature: MegaKernel diagnostics.
    Description: Export and validate one forward L0 dump window.
    Expectation: Mega moe forward tensor dump.
    """
    _run_dump_case(monkeypatch, tmp_path, "test_mega_moe_forward_tensor_dump_worker")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_forward_l1_tensor_dump(monkeypatch, tmp_path: Path) -> None:
    """Feature: MegaKernel diagnostics.
    Description: Export and validate one forward L1 dump window.
    Expectation: Mega moe forward l1 tensor dump.
    """
    _run_dump_case(monkeypatch, tmp_path, "test_mega_moe_forward_l1_tensor_dump_worker")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mega_moe_shared_l1_tails(monkeypatch, tmp_path: Path) -> None:
    """Verify full-fusion shared F+B task snapshots at unequal 129/257 token tails."""
    _run_dump_case(monkeypatch, tmp_path, "test_mega_moe_shared_l1_tail_worker")

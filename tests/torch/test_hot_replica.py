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
"""Launch the HotReplica two-rank device regression."""
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark

from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._test_env import prepare_multicore_test_environment, without_inherited_rank_environment
from tests.common.port_utils import allocate_port


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize(("dispatch_mode", "layout"),
                         [("push", "EIO"), ("push", "EOI"), ("pull", "EIO"), ("pull", "EOI")])
def test_placement_checkpoint_and_close(monkeypatch, tmp_path, dispatch_mode, layout):
    """Run the checkpoint and bank cleanup regression for one transport and weight layout."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HOT_REPLICA_DISPATCH_MODE", dispatch_mode)
    monkeypatch.setenv("HOT_REPLICA_WEIGHT_LAYOUT", layout)
    monkeypatch.setenv("HOT_REPLICA_TEST_RESULTS", str(tmp_path))
    monkeypatch.setenv("HYPER_MULTICORE_HOT_REPLICA_CHECKPOINT_BYTES", str(2 << 30))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                           "test_placement_checkpoint_and_close", num_proc=2, timeout=420)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_nonreentrant_checkpoint_replay(monkeypatch, tmp_path):
    """Exercise non-reentrant layout reuse and all-rank input rejection on two A3 devices."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HOT_REPLICA_TEST_RESULTS", str(tmp_path))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                           "test_nonreentrant_checkpoint_replay", num_proc=2, timeout=420)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_wide_placement_and_hard_receive_limit(monkeypatch, tmp_path):
    """Exercise both transports with the same explicit limits and a wide fused placement."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HOT_REPLICA_TEST_RESULTS", str(tmp_path))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                           "test_wide_placement_and_hard_receive_limit", num_proc=2, timeout=360)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_placement_diagnostics(monkeypatch, tmp_path):
    """Validate the EP dump protocol and real fused replica trace events."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HOT_REPLICA_TEST_RESULTS", str(tmp_path))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                           "test_placement_diagnostics", num_proc=2, timeout=360)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("ranks", [2, 4])
def test_placement_edge_cases(monkeypatch, tmp_path, ranks):
    """Exercise both layouts, probability positions, splits and graph lifetimes."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("EDGE_OUT", str(tmp_path / "edge-results"))
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_hot_replica_edge_cases.py")),
                           "test_placement_edge_cases", num_proc=ranks, timeout=600)

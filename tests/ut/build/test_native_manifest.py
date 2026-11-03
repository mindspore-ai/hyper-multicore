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
"""Behavioral checks for paired adapters, relaxed external versions and terminal loading."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time
from unittest.mock import patch

import pytest

from hyper_multicore import _loader
from hyper_multicore._native_manifest import check_integrity, read_manifest, resource_path, select_variant
from tests.ut.build.test_artifacts import _payload_fixture


@pytest.fixture(name="payload")
def _payload(tmp_path):
    """Provide a complete synthetic payload and matching build records."""
    info = _payload_fixture(tmp_path)
    return tmp_path, info, read_manifest(tmp_path)


@pytest.mark.parametrize("pta", ["2.12.0", "2.12.0.post99+local"])
def test_same_branch_has_no_build_version_floor(payload, pta):
    """Earlier and later post versions select the same branch without hash or CANN gates."""
    _, _, manifest = payload
    actual = {"torch_version": "2.12.0+cpu", "pta_version": pta, "python_abi": "cp310",
              "host_arch": "aarch64", "torch_cxx11_abi": True, "cann_version": "unknown"}
    assert select_variant(manifest, actual)["id"] == "fixture"


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside"])
def test_resource_paths_cannot_escape(payload, relative):
    """A manifest cannot redirect loading outside component-owned resources."""
    with pytest.raises(ValueError):
        resource_path(payload[0], relative)


def test_pair_and_selector_ambiguity_are_rejected(payload):
    """Duplicated selectors, swapped paths and missing companions fail before loading."""
    root, _, manifest = payload
    invalid = copy.deepcopy(manifest)
    invalid["variants"].append(copy.deepcopy(invalid["variants"][0]))
    (root / "adapters/manifest.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(root)
    invalid = copy.deepcopy(manifest)
    invalid["variants"][0]["shmem_so"] = invalid["variants"][0]["torch_so"]
    (root / "adapters/manifest.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="shmem_so"):
        read_manifest(root)


def test_requested_targets_cannot_be_removed_with_rewritten_hashes(payload):
    """A self-consistent file list cannot hide missing requested adapter pairs."""
    root, info, manifest = payload
    info["requested_target_ids"].append("missing-target")
    with pytest.raises(ValueError, match="requested"):
        check_integrity(root, info, manifest)


def test_shared_registration_lock_and_terminal_failure():
    """Concurrent callers register once; a partial failure prevents the other adapter loading."""
    barrier = Barrier(6)
    calls = []
    with patch.object(_loader, "_loaded_kinds", set()), patch.object(_loader, "_native_failure", None):
        def first_use(_):
            barrier.wait()
            with _loader.native_loading("torch") as needed:
                if needed:
                    time.sleep(0.01)
                    calls.append("torch")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(first_use, range(6)))
        assert calls == ["torch"]
        assert _loader.native_ready("torch")
        with pytest.raises(OSError, match="missing symbol"):
            with _loader.native_loading("shmem"):
                raise OSError("missing symbol")
        with pytest.raises(_loader.NativeComponentUnavailableError, match="Restart.*missing symbol"):
            with _loader.native_loading("torch"):
                pytest.fail("terminal state must not retry")
        with pytest.raises(_loader.NativeComponentUnavailableError, match="Restart"):
            _loader.native_ready("torch")


def test_pta_absence_and_import_error_have_distinct_causes():
    """Missing distribution metadata is distinguished from a broken installed backend."""
    _loader.load_torch_npu.cache_clear()
    with patch.object(_loader.metadata, "version", side_effect=_loader.metadata.PackageNotFoundError("torch-npu")):
        with pytest.raises(_loader.NativeComponentUnavailableError, match="PTA-NOT-INSTALLED"):
            _loader.load_torch_npu()
    error = OSError("libascendcl.so not found")
    with patch.object(_loader.metadata, "version", return_value="2.12.0"), patch.object(_loader.importlib, "import_module", side_effect=error):
        with pytest.raises(_loader.NativeComponentUnavailableError, match="PTA-IMPORT-FAILED.*libascendcl") as caught:
            _loader.load_torch_npu()
        assert caught.value.__cause__ is error
    _loader.load_torch_npu.cache_clear()


@pytest.mark.parametrize("first,second", [("torch", "shmem"), ("shmem", "torch")])
def test_other_adapter_failure_blocks_successful_fast_path(first, second):
    """A successful SO cannot keep running after its companion partially fails to initialize."""
    from hyper_multicore.torch import ops
    from hyper_multicore.shmem import _runtime

    with patch.object(_loader, "_loaded_kinds", {first}), patch.object(_loader, "_native_failure", None), \
            patch.object(_runtime, "_native", object()):
        with pytest.raises(OSError):
            with _loader.native_loading(second):
                raise OSError("companion partial initialization")
        entry = ops._load_native if first == "torch" else _runtime._load_native
        with pytest.raises(_loader.NativeComponentUnavailableError, match="Restart.*companion"):
            entry()


def test_launcher_keeps_corrupt_manifest_as_a_worker_failure(tmp_path):
    """Framework-free launchers must not turn a damaged schema-2 payload into a skip."""
    from tests.common.multicore import _test_env

    payload = tmp_path / "hyper_multicore"
    vendor = payload / "lib/vendors/hyper_multicore_nn"
    vendor.mkdir(parents=True)
    with patch.object(_test_env, "_active_multicore_vendor_roots", return_value=[vendor]):
        assert not _test_env.multicore_adapter_is_available()
        (payload / "adapters").mkdir()
        (payload / "adapters/manifest.json").write_bytes(b"")
        assert _test_env.multicore_adapter_is_available()

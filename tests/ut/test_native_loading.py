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
"""Installed-package adapter selection and integrity, without build sources or native loading."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

from hyper_multicore import _loader
from hyper_multicore import _native_manifest as manifest


class TestNativeLoading(unittest.TestCase):
    """Reject wrong or damaged adapters and preserve shared initialization state."""

    def setUp(self):
        """Create a minimal paired payload using only the installed runtime contract."""
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.variant = {"id": "fixture", "torch_release": "2.12.0", "pta_release": "2.12.0",
                        "torch_version": "2.12.0+cpu", "pta_version": "2.12.0.post3", "torch_cxx11_abi": True,
                        "torch_so": "adapters/fixture/libhyper_multicore_torch.so",
                        "shmem_so": "adapters/fixture/hyper_multicore_shmem_torch.so"}
        for field in ("torch_so", "shmem_so"):
            path = self.root / self.variant[field]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic native library")
        self.index = {"schema_version": 1, "python_abi": "cp310", "host_arch": "aarch64",
                      "variants": [self.variant]}
        self.index_path = self.root / "adapters/manifest.json"
        self.index_path.write_text(json.dumps(self.index), encoding="utf-8")
        hashes = {path.relative_to(self.root).as_posix(): manifest.file_sha256(path)
                  for path in self.root.rglob("*") if path.is_file()}
        self.info = {"schema_version": 2, "python_abi": "cp310", "host_arch": "aarch64",
                     "variants": [dict(self.variant)], "requested_target_ids": ["fixture"],
                     "payload_sha256": hashes,
                     "native_sha256": {name: value for name, value in hashes.items() if name.endswith(".so")}}

    def test_branch_and_abi_select_exactly_one_pair(self):
        """Post suffixes do not gate selection; another branch or ABI must not load this pair."""
        actual = {"torch_version": "2.12.0+cpu", "pta_version": "2.12.0.post1", "python_abi": "cp310",
                  "host_arch": "aarch64", "torch_cxx11_abi": True}
        self.assertEqual(manifest.select_variant(self.index, actual)["id"], "fixture")
        for key, value in (("pta_version", "2.13.0"), ("torch_cxx11_abi", False), ("python_abi", "cp311")):
            with self.subTest(field=key), self.assertRaises(ValueError):
                manifest.select_variant(self.index, {**actual, key: value})

    def test_manifest_rejects_ambiguous_and_redirected_pairs(self):
        """A pair cannot be ambiguous, swapped or redirected outside its payload."""
        invalid = copy.deepcopy(self.index)
        invalid["variants"].append(copy.deepcopy(invalid["variants"][0]))
        self.index_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            manifest.read_manifest(self.root)
        invalid = copy.deepcopy(self.index)
        invalid["variants"][0]["shmem_so"] = self.variant["torch_so"]
        self.index_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shmem_so"):
            manifest.read_manifest(self.root)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            manifest.resource_path(self.root, "../external.so")

    def test_complete_payload_integrity_rejects_changes_and_missing_files(self):
        """Both membership and bytes are authenticated before loading any selected library."""
        manifest.check_integrity(self.root, self.info, self.index)
        library = self.root / self.variant["torch_so"]
        library.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed payload"):
            manifest.check_integrity(self.root, self.info, self.index)
        library.unlink()
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            manifest.check_integrity(self.root, self.info, self.index)

    def test_concurrent_first_use_registers_once(self):
        """Concurrent callers serialize registration without repeating its side effects."""
        barrier = Barrier(4)
        calls = []

        def first_use(index):
            barrier.wait(timeout=5)
            with _loader.native_loading("torch") as needed:
                if needed:
                    calls.append(index)

        with patch.object(_loader, "_loaded_kinds", set()), patch.object(_loader, "_native_failure", None):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(first_use, range(4)))
            self.assertEqual(len(calls), 1)
            self.assertTrue(_loader.native_ready("torch"))

    def test_partial_failure_blocks_both_adapters(self):
        """One failed companion prevents retries and disables an already registered adapter."""
        with patch.object(_loader, "_loaded_kinds", {"torch"}), patch.object(_loader, "_native_failure", None):
            with self.assertRaisesRegex(OSError, "missing symbol"):
                with _loader.native_loading("shmem"):
                    raise OSError("missing symbol")
            with self.assertRaisesRegex(_loader.NativeComponentUnavailableError, "Restart.*missing symbol"):
                _loader.native_ready("torch")
            with self.assertRaisesRegex(_loader.NativeComponentUnavailableError, "Restart.*missing symbol"):
                with _loader.native_loading("shmem"):
                    self.fail("A terminal loading failure must not retry")

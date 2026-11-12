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
import ctypes
import json
from pathlib import Path
import tempfile
import sys
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hyper_multicore import _loader
from hyper_multicore import _native_manifest as manifest
from hyper_multicore.shmem import _runtime as shmem_runtime


class TestNativeLoading(unittest.TestCase):
    """Reject wrong or damaged adapters and preserve shared initialization state."""

    def setUp(self):
        """Create a minimal multi-backend payload using only the installed runtime contract."""
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.backends = {
            "arch22": {"npu_arch": 2201, "shmem_root": "shmem/lib/arch22",
                           "soc_list": ["ascend910b", "ascend910_93"]},
            "arch35": {"npu_arch": 3510, "shmem_root": "shmem/lib/arch35", "soc_list": ["ascend950"]},
        }
        for name, backend in self.backends.items():
            backend["kernel_root"] = f"lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe/kernel/{name}"
            backend["sdk_soc_type"] = "Ascend910B" if name == "arch22" else "Ascend950"
        self.variant = {"id": "fixture", "torch_release": "2.12.0", "pta_release": "2.12.0",
                        "torch_version": "2.12.0+cpu", "pta_version": "2.12.0.post3", "torch_cxx11_abi": True,
                        "torch_so": "adapters/fixture/libhyper_multicore_torch.so",
                        "shmem_so": {name: f"adapters/fixture/shmem/{name}/hyper_multicore_shmem_torch.so"
                                           for name in self.backends}}
        files = [self.variant["torch_so"], *self.variant["shmem_so"].values()]
        files.extend(f"{backend['shmem_root']}/{name}" for backend in self.backends.values()
                     for name in manifest.SHMEM_RESOURCES)
        for relative in files:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic native library")
        self.index = {"schema_version": 2, "python_abi": "cp310", "host_arch": "aarch64",
                      "variants": [self.variant], "architectures": self.backends}
        self.index_path = self.root / "adapters/manifest.json"
        self.index_path.write_text(json.dumps(self.index), encoding="utf-8")
        hashes = {path.relative_to(self.root).as_posix(): manifest.file_sha256(path)
                  for path in self.root.rglob("*") if path.is_file()}
        self.info = {"schema_version": 3, "python_abi": "cp310", "host_arch": "aarch64",
                     "architectures": copy.deepcopy(self.backends),
                     "soc_list": [soc for backend in self.backends.values() for soc in backend["soc_list"]],
                     "variants": [copy.deepcopy(self.variant)], "requested_target_ids": ["fixture"],
                     "payload_sha256": hashes,
                     "native_sha256": {name: value for name, value in hashes.items() if name.endswith(".so")}}

    def test_branch_and_abi_select_exactly_one_framework(self):
        """Post suffixes do not gate selection; another branch or ABI must not load this variant."""
        actual = {"torch_version": "2.12.0+cpu", "pta_version": "2.12.0.post1", "python_abi": "cp310",
                  "host_arch": "aarch64", "torch_cxx11_abi": True}
        self.assertEqual(manifest.select_variant(self.index, actual)["id"], "fixture")
        for key, value in (("pta_version", "2.13.0"), ("torch_cxx11_abi", False), ("python_abi", "cp311")):
            with self.subTest(field=key), self.assertRaises(ValueError):
                manifest.select_variant(self.index, {**actual, key: value})

    def test_manifest_rejects_ambiguous_and_redirected_adapters(self):
        """A framework variant cannot be ambiguous, swapped or redirected outside its payload."""
        invalid = copy.deepcopy(self.index)
        invalid["variants"].append(copy.deepcopy(invalid["variants"][0]))
        self.index_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            manifest.read_manifest(self.root)
        invalid = copy.deepcopy(self.index)
        invalid["variants"][0]["shmem_so"]["arch35"] = self.variant["torch_so"]
        self.index_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "adapter paths"):
            manifest.read_manifest(self.root)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            manifest.resource_path(self.root, "../external.so")

    def test_complete_payload_integrity_rejects_changes_and_missing_files(self):
        """Both membership and bytes are authenticated before loading any selected library."""
        manifest.check_integrity(self.root, self.info, self.index)
        for relative in (self.variant["torch_so"], "shmem/lib/arch35/shmem/aclshmem_bootstrap_uid.so"):
            library = self.root / relative
            original = library.read_bytes()
            library.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed payload"):
                manifest.check_integrity(self.root, self.info, self.index)
            library.unlink()
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                manifest.check_integrity(self.root, self.info, self.index)
            library.write_bytes(original)

    def test_device_arch_selects_only_a_packaged_backend(self):
        """Architectures select their own SDK, and absent backends never fall back."""
        selected = {"manifest": self.index}
        for arch, name in ((2201, "arch22"), (3510, "arch35")):
            with patch.object(_loader, "query_npu_arch", return_value=arch):
                self.assertEqual(_loader.select_native_shmem(selected)[0], name)
        del self.index["architectures"]["arch35"]
        for arch in (3510, 0):
            with patch.object(_loader, "query_npu_arch", return_value=arch), \
                    self.assertRaisesRegex(_loader.NativeComponentUnavailableError, "No packaged SHMEM backend"):
                _loader.select_native_shmem(selected)

    def test_arch_query_uses_visible_device_zero_and_propagates_failure(self):
        """The ACL query needs no selected device or framework initialization."""
        def query(device: int, attribute: int, output: object) -> int:
            """Write the fixture architecture through the ACL output pointer."""
            self.assertEqual((device, attribute), (0, 601))
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int64))[0] = 3510
            return 0

        function = Mock(side_effect=query)
        with patch.object(_loader.ctypes, "CDLL", return_value=SimpleNamespace(aclrtGetDeviceInfo=function)):
            self.assertEqual(_loader.query_npu_arch(), 3510)
            function.side_effect = None
            function.return_value = 507899
            with self.assertRaisesRegex(_loader.NativeComponentUnavailableError, "507899"):
                _loader.query_npu_arch()

    def test_shmem_first_use_loads_one_local_dependency_chain(self):
        """Concurrent first use loads only the chosen backend and subsequent calls reuse it."""
        # pylint: disable=protected-access
        selected = {"root": self.root, "manifest": self.index, "variant": self.variant}
        extension = SimpleNamespace()
        spec = SimpleNamespace(loader=Mock())
        with patch.multiple(shmem_runtime, require_multicore_environment=lambda: None,
                            load_torch_npu=lambda: None, validate_native_compatibility=lambda: selected,
                            _native=None, _native_libraries=[]), \
                patch.object(_loader, "_loaded_kinds", set()), patch.object(_loader, "_native_failure", None), \
                patch.object(_loader, "query_npu_arch", return_value=3510) as query, \
                patch.object(shmem_runtime.ctypes, "CDLL") as load, \
                patch.object(shmem_runtime.importlib.util, "spec_from_file_location", return_value=spec), \
                patch.object(shmem_runtime.importlib.util, "module_from_spec", return_value=extension), \
                patch.dict(sys.modules):
            with ThreadPoolExecutor(max_workers=4) as pool:
                natives = list(pool.map(lambda _: shmem_runtime._load_native(), range(4)))
            self.assertTrue(all(native is extension for native in natives))
            self.assertEqual(load.call_count, len(manifest.SHMEM_DEPENDENCIES))
            for call, relative in zip(load.call_args_list, manifest.SHMEM_DEPENDENCIES):
                self.assertEqual(call.args, (str(self.root / self.backends["arch35"]["shmem_root"] / relative),))
                self.assertEqual(call.kwargs, {"mode": ctypes.RTLD_LOCAL})
            spec.loader.exec_module.assert_called_once_with(extension)
            queries = query.call_count
            self.assertIs(shmem_runtime._load_native(), extension)
            self.assertEqual(query.call_count, queries)

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

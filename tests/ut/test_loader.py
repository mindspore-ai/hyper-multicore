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
"""Unit tests for unified multicore payload lookup and OPP diagnostics."""

import ctypes
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hyper_multicore import _loader
from hyper_multicore._loader import NativeComponentUnavailableError
from tests.common.mark_utils import arg_mark

class TestMulticoreNative(unittest.TestCase):
    """Verify component-owned OPP environment and payload lookup."""

    def setUp(self) -> None:
        """Create a local native root with one vendor and the Torch adapter directory."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.native_root = self.root / "hyper_multicore" / "lib"
        self.vendor_root = (
            self.native_root / "vendors" / "hyper_multicore_nn"
        )
        library = self.vendor_root / "op_api" / "lib" / "libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"vendor")
        adapter = (
            self.native_root / "framework" / "torch"
            / "libhyper_multicore_torch.so"
        )
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"adapter")
        self.shmem_root = self.native_root.parent / "shmem" / "lib"
        (self.shmem_root / "shmem").mkdir(parents=True)
        identity = patch.object(_loader, "validate_native_compatibility", return_value={
            "root": self.native_root.parent, "variant": {"torch_so": "adapters/test/libhyper_multicore_torch.so"}})
        identity.start()
        self.addCleanup(identity.stop)

    def test_vendor_preload_keeps_dependency_symbols_local(self) -> None:
        """Load the exact vendor locally and preserve the cause of any load failure."""
        library = self.vendor_root / "op_api" / "lib" / "libcust_opapi.so"
        error = OSError("missing dependency")
        for failure in (None, error):
            with patch.object(_loader.ctypes, "CDLL", side_effect=failure) as load:
                if failure is None:
                    _loader.preload_vendor_library(self.vendor_root)
                else:
                    with self.assertRaisesRegex(NativeComponentUnavailableError, "missing dependency") as raised:
                        _loader.preload_vendor_library(self.vendor_root)
                    self.assertIs(raised.exception.__cause__, error)
                    self.assertIn(str(library), str(raised.exception))
                load.assert_called_once_with(str(library), mode=ctypes.RTLD_LOCAL)

    def test_backend_import_failure_keeps_cause_and_metadata_context(self):
        """Broken PTA imports still expose the runtime and packaged matrix without another framework import."""
        (self.native_root.parent / "build_info.json").write_text(json.dumps({"variants": [
            {"id": "built-fixture", "torch_version": "2.7.1+cpu", "pta_version": "2.7.1.post11", "torch_cxx11_abi": True}]}))
        cann = self.root / "cann/opp"
        cann.mkdir(parents=True)
        (cann / "version.info").write_text("Version=9.2.0\n")
        error = ImportError("missing libhccl.so")
        _loader.load_torch_npu.cache_clear()
        self.addCleanup(_loader.load_torch_npu.cache_clear)
        with patch.object(_loader, "_component_root", return_value=self.native_root), patch.dict(
            os.environ, {"ASCEND_HOME_PATH": str(cann.parent)}
        ), patch.object(_loader.metadata, "version", side_effect=lambda name: {"torch": "2.7.1+cpu", "torch-npu": "2.7.1.post11"}[name]), patch.object(
            _loader.importlib, "import_module", side_effect=error
        ) as imports, self.assertRaisesRegex(NativeComponentUnavailableError, "PTA-IMPORT-FAILED") as raised:
            _loader.load_torch_npu()
        self.assertIs(raised.exception.__cause__, error)
        for expected in ("2.7.1.post11", "9.2.0", "built-fixture", "missing libhccl.so"):
            self.assertIn(expected, str(raised.exception))
        imports.assert_called_once_with("torch_npu")

    def test_component_paths_accept_sourced_environment_without_modifying_it(self):
        """Lookup accepts the sourced vendor paths without changing the process environment."""
        op_api_root = self.vendor_root / "op_api" / "lib"
        environment = {
            "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(["preexisting", str(self.vendor_root)]),
            "LD_LIBRARY_PATH": os.pathsep.join([
                "existing-lib",
                str(op_api_root),
                str(self.shmem_root),
                str(self.shmem_root / "shmem"),
            ]),
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ), patch.object(_loader, "sys", SimpleNamespace(modules={})):
            vendor_root, default_adapter = _loader.get_multicore_paths()
            opp_value = os.environ["ASCEND_CUSTOM_OPP_PATH"]
            library_value = os.environ["LD_LIBRARY_PATH"]

        self.assertEqual(vendor_root, self.vendor_root.resolve())
        self.assertEqual(default_adapter.name, "libhyper_multicore_torch.so")
        self.assertEqual(opp_value, environment["ASCEND_CUSTOM_OPP_PATH"])
        self.assertEqual(library_value, environment["LD_LIBRARY_PATH"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_opp_environment_requires_set_env(self):
        """Feature: MegaKernel diagnostics.
        Description: The loader requires explicit environment activation before a framework is loaded.
        Expectation: Missing opp environment requires set env.
        """
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _loader, "sys", SimpleNamespace(modules={})
        ), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ), self.assertRaisesRegex(
            NativeComponentUnavailableError,
            f"HP-NATIVE-OPP-NOT-ACTIVATED.*source {re.escape(str(self.native_root / 'set_env.bash'))}",
        ):
            _loader.require_multicore_environment()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_opp_environment_after_framework_import_is_too_late(self):
        """Feature: MegaKernel diagnostics.
        Description: The loader rejects activation after framework initialization.
        Expectation: Missing opp environment after framework import is too late.
        """
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _loader, "sys", SimpleNamespace(modules={"torch": object()})
        ), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ), self.assertRaisesRegex(
            NativeComponentUnavailableError,
            f"HP-NATIVE-OPP-ACTIVATION-TOO-LATE.*source {re.escape(str(self.native_root / 'set_env.bash'))}",
        ):
            _loader.require_multicore_environment()

# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Artifact-label and fail-closed payload tests without the component runtime."""

from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from packaging.version import Version

from tests.ut.build._build_test_support import ROOT, load_module, module_for

HELPER = load_module("artifacts")
from hyper_multicore import _loader as LOADER


def _payload_fixture(root, socs=("ascend910b",)):
    """Create a complete synthetic runtime layout without loading native code."""
    resources = [*HELPER.REQUIRED_LIBRARIES, "adapters/manifest.json",
                 "adapters/fixture/libhyper_multicore_torch.so", "adapters/fixture/hyper_multicore_shmem_torch.so", "lib/set_env.bash", "lib/vendors/hyper_multicore_nn/version.info",
                 "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
                 "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
                 "lib/vendors/hyper_multicore_nn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"]
    base = "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe"
    indexes = {}
    for soc in socs:
        binaries = {}
        resources.extend((f"{base}/config/{soc}/aic-{soc}-ops-info.json",
                          f"{base}/kernel/config/{soc}/binary_info_config.json"))
        for operator in ("hyper_mega_moe", "hyper_mega_moe_grad", "hyper_mega_mhc", "hyper_mega_mhc_grad",
                         "hyper_mega_gate_route", "hyper_mega_gate_route_grad"):
            variants = ("kernel", "alternative") if operator == "hyper_mega_moe" else ("kernel",)
            binary_list, bin_list = [], []
            for variant in variants:
                relative = f"{soc}/{operator}/{variant}"
                resources.extend((f"{base}/kernel/{relative}.o", f"{base}/kernel/{relative}.json"))
                binary_list.append({"binPath": relative + ".o", "jsonPath": relative + ".json"})
                bin_list.append({"binInfo": {"jsonFilePath": relative + ".json"}})
            resources.append(f"{base}/kernel/config/{soc}/{operator}.json")
            indexes[f"{base}/kernel/config/{soc}/{operator}.json"] = {"binList": bin_list}
            binaries[operator] = {"binaryList": binary_list}
        indexes[f"{base}/kernel/config/{soc}/binary_info_config.json"] = binaries
    for relative in resources:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"fixture")
    for relative, contents in indexes.items():
        (root / relative).write_text(json.dumps(contents), encoding="utf-8")
    variant = {"id": "fixture", "torch_release": "2.12.0", "pta_release": "2.12.0", "torch_version": "2.12.0+cpu",
               "pta_version": "2.12.0", "torch_cxx11_abi": True, "torch_so": "adapters/fixture/libhyper_multicore_torch.so",
               "shmem_so": "adapters/fixture/hyper_multicore_shmem_torch.so"}
    (root / "adapters/manifest.json").write_text(json.dumps({"schema_version": 1, "python_abi": "cp310",
                                                            "host_arch": "aarch64", "variants": [variant]}))
    hashes = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in resources}
    return {"schema_version": 2, "build_mode": "current", "variants": [variant], "requested_target_ids": ["fixture"], "python_abi": "cp310",
            "pta_version": "2.12.0", "artifact_version": "0.1.0.dev0+pt.2.12.0", "soc_list": list(socs), "host_arch": "aarch64",
            "native_sha256": {name: digest for name, digest in hashes.items() if name.endswith(".so")},
            "payload_sha256": hashes}


class TestArtifacts(unittest.TestCase):
    """Keep PTA labels and required native inputs explicit."""

    def test_pta_version_and_post_release(self):
        """PTA supplies the complete public label, not the Torch version."""
        self.assertEqual(HELPER.artifact_version("2.9.0"), "0.1.0.dev0+pt.2.9.0")
        self.assertEqual(HELPER.artifact_version("2.9.0.post4"), "0.1.0.dev0+pt.2.9.0.post4")
        self.assertEqual(HELPER.artifact_version("2.9.0.post4+custom.1"),
                         "0.1.0.dev0+pt.2.9.0.post4.custom.1")
        for value in ("2.15.0.dev20261005+git2c8682d", "internal build/42", "x" * 300):
            with self.subTest(value=value):
                version = HELPER.artifact_version(value)
                self.assertEqual(str(Version(version)), version)
        self.assertNotEqual(HELPER.artifact_version("internal build/42"), HELPER.artifact_version("internal/build 42"))

    def test_setup_reads_requirements_and_shared_component_version(self):
        """Source and native metadata leave framework versions unconstrained."""
        root = ROOT
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "setup.py").write_text((root / "setup.py").read_text(), encoding="utf-8")
            (project / "README.md").write_text("# Fixture\n", encoding="utf-8")
            (project / "requirements.txt").write_text(
                "# Runtime dependencies\ntorch\n\nnumpy>=1.26.4,<2\n", encoding="utf-8",
            )
            payload = project / "payload"
            payload.mkdir()
            (payload / "build_info.json").write_text(json.dumps({
                "artifact_version": "7.8.9.dev0+pt.2.12.0", "torch_version": "2.12.0+cpu",
            }), encoding="utf-8")
            for native in (False, True):
                with self.subTest(native=native), patch.dict(
                    os.environ, {"HYPER_MULTICORE_NATIVE_OUTPUT_ROOT": str(payload) if native else ""},
                ), patch.object(sys, "argv", [str(project / "setup.py"), "egg_info"]), patch(
                    "setuptools.setup",
                ) as setup, patch("_hyper_multicore_bootstrap.load_module", return_value=type("Metadata", (), {"COMPONENT_VERSION": "7.8.9.dev0"})):
                    runpy.run_path(str(project / "setup.py"))
                    setup_metadata = setup.call_args.kwargs
                    self.assertEqual(setup_metadata["version"], "7.8.9.dev0+pt.2.12.0" if native else "7.8.9.dev0")
                    self.assertEqual(setup_metadata["install_requires"], [
                        "torch", "numpy>=1.26.4,<2",
                    ])
                    self.assertNotIn("extras_require", setup_metadata)
                    self.assertEqual(setup_metadata["license_files"], [
                        "LICENSE", "Third_Party_Open_Source_Software_Notice",
                    ])

    def test_missing_pta_is_not_guessed(self):
        """Native identity creation must fail when torch-npu is not installed."""
        with patch.object(HELPER.sys, "version_info", (3, 13)), patch.object(
            HELPER._MANIFEST, "framework_package", side_effect=ModuleNotFoundError("torch_npu")
        ), self.assertRaises(ModuleNotFoundError):
            HELPER.build_identity(Path("."), "ascend910b")

    def test_missing_native_payload_is_rejected(self):
        """A pure-Python tree is never accepted as a complete native wheel payload."""
        with tempfile.TemporaryDirectory() as directory:
            info = {"pta_version": "2.9.0", "artifact_version": "0.1.0.dev0+pt.2.9.0"}
            with self.assertRaisesRegex(RuntimeError, "Missing native payload"):
                HELPER.validate_payload(Path(directory), info)

    def test_shared_adapter_rejects_extra_framework_libraries(self):
        """A stale split adapter cannot be packaged alongside the shared adapter."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = _payload_fixture(root)
            HELPER.validate_payload(root, info)
            extra = root / "lib/framework/torch/stale_adapter.so"
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_bytes(b"fixture")
            info["native_sha256"][extra.relative_to(root).as_posix()] = hashlib.sha256(extra.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "Expected only the shared Torch adapter"):
                HELPER.validate_payload(root, info)

    def test_payload_requires_vendor_resources_and_complete_manifest(self):
        """Libraries alone and an old SO-only manifest cannot certify a runtime."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = _payload_fixture(root)
            del info["payload_sha256"]
            with self.assertRaisesRegex(RuntimeError, "Missing complete payload checksum manifest"):
                HELPER.validate_payload(root, info)
            for path in (root / "lib/vendors/hyper_multicore_nn/op_impl").rglob("*"):
                if path.is_file():
                    path.unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing native payload resource"):
                HELPER.validate_payload(root, info)

    def test_payload_rejects_missing_resources_even_if_manifest_is_regenerated(self):
        """Each selected SoC needs all six kernels, metadata, indexes and configs."""
        base = "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe"
        cases = (
            "kernel/ascend910b/hyper_mega_moe/kernel.o",
            "kernel/ascend910b/hyper_mega_gate_route/kernel.json",
            "kernel/config/ascend910b/binary_info_config.json",
            "config/ascend910b/aic-ascend910b-ops-info.json",
            "kernel/config/ascend910b/hyper_mega_mhc_grad.json",
            "kernel/ascend910_93/hyper_mega_mhc/kernel.o",
            "op_tiling/liboptiling.so",
            "op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
        )
        for relative in cases:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                info = _payload_fixture(root, ("ascend910b", "ascend910_93"))
                (root / base / relative).unlink()
                info["payload_sha256"] = HELPER._payload_hashes(root)
                with self.assertRaisesRegex(RuntimeError, "native payload resource"):
                    HELPER.validate_payload(root, info)

    def test_finalize_rejects_missing_variant_and_proto_library(self):
        """Regenerating hashes cannot hide one missing variant or Host runtime library."""
        base = "lib/vendors/hyper_multicore_nn/"
        cases = (
            base + "op_impl/ai_core/tbe/kernel/ascend910b/hyper_mega_moe/alternative.o",
            base + "op_impl/ai_core/tbe/kernel/ascend910b/hyper_mega_moe/alternative.json",
            base + "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so",
        )
        for relative in cases:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "payload"
                info = _payload_fixture(root)
                (root / relative).unlink()
                identity = Path(directory) / "identity.json"
                identity.write_text(json.dumps(info), encoding="utf-8")
                with patch.object(sys, "argv", ["artifacts.py", "--output", str(root), "--finalize", str(identity)]):
                    with self.assertRaisesRegex(RuntimeError, "native payload resource"):
                        HELPER.main()
                self.assertFalse((root / "build_info.json").exists())

    def test_payload_rejects_changed_missing_and_unrecorded_files(self):
        """All resources, including scripts and optional vendor files, are protected."""
        cases = (
            ("change", "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe/kernel/ascend910b/hyper_mega_moe/kernel.json"),
            ("change", "lib/set_env.bash"),
            ("delete", "lib/vendors/hyper_multicore_nn/version.info"),
            ("add", "lib/unrecorded.dat"),
        )
        for action, relative in cases:
            with self.subTest(action=action, relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                info = _payload_fixture(root)
                path = root / relative
                if action == "delete":
                    path.unlink()
                else:
                    path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "Changed payload resource|Payload file set mismatch"):
                    HELPER.validate_payload(root, info)

    def test_finalize_records_resources_without_replacing_native_identity(self):
        """Manifest finalization is repeatable and preserves the original build identity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "payload"
            info = _payload_fixture(root)
            info.pop("payload_sha256")
            identity = {"build_commit": "original-commit", "source_sha256": "original-source",
                        "build_fingerprint": "original-fingerprint"}
            info.update(identity)
            identity_path = Path(directory) / "identity.json"
            identity_path.write_text(json.dumps(info), encoding="utf-8")
            for _ in range(2):
                with patch.object(sys, "argv", ["artifacts.py", "--output", str(root),
                                                 "--finalize", str(identity_path)]):
                    HELPER.main()
                actual = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
                self.assertEqual({key: actual[key] for key in identity}, identity)
                self.assertEqual(actual["payload_sha256"], HELPER._payload_hashes(root))
                self.assertNotIn("build_info.json", actual["payload_sha256"])
                HELPER.validate_payload(root, actual)

    def test_ini_changes_source_identity_without_git(self):
        """Operator build configuration contributes to the unpacked source identity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "hyper_multicore/ops/example/op_host/config/ascend910b/simplified_key.ini"
            config.parent.mkdir(parents=True)
            (root / "scripts").mkdir()
            for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
                (root / name).write_text("fixture", encoding="utf-8")
            config.write_text("key=1", encoding="utf-8")
            original = HELPER._source_identity(root)
            config.write_text("key=2", encoding="utf-8")
            changed = HELPER._source_identity(root)
            self.assertIsNone(changed["build_commit"])
            self.assertNotEqual(original["source_sha256"], changed["source_sha256"])
            config.write_text("key=1", encoding="utf-8")
            self.assertEqual(original, HELPER._source_identity(root))

    def test_sdist_does_not_inherit_an_enclosing_repository_commit(self):
        """Unpacked sources use their digest, never a parent directory's unrelated Git HEAD."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hyper_multicore").mkdir()
            (root / "scripts").mkdir()
            for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
                (root / name).write_text("fixture\n", encoding="utf-8")
            with patch.object(HELPER.subprocess, "run", side_effect=AssertionError("unexpected Git lookup")):
                identity = HELPER._source_identity(root)
            self.assertIsNone(identity["build_commit"])
            self.assertEqual(len(identity["source_sha256"]), 64)

    def test_native_ctest_source_changes_frozen_build_identity(self):
        """The compiled CTest input must be included even though it lives outside the package."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
                (root / name).write_text("fixture")
            native_test = root / "tests/ut/shmem/native_log_test.cpp"
            native_test.parent.mkdir(parents=True)
            before = HELPER._source_identity(root)
            native_test.write_text("first CTest source")
            present = HELPER._source_identity(root)
            native_test.write_text("changed CTest source")
            changed = HELPER._source_identity(root)
            self.assertNotEqual(before["source_sha256"], present["source_sha256"])
            self.assertNotEqual(present["source_sha256"], changed["source_sha256"])

    def test_native_environment_selects_branch_and_rejects_abi_mismatches(self):
        """External post/CANN/hash differences are diagnostic; branch and ABI boundaries remain."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = _payload_fixture(root)
            (root / "build_info.json").write_text(json.dumps(info))
            actual = {"torch_version": "2.12.0+cpu", "pta_version": "2.12.0", "python_abi": "cp310",
                      "host_arch": "aarch64", "torch_cxx11_abi": True, "cann_version": "9.2.0"}
            validate = LOADER.validate_native_compatibility
            for key, value in (("pta_version", "2.12.0.post99"), ("cann_version", "unknown"),
                               ("cann_version", "9.1.0"), ("libtorch_npu_sha256", "other"), ("torch_commit", "other")):
                with self.subTest(allowed=key), patch.multiple(module_for(validate), **{'_component_root': lambda : root / 'lib', '_selection': None, '_runtime_identity': lambda : {**actual, key: value}}):
                    self.assertEqual(validate()["variant"]["id"], "fixture")
            for key, value in (("pta_version", "2.9.0"), ("torch_version", "2.9.0"),
                               ("python_abi", "cp311"), ("host_arch", "x86_64"), ("torch_cxx11_abi", False)):
                with self.subTest(rejected=key), patch.multiple(module_for(validate), **{'_component_root': lambda : root / 'lib', '_selection': None, '_runtime_identity': lambda : {**actual, key: value}}), self.assertRaisesRegex(RuntimeError, "ABI-MISMATCH"):
                    validate()


if __name__ == "__main__":
    unittest.main()

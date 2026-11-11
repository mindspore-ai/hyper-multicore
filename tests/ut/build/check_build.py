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
"""Fast source-build checks using temporary data and no framework or device runtime."""

from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_hyper_multicore_bootstrap", ROOT / "hyper_multicore/_build/_hyper_multicore_bootstrap.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)
BUILD = BOOTSTRAP.load_module("build_controller")
PUBLICATION = BOOTSTRAP.load_module("build_publication")
FRAMEWORK = BOOTSTRAP.load_module("framework_environments")
VENDOR = BOOTSTRAP.load_module("vendor_cache")
WORKSPACE = BOOTSTRAP.load_module("build_workspace")


def write(path, text="fixture"):
    """Create one small fixture without touching checkout outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class BuildChecks(unittest.TestCase):
    """Protect dependency requests, cached kernels, output commits and owned paths."""

    def setUp(self):
        """Keep every filesystem effect and environment setting local to this check."""
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(
            tempfile.TemporaryDirectory(prefix="hyper-multicore-build-check-")))
        self.resources.enter_context(patch.dict(os.environ, HYPER_MULTICORE_VENV_PATH=str(self.root / "envs")))

    def test_framework_sources_and_environment_reuse(self):
        """Exact roots use configured mirrors and a successful venv is never reinstalled."""
        target = {"id": "fixture", "torch_version": "1.0.0", "pta_build_version": "1.0.0",
                  "torch_release": "1.0.0", "pta_release": "1.0.0", "build_requirements": ["numpy"],
                  "indexes": {"torch": "https://torch.example/simple", "pta": "https://pta.example/simple",
                              "dependencies": "https://pypi.example/simple",
                              "dependencies_extra": ["https://mirror.example/simple"]}}
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            if "--dest" in command:
                name, version = command[-1].split("==")
                name = name.replace("-", "_")
                destination = Path(command[command.index("--dest") + 1])
                with zipfile.ZipFile(destination / f"{name}-{version}-py3-none-any.whl", "w") as archive:
                    archive.writestr(f"{name}-{version}.dist-info/METADATA",
                                     f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
            if "--report" in command:
                write(Path(command[command.index("--report") + 1]), '{"install": []}')

        def probe(python, expected):
            self.assertEqual(python.parent.parent.parent, self.root / "envs")
            self.assertEqual(expected, {"torch": "1.0.0", "torch-npu": "1.0.0"})
            return [{"name": name, "version": version} for name, version in expected.items()]

        python = FRAMEWORK.prepare_environment(target, self.root, run, Path(sys.executable), identity_probe=probe)
        installs = [command for command in commands if "--index-url" in command]
        self.assertEqual(len(installs), 3)
        for command in installs:
            self.assertEqual(command[command.index("--extra-index-url") + 1], "https://mirror.example/simple")
        self.assertEqual([command[command.index("--index-url") + 1] for command in installs],
                         [target["indexes"][name] for name in ("torch", "pta", "dependencies")])
        commands.clear()
        self.assertEqual(FRAMEWORK.prepare_environment(
            target, self.root, run, Path(sys.executable), identity_probe=probe), python)
        self.assertFalse(commands)
        with patch.object(FRAMEWORK, "verify_environment", side_effect=RuntimeError("framework changed")):
            with self.assertRaisesRegex(RuntimeError, "framework changed"):
                FRAMEWORK.prepare_environment(target, self.root, run, Path(sys.executable))

    def test_vendor_reuse_rejects_changed_source_and_corrupt_output(self):
        """An intact snapshot is reusable; local header edits and damaged kernels are misses."""
        sdk = self.root / "sdk"
        component = self.root / "hyper_multicore"
        for name in VENDOR.SCRIPTS:
            write(component / name)
        for name in VENDOR.SOURCE_TREES:
            write(component / name / "input.h")
        for name in ("include", "src/device", "lib"):
            write(sdk / name / "input")
        header = write(component / "shmem/ccsrc/data_plane/rma.h", "original")
        cann = self.root / "cann"
        write(cann / "opp/version.info", "Version=9.2.0")
        candidate = self.root / "work/vendor"
        write(candidate / "kernel.o")
        request = self.root / "request.json"
        with patch.dict(os.environ, ASCEND_HOME_PATH=str(cann), HP_VENDOR_ORIGIN='{"build_commit":"original"}'), \
                patch.object(VENDOR, "tool_inputs", return_value={"compiler": "fixture"}):
            VENDOR.prepare(self.root, self.root / "native", "ascend910b", sdk, request)
            VENDOR.built(request, candidate, time.time())
            VENDOR.publish(self.root, [json.loads(request.read_text(encoding="utf-8"))])
            VENDOR.prepare(self.root, self.root / "native", "ascend910b", sdk, request)
            hit = json.loads(request.read_text(encoding="utf-8"))
            self.assertTrue(hit["reused"])
            write(header, "changed")
            VENDOR.prepare(self.root, self.root / "native", "ascend910b", sdk, request)
            self.assertFalse(json.loads(request.read_text(encoding="utf-8"))["reused"])
            write(header, "original")
            write(Path(hit["directory"]) / "vendor/kernel.o", "corrupted")
            VENDOR.prepare(self.root, self.root / "native", "ascend910b", sdk, request)
            self.assertFalse(json.loads(request.read_text(encoding="utf-8"))["reused"])

    @contextmanager
    def staged_build(self):
        """Run actual publication with only expensive preparation and native workers replaced."""
        candidate = write(self.root / "staging/candidate/new.so").parent
        wheel = write(self.root / "staging/candidate.whl")
        outputs = WORKSPACE.BuildOutputs(self.root / "dist", self.root / "native/payload/hyper_multicore")
        write(outputs.payload / "old.so")
        write(outputs.distribution / "old.whl")
        replacements = {
            "ROOT": self.root, "NATIVE": self.root / "native", "validate_toolchain": lambda: None,
            "prepare_frameworks": lambda *_: ([], [Path(sys.executable)], [{}]),
            "build_common": lambda *_: (self.root / "common", self.root / "sdk"),
            "build_adapters": lambda *_: ([], [{"pta_version": "fixture"}]),
            "assemble_payload": lambda *_: (candidate, {}), "package_wheel": lambda *_: wheel,
        }
        with patch.multiple(BUILD, **replacements), patch.object(PUBLICATION.SourceGuard, "observe", return_value={}):
            yield SimpleNamespace(all=False, tests=False, native_only=False), outputs

    def test_precommit_failure_leaves_no_formal_outputs(self):
        """A payload switch failure also removes the already published wheel."""
        with self.staged_build() as (args, outputs), \
                patch.object(BUILD, "publish_payload", side_effect=OSError("publication interrupted")):
            with self.assertRaisesRegex(OSError, "publication interrupted"), outputs:
                BUILD.build(args, outputs)
            self.assertFalse(outputs.committed)
            self.assertFalse(outputs.payload.exists())
            self.assertFalse(list(outputs.distribution.iterdir()))

    def test_maintenance_failure_retains_committed_outputs(self):
        """Maintenance cannot undo the successfully committed artifact pair."""
        with self.staged_build() as (args, outputs), \
                patch.object(BUILD.VENDOR, "publish", side_effect=OSError("maintenance interrupted")):
            with outputs:
                BUILD.build(args, outputs)
            self.assertTrue(outputs.committed)
            self.assertTrue((outputs.payload / "new.so").is_file())
            self.assertEqual([path.name for path in outputs.distribution.iterdir()], ["candidate.whl"])

    def test_workspace_reset_preserves_unknown_and_redirected_data(self):
        """Only owned real workspaces may be erased for a rebuild."""
        workspace = self.root / "variants/target"
        manual = write(workspace / "manual.txt")
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            WORKSPACE.reset_workspace(workspace, self.root)
        self.assertTrue(manual.is_file())
        shutil.rmtree(workspace)
        outside = write(self.root / "outside/keep.txt")
        workspace.symlink_to(outside.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            WORKSPACE.reset_workspace(workspace, self.root)
        self.assertTrue(outside.is_file())

    def test_wheel_integrity_rejects_changed_and_missing_payload(self):
        """Payload membership and hashes reject invalid staged wheel contents."""
        content = b"native fixture"
        payload = self.root / "payload"
        write(payload / "build_info.json", "{}")
        write(payload / "adapters/manifest.json", "{}")
        relative = "adapters/fixture/libhyper_multicore_torch.so"
        info = {"payload_sha256": {relative: hashlib.sha256(content).hexdigest(),
                                  "adapters/manifest.json": hashlib.sha256(b"{}").hexdigest()}}
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("hyper_multicore/" + relative, content)
            archive.writestr("hyper_multicore/build_info.json", "{}")
            archive.writestr("hyper_multicore/adapters/manifest.json", "{}")
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            PUBLICATION.validate_wheel_payload(archive, members, payload, info)
            info["payload_sha256"][relative] = "0" * 64
            with self.assertRaisesRegex(ValueError, "payload mismatch"):
                PUBLICATION.validate_wheel_payload(archive, members, payload, info)
            with self.assertRaisesRegex(ValueError, "membership mismatch"):
                PUBLICATION.validate_wheel_payload(archive, [], payload, info)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)

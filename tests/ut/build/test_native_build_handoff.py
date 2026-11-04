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
"""Exercise the shell SDK handoff independently of vendor cache directories."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


from tests.ut.build._build_test_support import ROOT

class TestNativeBuildHandoff(unittest.TestCase):
    """The SDK manifest belongs to the producer, not the vendor fingerprint."""

    def test_sdk_handoff_survives_vendor_work_root_fingerprinting(self) -> None:
        """Read the real handoff with an independently fingerprinted work root."""
        root = ROOT
        script = (root / "hyper_multicore/build.sh").read_text(encoding="utf-8")
        handoff = script.split('CURRENT_REASON_CODE="SHMEM_SDK_HANDOFF_FAILED"\n', 1)[1]
        handoff = handoff.split('if [[ ! -f "${SHMEM_INSTALL_ROOT}', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "native"
            manifest = native / "work/multicore/shmem/sdk.json"
            manifest.parent.mkdir(parents=True)
            sdk = str(native / "work/multicore/shmem/toolchain-fixture/shmem-install")
            manifest.write_text(json.dumps({"install_root": sdk, "toolchain_key": "fixture"}), encoding="utf-8")
            environment = dict(os.environ, NATIVE_ROOT=str(native),
                               WORK_ROOT=str(native / "work/multicore/build-fingerprint"), PYTHON_BIN=sys.executable)
            command = handoff + '\nprintf "%s\\n%s\\n" "$SHMEM_INSTALL_ROOT" "$HP_SHMEM_TOOLCHAIN_KEY"'
            result = subprocess.run(["bash", "-euc", command], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [sdk, "fixture"])


    def test_common_shmem_does_not_require_framework(self) -> None:
        """Run the actual shell preflight with a Python that rejects framework imports."""
        script = (ROOT / "hyper_multicore/shmem/build.sh").read_text()
        preflight = 'if ! command -v "${PYTHON_BIN}"' + script.split('if ! command -v "${PYTHON_BIN}"', 1)[1]
        preflight = preflight.split('source "${PROJECT_ROOT}/hyper_multicore/shmem/_build/shmem_sdk.sh"', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text(f"#!{sys.executable}\nimport os, sys\n"
                              "if 'import torch' in sys.argv[-1]:\n"
                              "    if os.environ.get('ALLOW_FRAMEWORK') == '1':\n"
                              "        compile(sys.argv[-1], '<probe>', 'exec')\n"
                              "        raise SystemExit(0)\n"
                              "    raise SystemExit(86)\n"
                              f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n")
            python.chmod(0o755)
            command = 'fail() { printf "%s\\n" "$1"; exit "$3"; };\n' + preflight
            for stage, available, expected in (("common", "0", 0), ("binding", "0", 4), ("binding", "1", 0)):
                env = dict(os.environ, PYTHON_BIN=str(python), HP_NATIVE_STAGE=stage, ALLOW_FRAMEWORK=available)
                result = subprocess.run(["bash", "-euc", command], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                if expected:
                    self.assertIn("TORCH_BUILD_DEPENDENCY_NOT_FOUND", result.stdout)


if __name__ == "__main__":
    unittest.main()


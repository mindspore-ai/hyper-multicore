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
"""Protect architecture isolation of upstream source adapters without an NPU."""

# The assertions verify internal ownership and the native ABI.
# pylint: disable=protected-access
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "hyper_multicore/_build"))
ASSEMBLE = importlib.import_module("assemble_multicore_source")


class SourceAdapters(unittest.TestCase):
    """Verify that architecture-specific source adapters remain isolated."""
    def test_arch35_adapter_is_not_applied_to_arch22_export(self):
        """Apply the source lock separately to arch22 and arch35 exports."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            inputs.mkdir()
            source = inputs / "entry.h"
            source.write_text("original\n")
            adapter = root / "entry.patch"
            adapter.write_text("--- a/entry.h\n+++ b/entry.h\n@@ -1 +1 @@\n-original\n+adapted\n")
            lock = {"patches": [{"path": "entry.patch", "architectures": ["arch35"],
                                 "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}]}
            with patch.object(ASSEMBLE, "_REPO_ROOT", root):
                ASSEMBLE._apply_locked_adapters(inputs, lock, architecture="arch22")
                self.assertEqual(source.read_text(), "original\n")
                ASSEMBLE._apply_locked_adapters(inputs, lock, architecture="arch35")
                self.assertEqual(source.read_text(), "adapted\n")


if __name__ == "__main__":
    unittest.main()

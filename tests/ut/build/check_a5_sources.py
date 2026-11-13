"""Protect architecture isolation of upstream source adapters without an NPU."""
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
    def test_a5_adapter_is_not_applied_to_a3_export(self):
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

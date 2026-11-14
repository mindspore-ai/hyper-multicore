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
"""Compile the production ready-selection loop with controlled task readiness."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import hyper_multicore


class TestVectorReadyQueue(unittest.TestCase):
    """Check rolling-window coverage and recording-independent placement/order."""

    def test_ready_queue_selection_and_recording(self):
        """Compile and run the production queue's selection and recording checks."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the native queue regression")
        root = Path(hyper_multicore.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="hmc-ready-queue-") as directory:
            executable = str(Path(directory) / "queue")
            subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(root), str(Path(__file__).with_name("vector_ready_queue_test.cpp")),
                            "-o", executable], check=True)
            subprocess.run([executable], check=True, timeout=30)

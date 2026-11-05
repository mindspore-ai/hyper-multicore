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
"""Installed matrix inspection remains available without importing Torch or PTA."""

import json
import os
import shutil
import subprocess
import sys

from tests.ut.build.test_artifacts import _payload_fixture
from tests.ut.build._build_test_support import ROOT


def test_matrix_json_works_without_framework_import(tmp_path):
    """Use an isolated fake installation to prove the CLI reads records, not native imports."""
    package = tmp_path / "hyper_multicore"
    info = _payload_fixture(package)
    (package / "build_info.json").write_text(json.dumps(info))
    shutil.copy2(ROOT / "hyper_multicore/_native_manifest.py", package / "_native_manifest.py")
    (package / "__init__.py").write_text("raise RuntimeError('must not import package')")
    for name in ("torch", "torch_npu"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("raise RuntimeError('must not import framework')")
    metadata = tmp_path / "hyper_multicore-0.1.0.dev0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: hyper-multicore\nVersion: 0.1.0.dev0\n")
    command = [sys.executable, str(ROOT / "scripts/hyper_multicore_build_info"), "--json"]
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["package_root"] == str(package)
    assert report["build_info"]["variants"][0]["id"] == "fixture"
    variant = report["build_info"]["variants"][0]
    assert variant["torch_so"] == "adapters/fixture/libhyper_multicore_torch.so"
    assert variant["shmem_so"] == "adapters/fixture/hyper_multicore_shmem_torch.so"


def test_text_matrix_aligns_every_vertical_separator():
    """Header, rule and data rows use the same positions even with long post versions."""
    import runpy

    formatter = runpy.run_path(str(ROOT / "scripts/hyper_multicore_build_info"))["format_text"]
    variants = [{"id": name, "torch_version": torch, "pta_version": pta, "torch_cxx11_abi": True}
                for name, torch, pta in [("pt2.7.1.post11-123456789abc", "2.7.1+cpu", "2.7.1.post11"),
                                         ("pt2.14.0-123456789abc", "2.14.0+cpu", "2.14.0")]]
    text = formatter({"package_version": "0.1.0.dev0", "package_root": "/package", "build_info": {
        "build_mode": "all", "python_abi": "cp310", "host_arch": "aarch64", "soc_list": ["ascend910b"],
        "variants": variants, "reference_cann": "9.2.0", "cann_version": "9.1.0"}, "runtime": {"python": "/python", "python_abi": "cp310", "host_arch": "aarch64",
        "torch": "not installed", "torch-npu": "not installed", "cann_version": "unknown"}})
    rows = [line for line in text.splitlines() if line.lstrip().startswith("|")]
    positions = [[i for i, char in enumerate(row) if char == "|"] for row in rows]
    assert len(rows) == 4 and len(positions[0]) == 4
    assert all(p == positions[0] for p in positions)
    assert "Variant" not in text and "adapters/" not in text
    assert all(variant["id"] not in text for variant in variants)
    assert "hyper_multicore_build_info --json" in text
    assert "Runtime" in text and "not installed" in text
    assert "CANN actual" in text and "9.1.0" in text
    assert "CANN reference" in text and "9.2.0" in text

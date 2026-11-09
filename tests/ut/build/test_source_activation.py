"""Source trees activate their own relocatable payload without an installed distribution."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.ut.build._build_test_support import ROOT
from tests.ut.build.test_artifacts import _payload_fixture


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source with spaces"
    payload = root / "build/native/payload/hyper_multicore"
    info = _payload_fixture(payload)
    shutil.copy2(ROOT / "hyper_multicore/set_env.bash", payload / "lib/set_env.bash")
    info["payload_sha256"]["lib/set_env.bash"] = hashlib.sha256((payload / "lib/set_env.bash").read_bytes()).hexdigest()
    (payload / "build_info.json").write_text(json.dumps(info))
    (root / "hyper_multicore").mkdir()
    shutil.copy2(ROOT / "hyper_multicore/_native_manifest.py", root / "hyper_multicore/_native_manifest.py")
    (root / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/hyper_multicore_build_info", root / "scripts/hyper_multicore_build_info")
    shutil.copy2(ROOT / "activate.bash", root / "activate.bash")
    return root


def test_activation_survives_move_and_is_idempotent(source_tree: Path, tmp_path: Path) -> None:
    moved = tmp_path / "relocated source"
    source_tree.rename(moved)
    command = 'source "$1/activate.bash" && source "$1/activate.bash" && exec "$2" -c "$3"'
    code = "import json,os; print(json.dumps({k:os.environ[k] for k in " + repr([
        "HYPER_MULTICORE_ROOT", "PYTHONPATH", "PATH", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"
    ]) + "}))"
    result = subprocess.run(["bash", "-c", command, "activation", str(moved), sys.executable, code],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["HYPER_MULTICORE_ROOT"] == str(moved)
    assert values["PYTHONPATH"].split(os.pathsep).count(str(moved)) == 1
    assert values["PATH"].split(os.pathsep).count(str(moved / "scripts")) == 1
    assert str(moved / "build/native/payload/hyper_multicore/lib/vendors/hyper_multicore_nn") in values[
        "ASCEND_CUSTOM_OPP_PATH"]
    assert str(source_tree) not in json.dumps(values)


def test_unbuilt_source_fails_without_an_install_fallback(tmp_path: Path) -> None:
    script = tmp_path / "activate.bash"
    shutil.copy2(ROOT / "activate.bash", script)
    result = subprocess.run(["bash", "-c", 'source "$1"', "activation", str(script)], text=True,
                            capture_output=True)
    assert result.returncode != 0
    assert "build.sh" in result.stderr


def test_build_info_reads_source_payload_without_distribution_metadata(source_tree: Path, tmp_path: Path) -> None:
    environment = {**os.environ, "HYPER_MULTICORE_ROOT": str(source_tree), "PYTHONPATH": ""}
    command = [sys.executable, str(source_tree / "scripts/hyper_multicore_build_info"), "--json"]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["package_root"] == str(source_tree / "build/native/payload/hyper_multicore")
    assert info["package_version"] == "0.1.0.dev0+pt.2.12.0"

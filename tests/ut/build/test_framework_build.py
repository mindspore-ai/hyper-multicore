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
"""Framework preparation, reuse and source isolation without real installation."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest
from tests.ut.build._build_test_support import ROOT, load_module, make_target, cached_resolution
ENV = load_module("framework_environments")
BUILD = load_module("build_controller")


def test_install_uses_verified_local_list_and_no_dependency_download(tmp_path):
    """A cached resolution installs offline without tests or manually pinned transitive inputs."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    resolution, record = cached_resolution(target, tmp_path, Path(sys.executable))
    commands = []
    def run(command, **_kwargs):
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            (Path(command[3]) / "bin").mkdir(parents=True)
            (Path(command[3]) / "bin/python").write_text("fixture")
    python = ENV.prepare_environment(target, tmp_path, run, Path(sys.executable))
    installation = next(c for c in commands if c[1:4] == ["-m", "pip", "install"])
    assert {"--no-index", "--no-cache-dir", "--no-deps", "--require-hashes", "--only-binary=:all:"} <= set(installation)
    listing = Path(installation[-1]).read_text()
    assert "https://" not in listing and "http://" not in listing
    assert all(str(ENV.STORE.wheel_path(tmp_path, p)) in listing for p in record["packages"])
    assert not any(c[1:4] == ["-m", "pip", "download"] for c in commands)
    assert python.parent.parent.joinpath("ready.json").exists()
    assert not {"pytest", "pybind11"} & {p["name"] for p in record["packages"]}
    commands.clear()
    assert ENV.prepare_environment(target, tmp_path, run, Path(sys.executable)) == python
    assert not any(c[1:4] in (["-m", "pip", "install"], ["-m", "pip", "download"]) for c in commands)


def test_environment_creation_failure_never_marks_ready(tmp_path):
    """A failed install leaves no ready marker and retries from verified local wheels."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    cached_resolution(target, tmp_path, Path(sys.executable))
    failing = True
    def _run(command, **_kwargs):
        if command[1:3] == ["-m", "venv"]:
            (Path(command[3]) / "bin").mkdir(parents=True)
        elif failing and command[1:4] == ["-m", "pip", "install"]:
            raise RuntimeError("offline missing dependency")
    with pytest.raises(RuntimeError, match="offline"):
        ENV.prepare_environment(target, tmp_path, _run, Path(sys.executable))
    assert not list(tmp_path.rglob("ready.json"))
    failing = False
    python = ENV.prepare_environment(target, tmp_path, _run, Path(sys.executable))
    assert (python.parent.parent / "ready.json").is_file()


def test_environment_isolation_preserves_cann_but_removes_framework_paths():
    """The selected interpreter cannot inherit another framework's CMake or SO paths."""
    with patch.dict(ENV.os.environ, {"PYTHONPATH": "wrong", "Torch_DIR": "wrong",
                    "LD_LIBRARY_PATH": ("/cann/lib64:/old/site-packages/torch/lib:"
                                        "/old/build/native/framework/envs/other/lib"),
                    "PIP_EXTRA_INDEX_URL": "https://wrong.invalid", "PIP_FIND_LINKS": "/wrong",
                    "PIP_CACHE_DIR": "/user/cache", "PIP_NO_CACHE_DIR": "1",
                    "ASCEND_HOME_PATH": "/cann", "CMAKE_PREFIX_PATH": "/cann/compiler:/old/site-packages/torch/share/cmake"}, clear=True):
        result = ENV.isolated_environment(Path("/owned/venv/bin/python"))
        pip = ENV.pip_environment(Path("/owned/venv/bin/python"))
    assert "PYTHONPATH" not in result and "Torch_DIR" not in result
    assert result["LD_LIBRARY_PATH"] == "/cann/lib64"
    assert result["ASCEND_HOME_PATH"] == "/cann"
    assert result["CMAKE_PREFIX_PATH"] == "/cann/compiler"
    assert result["PATH"].startswith("/owned/venv/bin")
    assert "PIP_EXTRA_INDEX_URL" not in pip and "PIP_FIND_LINKS" not in pip
    assert pip["PIP_CACHE_DIR"] == "/user/cache" and pip["PIP_NO_CACHE_DIR"] == "1"
    assert pip["PIP_CONFIG_FILE"] == os.devnull


def test_framework_compiler_override_does_not_change_common_toolchain():
    """A framework requiring newer C++ cannot redirect the CANN vendor compiler."""
    with patch.dict(ENV.os.environ, {"CXX": "/system/g++", "HYPER_MULTICORE_FRAMEWORK_CXX": "/framework/g++"}):
        common = ENV.isolated_environment(Path("/owned/venv/bin/python"))
        adapter = BUILD.adapter_environment(Path("/owned/venv/bin/python"))
        assert common["CXX"] == "/system/g++"
        assert adapter["CXX"] == "/framework/g++"
        assert ENV.os.environ["CXX"] == "/system/g++"


def test_cann_python_paths_are_scoped_to_compiler_workers(tmp_path):
    """CANN tools need their imports, while pip must not inspect unrelated CANN distributions."""
    cann = tmp_path / "cann"
    paths = [cann / "python/site-packages", cann / "opp/built-in/op_impl/ai_core/tbe"]
    for path in paths:
        path.mkdir(parents=True)
    python = Path("/owned/venv/bin/python")
    with patch.dict(ENV.os.environ, {"ASCEND_HOME_PATH": str(cann), "PYTHONPATH": "/old/framework:/user/source"}):
        installed = ENV.isolated_environment(python)
        compiler = ENV.cann_tool_environment(python)
    assert "PYTHONPATH" not in installed
    assert compiler["PYTHONPATH"].split(ENV.os.pathsep) == list(map(str, paths))
    assert compiler["HI_PYTHON"] == str(python)


def test_cann_precheck_fails_before_framework_installation(tmp_path):
    """A wrong CANN installation must be rejected before preparing the release matrix."""
    function = BUILD.validate_toolchain
    with patch.dict(BUILD.os.environ, {}, clear=True), pytest.raises(ValueError, match="source the CANN"):
        function()
    (tmp_path / "opp").mkdir()
    (tmp_path / "opp/version.info").write_text("Version=9.1.0\n")
    with patch.dict(BUILD.os.environ, {"ASCEND_HOME_PATH": str(tmp_path)}), pytest.raises(ValueError, match="9.2.0"):
        function()


@pytest.mark.parametrize("minor, forwarded", [(7, False), (14, True)])
def test_op_plugin_compilation_uses_actual_torch_branch(tmp_path, minor, forwarded):
    """Newer op-plugin headers must not select removed APIs for an older Torch build."""
    torch = tmp_path / "torch"
    pta = tmp_path / "torch_npu"
    header = torch / "include/torch/csrc/api/include/torch/version.h"
    plugin = pta / "include/third_party/op-plugin/op_plugin/utils/Version.h"
    header.parent.mkdir(parents=True)
    plugin.parent.mkdir(parents=True)
    definitions = f"#define TORCH_VERSION_MAJOR 2\n#define TORCH_VERSION_MINOR {minor}\n"
    if forwarded:
        header.write_text('#include <torch/headeronly/version.h>\n')
        header = torch / "include/torch/headeronly/version.h"
        header.parent.mkdir(parents=True)
    header.write_text(definitions)
    plugin.write_text("#define V2R7 6\n#define V2R12 11\n#define V2R14 13\n#define VERSION_NEWEST V2R14\n")
    script = tmp_path / "check.cmake"
    script.write_text('include("%s")\nhp_detect_op_plugin_version("%s" "%s" selected)\nif(NOT selected STREQUAL "V2R%d")\nmessage(FATAL_ERROR "wrong branch")\nendif()\n' %
                      (ROOT / "hyper_multicore/cmake/torch_compat.cmake", torch, pta, minor))
    completed = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plugin.write_text("#define VERSION_NEWEST 99\n")
    completed = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True, check=False)
    assert completed.returncode != 0 and "no version selector" in completed.stderr


@pytest.mark.parametrize("entry, selection, native_only", [
    ("build.sh", [], False),
    ("build.sh", ["--all"], False),
    ("hyper_multicore/build.sh", [], True),
])
def test_shell_entries_preserve_arguments_and_failure(tmp_path, entry, selection, native_only):
    """The surviving public shell path forwards selection and returns the controller's failure."""
    output = tmp_path / "arguments.json"
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                      f"Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\nraise SystemExit(23)\n")
    python.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / entry), *selection, "--soc-list", "ascend910b",
                             "--jobs", "2"], env={**os.environ, "PYTHON_BIN": str(python), "HP_NATIVE_STAGE": "",
                                "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
                            text=True, capture_output=True, check=False)
    assert result.returncode == 23, result.stderr
    arguments = json.loads(output.read_text())
    assert arguments[0] == str(ROOT / "hyper_multicore/_build/build_native.py")
    assert ("--native-only" in arguments) == native_only
    assert arguments[arguments.index("--soc-list") + 1] == "ascend910b"
    assert arguments[arguments.index("--jobs") + 1] == "2"
    assert ("--all" in arguments) == bool(selection)


def test_environment_key_separates_architecture_python_and_requirements(tmp_path):
    """Platform changes and requested build requirements cannot share a venv."""
    target = make_target(tmp_path)
    key = ENV.environment_key(target, Path(sys.executable))
    with patch.object(ENV.platform, "machine", return_value="other-architecture"):
        assert ENV.environment_key(target, Path(sys.executable)) != key
    with patch.object(ENV.sys, "version", "another-python"):
        assert ENV.environment_key(target, Path(sys.executable)) != key
    target["build_requirements"].append("another-build-tool")
    assert ENV.environment_key(target, Path(sys.executable)) != key


def test_installed_snapshot_changes_are_rejected():
    """Version verification compares installed packages with the recorded actual versions."""
    record = {"packages": [{"name": "packaging", "version": "0.0.0"}]}
    def run(command, **_kwargs):
        subprocess.run(command, check=True, capture_output=True)
    with pytest.raises(subprocess.CalledProcessError):
        ENV.verify_environment(Path(sys.executable), record, run)


@pytest.mark.parametrize("legacy", [False, True])
def test_resolver_report_supports_current_and_legacy_pip_hashes(legacy):
    """Pip need not be upgraded to consume its installation report."""
    archive = {"hash": "sha256=" + "a" * 64} if legacy else {"hashes": {"sha256": "a" * 64}}
    assert ENV.report_digest({"archive_info": archive}) == "a" * 64


def test_resolver_report_rejects_invalid_hash():
    """An incomplete report cannot produce a local installation snapshot."""
    with pytest.raises(ValueError):
        ENV.report_digest({"archive_info": {"hashes": {"sha256": "bad"}}})

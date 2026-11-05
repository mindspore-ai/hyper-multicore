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
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tests.ut.build._build_test_support import ROOT, load_module, make_target
ENV = load_module("framework_environments")
BUILD = load_module("build_controller")


@pytest.fixture(name="local_preparation")
def _local_preparation(monkeypatch, tmp_path):
    """Exercise real framework dispatch while substituting expensive capability probes."""
    version = "private build/42"
    identity = dict.fromkeys(("torch_version", "pta_version"), version)
    identity.update(cann_version="9.1.0.dev-local", cann_version_info="Version=9.1.0.dev-local\n")
    monkeypatch.setattr(BUILD.MANIFEST, "framework_package", lambda name: {
        "version": version, "root": str(tmp_path / "installed" / name)})
    monkeypatch.setattr(BUILD.HELPERS, "_source_identity", lambda root: {})
    run = BUILD.BuildContext(ROOT, tmp_path / "native")
    def _probe(*_args, **_kwargs):
        run.events.append({"stage": "probe"})
        return json.dumps({"identity": identity, "packages": [], "timings": {}})
    run.commands = _probe
    return SimpleNamespace(all=False, soc_list="ascend910b", jobs=1), run, identity


def test_current_framework_ignores_release_matrix(local_preparation, monkeypatch, capsys):
    """Local versions reach capability checks even when the release matrix is unavailable."""
    args, run, identity = local_preparation
    def _unavailable(_path):
        raise ValueError("missing or damaged release matrix")
    monkeypatch.setattr(ENV, "load_targets", _unavailable)
    targets, _, identities = BUILD.prepare_frameworks(args, run)
    assert targets == [] and identities[0]["pta_version"] == identity["pta_version"]
    output = capsys.readouterr().out
    assert identity["pta_version"] in output and "Torch" in output and "torch_npu" in output
    assert "installed/torch" in output and "installed/torch_npu" in output


def test_environment_isolation_preserves_cann_but_removes_framework_paths():
    """The selected interpreter cannot inherit another framework's CMake or SO paths."""
    with patch.dict(ENV.os.environ, {"PYTHONPATH": "wrong", "Torch_DIR": "wrong",
                    "LD_LIBRARY_PATH": ("/cann/lib64:/old/site-packages/torch/lib:"
                                        "/old/build/native/envs/other/lib"),
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


def test_cann_tool_check_has_no_version_gate(tmp_path, monkeypatch):
    """Unknown or different versions can compile, while missing tools remain actionable."""
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr(BUILD.shutil, "which", lambda name: f"/tools/{name}")
    BUILD.validate_toolchain()
    (tmp_path / "opp").mkdir()
    (tmp_path / "opp/version.info").write_text("Version=9.1.0\n")
    BUILD.validate_toolchain()
    monkeypatch.setattr(BUILD.shutil, "which", lambda name: None if name == "bisheng" else f"/tools/{name}")
    with pytest.raises(ValueError, match="Missing native build tools.*bisheng"):
        BUILD.validate_toolchain()


def test_local_environment_preserves_source_paths_and_partial_cann_layout(tmp_path):
    """Developer imports and user-site settings survive all local build phases."""
    cann = tmp_path / "cann"
    modules = cann / "python/site-packages"
    modules.mkdir(parents=True)
    caller = {"ASCEND_HOME_PATH": str(cann), "PYTHONPATH": "/source/torch:/source/pta",
              "PYTHONNOUSERSITE": "0", "CPATH": "/custom/headers", "Torch_DIR": "/source/cmake"}
    python = Path(sys.executable)
    with patch.dict(os.environ, caller):
        environment = BUILD.adapter_environment(python)
        compiler = ENV.cann_tool_environment(python)
        assert all(environment[k] == v for k, v in caller.items())
        assert compiler["PYTHONPATH"].split(os.pathsep) == ["/source/torch", "/source/pta", str(modules)]
        assert compiler["PYTHONNOUSERSITE"] == "0"
        isolated = ENV.cann_tool_environment(python, isolated=True)
        assert isolated["PYTHONPATH"] == str(modules)
        assert isolated["HI_PYTHON"] == str(python)
        assert "PYTHONPATH" not in ENV.build_environment(python, isolated=True)


def test_source_package_identity_ignores_unrelated_distribution(tmp_path, monkeypatch):
    """No dist-info is required; build and runtime share the resolved source identity."""
    manifest = BUILD.MANIFEST
    package = tmp_path / "torch_npu"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    with patch.object(manifest.metadata, "distribution", side_effect=manifest.metadata.PackageNotFoundError):
        assert manifest.framework_package("torch_npu") == {"root": str(package), "version": "unknown"}
        (package / "version.py").write_text("__version__ = 'private build/42'\n")
        assert manifest.framework_package("torch_npu")["version"] == "private build/42"


def test_shmem_sdk_uses_tool_checks_without_a_cann_version_floor(tmp_path):
    """The real shell SDK path reaches its tool checks on an older local CANN."""
    cann = tmp_path / "cann"
    (cann / "opp").mkdir(parents=True)
    (cann / "opp/version.info").write_text("Version=9.1.0\n")
    empty_path = tmp_path / "bin"
    empty_path.mkdir()
    result = subprocess.run(["/bin/bash", "-c", 'source "$1"; hp_prepare_shmem_sdk "$2" ascend910b 1',
        "sdk-check", str(ROOT / "hyper_multicore/shmem/_build/shmem_sdk.sh"), str(ROOT)],
        env={"PATH": str(empty_path), "PYTHON_BIN": sys.executable, "ASCEND_HOME_PATH": str(cann)},
        text=True, capture_output=True, check=False)
    assert result.returncode == 4 and "BUILD_TOOL_NOT_FOUND" in result.stdout
    assert "UNSUPPORTED_CANN_VERSION" not in result.stdout + result.stderr


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
    cann = tmp_path / "cann"
    (cann / "opp").mkdir(parents=True)
    (cann / "opp/version.info").write_text("Version=9.2.0\n")
    module = SimpleNamespace(__file__=str(torch / "__init__.py"),
        version=SimpleNamespace(git_version="fixture"), _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True))
    with patch.dict(os.environ, ASCEND_HOME_PATH=str(cann)):
        identity = BUILD.HELPERS.identity_metadata(ROOT, "ascend910b", module, "private torch/42", "custom pta/42", {})
    assert identity["op_plugin_version"] == f"V2R{minor}" and identity["torch_version"] == "private torch/42"
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


@pytest.mark.parametrize("reference,actual,warnings", [
    ("9.2.0", "9.2.0", 0), ("9.2.0", "9.1.0", 1), ("9.2.0", "unknown", 1), (None, "unknown", 0)])
def test_cann_reference_only_warns(reference, actual, warnings, capsys):
    BUILD.report_cann_reference(reference, actual)
    output = capsys.readouterr().out
    assert actual in output and output.count("WARN") == warnings


def test_environment_install_and_reuse_only_locks_framework_pair(tmp_path):
    """The actual manager installs exact roots once, then accepts other package changes."""
    wheels = tmp_path / "downloads"
    wheels.mkdir()
    target = make_target(wheels)
    commands = []
    def runner(command, **_kwargs):
        commands.append(command)
        if "--dest" in command:
            import shutil
            name = command[-1].split("==")[0].replace("-", "_")
            destination = Path(command[command.index("--dest") + 1])
            source = wheels / f"{name}-1.0.0-py3-none-any.whl"
            shutil.copyfile(source, destination / source.name)
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text(json.dumps({"install": []}))
    actual = [{"name": "numpy", "version": "1.26.4"}]
    def probe(_python, expected):
        assert expected == {"torch": "1.0.0", "torch-npu": "1.0.0"}
        return actual
    python = ENV.prepare_environment(target, tmp_path, runner, Path(sys.executable), identity_probe=probe)
    assert python.parent.parent.parent == tmp_path / "envs"
    install = next(c for c in commands if "--report" in c)
    assert "--only-binary=:all:" in install and "--no-deps" not in install
    assert not list(python.parent.parent.glob("framework-install-*"))
    actual[0]["version"] = "2.0.0"
    commands.clear()
    ENV.prepare_environment(target, tmp_path, runner, Path(sys.executable), identity_probe=probe)
    assert commands == []
    record = json.loads((python.parent.parent / "ready.json").read_text())
    assert record["installed_packages"] == actual
    with pytest.raises(RuntimeError, match="changed"):
        ENV.prepare_environment(target, tmp_path, runner, Path(sys.executable),
                                identity_probe=lambda *_: (_ for _ in ()).throw(RuntimeError("framework changed")))


def test_failed_install_cleans_downloads_without_publishing_ready(tmp_path, monkeypatch):
    target = {"id": "fixture", "torch_version": "1.0.0", "torch_release": "1.0.0",
              "pta_build_version": "1.0.0", "pta_release": "1.0.0", "indexes":
              dict.fromkeys(("torch", "pta", "dependencies"), "https://example.invalid/simple"),
              "build_requirements": []}
    monkeypatch.setattr(ENV, "download_roots", lambda *_: [])
    def runner(command, **_kwargs):
        if "--report" in command:
            raise RuntimeError("dependency installation failed")
    with pytest.raises(RuntimeError, match="installation failed"):
        ENV.prepare_environment(target, tmp_path, runner, Path(sys.executable))
    assert not list(tmp_path.rglob("ready.json"))
    assert not list(tmp_path.rglob("framework-install-*"))


def test_matrix_reference_does_not_change_environment_key(tmp_path):
    path = ROOT / "hyper_multicore/_build/frameworks.lock.json"
    original = json.loads(path.read_text())
    target = ENV.load_targets(path)[0]
    for value in (None, "9.0.0"):
        original["reference_cann"] = value
        path = tmp_path / "matrix.json"
        path.write_text(json.dumps(original))
        assert ENV.environment_key(ENV.load_targets(path)[0], Path(sys.executable)) == ENV.environment_key(target, Path(sys.executable))
    original["targets"] = original["targets"][:1]
    original["targets"][0]["pta_build_version"] = "2.7.1.post12"
    path.write_text(json.dumps(original))
    assert len(ENV.load_targets(path)) == 1 and ENV.load_targets(path)[0]["pta_build_version"] == "2.7.1.post12"

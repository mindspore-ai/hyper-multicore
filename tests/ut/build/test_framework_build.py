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
from tests.ut.build._build_test_support import ROOT, load_module, make_target, cached_resolution
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
    monkeypatch.setattr(BUILD, "compiler_identity", lambda *_: {})
    run = BUILD.BuildContext(ROOT, tmp_path / "native")
    def _probe(*_args, **_kwargs):
        run.events.append({"stage": "probe"})
        return json.dumps({"identity": identity, "paths": {}, "timings": {}})
    run.commands = _probe
    return SimpleNamespace(all=False, soc_list="ascend910b", jobs=1), run, identity


def test_current_framework_ignores_release_matrix(local_preparation, monkeypatch, capsys):
    """Local versions reach capability checks even when the release matrix is unavailable."""
    args, run, identity = local_preparation
    def _unavailable(_path):
        raise ValueError("missing or damaged release matrix")
    monkeypatch.setattr(ENV, "load_targets", _unavailable)
    probed = []
    monkeypatch.setattr(BUILD, "preflight_frameworks", lambda *values: probed.extend(values[3]))
    targets, _, identities, _ = BUILD.prepare_frameworks(args, run)
    assert targets == [] and probed == identities
    output = capsys.readouterr().out
    assert identity["pta_version"] in output and "Torch" in output and "torch_npu" in output
    assert "installed/torch" in output and "installed/torch_npu" in output


@pytest.mark.parametrize("mismatch", ["framework", "cann"])
def test_matrix_framework_still_requires_exact_pair(local_preparation, monkeypatch, mismatch):
    """Matrix preparation cannot substitute the caller's development release for a locked pair."""
    args, run, _ = local_preparation
    args.all = True
    target = ENV.load_targets(ROOT / "hyper_multicore/_build/frameworks.lock.json")[0]
    monkeypatch.setattr(ENV, "load_targets", lambda _: [target])
    identity = {"torch_version": target["torch_version"], "pta_version": target["pta_build_version"], "cann_version": "9.2.0"}
    field = "pta_version" if mismatch == "framework" else "cann_version"
    identity[field] = "2.15.0.dev20261005+git42" if mismatch == "framework" else "9.1.0"
    monkeypatch.setattr(BUILD, "prepare_matrix_environments", lambda *_: ([Path(sys.executable)], [identity]))
    monkeypatch.setattr(BUILD, "attach_framework_environments", lambda *_: None)
    with pytest.raises(ValueError, match="Target framework identity changed|--all builds require CANN"):
        BUILD.prepare_frameworks(args, run)


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


def test_cann_precheck_only_restricts_the_full_matrix(tmp_path, monkeypatch):
    """Local toolkits reach capability checks; the full matrix retains its CANN baseline."""
    function = BUILD.validate_toolchain
    with patch.dict(BUILD.os.environ, {}, clear=True), pytest.raises(ValueError, match="source the CANN"):
        function()
    (tmp_path / "opp").mkdir()
    version_file = tmp_path / "opp/version.info"
    version_file.write_text("Version=9.1.0.dev-local\n")
    monkeypatch.setattr(BUILD.shutil, "which", lambda name: f"/tools/{name}")
    with patch.dict(BUILD.os.environ, {"ASCEND_HOME_PATH": str(tmp_path)}):
        function()
        with pytest.raises(ValueError, match="--all builds require CANN 9.2.0"):
            function(full_matrix=True)
        version_file.unlink()
        assert BUILD.HELPERS.cann_identity(tmp_path)["cann_version_sha256"] is None
        function()
        version_file.write_text("Version=9.2.0\n")
        function(full_matrix=True)
        monkeypatch.setattr(BUILD.shutil, "which", lambda name: None if name == "bisheng" else f"/tools/{name}")
        with pytest.raises(ValueError, match="Missing native build tools.*bisheng"):
            function()


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
        identity = BUILD.HELPERS.identity_metadata(ROOT, "ascend910b", module, "private torch/42", "custom pta/42", {}, {})
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

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
"""Native input authentication and common, adapter and preflight cache behavior."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tests.ut.build._build_test_support import ROOT, load_module, module_for, put
BUILD = load_module("build_controller")
INPUTS = load_module("native_inputs")
ENV = BUILD.FRAMEWORK
PROBE = BUILD.PREFLIGHT
HELPERS = BUILD.HELPERS


class ProbeWorker:
    """Model configure/compile/link stages while retaining real cache/hash functions."""
    def __init__(self, work, library, action=None):
        self.work, self.library, self.action = work, library, action
        self.commands = []

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        if command[:2] == ["cmake", "-S"]:
            self.work.mkdir(parents=True, exist_ok=True)
        elif command[:2] == ["cmake", "--build"]:
            if self.action:
                self.action()
            put(self.work / "libhyper_multicore_framework_probe.so", "compiled output")
        elif command[0] == "ldd":
            return f"libfixture.so => {self.library} (0x1)\n"
        return ""


def probe_fixture(tmp_path, action=None):
    work = tmp_path / "probe"
    library = put(tmp_path / "libfixture.so")
    worker = ProbeWorker(work, library, action)
    return work, worker


def ensure(work, worker, *, clean=False, key="key", digest=None, guard=None):
    return PROBE.ensure_probe(work, key, SimpleNamespace(clean=clean, jobs=1), Path(sys.executable), {},
        worker, "2.12.0", digest or HELPERS.ProbeHashes(), guard, ROOT, BUILD.write_json)


def test_probe_success_then_verified_reuse_and_forced_clean(tmp_path):
    work, worker = probe_fixture(tmp_path)
    assert ensure(work, worker)[0] is False
    worker.commands.clear()
    assert ensure(work, worker)[0] is True
    assert [c[0] for c in worker.commands] == ["ldd"]
    worker.commands.clear()
    assert ensure(work, worker, clean=True)[0] is False
    assert any(c[:2] == ["cmake", "--build"] for c in worker.commands)


@pytest.mark.parametrize("changed", ["output", "dependency", "key"])
def test_changed_probe_requires_revalidation(tmp_path, changed):
    work, worker = probe_fixture(tmp_path)
    ensure(work, worker)
    if changed == "output":
        (work / "libhyper_multicore_framework_probe.so").write_text("tampered")
    elif changed == "dependency":
        worker.library.write_text("updated dependency")
    worker.commands.clear()
    assert ensure(work, worker, key="changed" if changed == "key" else "key")[0] is False
    assert any(c[:2] == ["cmake", "--build"] for c in worker.commands)


@pytest.mark.parametrize("error", [RuntimeError("compile failed"), KeyboardInterrupt()])
def test_failed_probe_never_leaves_success_and_can_retry(tmp_path, error):
    def fail():
        raise error
    work, worker = probe_fixture(tmp_path, fail)
    with pytest.raises(type(error)):
        ensure(work, worker)
    assert not (work / "complete.json").exists()
    worker.action = None
    assert ensure(work, worker)[0] is False


def test_input_changed_and_restored_during_probe_cannot_be_certified(tmp_path):
    header = put(tmp_path / "include/input.h", "A")
    digest = HELPERS.ProbeHashes()
    digest(header)
    def change():
        header.write_text("B")
        put(header.with_name("replacement.h"), "A").replace(header)
    work, worker = probe_fixture(tmp_path, change)
    with pytest.raises(RuntimeError, match="changed"):
        ensure(work, worker, digest=digest)
    assert not (work / "complete.json").exists()


def test_parent_transaction_revokes_new_probe_success_on_later_source_change(tmp_path):
    work, worker = probe_fixture(tmp_path)
    state = {"source": "A"}
    guard_class = BUILD.CacheTransaction
    with patch.multiple(module_for(guard_class.__init__), **{'source_observation': lambda : dict(state)}):
        with pytest.raises(RuntimeError, match="changed"):
            with guard_class() as guard:
                ensure(work, worker, guard=guard)
                assert (work / "complete.json").exists()
                state["source"] = "B"
    assert not (work / "complete.json").exists()


def test_make_dependencies_include_escaped_paths_and_cmake_modules(tmp_path):
    header = put(tmp_path / "with space/header.h")
    module = put(tmp_path / "module.cmake")
    escaped = str(header).replace(" ", "\\ ")
    put(tmp_path / "probe.cpp.o.d", "output: " + escaped + "\n")
    put(tmp_path / "CMakeFiles/Makefile.cmake", f'set(CMAKE_MAKEFILE_DEPENDS\n  "{module}"\n  )\n')
    assert PROBE.dependency_files(tmp_path, None, {}, "2.12.0") == sorted([header, module])


def native_inputs():
    return INPUTS.NativeInputs(BUILD.HELPERS.ProbeHashes())


@pytest.mark.parametrize("relative", ["include/acl/acl.h", "compiler/tikcpp/compiler.py", "lib64/libascendcl.so"])
def test_consumed_cann_changes_affect_identity_but_docs_do_not(tmp_path, relative):
    cann = tmp_path / "cann"
    path = put(cann / relative)
    collect = lambda: native_inputs().cann(cann, {}, lambda *a, **k: "")
    before = collect()
    put(cann / "docs/unrelated.md", "new documentation")
    assert collect() == before
    path.write_text("B")
    assert collect() != before


def test_symlinked_headers_are_content_inputs_and_new_members_are_detected(tmp_path):
    real = tmp_path / "real"
    header = put(real / "acl.h")
    (tmp_path / "include").symlink_to(real, target_is_directory=True)
    inputs = native_inputs()
    record = inputs.cann(tmp_path, {}, lambda *a, **k: "")
    assert record[str(tmp_path / "include/acl.h")]["path"] == str(header)
    put(real / "new.h")
    with pytest.raises(RuntimeError, match="changed"):
        inputs.validate()


def test_unregistered_consumed_cann_file_is_rejected(tmp_path):
    cann = tmp_path / "cann"
    unregistered = put(cann / "unregistered/header.h")
    dep = put(tmp_path / "work/object.o.d", f"object.o: {unregistered}\n")
    with pytest.raises(RuntimeError, match="missing from native identity"):
        native_inputs().verify_consumed(dep.parent, cann)


def test_same_version_compiler_replacement_changes_native_contract(tmp_path):
    compiler = tmp_path / "g++"
    def contract(flag):
        put(compiler, f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "same compiler"; else exec /usr/bin/g++ -DTEST={flag} "$@"; fi\n')
        compiler.chmod(0o755)
        run = lambda cmd, **kwargs: subprocess.check_output(cmd, env=kwargs.get("env"), text=True)
        identity = BUILD.compiler_identity(str(compiler), run, "version")
        return identity, native_inputs().compiler(identity, dict(os.environ), run)
    first, a = contract(1)
    second, b = contract(2)
    assert first == second and a != b
    assert a["programs"]["compiler"]["sha256"] != b["programs"]["compiler"]["sha256"]


def test_native_completion_input_record_must_match_its_key(tmp_path):
    inputs = {"tools": "recorded bytes", "cann_development": "recorded headers"}
    key = ENV.digest_json(inputs)
    (tmp_path / "payload").mkdir()
    record = {"fingerprint": key, "cache_schema": BUILD.CACHE_SCHEMA_VERSION, "inputs": inputs, "hashes": {}}
    marker = put(tmp_path / "complete.json", json.dumps(record))
    assert BUILD.cache_valid(tmp_path, key)
    record["inputs"]["tools"] = "unrelated compiler"
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="input record mismatch"):
        BUILD.cache_valid(tmp_path, key)


def test_preflight_reuses_interfaces_across_unrelated_source_identity_changes(tmp_path):
    """An assembly commit cannot force recompilation of unchanged framework interfaces."""
    for name in ("hyper_multicore/cmake/framework_probe/probe.cpp", "hyper_multicore/cmake/torch_compat.cmake",
                 "hyper_multicore/torch/csrc/cached_op_api.h"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("probe input")
    identity = {k: "fixture" for k in ("torch_version", "pta_version", "torch_cxx11_abi", "python_abi", "host_arch",
        "torch_headers_sha256", "pta_headers_sha256", "torch_libraries_sha256", "libtorch_npu_sha256", "cann_root", "cann_version_sha256")}
    function = BUILD.preflight_fingerprint
    with patch.multiple(module_for(function), **{'ROOT': tmp_path}):
        first = function(identity, Path("/venv/bin/python"), {"version": "GCC12"}, {})
        identity.update(build_commit="later assembly commit", source_sha256="unrelated packaging input")
        assert function(identity, Path("/venv/bin/python"), {"version": "GCC12"}, {}) == first
        assert function(identity, Path("/venv/bin/python"), {"version": "GCC10"}, {}) != first
        (tmp_path / "hyper_multicore/torch/csrc/cached_op_api.h").write_text("changed consumed header")
        assert function(identity, Path("/venv/bin/python"), {"version": "GCC12"}, {}) != first


def test_activation_script_is_a_consumed_common_input(tmp_path):
    """Changing activation bytes changes the public common-source fingerprint."""
    function = BUILD.common_source_fingerprint
    def digest(_directories, extra):
        return hashlib.sha256(b"".join((tmp_path / name).read_bytes() for name in extra)).hexdigest()
    for name in BUILD.COMMON_EXTRA_INPUTS:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("input")
    with patch.multiple(module_for(function), **{'fingerprint_files': digest}):
        before = function()
        (tmp_path / "hyper_multicore/set_env.bash").write_text("new activation")
        assert function() != before


def test_native_cache_checks_resources_and_sdk_handoff(tmp_path):
    """A valid marker does not mask modified shared resources or missing SDK headers."""
    (tmp_path / "payload").mkdir()
    (tmp_path / "payload/lib.so").write_bytes(b"native")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "shmem.h").write_text("header")
    (tmp_path / "sdk.json").write_text(json.dumps({"install_root": str(sdk)}))
    data = {"cache_schema": BUILD.CACHE_SCHEMA_VERSION, "fingerprint": "key", "hashes": BUILD.hashes(tmp_path / "payload"), "sdk_root": str(sdk),
            "sdk_hashes": BUILD.hashes(sdk), "sdk_manifest_sha256": hashlib.sha256((tmp_path / "sdk.json").read_bytes()).hexdigest()}
    (tmp_path / "complete.json").write_text(json.dumps(data))
    assert BUILD.cache_valid(tmp_path, "key")
    (sdk / "shmem.h").write_text("changed")
    with pytest.raises(ValueError, match="SDK inputs"):
        BUILD.cache_valid(tmp_path, "key")


def test_fixed_slot_input_change_is_a_miss_but_matching_corruption_is_rejected(tmp_path):
    payload = put(tmp_path / "payload/library.so", "old bytes")
    marker = {"cache_schema": BUILD.CACHE_SCHEMA_VERSION, "fingerprint": "old", "hashes": BUILD.hashes(payload.parent)}
    put(tmp_path / "complete.json", json.dumps(marker))
    assert BUILD.cache_valid(tmp_path, "old")
    assert not BUILD.cache_valid(tmp_path, "new")
    payload.write_text("damaged")
    with pytest.raises(ValueError, match="integrity"):
        BUILD.cache_valid(tmp_path, "old")


def test_development_trees_are_content_inputs(tmp_path):
    paths = {"python_include": str(tmp_path / "python/include"),
             "torch_cmake": str(tmp_path / "torch/share/cmake")}
    files = [put(Path(paths["python_include"]) / "Python.h"),
             put(Path(paths["torch_cmake"]) / "TorchConfig.cmake")]
    (tmp_path / "cann/include").mkdir(parents=True)
    digest = HELPERS._sha256
    before = PROBE.development_inputs(paths, tmp_path / "cann", digest)
    for path in files:
        original = path.read_text(encoding="utf-8")
        path.write_text("changed", encoding="utf-8")
        assert PROBE.development_inputs(paths, tmp_path / "cann", digest) != before
        path.write_text(original, encoding="utf-8")


def test_adapter_pollution_is_rejected_then_rebuilt_and_reused(tmp_path, monkeypatch):
    """Restoring A must rebuild instead of reusing the rejected B output."""
    identity = {"pta_version": "2.12.0", "torch_version": "2.12.0+cpu", "python_abi": "cp310",
        "torch_cxx11_abi": True, "host_arch": "aarch64", "build_commit": "original-build",
        "source_sha256": "A", "build_fingerprint": "fixture"}
    key = ENV.digest_json({})
    directory = tmp_path / "variants" / BUILD.WORKSPACE.framework_slot(identity)
    state, commands = {"source": "A", "mutate": True}, []
    def _worker(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["cmake", "--install"]:
            data = "B" if state["mutate"] else "A"
            put(directory / "payload/libhyper_multicore_torch.so", data)
            put(directory / "payload/hyper_multicore_shmem_torch.so", data)
            state["source"] = data
    monkeypatch.setattr(BUILD, "NATIVE", tmp_path)
    monkeypatch.setattr(BUILD, "adapter_environment", lambda _, **_kw: {})
    monkeypatch.setattr(BUILD, "validate_elf", lambda *_: None)
    monkeypatch.setattr(load_module("native_cache"), "source_observation", lambda: state["source"])
    context = BUILD.BuildContext(ROOT, tmp_path)
    context.commands = _worker
    def _build():
        return BUILD.cmake_pair(Path(sys.executable), identity, tmp_path, tmp_path, key,
            SimpleNamespace(clean=False, jobs=1), context, inputs={}, probe_inputs={})
    with pytest.raises(RuntimeError, match="changed"):
        _build()
    assert not (directory / "complete.json").exists()
    state.update(source="A", mutate=False)
    payload, record = _build()
    assert (payload / "libhyper_multicore_torch.so").read_text() == "A"
    assert BUILD.cache_valid(directory, key)
    commands.clear()
    assert _build()[1]["native_build_identity"] == record["native_build_identity"]
    assert not commands
    context.common_cacheable = False
    commands.clear()
    _build()
    assert commands and not (directory / "complete.json").exists()


def test_common_pollution_is_rejected_then_rebuilt_and_reused(tmp_path, monkeypatch):
    directory = tmp_path / "common" / BUILD.WORKSPACE.common_slot("ascend910b")
    state, commands = {"source": "A", "mutate": True}, []
    key = ENV.digest_json({})
    def _worker(command, **_kwargs):
        commands.append(command)
        data = "B" if state["mutate"] else "A"
        put(directory / "payload/lib.so", data)
        put(directory / "sdk/header.h", data)
        put(directory / "sdk.json", json.dumps({"install_root": str(directory / "sdk")}))
        state["source"] = data
    monkeypatch.setattr(BUILD, "NATIVE", tmp_path)
    monkeypatch.setattr(load_module("native_cache"), "source_observation", lambda: state["source"])
    context = BUILD.BuildContext(ROOT, tmp_path)
    context.commands = _worker
    def _build():
        with BUILD.CacheTransaction() as guard:
            return BUILD.build_common(SimpleNamespace(all=False, clean=False, jobs=1, soc_list="ascend910b"),
                context, Path(sys.executable), key, {"ASCEND_HOME_PATH": str(tmp_path / "cann")}, guard)
    with pytest.raises(RuntimeError, match="changed"):
        _build()
    assert not (directory / "complete.json").exists()
    state.update(source="A", mutate=False)
    payload, _ = _build()
    assert (payload / "lib.so").read_text() == "A" and BUILD.cache_valid(directory, key)
    commands.clear()
    _build()
    assert not commands
    context.common_cacheable = False
    commands.clear()
    _build()
    assert "--clean" in commands[0] and not (directory / "complete.json").exists()

    context.common_cacheable = True
    def incomplete(*_args):
        raise RuntimeError("Consumed CANN input is missing from native identity")
    monkeypatch.setattr(context.native_inputs, "verify_consumed", incomplete)
    _build()
    assert context.common_cacheable is False and not (directory / "complete.json").exists()


def test_restoring_source_bytes_during_build_is_still_rejected(tmp_path):
    """A write-and-restore changes the file generation even if its digest is unchanged."""
    source = tmp_path / "hyper_multicore/input.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("A")
    cls = BUILD.CacheTransaction
    helpers = SimpleNamespace(_source_identity=lambda _root: {"source_sha256": "A"})
    with patch.multiple(module_for(cls.check), **{'ROOT': tmp_path, 'HELPERS': helpers}):
        transaction = cls()
        # Linux may reuse a coarse filesystem timestamp for writes in the same tick.
        import time
        time.sleep(0.02)
        source.write_text("B")
        source.write_text("A")
        with pytest.raises(RuntimeError, match="changed"):
            transaction.check()


def test_python_headers_invalidate_preflight_and_adapter_across_builds(tmp_path, monkeypatch):
    """New contexts must rebuild on header changes and reuse the next unchanged build."""
    python = Path(sys.executable)
    include = tmp_path / "python/include"
    header = put(include / "Python.h", "A")
    put(include / "pyconfig.h", "configuration")
    put(include / "cpython/object.h", "nested header")
    library = put(tmp_path / "python/libpython.so")
    cmake = put(tmp_path / "torch/cmake/TorchConfig.cmake").parent
    put(tmp_path / "cann/include/acl.h")
    native = tmp_path / "native"
    identity = dict.fromkeys(("torch_commit", "source_sha256", "build_commit", "build_fingerprint",
        "op_plugin_headers_sha256", "torch_headers_sha256", "pta_headers_sha256",
        "torch_libraries_sha256", "libtorch_npu_sha256", "cann_version_sha256"), "fixed")
    identity.update(torch_version="2.12.0+cpu", pta_version="2.12.0.post3", pta_release="2.12.0",
        python_abi="cp310", host_arch="aarch64", torch_cxx11_abi=True, cann_root=str(tmp_path / "cann"))
    development = {"python_include": str(include), "torch_cmake": str(cmake)}
    slot = BUILD.WORKSPACE.framework_slot(identity)
    common_dir = native / "common" / BUILD.WORKSPACE.common_slot("ascend910b")
    args = SimpleNamespace(clean=False, jobs=1, soc_list="ascend910b")
    probe = ProbeWorker(native / "preflight" / slot, library)
    monkeypatch.setattr(BUILD, "NATIVE", native)
    monkeypatch.setattr(BUILD, "tool_inputs", lambda *_args: None)
    monkeypatch.setattr(BUILD, "validate_elf", lambda *_args: None)
    commands, results = [], []
    def _run(command, **kwargs):
        commands.append(kwargs["label"])
        if kwargs["label"].startswith("preflight"):
            return probe(command, **kwargs)
        if kwargs["label"] == "common-resources":
            put(common_dir / "payload/common.so")
            put(common_dir / "sdk/include/shmem.h")
            put(common_dir / "sdk.json", json.dumps({"install_root": str(common_dir / "sdk")}))
        elif command[:2] == ["cmake", "--install"]:
            for filename in ("libhyper_multicore_torch.so", "hyper_multicore_shmem_torch.so"):
                put(native / "variants" / slot / "payload" / filename, header.read_text(encoding="utf-8"))
        return ""
    for contents in ("A", "B", "B"):
        if header.read_text(encoding="utf-8") != contents:
            header.write_text(contents, encoding="utf-8")
        context = BUILD.BuildContext(ROOT, native)
        context.commands = _run
        context.frameworks[str(python)] = BUILD.PreparedFramework(python, identity, development)
        context.native_inputs.enrich_tools = lambda *_args: {}
        commands.clear()
        with BUILD.CacheTransaction() as guard:
            guard.input_checks.append(context.native_inputs.validate)
            BUILD.preflight_frameworks(args, context, [python], [identity], {"command": ["c++"]}, guard)
            common_key = ENV.digest_json({})
            common, sdk = BUILD.build_common(args, context, python, common_key,
                {"ASCEND_HOME_PATH": identity["cann_root"]}, guard)
            BUILD.build_adapters(args, context, [python], [identity], common_key, common, sdk, guard)
        preflight_hit = next(e["preflight_cache_hit"] for e in context.events if "preflight_cache_hit" in e)
        cache_hits = [e["cache_hit"] for e in context.events if "cache_hit" in e]
        results.append((preflight_hit, *cache_hits))
    assert results == [(False, False, False), (False, True, False), (True, True, True)]
    assert not any(label.startswith(("compile-", "preflight-compile-")) for label in commands)
    assert (native / "variants" / slot / "payload/libhyper_multicore_torch.so").read_text(encoding="utf-8") == "B"


def test_default_incomplete_cann_discovery_builds_without_certification(tmp_path):
    """An alternate toolkit reaches actual compilation; the release matrix remains strict."""
    context = BUILD.BuildContext(ROOT, tmp_path / "native")
    python = Path(sys.executable)
    env = {"ASCEND_HOME_PATH": str(tmp_path)}
    args = SimpleNamespace(all=False)
    assert BUILD.discover_cann_inputs(args, context, python, tmp_path, env) == {
        "asc_configuration": {}, "cann_development": {}}
    assert context.common_cacheable is False
    args.all = True
    with pytest.raises(ValueError, match="ASC CMake configuration"):
        BUILD.discover_cann_inputs(args, context, python, tmp_path, env)


def test_probe_load_error_does_not_certify_success(tmp_path):
    """Successful linking alone cannot hide an unresolved extension symbol."""
    work, worker = probe_fixture(tmp_path)
    def run(command, **kwargs):
        if kwargs.get("label", "").startswith("preflight-load-"):
            raise RuntimeError("undefined symbol: PyUnresolved")
        return worker(command, **kwargs)
    with pytest.raises(RuntimeError, match="undefined symbol"):
        ensure(work, run)
    assert not (work / "complete.json").exists()


def test_asc_overrides_never_authenticate_a_different_configuration(tmp_path):
    """ASC_DIR takes precedence; unknown external search inputs disable local reuse."""
    cann = tmp_path / "cann"
    first = put(cann / "selected/ASCConfig.cmake", "compiler/tikcpp/ascendc_kernel_cmake")
    other = put(cann / "other/ASCConfig.cmake", "different configuration")
    put(cann / "compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py")
    env = {"ASC_DIR": str(first.parent), "CMAKE_PREFIX_PATH": str(other.parent)}
    assert BUILD.asc_compilation_inputs(cann, env)[1] == first
    env["ASC_DIR"] = str(put(tmp_path / "external/ASCConfig.cmake").parent)
    context = BUILD.BuildContext(ROOT, tmp_path / "native")
    BUILD.discover_cann_inputs(SimpleNamespace(all=False), context, Path(sys.executable), cann, env)
    assert context.common_cacheable is False
    with pytest.raises(ValueError, match="External ASC search paths"):
        BUILD.discover_cann_inputs(SimpleNamespace(all=True), context, Path(sys.executable), cann, env)

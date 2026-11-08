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
"""Real native scheduling rebuilds fresh workspaces while sharing common work per run."""

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.ut.build._build_test_support import ROOT, load_module, put

BUILD = load_module("build_controller")


@pytest.mark.parametrize("count", [1, 7])
def test_repeated_native_scheduling_always_compiles(tmp_path, monkeypatch, count):
    """Old outputs cannot satisfy a second invocation; all pairs share one new common payload."""
    monkeypatch.setattr(BUILD, "NATIVE", tmp_path)
    identity = {"python_abi": "cp310", "host_arch": "aarch64", "torch_version": "2.12.0+cpu",
                "torch_cxx11_abi": True, "build_commit": "commit", "source_sha256": "source", "build_fingerprint": "f" * 64}
    identities = [{**identity, "pta_version": f"2.{i}.0"} for i in range(count)]
    args = SimpleNamespace(all=count == 7, soc_list="ascend910b", jobs=1, clean=count == 7)
    calls = []
    def command(_command, **kwargs):
        if kwargs["label"] == "common-resources":
            env = kwargs["env"]
            assert env["HP_VENDOR_CACHE_ROOT"] == str(tmp_path)
            assert (env.get("HP_VENDOR_CACHE_CLEAN") == "1") == args.clean
            directory = Path(env["HP_COMPONENT_ROOT"]).parent
            assert not (directory / "stale.o").exists()
            put(directory / "stale.o")
            put(directory / "sdk.json", json.dumps({"install_root": str(tmp_path / "sdk")}))
            calls.append("common")
    def compile_pair(_python, directory, payload, common, *_args):
        assert not (directory / "stale.o").exists()
        put(directory / "stale.o")
        put(directory / "torch/torch_cxx_standard.txt", "20")
        put(payload / "libhyper_multicore_torch.so")
        put(payload / "hyper_multicore_shmem_torch.so")
        assert common == tmp_path / "common" / BUILD.WORKSPACE.common_slot(args.soc_list) / "payload"
        calls.append("adapter")
    monkeypatch.setattr(BUILD, "compile_pair", compile_pair)
    monkeypatch.setattr(BUILD, "load_pair", lambda *_: None)
    monkeypatch.setattr(BUILD, "validate_elf", lambda *_: None)
    monkeypatch.setattr(BUILD.FRAMEWORK, "cann_tool_environment", lambda *_a, **_kw: {})
    class Workers(BUILD.CONSOLE.Commands):
        def __call__(self, arguments, **kwargs):
            return command(arguments, **kwargs)
    for _ in range(2):
        context = BUILD.BuildContext(ROOT, tmp_path)
        context.commands = Workers(tmp_path / "logs", events=context.events)
        common, sdk = BUILD.build_common(args, context, Path("python"))
        BUILD.build_adapters(args, context, [Path("python")] * count, identities, common, sdk,
                             SimpleNamespace(check_source=lambda: None))
    assert calls == (["common"] + ["adapter"] * count) * 2
    assert not list(tmp_path.rglob("complete.json"))


@pytest.mark.parametrize("count,jobs,expected", [
    (7, 3, [3]), (7, 4, [2, 2]), (7, 14, [2] * 7),
    (7, 24, [4, 4, 4, 3, 3, 3, 3]), (3, 7, [3, 2, 2]), (1, 1, [1])])
def test_adapter_budget_tiers(count, jobs, expected):
    assert BUILD.adapter_budgets(count, jobs) == expected
    assert sum(expected) == jobs


def test_parallel_refill_preserves_budget_and_matrix_order(tmp_path, monkeypatch):
    """A fast slot refills out of order without borrowing another slot's fixed allocation."""
    run = BUILD.BuildContext(ROOT, tmp_path)
    identities = [{"pta_version": str(index), "torch_version": "fixture"} for index in range(5)]
    guard = SimpleNamespace(check_source=lambda: None)
    lock, initial = threading.Lock(), threading.Barrier(2)
    active = 0
    peak = 0
    finished = []
    quotas = {}
    def worker(_python, identity, _common, _sdk, args, _run):
        nonlocal active, peak
        index = int(identity["pta_version"])
        with lock:
            active += args.jobs
            peak = max(peak, active)
            quotas.setdefault(threading.get_ident(), set()).add(args.jobs)
        if index < 2:
            initial.wait(timeout=5)
        time.sleep(0.08 if index == 0 else 0.005)
        with lock:
            active -= args.jobs
            finished.append(index)
        return tmp_path / str(index), identity
    monkeypatch.setattr(BUILD, "cmake_pair", worker)
    args = SimpleNamespace(all=True, jobs=9)
    outputs, records = BUILD.build_adapters(args, run, [Path("python")] * 5, identities,
                                           tmp_path, tmp_path, guard)
    assert peak == 9 and active == 0
    assert finished != list(range(5))
    assert [record["pta_version"] for record in records] == list(map(str, range(5)))
    assert outputs == [tmp_path / str(index) for index in range(5)]
    assert sorted(next(iter(values)) for values in quotas.values()) == [4, 5]
    assert all(len(values) == 1 for values in quotas.values())
    assert args.jobs == 9


VENDOR = load_module("vendor_cache")


@pytest.fixture
def vendor_fixture(tmp_path, monkeypatch):
    root, sdk = tmp_path / "repo", tmp_path / "sdk"
    for name in VENDOR.SCRIPTS:
        put(root / "hyper_multicore" / name)
    for name in VENDOR.SOURCE_TREES:
        put(root / "hyper_multicore" / name / "input.h")
    for name in ("include", "src/device", "lib"):
        put(sdk / name / "input")
    cann = tmp_path / "cann"
    put(cann / "opp/version.info", "Version=9.2.0")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    monkeypatch.setenv("HP_VENDOR_ORIGIN", '{"build_commit":"original"}')
    tool = put(tmp_path / "compiler", "tool A")
    monkeypatch.setattr(VENDOR, "tool_inputs", lambda _env: {"compiler": VENDOR.sha256(tool)})
    candidate = tmp_path / "work/vendor"
    put(candidate / "kernel.o", "binary")
    merge = load_module("merge_multicore_vendors")
    for directory in merge._soc_payload_paths(candidate, "ascend910b"):
        put(directory / "binary")
    request = tmp_path / "request.json"
    return root, sdk, candidate, request, tool


def vendor_prepare(fixture, clean=False):
    root, sdk, _, request, _ = fixture
    # Each call represents fresh common work, while its one vendor snapshot remains.
    reused = request.parent / "vendor-reuse-ascend910b"
    if reused.exists():
        import shutil
        shutil.rmtree(reused)
    VENDOR.prepare(root, root / "build/native", "ascend910b", sdk, request, clean)
    return json.loads(request.read_text())


def test_vendor_snapshot_commit_hit_corruption_and_clean(vendor_fixture):
    root, _sdk, candidate, request, _tool = vendor_fixture
    first = vendor_prepare(vendor_fixture)
    assert not first["reused"] and not Path(first["directory"]).exists()
    VENDOR.built(request, candidate, time.time())
    VENDOR.publish(root, [json.loads(request.read_text())])
    second = vendor_prepare(vendor_fixture)
    assert second["reused"] and second["origin"]["build_commit"] == "original"
    merge = load_module("merge_multicore_vendors")
    normalized, _ = merge._normalize_inputs([("ascend910b", Path(second["candidate"]))])
    assert normalized[0][1] == Path(second["candidate"]).resolve()
    put(Path(second["directory"]) / "vendor/kernel.o", "corrupted")
    assert not vendor_prepare(vendor_fixture)["reused"]
    assert not vendor_prepare(vendor_fixture, clean=True)["reused"]
    assert not Path(second["directory"]).exists()


@pytest.mark.parametrize("change", ["source", "shmem_header", "shmem_file", "sdk", "tool", "flags"])
def test_vendor_consumed_inputs_change(vendor_fixture, monkeypatch, change):
    root, sdk, candidate, request, tool = vendor_fixture
    header = root / "hyper_multicore/shmem/ccsrc/data_plane/rma.h"
    if change == "shmem_header":
        put(header, "header A")
    vendor_prepare(vendor_fixture)
    VENDOR.built(request, candidate, time.time())
    VENDOR.publish(root, [json.loads(request.read_text())])
    assert vendor_prepare(vendor_fixture)["reused"]
    if change == "source":
        put(root / "hyper_multicore/ops/new.cpp")
    elif change == "shmem_header":
        put(header, "header B")
    elif change == "shmem_file":
        put(root / "hyper_multicore/shmem/ccsrc/cann/new.h")
    elif change == "sdk":
        put(sdk / "src/device/input", "new device header")
    elif change == "tool":
        put(tool, "tool B")
    else:
        monkeypatch.setenv("CXXFLAGS", "-g")
    assert not vendor_prepare(vendor_fixture)["reused"]


def test_vendor_rejects_changed_build_and_preserves_unknown_data(vendor_fixture):
    root, sdk, candidate, request, _tool = vendor_fixture
    first = vendor_prepare(vendor_fixture)
    VENDOR.built(request, candidate, time.time())
    put(sdk / "include/input", "changed while compiling")
    VENDOR.publish(root, [json.loads(request.read_text())])
    assert not Path(first["directory"]).exists()
    directory = Path(first["directory"])
    put(directory / "manual.txt")
    with pytest.raises(ValueError, match="Unattributed"):
        vendor_prepare(vendor_fixture, clean=True)
    import shutil
    shutil.rmtree(directory)
    directory.symlink_to(candidate, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        vendor_prepare(vendor_fixture)
    assert (candidate / "kernel.o").is_file()
    directory.unlink()


def test_vendor_missing_identity_rebuilds_without_snapshot(vendor_fixture):
    root, sdk, candidate, request, _tool = vendor_fixture
    (sdk / "src/device/input").unlink()
    first = vendor_prepare(vendor_fixture)
    assert not first["reused"] and "fingerprint" not in first
    VENDOR.built(request, candidate, time.time())
    VENDOR.publish(root, [json.loads(request.read_text())])
    assert not Path(first["directory"]).exists()


def test_vendor_single_slot_replaces_only_after_success(vendor_fixture, monkeypatch):
    """Default/matrix environment changes miss safely without separate cache slots."""
    root, _sdk, candidate, request, _tool = vendor_fixture
    original = vendor_prepare(vendor_fixture)
    VENDOR.built(request, candidate, time.time())
    VENDOR.publish(root, [json.loads(request.read_text())])
    record = Path(original["directory"]) / "record.json"
    previous = record.read_bytes()
    with monkeypatch.context() as context:
        context.setenv("PYTHONPATH", "/isolated/cann/modules")
        changed = vendor_prepare(vendor_fixture)
        assert not changed["reused"] and changed["directory"] == original["directory"]
        assert record.read_bytes() == previous
        VENDOR.built(request, candidate, time.time())
        VENDOR.publish(root, [json.loads(request.read_text())])
        assert vendor_prepare(vendor_fixture)["reused"]
    assert not vendor_prepare(vendor_fixture)["reused"]


def test_common_dispatch_and_cache_ignore_first_framework(vendor_fixture, tmp_path, monkeypatch):
    """The real dispatcher preserves caller tools while adapters keep their framework isolation."""
    root, sdk, _candidate, _request, _tool = vendor_fixture
    monkeypatch.setattr(BUILD, "NATIVE", tmp_path / "native")
    monkeypatch.setattr(BUILD, "validate_toolchain", lambda: None)
    monkeypatch.setenv("PYTHONPATH", "/developer/cann-overrides")
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    caller = Path(sys.executable).absolute()
    environments = []
    class Workers(BUILD.CONSOLE.Commands):
        def __call__(self, _command, **kwargs):
            env = kwargs["env"]
            environments.append(env)
            put(Path(env["HP_SHMEM_HANDOFF"]), json.dumps({"install_root": str(sdk)}))
    def stop_before_adapters(*_args):
        raise RuntimeError("stop before expensive adapters")
    monkeypatch.setattr(BUILD, "build_adapters", stop_before_adapters)
    for all_mode, framework in ((False, caller), (True, tmp_path / "pta27/bin/python"),
                                (True, tmp_path / "pta214/bin/python")):
        context = BUILD.BuildContext(ROOT, BUILD.NATIVE)
        context.isolated = all_mode
        context.commands = Workers(context.directory, events=context.events)
        monkeypatch.setattr(BUILD, "prepare_frameworks", lambda *_a, selected=framework: ([], [selected], [{}]))
        args = SimpleNamespace(all=all_mode, soc_list="ascend910b", jobs=1, clean=False)
        with pytest.raises(RuntimeError, match="stop before expensive adapters"):
            BUILD._build(args, context, SimpleNamespace(check_source=lambda: None))
        if all_mode:
            adapter = BUILD.adapter_environment(framework, isolated=True)
            assert adapter["PATH"].startswith(str(framework.parent) + ":")
            assert "/developer/cann-overrides" not in adapter.get("PYTHONPATH", "")
            assert adapter["PYTHONNOUSERSITE"] == "1"
    assert all(env["PYTHON_BIN"] == str(caller) for env in environments)
    assert all(env["PYTHONNOUSERSITE"] == "0" for env in environments)
    assert all(env["PYTHONPATH"].startswith("/developer/cann-overrides") for env in environments)
    identities = [VENDOR.vendor_inputs(root, sdk, "ascend910b", {key: env.get(key, "")
                   for key in VENDOR.ENVIRONMENT}) for env in environments]
    assert len({VENDOR.fingerprint(identity) for identity in identities}) == 1

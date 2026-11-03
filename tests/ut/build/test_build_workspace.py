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
"""Workspaces, locks, bounded cleanup and observable command execution."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tests.ut.build._build_test_support import ROOT, load_module, module_for, put
WORKSPACE = load_module("build_workspace")
BUILD = load_module("build_controller")
RUNNER = load_module("command_runner")
ENV = BUILD.FRAMEWORK
LAYOUT = BUILD.WORKSPACE
LIFECYCLE = load_module("preflight_cache")
RUN_ID = "20261102-143349-732599"


def test_latest_replaces_only_generated_logs_and_scratch(tmp_path):
    """Cleanup is bounded even when a generated child redirects outside the build tree."""
    previous = put(tmp_path / "logs/build.log", "failure")
    staged = put(tmp_path / "staging" / RUN_ID / "wheel.whl")
    orphan = put(tmp_path / "payload" / f".payload-{RUN_ID}" / "lib.so")
    protected = [put(tmp_path / name / "sentinel") for name in (
        "framework/envs", "framework/resolutions", "deps", "common", "variants", "preflight",
        "payload/hyper_multicore", "dist", "logs/manual", "staging/manual")]
    external = put(tmp_path / "external/keep")
    latest = WORKSPACE.prepare_logs(tmp_path)
    assert latest == tmp_path / "logs"
    assert not any(p.exists() for p in (previous, staged, orphan))
    assert json.loads((latest / "stages.json").read_text()) == []
    assert all(p.read_text() == "retained" for p in [*protected, external])


@pytest.mark.parametrize("directory, cleanup", [
    ("logs", WORKSPACE.prepare_logs),
    ("framework", lambda native: ENV.prune_environments(native, [], set())),
])
def test_cleanup_refuses_redirected_roots(tmp_path, directory, cleanup):
    external = tmp_path / "external"
    external.mkdir()
    sentinel = put(external / RUN_ID / "keep")
    (tmp_path / directory).symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        cleanup(tmp_path)
    assert sentinel.exists()


@pytest.mark.parametrize("arguments", [["--help"], ["--jobs", "0"]])
def test_help_and_bad_arguments_leave_previous_log_untouched(tmp_path, arguments):
    log = put(tmp_path / "logs/build.log", "previous failure")
    main = BUILD.main
    with patch.multiple(module_for(main), **{'NATIVE': tmp_path}), patch.object(sys, "argv", ["build.sh", *arguments]):
        with pytest.raises(SystemExit):
            main()
    assert log.read_text() == "previous failure"
    assert not (tmp_path / ".build.lock").exists()


def wait_text(path, token, process):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if path.exists() and token in path.read_text():
            return
        if process.poll() is not None:
            pytest.fail(f"lock fixture exited early: {path.read_text()}")
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for {token}: {path.read_text()}")


@contextmanager
def lock_process(tmp_path, name, hold):
    """A real independent process owns or waits for the same persistent lock inode."""
    script = ("import time; from pathlib import Path\n"
              f"import sys; sys.path.insert(0,{str(ROOT / 'hyper_multicore/_build')!r})\n"
              "from _hyper_multicore_bootstrap import load_module\n"
              f"h=load_module('build_workspace'); native=Path({str(tmp_path)!r})\n"
              "try:\n"
              " with h.build_lock(native, 'all'):\n"
              "  h.prepare_logs(native)\n"
              "  print('ACQUIRED',flush=True)\n"
              f"  while {hold!r} and not (native/'release').exists(): time.sleep(0.02)\n"
              "except KeyboardInterrupt: print('CANCELLED',flush=True)\n")
    log = tmp_path / f"{name}.log"
    with log.open("w") as output:
        process = subprocess.Popen([sys.executable, "-c", script], stdout=output, stderr=subprocess.STDOUT)
        try:
            yield process, log
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_wait_cancellation_preserves_owner_metadata_logs_and_inode(tmp_path):
    """A waiting build cannot erase active logs, metadata or the persistent lock file."""
    with lock_process(tmp_path, "owner", True) as (owner, owner_log):
        wait_text(owner_log, "ACQUIRED", owner)
        lock = tmp_path / ".build.lock"
        inode, metadata = lock.stat().st_ino, lock.read_text()
        latest = put(tmp_path / "logs/build.log", "active owner")
        with lock_process(tmp_path, "waiter", False) as (waiter, waiter_log):
            wait_text(waiter_log, "Waiting for build lock", waiter)
            assert lock.read_text() == metadata and latest.read_text() == "active owner"
            waiter.send_signal(signal.SIGINT)
            waiter.wait(timeout=5)
            assert "CANCELLED" in waiter_log.read_text()
            assert owner.poll() is None and lock.read_text() == metadata
            assert latest.read_text() == "active owner"
        put(tmp_path / "release")
        owner.wait(timeout=5)
        assert lock.exists() and lock.stat().st_ino == inode and lock.read_text() == ""
    with lock_process(tmp_path, "next", False) as (process, log):
        wait_text(log, "ACQUIRED", process)
        process.wait(timeout=5)
    assert lock.stat().st_ino == inode and not latest.exists()


def test_distribution_rotation_refuses_external_symlink(tmp_path):
    external = tmp_path / "external"
    old = put(external / "old.whl")
    (tmp_path / "dist").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        with WORKSPACE.BuildOutputs(tmp_path / "dist", tmp_path / "payload/hyper_multicore") as output:
            pytest.fail("must not enter")
    assert old.exists()


def test_success_prunes_only_owned_old_environments(tmp_path):
    """Retain current and latest successful rollback; never touch external symlinks."""
    base = tmp_path / "framework/envs"
    base.mkdir(parents=True)
    current = base / "current"
    for name, timestamp in (("current", 3), ("rollback", 2), ("older", 1), ("removed-branch", 5)):
        directory = base / name
        (directory / "bin").mkdir(parents=True)
        (directory / "ready.json").write_text(json.dumps({"pta_release": "2.8.0" if name == "removed-branch" else "2.12.0",
                                                         "last_successful_build": timestamp}))
    abandoned = base / ("failed-" + "f" * 20)
    abandoned.mkdir()
    (abandoned / "pending.json").write_text(json.dumps({"owner": "hyper-multicore-framework-env",
        "environment_key": "f" * 64, "pta_release": "2.12.0"}))
    unknown = base / "manual"
    unknown.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (base / "external-link").symlink_to(external, target_is_directory=True)
    removed = ENV.prune_environments(tmp_path, [current / "bin/python"], {"2.12.0"})
    assert {Path(p).name for p in removed} == {"older", "removed-branch", abandoned.name}
    assert current.exists() and (base / "rollback").exists() and external.exists() and unknown.exists()


def test_interruption_kills_grandchild_before_controller_returns(tmp_path):
    """TERM-resistant descendants must not outlive their interrupted build controller."""
    child_pid = tmp_path / "grandchild.pid"
    group_pid = tmp_path / "child-group.pid"
    grandchild = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text(str(os.getpid())); time.sleep(120)" % str(child_pid)
    child = "import os,subprocess,sys,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); subprocess.Popen([sys.executable,'-c',%r]); time.sleep(120)" % (str(group_pid), grandchild)
    wrapper = tmp_path / "controller.py"
    wrapper.write_text("import sys; from pathlib import Path\nsys.path.insert(0,%r)\nfrom _hyper_multicore_bootstrap import load_module\nh=load_module('command_runner')\nh.Commands(Path(%r))([sys.executable,'-c',%r],label='child')\n" %
                       (str(ROOT / "hyper_multicore/_build"), str(tmp_path / "logs"), child))
    with subprocess.Popen([sys.executable, str(wrapper)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as controller:
        try:
            deadline = time.monotonic() + 20
            while not child_pid.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert child_pid.exists(), "test grandchild did not start"
            controller.send_signal(signal.SIGINT)
            controller.wait(timeout=25)
            stat = Path("/proc") / child_pid.read_text() / "stat"
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait()
            if group_pid.exists():
                try:
                    os.killpg(int(group_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if child_pid.exists():
                try:
                    os.kill(int(child_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_tool_output_is_visible_before_child_exits(tmp_path):
    """The terminal receives actual installer/compiler output while the tool is running."""
    gate = tmp_path / "release"
    child = f"import time; from pathlib import Path; print('INSTALLING-VISIBLE-NOW',flush=True)\nwhile not Path({str(gate)!r}).exists(): time.sleep(0.02)"
    script = tmp_path / "live.py"
    script.write_text("import sys; from pathlib import Path\nsys.path.insert(0,%r)\nfrom _hyper_multicore_bootstrap import load_module\nh=load_module('command_runner')\nh.Commands(Path(%r))([sys.executable,'-c',%r],label='install-fixture')\n" %
                      (str(ROOT / "hyper_multicore/_build"), str(tmp_path / "logs"), child))
    output = b""
    with subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 5
                while b"INSTALLING-VISIBLE-NOW" not in output and time.monotonic() < deadline:
                    for key, _ in selector.select(timeout=0.1):
                        output += os.read(key.fileobj.fileno(), 8192)
            assert b"INSTALLING-VISIBLE-NOW" in output
            assert process.poll() is None, "output was buffered until the child exited"
        finally:
            gate.write_text("release")
            process.communicate(timeout=5)
    assert process.returncode == 0
    assert "INSTALLING-VISIBLE-NOW" in next((tmp_path / "logs").glob("*.log")).read_text()


def test_merged_log_ranges_keep_errors_and_private_json_separate(tmp_path, capsys):
    run = RUNNER.Commands(tmp_path)
    run([sys.executable, "-c", "print('error: earlier stage'); print('Building CXX object earlier.o')"], label="fixture")
    assert run([sys.executable, "-c", "print('JSON-ONLY')"], label="metadata", capture=True) == "JSON-ONLY\n"
    with pytest.raises(RuntimeError) as caught:
        run([sys.executable, "-c", "print('error: current linker failure'); raise SystemExit(2)"], label="preflight-compile-fixture")
    assert "current linker failure" in str(caught.value) and "earlier stage" not in str(caught.value)
    assert list(tmp_path.glob("*.log")) == [tmp_path / "tools.log"]
    records = json.loads((tmp_path / "stages.json").read_text())
    chunks = [RUNNER.read_log_range(Path(e["log"]), e["log_start"], e["log_end"]) for e in records]
    assert chunks[1] == "JSON-ONLY\n" and "JSON-ONLY" not in chunks[2]
    assert "JSON-ONLY" not in capsys.readouterr().out
    assert records[-1]["status"] == "failed"


def workspace(native, key, root=ROOT):
    directory = native / "preflight" / ("cp310-aarch64-torch2.12.0-pt2.12.0." + key[0] + "-abi1")
    directory.mkdir(parents=True)
    (directory / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={root / 'hyper_multicore/cmake/framework_probe'}\n")
    return directory


def success(native, mode, *keys):
    events = [{"fingerprint": key[0] * 64 if key.startswith("cp") else key, "directory": key, "preflight_cache_hit": True} for key in keys]
    LIFECYCLE.finish(native, ROOT, mode, events)
    return events[-1]


def test_full_and_default_protect_each_other_and_collect_only_old_owned_workspaces(tmp_path):
    old, full, default = (workspace(tmp_path, key * 64) for key in ("a", "b", "c"))
    foreign = workspace(tmp_path, "d" * 64, tmp_path / "other-project")
    unknown = full.parent / "manual"
    unknown.mkdir()
    linked = full.parent / "external-link"
    linked.symlink_to(foreign, target_is_directory=True)
    success(tmp_path, "current", default.name)
    event = success(tmp_path, "all", full.name)
    assert not old.exists() and full.exists() and default.exists()
    assert event["removed"] == [old.name]
    assert foreign.exists() and unknown.exists() and linked.is_symlink()
    success(tmp_path, "current", default.name)
    success(tmp_path, "all", full.name)
    assert full.exists() and default.exists()


@pytest.mark.parametrize("value", ["bad JSON"])
def test_corrupt_index_never_authorizes_deletion(tmp_path, value):
    old, full = (workspace(tmp_path, key * 64) for key in ("a", "b"))
    index = full.parent / "index.json"
    index.write_text(value)
    event = success(tmp_path, "all", full.name)
    assert event["status"] == "invalid-index" and old.exists()
    assert index.read_text() == value


def test_unknown_and_redirected_fixed_slots_cannot_be_replaced(tmp_path):
    slot = tmp_path / "variants" / "target"
    sentinel = put(slot / "manual", "keep")
    with pytest.raises(ValueError, match="Unattributed"):
        LAYOUT.check_workspace(slot, ROOT)
    assert sentinel.read_text() == "keep"
    LAYOUT.claim_workspace(slot, ROOT)
    LAYOUT.check_workspace(slot, ROOT)
    with pytest.raises(ValueError, match="Foreign"):
        LAYOUT.check_workspace(slot, tmp_path)
    linked = slot.with_name("linked")
    linked.symlink_to(slot, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        LAYOUT.check_workspace(linked, ROOT)


@pytest.mark.parametrize("diagnostic, language_error", [
    ("error: C++20 or later compatible compiler is required to use PyTorch", True),
    ("ld: undefined reference to aclnnExample", False),
])
def test_preflight_failure_preserves_cause_and_offers_applicable_remedy(tmp_path, diagnostic, language_error):
    """A redirected compiler diagnostic must reach the top-level failure message."""
    commands = RUNNER.Commands(tmp_path / "logs")
    def _replay(_command, **kwargs):
        if kwargs["label"].startswith("preflight-compile-"):
            return commands([sys.executable, "-c", f"print({diagnostic!r}); raise SystemExit(2)"], **kwargs)
        return None
    identity = {"torch_version": "2.14.0+cpu", "pta_version": "2.14.0", "python_abi": "cp310", "host_arch": "aarch64", "torch_cxx11_abi": 1}
    compiler = {"command": ["/usr/bin/g++"], "version": "g++ (GCC) 10.3.1\nCopyright\n"}
    context = BUILD.BuildContext(ROOT, tmp_path)
    context.commands = _replay
    context.frameworks[str(Path(sys.executable))] = BUILD.PreparedFramework(Path(sys.executable), identity, {})
    context.native_inputs.enrich_tools = lambda contract, *args: contract
    function = BUILD.preflight_frameworks
    with patch.multiple(module_for(function), **{'preflight_inputs': lambda *_: {'fixture': True, 'compiler': None}, 'NATIVE': tmp_path}):
        with pytest.raises(RuntimeError) as caught:
            function(SimpleNamespace(all=True, clean=False, soc_list="ascend910b", jobs=24), context, [Path(sys.executable)], [identity], compiler)
    message = str(caught.value)
    assert diagnostic in message
    assert "2.14.0+cpu" in message and "/usr/bin/g++" in message and "10.3.1" in message
    assert "tools.log" in message and "preflight-compile-2.14.0" in message
    if language_error:
        assert "HYPER_MULTICORE_FRAMEWORK_CXX=" in message
        assert "--all --soc-list ascend910b --jobs 24" in message
    else:
        assert "C++20" not in message

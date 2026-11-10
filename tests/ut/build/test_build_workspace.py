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

from concurrent.futures import ThreadPoolExecutor
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
RUN_ID = "20261102-143349-732599"


def test_latest_replaces_only_generated_logs_and_scratch(tmp_path):
    """Cleanup is bounded even when a generated child redirects outside the build tree."""
    previous = put(tmp_path / "logs/build.log", "failure")
    staged = put(tmp_path / "staging" / RUN_ID / "wheel.whl")
    orphan = put(tmp_path / "payload" / f".payload-{RUN_ID}" / "lib.so")
    protected = [put(tmp_path / name / "sentinel") for name in (
        "envs", "deps", "common", "variants",
        "payload/hyper_multicore", "dist", "logs/manual", "staging/manual")]
    external = put(tmp_path / "external/keep")
    latest = WORKSPACE.prepare_logs(tmp_path)
    assert latest == tmp_path / "logs"
    assert not any(p.exists() for p in (previous, staged, orphan))
    assert json.loads((latest / "stages.json").read_text()) == []
    assert all(p.read_text() == "retained" for p in [*protected, external])


@pytest.mark.parametrize("directory, cleanup", [
    ("logs", WORKSPACE.prepare_logs),
    ("envs", lambda native: ENV.prune_environments(native, [], set())),
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


@pytest.mark.parametrize("custom", [False, True], ids=["default-root", "custom-root"])
def test_success_prunes_only_owned_old_environments(tmp_path, monkeypatch, custom):
    """Retain current and latest successful rollback; never touch external symlinks."""
    base = tmp_path / ("custom venvs" if custom else "envs")
    if custom:
        monkeypatch.setenv("HYPER_MULTICORE_VENV_PATH", str(base))
    protected = put(tmp_path / "other-storage/sentinel")
    base.mkdir(parents=True)
    current = base / ("current-" + "a" * 20)
    for name, timestamp in (("current", 3), ("rollback", 2), ("older", 1), ("removed-branch", 5)):
        directory = base / (name + "-" + "a" * 20)
        (directory / "bin").mkdir(parents=True)
        (directory / "ready.json").write_text(json.dumps({"environment_key": "a" * 64, "request": {}, "pta_release": "2.8.0" if name == "removed-branch" else "2.12.0",
                                                         "last_successful_build": timestamp}))
    abandoned = base / ("failed-" + "f" * 20)
    abandoned.mkdir()
    (abandoned / "pending.json").write_text(json.dumps({"owner": "hyper-multicore-framework-env",
        "environment_key": "f" * 64, "pta_release": "2.12.0"}))
    unknown = base / "manual"
    unknown.mkdir()
    (unknown / "ready.json").write_text("{}")
    redirected = base / ("redirected-" + "a" * 20)
    redirected.mkdir()
    (redirected / "ready.json").symlink_to(current / "ready.json")
    sentinel = put(redirected / "manual")
    external = tmp_path / "external"
    external.mkdir()
    (base / "external-link").symlink_to(external, target_is_directory=True)
    removed = ENV.prune_environments(tmp_path, [current / "bin/python"], {"2.12.0"})
    assert {Path(p).name for p in removed} == {"older-" + "a" * 20, "removed-branch-" + "a" * 20, abandoned.name}
    assert current.exists() and (base / ("rollback-" + "a" * 20)).exists() and external.exists() and unknown.exists()
    assert sentinel.exists() and (redirected / "ready.json").is_symlink()
    assert protected.exists()


def test_environment_storage_selection_and_redirect_protection(tmp_path, monkeypatch):
    """Only explicit absolute roots replace the default; redirected storage stays protected."""
    assert WORKSPACE.environment_directory(tmp_path) == tmp_path / "envs"
    monkeypatch.setenv("HYPER_MULTICORE_VENV_PATH", "")
    assert WORKSPACE.environment_directory(tmp_path) == tmp_path / "envs"
    storage = tmp_path / "external venvs"
    monkeypatch.setenv("HYPER_MULTICORE_VENV_PATH", str(tmp_path / "unused/../external venvs"))
    assert WORKSPACE.environment_directory(tmp_path) == storage
    storage.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(storage, target_is_directory=True)
    monkeypatch.setenv("HYPER_MULTICORE_VENV_PATH", str(redirected / "envs"))
    with pytest.raises(ValueError, match="symlinked"):
        WORKSPACE.environment_directory(tmp_path)


@pytest.mark.parametrize("configured", ["relative/venvs", "~/venvs"])
def test_relative_environment_storage_fails_before_install(tmp_path, monkeypatch, configured):
    monkeypatch.setattr(BUILD, "NATIVE", tmp_path)
    monkeypatch.setenv("HYPER_MULTICORE_VENV_PATH", configured)
    with pytest.raises(ValueError, match="HYPER_MULTICORE_VENV_PATH must be an absolute path"):
        BUILD.prepare_matrix_environments([], None, Path(sys.executable), "ascend910b", {})
    assert not (tmp_path / "envs").exists()


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


def test_failed_tool_cleans_up_descendants_and_preserves_diagnostic(tmp_path):
    """A failed parent must not leave its TERM-resistant descendants outside cancellation."""
    pidfile = tmp_path / "descendant.pid"
    descendant = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}])\n"
        f"while not Path({str(pidfile)!r}).exists(): time.sleep(0.02)\n"
        "print('error: initiating compiler failure',flush=True); raise SystemExit(2)"
    )
    run = RUNNER.Commands(tmp_path / "logs")
    try:
        with pytest.raises(RuntimeError, match="initiating compiler failure"):
            run([sys.executable, "-c", parent], label="compile-fixture")
        stat = Path("/proc") / pidfile.read_text() / "stat"
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        record = json.loads((run.directory / "stages.json").read_text())[-1]
        assert record["exit_code"] == 2 and record["status"] == "failed"
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
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
        run([sys.executable, "-c", "print('error: current linker failure'); raise SystemExit(2)"], label="compile-fixture")
    assert "current linker failure" in str(caught.value) and "earlier stage" not in str(caught.value)
    assert list(tmp_path.glob("*.log")) == [tmp_path / "tools.log"]
    records = json.loads((tmp_path / "stages.json").read_text())
    chunks = [RUNNER.read_log_range(Path(e["log"]), e["log_start"], e["log_end"]) for e in records]
    assert chunks[1] == "JSON-ONLY\n" and "JSON-ONLY" not in chunks[2]
    assert "JSON-ONLY" not in capsys.readouterr().out
    assert records[-1]["status"] == "failed"


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


@pytest.mark.parametrize("diagnostic,language_error", [
    ("error: C++20 or later compatible compiler is required to use PyTorch", True),
    ("ld: undefined reference to aclnnExample", False)])
def test_real_compile_failure_preserves_diagnostic(tmp_path, monkeypatch, diagnostic, language_error):
    context = BUILD.BuildContext(ROOT, tmp_path)
    def fail(_python, _identity, _common, _sdk, selected, run):
        assert selected.jobs == 3
        run.commands([sys.executable, "-c", f"print({diagnostic!r}); raise SystemExit(2)"],
                     label="compile-pt2.14.0-torch")
    monkeypatch.setattr(BUILD, "cmake_pair", fail)
    identity = {"torch_version": "2.14.0+cpu", "pta_version": "2.14.0"}
    args = SimpleNamespace(all=True, soc_list="ascend910b", jobs=24)
    with pytest.raises(RuntimeError) as caught:
        BUILD.build_adapter(0, 7, Path(sys.executable), identity, 3, args, context, tmp_path, tmp_path,
                            SimpleNamespace(check_source=lambda: None))
    message = str(caught.value)
    assert diagnostic in message and "tools.log" in message and "2.14.0+cpu" in message
    assert ("HYPER_MULTICORE_FRAMEWORK_CXX=" in message) == language_error
    if language_error:
        assert "--all --soc-list ascend910b --jobs 24" in message


def test_parallel_output_has_exclusive_raw_ranges_and_live_context(tmp_path, capsys):
    logs = tmp_path / "logs"
    scratch = tmp_path / "staging/output"
    commands = RUNNER.Commands(logs, scratch=scratch)
    def invoke(name):
        code = (f"import time; print('partial-{name}-',end='',flush=True); time.sleep(0.1); "
                f"print('done'); print('last-{name}',end='',flush=True)")
        with commands.scope(name):
            commands([sys.executable, "-c", code], label="compile-fixture")
    with RUNNER.capture_log(logs / "build.log"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(invoke, ("left", "right")))
        assert commands([sys.executable, "-c", "print('{\"private\": true}')"],
                        label="metadata", capture=True) == '{"private": true}\n'
    visible = capsys.readouterr().out
    assert '[left] partial-left-done' in visible and '[right] partial-right-done' in visible
    assert '[left] last-left' in visible and '[right] last-right' in visible
    assert '"private"' not in visible
    records = json.loads((logs / "stages.json").read_text())
    previous = 0
    for record in records:
        assert record["log_start"] == previous
        previous = record["log_end"]
        raw = RUNNER.read_log_range(commands.log, record["log_start"], record["log_end"])
        if record["context"]:
            assert record["context"] in raw
            assert ("right" if record["context"] == "left" else "left") not in raw
    assert previous == commands.log.stat().st_size
    assert sorted(p.name for p in logs.iterdir()) == ["build.log", "stages.json", "tools.log"]
    assert not list(scratch.iterdir())


def parallel_driver(tmp_path, failure):
    """Run the actual adapter scheduler around real child groups and the output transaction."""
    return f'''import os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(ROOT / "hyper_multicore/_build")!r})
from _hyper_multicore_bootstrap import load_module
build=load_module('build_controller')
root=Path({str(tmp_path)!r})
run=build.BuildContext(Path({str(ROOT)!r}),root/'native')
started=[]
def pair(python,identity,common,sdk,args,context):
 index=int(identity['pta_version'])
 started.append(index)
 if index==1 and {failure!r}:
  deadline=time.monotonic()+8
  while not (root/'grandchild-0.pid').exists():
   if time.monotonic()>deadline:raise RuntimeError('fixture child did not start')
   time.sleep(0.02)
  context.commands([sys.executable,'-c',"print('fatal error: originating compile failure'); raise SystemExit(7)"],label='compile-origin')
 else:
  pid=root/f'grandchild-{{index}}.pid'
  grandchild="import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text(str(os.getpid())); time.sleep(120)" % str(pid)
  parent="import subprocess,sys,time; from pathlib import Path; subprocess.Popen([sys.executable,'-c',%r]); p=Path(%r)\\nwhile not p.exists(): time.sleep(0.02)\\nprint('WORKER_READY %d',flush=True); time.sleep(120)" % (grandchild,str(pid),index)
  context.commands([sys.executable,'-c',parent],label='compile-worker')
 return root/str(index),identity
build.cmake_pair=pair
signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
output=build.WORKSPACE.BuildOutputs(root/'dist',root/'payload')
try:
 with output:
  build.build_adapters(SimpleNamespace(all=True,jobs=8),run,[Path(sys.executable)]*7,
    [{{'pta_version':str(i),'torch_version':'fixture'}} for i in range(7)],root,root,
    SimpleNamespace(check_source=lambda:None))
except BaseException as error:
 print('RESULT',type(error).__name__,str(error),flush=True)
 (root/'started.json').write_text(__import__('json').dumps(started))
 raise SystemExit(130 if isinstance(error,KeyboardInterrupt) else 9)
'''


@pytest.mark.parametrize("interrupt", [None, signal.SIGINT, signal.SIGTERM])
def test_parallel_failure_and_signals_stop_all_groups_before_output_cleanup(tmp_path, interrupt):
    """An originating failure or user signal cannot leave siblings, descendants or partial outputs."""
    script = put(tmp_path / "driver.py", parallel_driver(tmp_path, interrupt is None))
    put(tmp_path / "dist/old.whl")
    put(tmp_path / "payload/old.so")
    log = tmp_path / "driver.log"
    with log.open("w", encoding="utf-8") as output:
        controller = subprocess.Popen([sys.executable, str(script)], stdout=output, stderr=subprocess.STDOUT)
        try:
            if interrupt is not None:
                wait_text(log, "WORKER_READY 1", controller)
                controller.send_signal(interrupt)
            controller.wait(timeout=25)
            text = log.read_text()
            assert controller.returncode == (9 if interrupt is None else 130), text
            assert not list((tmp_path / "dist").iterdir()) and not (tmp_path / "payload").exists()
            assert sorted(json.loads((tmp_path / "started.json").read_text())) == [0, 1]
            assert ("RESULT RuntimeError" if interrupt is None else "RESULT KeyboardInterrupt") in text
            if interrupt is None:
                assert "originating compile failure" in text
            records = json.loads((tmp_path / "native/logs/stages.json").read_text())
            assert any(r.get("status") == "cancelled" for r in records)
            assert not list((tmp_path / "native/staging").rglob("*.tmp"))
            for marker in tmp_path.glob("grandchild-*.pid"):
                stat = Path("/proc") / marker.read_text() / "stat"
                assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait(timeout=5)
            for marker in tmp_path.glob("grandchild-*.pid"):
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

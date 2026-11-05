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
"""Execute tools with live console output, transcripts and bounded stage logs."""

import codecs
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from .build_utils import atomic_json as write_json

ROOT = Path(__file__).resolve().parents[2]
CONSOLE_LOCK = threading.RLock()


def format_message(text: str, level="INFO") -> str:
    """Keep controller feedback consistent in terminals, redirected output and CI."""
    return f"{time.strftime('%H:%M:%S')} {level:<5} {text}\n"


def message(text: str, level="INFO"):
    """Write one complete message so tool-reader threads cannot split its prefix."""
    stream = sys.stderr if level == "ERROR" else sys.stdout
    with CONSOLE_LOCK:
        stream.write(format_message(text, level))
        stream.flush()


class Transcript:
    """Mirror an existing stream without taking ownership of its terminal descriptor."""

    def __init__(self, stream, log, lock):
        self.stream, self.log, self.lock = stream, log, lock

    def write(self, text):
        with self.lock:
            self.log.write(text)
            self.log.flush()
            return self.stream.write(text)

    def flush(self):
        with self.lock:
            self.log.flush()
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def capture_log(path: Path):
    """Retain controller, stdout and stderr output together, including failure guidance."""
    lock = threading.RLock()
    with path.open("w", encoding="utf-8") as log:
        with redirect_stdout(Transcript(sys.stdout, log, lock)), redirect_stderr(Transcript(sys.stderr, log, lock)):
            yield


def stage_title(label: str) -> str:
    """Translate internal task identifiers into readable progress titles."""
    label = re.sub(r"cp\d+-[^ ]+-torch[^ ]+-pta([^-]+)-[0-9a-f]{20}", r"PTA \1", label)
    label = re.sub(r"pt([\d.]+(?:post\d+)?)-[0-9a-f]{12}", r"PTA \1", label)
    titles = {
        "package-wheel": "Packaging wheel", "common-resources": "Building shared native resources",
        "install-adapter-": "Installing adapter: ",
        "venv-": "Creating framework environment: ", "install-": "Installing framework packages: ",
        "download-": "Downloading framework wheels: ",
        "configure-": "Configuring adapter: ",
        "compile-": "Building adapter: ", "ctest-": "Testing SHMEM: ",
    }
    for prefix, title in titles.items():
        if label.startswith(prefix):
            return title + label[len(prefix):]
    return label.replace("-", " ").capitalize()


def visible_stage(label: str, capture: bool) -> bool:
    """Internal identity/hash probes remain in stage records rather than terminal progress."""
    return not capture and not label.startswith("identity-")


def compiler_metrics(output: str) -> dict:
    """Count logged compiler commands and separately expose observable CMake/OPC actions."""
    commands = 0
    for line in output.splitlines():
        if not re.search(r"(?:^|\s)-c(?:\s|$)", line):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if "-c" in tokens and any(
            re.search(r"(?:^|-)(?:g\+\+|c\+\+|clang\+\+|bisheng)(?:-\d+(?:\.\d+)*)?$", Path(token).name)
            for token in tokens
        ):
            commands += 1
    object_actions = len(re.findall(r"Building (?:C|CXX) object ", output))
    return {
        "compiler_invocations": max(commands, object_actions),
        "compiler_commands_logged": commands,
        "host_object_compile_actions": object_actions,
        "kernel_compile_jobs": output.count("Opc tool start working now"),
    }


def stage_failure(label: str, status: int, log: Path, start=0, end=None) -> str:
    """Surface the first tool diagnostic while keeping full output in its log."""
    detail = f"Stage {label} failed ({status}); inspect {log}"
    lines = read_log_range(log, start, end).splitlines()
    errors = [
        (i, line)
        for i, line in enumerate(lines)
        if re.search(r"fatal error:|error:|CMake Error|undefined reference|(?:Error|Exception):", line, re.IGNORECASE)
    ]
    if errors:
        index = errors[0][0]
        detail += "\nDiagnostic:\n" + "\n".join(line[:600] for line in lines[index : index + 3])
    return detail


class StageOutput:
    """Mirror tool logs live and report quiet stages without piping or blocking the child."""

    def __init__(self, log: Path, label: str, started: float, *, capture=False, heartbeat=10.0, context=""):
        self.log, self.label, self.started = log, label, started
        self.capture, self.heartbeat, self.context = capture, heartbeat, context
        self.pending = ""
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self.follow, name="native-build-output", daemon=True)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.worker.join()

    def emit(self, text: str) -> bool:
        """A closed terminal must not leave the log reader alive or lose disk evidence."""
        try:
            if text:
                with CONSOLE_LOCK:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            return True
        except (OSError, ValueError):
            return False

    def follow(self):
        """Drain final bytes before exiting; captured machine stdout is never echoed."""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        last_output = time.monotonic()
        with self.log.open("rb") as source:
            while True:
                data = b"" if self.capture else source.read(8192)
                if data:
                    if not self.emit_lines(decoder.decode(data)):
                        return
                    last_output = time.monotonic()
                    continue
                if self.stopped.is_set():
                    break
                now = time.monotonic()
                if now - last_output >= self.heartbeat:
                    if not self.emit(
                        format_message(
                            f"{self.context + ': ' if self.context else ''}"
                            f"{stage_title(self.label)}: still running ({now - self.started:.0f}s elapsed)"
                        )
                    ):
                        return
                    last_output = now
                self.stopped.wait(0.1)
        self.emit_lines(decoder.decode(b"", final=True), final=True)

    def emit_lines(self, text: str, *, final=False) -> bool:
        """Keep each contextual line intact even when several children write partial chunks."""
        if not self.context:
            return self.emit(text)
        self.pending += text.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if not self.emit(f"[{self.context}] {line}\n"):
                return False
        if final and self.pending:
            return self.emit(f"[{self.context}] {self.pending}\n")
        return True


def signal_groups(processes, number):
    """Signal every group before waiting for any individual child."""
    for process in processes:
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass


def wait_for_groups(processes):
    """Bound descendant cleanup by one shared deadline, rather than one per target."""
    groups = {process.pid for process in processes}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        remaining = False
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                remaining |= fields[0] != "Z" and int(fields[2]) in groups
            except (OSError, ValueError, IndexError):
                continue
        if not remaining:
            return
        time.sleep(0.05)


def stop_stages(processes) -> None:
    """Terminate all active groups under common grace periods, including descendants."""
    signal_groups(processes, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
        time.sleep(0.05)
    signal_groups(processes, signal.SIGKILL)
    for process in processes:
        process.wait()
    wait_for_groups(processes)


def read_log_range(path: Path, start=0, end=None) -> str:
    """Decode only this command's byte interval, never preceding tool output."""
    with path.open("rb") as source:
        source.seek(start)
        return source.read(-1 if end is None else end - start).decode("utf-8", errors="replace")


class CommandCancelled(RuntimeError):
    """An otherwise valid command was stopped by another failure or user cancellation."""


class Commands:
    """Execute concurrent tools while serializing raw-log blocks and stage records."""

    def __init__(self, directory: Path, *, root: Path = ROOT, events=None, scratch=None):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = directory / "tools.log"
        self.log.touch(exist_ok=True)
        self.root = root
        self.events = [] if events is None else events
        self.scratch = scratch or directory
        self.cancelled = threading.Event()
        self.failure = None
        self._processes = {}
        self._process_lock = threading.RLock()
        self._output_lock = threading.RLock()
        self._local = threading.local()

    @contextmanager
    def scope(self, name):
        """Attach an opaque output label to one calling thread, without shared environment mutation."""
        previous = getattr(self._local, "context", "")
        self._local.context = f"{previous} / {name}" if previous else name
        try:
            yield
        finally:
            self._local.context = previous

    def add_event(self, record):
        """Append complete stage metadata and replace JSON under the same writer lock."""
        with self._output_lock:
            self.events.append(record)
            write_json(self.directory / "stages.json", self.events)

    def cancel(self, error):
        """The first cause wins; no process may start after the cancellation snapshot."""
        with self._process_lock:
            if self.cancelled.is_set():
                return
            self.failure = error
            self.cancelled.set()
            processes = tuple(self._processes.values())
        if processes:
            message(f"Stopping {len(processes)} active tool process groups", "WARN")
            stop_stages(processes)

    def start(self, command, output, env, capture):
        """Make cancellation and process registration atomic with respect to worker starts."""
        with self._process_lock:
            if self.cancelled.is_set():
                raise CommandCancelled("Build cancelled; no further commands will start")
            process = subprocess.Popen(
                command, cwd=self.root, env=env, stdout=subprocess.PIPE if capture else output,
                stderr=output, text=True, start_new_session=True,
            )
            self._processes[process.pid] = process
            return process

    def record(self, command, label, started, wall_started, spool, status, error=None):
        """Merge one complete raw block and give it an exclusive byte interval."""
        text = spool.read_text(encoding="utf-8", errors="replace")
        record = {
            "stage": label, "context": getattr(self._local, "context", ""),
            "started_at": wall_started, "ended_at": time.time(),
            "elapsed_seconds": time.monotonic() - started, "exit_code": status,
            "status": command_status(status, error), "log": str(self.log), "command": list(command),
            "disk_free_bytes": shutil.disk_usage(self.directory).free,
            **compiler_metrics(text),
        }
        if error:
            record["error"] = str(error) or type(error).__name__
        with self._output_lock:
            with self.log.open("ab") as output, spool.open("rb") as source:
                record["log_start"] = output.tell()
                shutil.copyfileobj(source, output)
                record["log_end"] = output.tell()
            self.add_event(record)
        return record

    def execute(self, command, label, spool, started, env, capture):
        """Each child owns its output file; only terminal rendering is shared while running."""
        stdout = None
        with spool.open("wb", buffering=0) as output:
            process = self.start(command, output, env, capture)
            try:
                with StageOutput(spool, label, started, capture=capture,
                                 context=getattr(self._local, "context", "")):
                    stdout, _ = process.communicate()
            except BaseException as error:
                self.cancel(error)
                if capture:
                    stdout, _ = process.communicate()
                raise
            finally:
                if process.returncode and not self.cancelled.is_set():
                    stop_stages((process,))
                with self._process_lock:
                    self._processes.pop(process.pid, None)
                if capture and stdout:
                    output.write(bytes(stdout, "utf-8"))
        return stdout, process.returncode

    def __call__(self, command, *, label: str, env=None, capture=False):
        if self.cancelled.is_set():
            raise CommandCancelled("Build cancelled; no further commands will start")
        self.scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.scratch, prefix="tool-", suffix=".tmp", delete=False) as file:
            spool = Path(file.name)
        started, wall_started = time.monotonic(), time.time()
        stdout, status, error = None, None, None
        title = stage_title(label)
        context = getattr(self._local, "context", "")
        if context:
            title = f"{context}: {title}"
        visible = visible_stage(label, capture)
        if visible:
            message(title)
        try:
            try:
                stdout, status = self.execute(command, label, spool, started, env, capture)
                if self.cancelled.is_set():
                    error = CommandCancelled("Stage cancelled after another failure or interruption")
            except BaseException as caught:
                error = caught
            record = self.record(command, label, started, wall_started, spool, status, error)
        finally:
            spool.unlink(missing_ok=True)
        if error is not None:
            raise error
        if status or visible:
            outcome = "failed" if status else "completed"
            message(f"{title}: {outcome} in {record['elapsed_seconds']:.1f}s (exit {status})",
                    "ERROR" if status else "INFO")
        if status:
            diagnostic = stage_failure(label, status, self.log, record["log_start"], record["log_end"])
            raise RuntimeError(f"{diagnostic}\nCommand executable: {command[0]}")
        return stdout


def command_status(status, error):
    """Keep collateral cancellation distinct from the initiating command failure."""
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, CommandCancelled):
        return "cancelled"
    return "failed" if error or status else "succeeded"

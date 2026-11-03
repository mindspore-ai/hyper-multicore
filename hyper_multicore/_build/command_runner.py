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
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from .build_utils import atomic_json as write_json

ROOT = Path(__file__).resolve().parents[2]


def format_message(text: str, level="INFO") -> str:
    """Keep controller feedback consistent in terminals, redirected output and CI."""
    return f"{time.strftime('%H:%M:%S')} {level:<5} {text}\n"


def message(text: str, level="INFO"):
    """Write one complete message so tool-reader threads cannot split its prefix."""
    stream = sys.stderr if level == "ERROR" else sys.stdout
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
        "preflight-configure-": "Configuring framework preflight: PTA ",
        "preflight-compile-": "Checking framework compile/link support: PTA ",
        "venv-": "Creating framework environment: ", "install-": "Installing framework packages: ",
        "pip-check-": "Checking installed dependencies: ", "download-": "Downloading framework wheels: ",
        "resolve-": "Resolving framework dependencies: ", "configure-": "Configuring adapter: ",
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
    """Surface the first preflight diagnostic while keeping full output in its log."""
    detail = f"Stage {label} failed ({status}); inspect {log}"
    lines = read_log_range(log, start, end).splitlines()
    errors = [
        (i, line)
        for i, line in enumerate(lines)
        if re.search(r"fatal error:|error:|CMake Error|undefined reference|(?:Error|Exception):", line, re.IGNORECASE)
    ]
    if errors:
        index = errors[0][0] if label.startswith("preflight-") else errors[-1][0]
        detail += "\nDiagnostic:\n" + "\n".join(line[:600] for line in lines[index : index + 3])
    return detail


class StageOutput:
    """Mirror tool logs live and report quiet stages without piping or blocking the child."""

    def __init__(self, log: Path, label: str, started: float, *, capture=False, heartbeat=10.0, offset=0):
        self.log, self.label, self.started = log, label, started
        self.capture, self.heartbeat, self.offset = capture, heartbeat, offset
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
            source.seek(self.offset)
            while True:
                data = b"" if self.capture else source.read(8192)
                if data:
                    if not self.emit(decoder.decode(data)):
                        return
                    last_output = time.monotonic()
                    continue
                if self.stopped.is_set():
                    break
                now = time.monotonic()
                if now - last_output >= self.heartbeat:
                    if not self.emit(
                        format_message(
                            f"{stage_title(self.label)}: still running ({now - self.started:.0f}s elapsed)"
                        )
                    ):
                        return
                    last_output = now
                self.stopped.wait(0.1)
        self.emit(decoder.decode(b"", final=True))


def stop_stage(process) -> None:
    """Preserve bounded cleanup of the child process group, including surviving grandchildren."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit while a grandchild still holds the process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        remaining = []
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == process.pid:
                    remaining.append(stat)
            except (OSError, ValueError, IndexError):
                continue
        if not remaining:
            break
        time.sleep(0.05)


def read_log_range(path: Path, start=0, end=None) -> str:
    """Decode only this command's byte interval, never preceding tool output."""
    with path.open("rb") as source:
        source.seek(start)
        return source.read(-1 if end is None else end - start).decode("utf-8", errors="replace")


class Commands:
    """Keep raw tools in one append-only stream with explicit command byte ranges."""

    def __init__(self, directory: Path, *, root: Path = ROOT, events=None):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = directory / "tools.log"
        self.log.touch(exist_ok=True)
        self.root = root
        self.events = [] if events is None else events

    def record(self, command, label, started, offset, status, error=None):
        end = self.log.stat().st_size
        record = {
            "stage": label,
            "elapsed_seconds": time.monotonic() - started,
            "exit_code": status,
            "status": (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else ("failed" if error or status else "succeeded")
            ),
            "log": str(self.log),
            "log_start": offset,
            "log_end": end,
            "command": list(command),
            "disk_free_bytes": shutil.disk_usage(self.directory).free,
        }
        record.update(compiler_metrics(read_log_range(self.log, offset, end)))
        if error:
            record["error"] = str(error) or type(error).__name__
        self.events.append(record)
        write_json(self.directory / "stages.json", self.events)
        return record

    def execute(self, command, label, offset, started, env, capture):
        stdout = None
        with self.log.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=subprocess.PIPE if capture else output,
                stderr=output,
                text=True,
                start_new_session=True,
            )
            with StageOutput(self.log, label, started, capture=capture, offset=offset):
                try:
                    stdout, _ = process.communicate()
                except BaseException:
                    message(f"{stage_title(label)}: stopping the stage process group", "WARN")
                    stop_stage(process)
                    if capture:
                        stdout, _ = process.communicate()
                    raise
                finally:
                    if capture and stdout:
                        output.write(bytes(stdout, "utf-8"))
        return stdout, process.returncode

    def __call__(self, command, *, label: str, env=None, capture=False):
        started, offset = time.monotonic(), self.log.stat().st_size
        title = stage_title(label)
        visible = visible_stage(label, capture)
        if visible:
            message(title)
        try:
            stdout, status = self.execute(command, label, offset, started, env, capture)
        except BaseException as error:
            self.record(command, label, started, offset, None, error)
            raise
        record = self.record(command, label, started, offset, status)
        if status or visible:
            outcome = "failed" if status else "completed"
            message(
                f"{title}: {outcome} in {record['elapsed_seconds']:.1f}s (exit {status})", "ERROR" if status else "INFO"
            )
        if status:
            diagnostic = stage_failure(label, status, self.log, offset, record["log_end"])
            raise RuntimeError(f"{diagnostic}\nCommand executable: {command[0]}")
        return stdout

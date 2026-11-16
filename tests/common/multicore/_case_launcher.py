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
"""Framework-free device isolation for multicore system-test workers."""

import fcntl
import os
from contextlib import ExitStack
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from tests.common.distributed_launcher import torchrun_case
from tests.common.multicore._test_env import INHERITED_RANK_VARIABLES

def _worker_environment(device_count: int) -> tuple[dict[str, str], list[int]]:
    """Isolate the assigned device set and strip inherited rank variables."""
    env = os.environ.copy()
    visible_devices = "ASCEND_RT_VISIBLE_DEVICES"
    if env.get("HYPER_MULTICORE_TEST_DEVICE_TYPE", "").strip().lower() == "cpu":
        env.pop(visible_devices, None)
        devices = []
    else:
        allocated = env.get(visible_devices)
        devices = list(map(int, allocated.split(","))) if allocated else list(range(device_count))
        if len(devices) < device_count or min(devices) < 0 or len(set(devices)) != len(devices):
            raise ValueError("invalid or insufficient ASCEND_RT_VISIBLE_DEVICES allocation")
        devices = devices[:device_count]
        env[visible_devices] = ",".join(map(str, devices))

    for variable in INHERITED_RANK_VARIABLES:
        env.pop(variable, None)
    # CI may invoke pytest from tests/ without placing the archive root in its environment.
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    return env, devices


def run_multicore_case(
    file_name: str, case_name: str, num_proc: int = 1, timeout: float = 1200,
    *, num_devices: int | None = None, device_timeout: float | None = None,
) -> None:
    """Reserve physical NPUs before starting the bounded worker process group.

    Device indices follow the inherited ASCEND_RT_VISIBLE_DEVICES allocation.
    Only multicore launchers share these locks; unrelated tests are unaffected.
    Lock waiting shares the case deadline by default. If device_timeout is
    set, the worker gets its full timeout after all devices are acquired.
    """
    if num_proc < 1:
        raise ValueError("num_proc must be positive")
    device_count = num_proc if num_devices is None else num_devices
    if device_count < num_proc:
        raise ValueError("num_devices must cover every worker")
    if timeout is None or timeout <= 0:
        raise ValueError("timeout must be positive and bounded")
    lease_timeout = timeout if device_timeout is None else device_timeout
    if lease_timeout <= 0:
        raise ValueError("device_timeout must be positive")
    env, devices = _worker_environment(device_count)
    command = [sys.executable, "-m", "tests.common.multicore._case_launcher", file_name, case_name, str(num_proc)]
    started = time.monotonic()
    print(f"Multicore {case_name}: waiting for devices={devices}", flush=True)
    with ExitStack() as locks:
        lock_files = []
        # A fixed order prevents deadlock when two allocations partly overlap.
        for device in sorted(devices):
            path = Path(tempfile.gettempdir()) / f"hp_multicore_npu_{os.getuid()}_{device}.lock"
            lock = locks.enter_context(path.open("a", encoding="utf-8"))
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() - started >= lease_timeout:
                        raise TimeoutError(f"Device lease timeout for {case_name}: {devices}") from exc
                    time.sleep(0.2)
            lock_files.append(lock)
        print(
            f"Multicore {case_name}: acquired devices={devices}, waited={time.monotonic() - started:.3f}s", flush=True,
        )
        # The supervisor retains leases if its pytest parent is interrupted.
        with subprocess.Popen(
            command, env=env, start_new_session=True, pass_fds=tuple(lock.fileno() for lock in lock_files),
        ) as process:
            try:
                worker_timeout = timeout
                if device_timeout is None:
                    worker_timeout = max(0.001, timeout - (time.monotonic() - started))
                returncode = process.wait(timeout=worker_timeout)
            finally:
                # Release devices only after workers and any surviving descendants stop.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)


if __name__ == "__main__":
    torchrun_case(sys.argv[1], sys.argv[2], num_proc=int(sys.argv[3]))

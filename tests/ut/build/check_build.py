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
"""Fast source-build checks using temporary data and no framework or device runtime."""

from contextlib import contextmanager, ExitStack, nullcontext, redirect_stderr
import fcntl
import hashlib
from itertools import combinations
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_hyper_multicore_bootstrap", ROOT / "hyper_multicore/_build/_hyper_multicore_bootstrap.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)
BUILD = BOOTSTRAP.load_module("build_controller")
PUBLICATION = BOOTSTRAP.load_module("build_publication")
FRAMEWORK = BOOTSTRAP.load_module("framework_environments")
VENDOR = BOOTSTRAP.load_module("vendor_cache")
WORKSPACE = BOOTSTRAP.load_module("build_workspace")
ARTIFACTS = BOOTSTRAP.load_module("artifacts")
MERGE = BOOTSTRAP.load_module("merge_multicore_vendors")


def write(path, text="fixture"):
    """Create one small fixture without touching checkout outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def git(directory: Path, *args: str) -> str:
    """Run Git only against a check's temporary source repository."""
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


class BuildChecks(unittest.TestCase):
    """Protect dependency requests, cached kernels, output commits and owned paths."""

    def setUp(self):
        """Keep every filesystem effect and environment setting local to this check."""
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(
            tempfile.TemporaryDirectory(prefix="hyper-multicore-build-check-")))
        self.resources.enter_context(patch.dict(os.environ, HYPER_MULTICORE_VENV_PATH=str(self.root / "envs")))

    def test_soc_arguments_group_all_supported_subsets(self):
        """Every nonempty subset selects only its required SHMEM backends."""
        socs = ("ascend910b", "ascend910_93", "ascend950")
        for size in (1, 2, 3):
            for subset in combinations(socs, size):
                with self.subTest(subset=subset):
                    args = BUILD.parse_arguments(["--soc-list", ",".join(subset)])
                    expected = {}
                    for soc in subset:
                        expected.setdefault("arch35" if soc == "ascend950" else "arch22", []).append(soc)
                    self.assertEqual(args.architectures, expected)
        for flags in ([], ["--all"]):
            self.assertEqual(BUILD.parse_arguments(flags).soc_list, ",".join(socs))
            self.assertEqual(BUILD.parse_arguments([*flags, "--soc-list", " ascend950 , ascend910_93 "]).soc_list,
                             "ascend950,ascend910_93")
        for value in ("ascend950,ascend950", "", ",ascend950", "unknown", "ascend 950"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                BUILD.parse_arguments(["--soc-list", value])
            self.assertEqual(error.exception.code, 2)

    def test_native_test_changes_update_identity_and_stop_publication(self):
        """Freeze the contents and membership of the compiled SHMEM test sources."""
        for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
            write(self.root / name)
        sources = self.root / "tests/ut/shmem/cpp"
        for name in ("native_config_test.cpp", "native_device_identity_test.cpp", "native_log_test.cpp"):
            path = write(sources / name, "original")
            guard = PUBLICATION.SourceGuard(self.root)
            before = guard.frozen[0]["source_sha256"]
            guard.check_source()
            path.write_text("changed")
            with self.subTest(source=name):
                self.assertNotEqual(guard.observe()[0]["source_sha256"], before)
                with self.assertRaises(RuntimeError):
                    guard.check_source()
        added = sources / "additional_test.cpp"
        for operation in ("add", "remove"):
            guard = PUBLICATION.SourceGuard(self.root)
            before = guard.frozen[0]["source_sha256"]
            if operation == "add":
                write(added)
            else:
                added.unlink()
            with self.subTest(operation=operation):
                self.assertNotEqual(guard.observe()[0]["source_sha256"], before)
                with self.assertRaises(RuntimeError):
                    guard.check_source()

    def test_invalid_arguments_leave_outputs_and_tools_untouched(self):
        """Each distinct validator must reject bad arguments before cleanup or tools."""
        root = self.root / "repo"
        for name in ("build.sh", "hyper_multicore/build.sh", "hyper_multicore/shmem/build.sh",
                     "scripts/native/cann_version.sh", "hyper_multicore/_native_manifest.py"):
            write(root / name, (ROOT / name).read_text())
        shutil.copytree(ROOT / "hyper_multicore/_build", root / "hyper_multicore/_build",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("dist/previous.whl", "build/native/payload/hyper_multicore/library.so",
                     "build/native/logs/build.log", "build/native/logs/tools.log", "build/native/logs/stages.json",
                     f"build/native/common/{os.uname().machine}/work/input",
                     f"build/native/common/{os.uname().machine}/payload/shmem/lib/previous.so"):
            write(root / name)
        tools = self.root / "tools"
        for name in ("cmake", "gcc", "g++", "bisheng", "git"):
            write(tools / name, '#!/bin/sh\necho called >> "$TOOL_CALLS"\nexit 99\n').chmod(0o755)
        env = {key: value for key, value in os.environ.items() if not key.startswith("HP_")}
        env.update(PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1",
                   PATH=str(tools) + os.pathsep + os.environ["PATH"], TOOL_CALLS=str(self.root / "calls"))
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        for entry, stage in (("build.sh", ""), ("hyper_multicore/build.sh", "common"),
                             ("hyper_multicore/shmem/build.sh", "")):
            for arguments in (["--soc-list", "ascend950,unknown"], ["--jobs", "0"]):
                with self.subTest(entry=entry, arguments=arguments):
                    result = subprocess.run(["bash", str(root / entry), "--clean", *arguments], cwd=root,
                                            env={**env, "HP_NATIVE_STAGE": stage}, capture_output=True, text=True,
                                            check=False)
                    expected = 2 if entry == "build.sh" or arguments[0] == "--jobs" else 10
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual({p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()},
                                     before)
                    self.assertFalse(Path(env["TOOL_CALLS"]).exists())
                    self.assertFalse((root / "build/native/.build.lock").exists())

    def test_backend_handoff_reaches_bindings_and_rejects_mismatch(self):
        """Share Torch once and compile every requested binding against its own SDK."""
        class Commands:
            """Record commands and publish the controlled SDK handoff."""

            def __init__(self, handoff: dict[str, dict[str, object]]) -> None:
                """Keep the SDK handoff and recorded build commands."""
                self.handoff, self.calls = handoff, []
                self.events = []
                self.add_event = self.events.append

            def scope(self, name: str) -> nullcontext[str]:
                """Accept a command scope without changing command recording."""
                return nullcontext(name)

            def __call__(self, command: list[str], **kwargs: Any) -> None:
                """Record the command and emit the handoff during the common build."""
                self.calls.append(command)
                if kwargs.get("label") == "common-resources":
                    write(Path(kwargs["env"]["HP_SHMEM_HANDOFF"]), json.dumps({"architectures": self.handoff}))

        previous_common = None
        for socs, mismatch in (("ascend910b,ascend910_93,ascend950", False),
                               ("ascend910_93", False), ("ascend950", True)):
            args = BUILD.parse_arguments(["--soc-list", socs, "--jobs", "1"])
            handoff = {name: {"install_root": str(self.root / name),
                              "sdk_soc_type": BUILD.ARCHITECTURES[name]["sdk_soc_type"],
                              "soc_list": selected, "toolchain_key": "fixture"}
                       for name, selected in args.architectures.items()}
            if mismatch:
                handoff["arch35"]["sdk_soc_type"] = "Ascend910B"
            commands = Commands(handoff)
            run = SimpleNamespace(commands=commands, events=[], isolated=False)
            identity = {"python_abi": "cp310", "host_arch": "aarch64", "torch_version": "2.12.0",
                        "pta_version": "2.12.0.post3", "torch_cxx11_abi": True,
                        "build_fingerprint": "f" * 64, "source_sha256": "f" * 64, "build_commit": "fixture"}
            with self.subTest(socs=socs, mismatch=mismatch), \
                    patch.multiple(BUILD, ROOT=self.root, NATIVE=self.root / "native", _source_identity=lambda _: {},
                                   adapter_environment=lambda *a, **kw: {}, validate_elf=lambda *a: None,
                                   load_backend=lambda *a: None), \
                    patch.object(FRAMEWORK, "cann_tool_environment", return_value={}):
                if mismatch:
                    with self.assertRaises(ValueError):
                        BUILD.build_common(args, run, Path(sys.executable))
                    continue
                common, sdks = BUILD.build_common(args, run, Path(sys.executable))
                self.assertEqual(common, self.root / "native/common" / os.uname().machine / "payload")
                if previous_common is not None:
                    self.assertEqual(common, previous_common)
                    self.assertFalse((common / "stale-selection").exists())
                write(common / "stale-selection")
                previous_common = common
                _, record = BUILD.build_framework(Path(sys.executable), identity, common, sdks, args, run)
                configured = [call for call in commands.calls if call[:2] == ["cmake", "-S"]]
                self.assertEqual(len(configured), 1 + len(sdks))
                self.assertEqual(set(record["shmem_so"]), set(sdks))
                self.assertEqual(record["id"], "pt2.12.0-pta2.12.0.post3-ffffffffffff")
                self.assertIn(record["id"], record["torch_so"])
                for backend in sdks:
                    flag = "-DHP_SHMEM_SOC_TYPE=" + sdks[backend]["sdk_soc_type"]
                    binding = next(call for call in configured if flag in call)
                    self.assertIn(f"-DSHMEM_OPS_LIB={common}/shmem/lib/{backend}/libhyper_multicore_shmem_ops.so",
                                  binding)

    def test_activation_checks_backends_without_exporting_sdk_search_paths(self):
        """Activation requires all packaged SDKs while only the vendor enters global lookup."""
        payload = self.root / "payload"
        script = write(payload / "lib/set_env.bash", (ROOT / "hyper_multicore/set_env.bash").read_text())
        op_api = write(payload / "lib/vendors/hyper_multicore_nn/op_api/lib/libcust_opapi.so")
        backends = {name: {"shmem_root": f"shmem/lib/{name}"} for name in ("arch22", "arch35")}
        for backend in backends.values():
            for name in ("libhyper_multicore_shmem_ops.so", "shmem/libhyper_multicore_shmem.so"):
                write(payload / backend["shmem_root"] / name)
        write(payload / "adapters/manifest.json", json.dumps({"schema_version": 2, "architectures": backends}))
        command = ["bash", "-euc", 'source "$1"; printf "%s\\n" "$LD_LIBRARY_PATH"', "activation", str(script)]
        environment = {**os.environ, "LD_LIBRARY_PATH": "/existing"}
        result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), str(op_api.parent) + ":/existing")
        (payload / "shmem/lib/arch35/shmem/libhyper_multicore_shmem.so").unlink()
        result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HP-NATIVE-PAYLOAD-MISSING", result.stderr)

    def _sdk_fixture(self, directory):
        """Use real locked Git objects and recording tools in an isolated SDK project."""
        project, shmem, source = (directory / name for name in ("project", "shmem", "custom-json"))
        for repository in (shmem, source):
            repository.mkdir(parents=True)
            git(repository, "init", "-q")
            git(repository, "config", "user.name", "SDK Test")
            git(repository, "config", "user.email", "sdk-test@example.invalid")
            git(repository, "config", "commit.gpgsign", "false")
        write(shmem / "CMakeLists.txt", "project(test)")
        write(shmem / "src/input.cpp")
        git(shmem, "add", ".")
        git(shmem, "commit", "-qm", "SDK")
        header = write(source / "single_include/nlohmann/json.hpp", "locked A\n")
        git(source, "add", ".")
        git(source, "commit", "-qm", "A")
        pinned = git(source, "rev-parse", "HEAD")
        header.write_text("current B\n")
        git(source, "commit", "-qam", "B")
        dependencies = {"shmem": {"source_dir": str(shmem), "commit": git(shmem, "rev-parse", "HEAD")},
                        "nlohmann_json": {"source_dir": str(source), "commit": pinned}}
        lock = write(project / "hyper_multicore/_build/dependencies.lock.json",
                     json.dumps({"components": {"multicore": dependencies}}))
        write(project / "hyper_multicore/_build/prepare_dependencies.py",
              "import json, os, sys\n"
              "name = sys.argv[sys.argv.index('--dependency') + 1]\n"
              "if name == 'nlohmann_json' and os.environ.get('FAIL_JSON'): raise SystemExit(7)\n"
              f"print(json.dumps({dependencies!r}))\n")
        for name in ("hyper_multicore/cmake/hardening.cmake",
                     "hyper_multicore/shmem/cmake/shmem_wrapper/CMakeLists.txt"):
            write(project / name)
        for name in ("hyper_multicore/shmem/_build/shmem_sdk.sh", "hyper_multicore/_build/soc_selection.py"):
            write(project / name, (ROOT / name).read_text())
        for name in ("__init__.py", "_hyper_multicore_bootstrap.py", "build_workspace.py",
                     "build_utils.py", "command_runner.py"):
            relative = "hyper_multicore/_build/" + name
            write(project / relative, (ROOT / relative).read_text())
        tools = directory / "tools"
        for name in ("bisheng", "cmake", "gcc", "g++"):
            write(tools / name, '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TOOL_CALLS"\n').chmod(0o755)
        cann = directory / "cann"
        cann.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith("HP_")}
        env.update(PATH=str(tools) + os.pathsep + os.environ["PATH"], PYTHON_BIN=sys.executable,
                   ASCEND_HOME_PATH=str(cann), TOOL_CALLS=str(directory / "calls"))
        return SimpleNamespace(project=project, source=source, lock=lock, env=env)

    def _prepare_sdk(self, fixture, soc):
        """Execute SDK preparation without invoking the compiler or CMake."""
        result = subprocess.run(
            ["bash", "-euc", 'source "$1"; hp_prepare_shmem_sdk "$2" "$3" 1; '
             'printf "BACKEND=%s\\nKEY=%s\\n" "$HP_SHMEM_SOC_TYPE" "$HP_SHMEM_TOOLCHAIN_KEY"', "sdk-test",
             str(fixture.project / "hyper_multicore/shmem/_build/shmem_sdk.sh"), str(fixture.project), soc],
            env=fixture.env, capture_output=True, text=True, check=False)
        return result

    def test_sdk_exports_locked_json_without_changing_source(self):
        """The 950 SDK must use the returned locked object rather than source HEAD."""
        fixture = self._sdk_fixture(self.root)
        head = git(fixture.source, "rev-parse", "HEAD")
        result = self._prepare_sdk(fixture, "ascend950")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        workspace = fixture.project / "build/native/sdk" / os.uname().machine / "arch35"
        header = workspace / "shmem-source/3rdparty/json/single_include/nlohmann/json.hpp"
        self.assertEqual(header.read_text(), "locked A\n")
        self.assertEqual(git(fixture.source, "rev-parse", "HEAD"), head)
        self.assertEqual(git(fixture.source, "status", "--porcelain"), "")
        self.assertEqual((fixture.source / "single_include/nlohmann/json.hpp").read_text(), "current B\n")
        self.assertIn("BACKEND=Ascend950", result.stdout)
        self.assertIn("-DSOC_TYPE=Ascend950", Path(fixture.env["TOOL_CALLS"]).read_text(encoding="utf-8"))
        stale = write(workspace / "previous-build/input")
        result = self._prepare_sdk(fixture, "ascend950")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(stale.exists())
        self.assertTrue((workspace / ".workspace.json").is_file())

    def test_sdk_workspace_rejects_unknown_and_redirected_data(self):
        """SDK preparation must reject unsafe roots before any compiler is called."""
        fixture = self._sdk_fixture(self.root)
        outside = write(self.root / "outside/toolchain/user.txt", "keep")
        unknown = write(self.root / "unknown/manual.txt", "keep")
        foreign = self.root / "foreign"
        WORKSPACE.claim_workspace(foreign, self.root / "other-checkout")
        foreign_data = write(foreign / "manual.txt", "keep")
        link = self.root / "redirected"
        link.symlink_to(outside.parent.parent, target_is_directory=True)
        for workspace in (unknown.parent, foreign, link, link / "nested/arch35"):
            with self.subTest(workspace=workspace):
                fixture.env["HP_SHMEM_WORK_ROOT"] = str(workspace)
                result = self._prepare_sdk(fixture, "ascend950")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SHMEM_WORKSPACE_INVALID", result.stdout)
                self.assertEqual(outside.read_text(), "keep")
                self.assertEqual(unknown.read_text(), "keep")
                self.assertEqual(foreign_data.read_text(), "keep")
                self.assertFalse(Path(fixture.env["TOOL_CALLS"]).exists())

    def test_shmem_stage_builds_only_sdk_and_ops_without_torch(self):
        """The SDK/ops entry must run without framework packages or binding compilation."""
        fixture = self._sdk_fixture(self.root)
        script = "hyper_multicore/shmem/build.sh"
        write(fixture.project / script, (ROOT / script).read_text())
        tools = self.root / "tools"
        python = write(tools / "without-framework", f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
        python.chmod(0o755)
        cmake = write(tools / "cmake", '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CMAKE_CALLS"], "a") as output:
    output.write(json.dumps(args) + "\\n")
if "-S" in args:
    work = Path(args[args.index("-B") + 1]); work.mkdir(parents=True, exist_ok=True)
    options = dict(a[2:].split("=", 1) for a in args if a.startswith("-D"))
    (work / "fixture.json").write_text(json.dumps(options))
elif "--install" in args:
    work = Path(args[1]); options = json.loads((work / "fixture.json").read_text())
    root = Path(args[args.index("--prefix") + 1]) if "--prefix" in args else Path(options["CMAKE_INSTALL_PREFIX"])
    if "HP_SHMEM_RESOURCE_ARCH" in options:
        root = root / "lib" / options["HP_SHMEM_RESOURCE_ARCH"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "libhyper_multicore_shmem_ops.so").write_text("fixture")
        root = root / "shmem"
    root = root / "lib" if "--prefix" in args else root
    root.mkdir(parents=True, exist_ok=True)
    for name in ("libhyper_multicore_shmem.so", "libhyper_multicore_shmem_utils.so",
                 "aclshmem_bootstrap_uid.so", "aclshmem_bootstrap_config_store.so"):
        (root / name).write_text("fixture")
''')
        cmake.chmod(0o755)
        readelf = write(tools / "readelf",
                        '#!/bin/bash\necho " (SONAME) [$(basename "${!#}")]"\necho " 1 hpshmem_log"\n')
        readelf.chmod(0o755)
        common = fixture.project / "build/native/common" / os.uname().machine
        component = common / "payload"
        calls = self.root / "cmake-calls"
        result = subprocess.run(["bash", str(fixture.project / script), "--soc-list", "ascend950"],
                                env={**fixture.env, "PYTHON_BIN": str(python),
                                     "CMAKE_CALLS": str(calls)}, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        configurations = [json.loads(line) for line in calls.read_text().splitlines() if '"-S"' in line]
        self.assertEqual({args[args.index("-S") + 1] for args in configurations}, {
            str(fixture.project / "hyper_multicore/shmem/cmake/shmem_wrapper"),
            str(fixture.project / "hyper_multicore/shmem/ccsrc/ops")})
        self.assertTrue((component / "shmem/lib/arch35/libhyper_multicore_shmem_ops.so").is_file())
        handoff = json.loads((common / "sdk.json").read_text())["architectures"]["arch35"]
        sdk = fixture.project / "build/native/sdk" / os.uname().machine / "arch35"
        self.assertEqual(Path(handoff["install_root"]), sdk / "shmem-install")
        self.assertTrue((common / ".workspace.json").is_file())

    def test_shmem_cancellation_reaps_children_before_unlock(self):
        """Cancelling the SDK entry must stop its compiler before releasing the shared lock."""
        fixture = self._sdk_fixture(self.root)
        script = "hyper_multicore/shmem/build.sh"
        write(fixture.project / script, (ROOT / script).read_text())
        pid_file = self.root / "compiler.pid"
        write(self.root / "tools/cmake", '''#!/usr/bin/env python3
import os, time
from pathlib import Path
pid = Path(os.environ["COMPILER_PID"])
temporary = pid.with_suffix(".tmp")
temporary.write_text(str(os.getpid()))
temporary.replace(pid)
time.sleep(30)
''').chmod(0o755)
        with (self.root / "cancel.log").open("w", encoding="utf-8") as output, subprocess.Popen(
            ["bash", str(fixture.project / script), "--soc-list", "ascend950"],
            env={**fixture.env, "COMPILER_PID": str(pid_file)},
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        ) as process:
            compiler_group = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_file.exists(), "SDK build did not reach the compiler")
                compiler_pid = int(pid_file.read_text())
                compiler_group = os.getpgid(compiler_pid)
                with (fixture.project / "build/native/.build.lock").open("a+", encoding="utf-8") as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    process.terminate()
                    self.assertEqual(process.wait(timeout=5), 130)
                    stat = Path(f"/proc/{compiler_pid}/stat")
                    try:
                        state = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
                    except FileNotFoundError:
                        state = None
                    self.assertIn(state, (None, "Z"))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                for group in {process.pid, compiler_group} - {None}:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)

    def test_sdk_dependency_failure_stops_before_compilation(self):
        """Both dependency preparation and locked-header export failures stop SDK builds."""
        for failure in ("prepare", "missing-header"):
            with self.subTest(failure=failure):
                fixture = self._sdk_fixture(self.root / failure)
                if failure == "prepare":
                    fixture.env["FAIL_JSON"] = "1"
                else:
                    helper = fixture.project / "hyper_multicore/_build/prepare_dependencies.py"
                    pinned = git(fixture.source, "rev-parse", "HEAD~")
                    git(fixture.source, "rm", "single_include/nlohmann/json.hpp")
                    git(fixture.source, "commit", "-qm", "no header")
                    helper.write_text(helper.read_text().replace(pinned, git(fixture.source, "rev-parse", "HEAD")))
                result = self._prepare_sdk(fixture, "ascend950")
                self.assertEqual(result.returncode, 6, result.stdout + result.stderr)
                calls = Path(fixture.env["TOOL_CALLS"]).read_text(encoding="utf-8").splitlines()
                self.assertFalse(any(call.startswith(("-S ", "--build ", "--install ")) for call in calls))

    def test_sdk_json_lock_changes_only_950_build_identity(self):
        """Only SDKs consuming JSON may depend on its lock record."""
        for soc, backend in (("ascend950", "Ascend950"), ("ascend910b", "Ascend910B")):
            with self.subTest(soc=soc):
                fixture = self._sdk_fixture(self.root / soc)
                first = self._prepare_sdk(fixture, soc)
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                record = json.loads(fixture.lock.read_text())
                record["components"]["multicore"]["nlohmann_json"]["version"] = "new identity"
                fixture.lock.write_text(json.dumps(record))
                if soc == "ascend910b":
                    fixture.env["FAIL_JSON"] = "1"
                second = self._prepare_sdk(fixture, soc)
                self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
                first_key = next(line for line in first.stdout.splitlines() if line.startswith("KEY="))
                second_key = next(line for line in second.stdout.splitlines() if line.startswith("KEY="))
                self.assertEqual(first_key != second_key, soc == "ascend950")
                self.assertIn(f"BACKEND={backend}", second.stdout)
                self.assertIn(f"-DSOC_TYPE={backend}", Path(fixture.env["TOOL_CALLS"]).read_text(encoding="utf-8"))

    def vendor_fixture(self, compile_soc: str, series: list[str]) -> Path:
        """Build the smallest complete six-operator vendor with shared host fixtures."""
        vendor = self.root / compile_soc / "hyper_multicore_nn"
        for name in ("op_api/lib/libcust_opapi.so", "op_impl/ai_core/tbe/op_tiling/liboptiling.so",
                     "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
                     "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"):
            write(vendor / name)
        tbe = vendor / "op_impl/ai_core/tbe"
        for soc in series:
            write(tbe / f"config/{soc}/aic-{soc}-ops-info.json", "{}")
        index = {}
        for operator in ("hyper_mega_moe", "hyper_mega_moe_grad", "hyper_mega_mhc", "hyper_mega_mhc_grad",
                         "hyper_mega_gate_route", "hyper_mega_gate_route_grad"):
            relative = f"{compile_soc}/{operator}/kernel"
            write(tbe / f"kernel/{relative}.o")
            write(tbe / f"kernel/{relative}.json", "{}")
            write(tbe / f"kernel/config/{compile_soc}/{operator}.json", json.dumps({
                "binList": [{"binInfo": {"jsonFilePath": relative + ".json"}}]}))
            index[operator] = {"binaryList": [{"binPath": relative + ".o", "jsonPath": relative + ".json"}]}
        write(tbe / f"kernel/config/{compile_soc}/binary_info_config.json", json.dumps(index))
        write(vendor / "framework/tensorflow/npu_supported_ops.json", "{}")
        return vendor

    def test_architecture_vendor_shares_kernels_and_preserves_soc_registration(self):
        """Shared binaries retain every selected lookup and reject wrong or missing resources."""
        # Host ELF inspection is covered by real builds; no compiler runs in these source checks.
        # pylint: disable=protected-access
        b = self.vendor_fixture("ascend910b", ["ascend910b", "ascend910_93"])
        a5 = self.vendor_fixture("ascend950", ["ascend950"])
        socs = ["ascend910b", "ascend910_93", "ascend950"]
        output = self.root / "payload/lib/vendors/hyper_multicore_nn"
        identities = {soc: "f" * 64 for soc in ("ascend910b", "ascend950")}
        with patch.object(MERGE, "_host_elf_abi_fingerprint", return_value={}):
            MERGE.merge_vendors([("ascend910b", b), ("ascend950", a5)], output, identities, socs)
            ARTIFACTS._validate_vendor_resources(self.root / "payload", socs, "aarch64")
            kernel = output / "op_impl/ai_core/tbe/kernel"
            self.assertEqual({p.name for p in kernel.iterdir()}, {"arch22", "arch35", "config"})
            left = kernel / "config/ascend910b/hyper_mega_moe.json"
            right = kernel / "config/ascend910_93/hyper_mega_moe.json"
            self.assertEqual(json.loads(left.read_text()), json.loads(right.read_text()))
            original = left.read_text()
            left.write_text(original.replace("arch22/", "arch35/"))
            with self.assertRaises(RuntimeError):
                ARTIFACTS._validate_vendor_resources(self.root / "payload", socs, "aarch64")
            left.write_text(original)
            (kernel / "arch35/hyper_mega_moe/kernel.o").unlink()
            with self.assertRaises(RuntimeError):
                ARTIFACTS._validate_vendor_resources(self.root / "payload", socs, "aarch64")
            c_only = self.root / "c_only/lib/vendors/hyper_multicore_nn"
            MERGE.merge_vendors([("ascend910b", b)], c_only, {"ascend910b": "f" * 64}, ["ascend910_93"])
            ARTIFACTS._validate_vendor_resources(self.root / "c_only", ["ascend910_93"], "aarch64")
            self.assertEqual({p.name for p in (c_only / "op_impl/ai_core/tbe/config").iterdir()}, {"ascend910_93"})
            missing = c_only / "op_impl/ai_core/tbe/kernel/config/ascend910_93/hyper_mega_gate_route_grad.json"
            missing.unlink()
            with self.assertRaises(RuntimeError):
                ARTIFACTS._validate_vendor_resources(self.root / "c_only", ["ascend910_93"], "aarch64")
            write(b / "op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json", '{"different":true}')
            with self.assertRaises(ValueError):
                MERGE.merge_vendors([("ascend910b", b)], c_only, {"ascend910b": "f" * 64}, ["ascend910_93"])

    def test_framework_sources_and_environment_reuse(self):
        """Exact roots use configured mirrors and a successful venv is never reinstalled."""
        target = {"id": "fixture", "torch_version": "1.0.0", "pta_build_version": "1.0.0",
                  "torch_release": "1.0.0", "pta_release": "1.0.0", "build_requirements": ["numpy"],
                  "indexes": {"torch": "https://torch.example/simple", "pta": "https://pta.example/simple",
                              "dependencies": "https://pypi.example/simple",
                              "dependencies_extra": ["https://mirror.example/simple"]}}
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            if "--dest" in command:
                name, version = command[-1].split("==")
                name = name.replace("-", "_")
                destination = Path(command[command.index("--dest") + 1])
                with zipfile.ZipFile(destination / f"{name}-{version}-py3-none-any.whl", "w") as archive:
                    archive.writestr(f"{name}-{version}.dist-info/METADATA",
                                     f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
            if "--report" in command:
                write(Path(command[command.index("--report") + 1]), '{"install": []}')

        def probe(python, expected):
            self.assertEqual(python.parent.parent.parent, self.root / "envs")
            self.assertEqual(expected, {"torch": "1.0.0", "torch-npu": "1.0.0"})
            probes.append(expected)
            return [{"name": name, "version": version} for name, version in expected.items()]

        probes = []
        python = FRAMEWORK.prepare_environment(target, self.root, run, Path(sys.executable), identity_probe=probe)
        installs = [command for command in commands if "--index-url" in command]
        self.assertEqual(len(installs), 3)
        for command in installs:
            self.assertEqual(command[command.index("--extra-index-url") + 1], "https://mirror.example/simple")
        self.assertEqual([command[command.index("--index-url") + 1] for command in installs],
                         [target["indexes"][name] for name in ("torch", "pta", "dependencies")])
        commands.clear()
        installation = json.loads((python.parent.parent / "ready.json").read_text())["installation"]
        self.assertEqual(FRAMEWORK.prepare_environment(
            target, self.root, run, Path(sys.executable), identity_probe=probe), python)
        self.assertFalse(commands)
        renamed = {**target, "id": "renamed"}
        renamed["indexes"] = {**target["indexes"], "dependencies_extra": ["https://new-mirror.example/simple"]}
        self.assertEqual(FRAMEWORK.prepare_environment(
            renamed, self.root, run, Path(sys.executable), identity_probe=probe), python)
        self.assertFalse(commands)
        self.assertEqual(len(probes), 3)
        ready = json.loads((python.parent.parent / "ready.json").read_text())
        self.assertEqual(ready["request"], renamed)
        self.assertEqual(ready["target"], "renamed")
        self.assertEqual(ready["installation"], installation)
        requests = ({**target, "torch_version": "1.0.1", "torch_release": "1.0.1"},
                    {**target, "build_requirements": ["numpy>=2"]},
                    {**target, "indexes": {**target["indexes"], "torch": "https://new-torch.example/simple"}})
        for changed in requests:
            self.assertNotEqual(FRAMEWORK.environment_key(target, Path(sys.executable)),
                                FRAMEWORK.environment_key(changed, Path(sys.executable)))
        with patch.object(FRAMEWORK, "verify_environment", side_effect=RuntimeError("framework changed")):
            with self.assertRaisesRegex(RuntimeError, "framework changed"):
                FRAMEWORK.prepare_environment(target, self.root, run, Path(sys.executable))

    def test_environment_pruning_keeps_only_selected_managed_environments(self):
        """Discard old managed installs while preserving current and unrelated data."""
        environments = self.root / "envs"
        selected = environments / ("current-" + "a" * 20)
        old = environments / ("old-" + "b" * 20)
        pending = environments / ("pending-" + "c" * 20)
        for path, key, marker in ((selected, "a", "ready.json"), (old, "b", "ready.json"),
                                  (pending, "c", "pending.json")):
            write(path / marker, json.dumps({"owner": "hyper-multicore-framework-env",
                                            "environment_key": key * 64, "request": {},
                                            "last_successful_build": 1}))
        python = write(selected / "bin/python")
        unrelated = write(environments / "manual/keep.txt")
        external = write(self.root / "external/keep.txt")
        link = environments / "redirected"
        link.symlink_to(external.parent, target_is_directory=True)
        self.assertEqual(set(FRAMEWORK.prune_environments(self.root, [python])), {str(old), str(pending)})
        self.assertTrue(python.is_file())
        self.assertFalse(old.exists())
        self.assertFalse(pending.exists())
        self.assertEqual(unrelated.read_text(), "fixture")
        self.assertEqual(external.read_text(), "fixture")
        self.assertTrue(link.is_symlink())

    def test_vendor_reuse_rejects_changed_source_and_corrupt_output(self):
        """An intact snapshot is reusable; local header edits and damaged kernels are misses."""
        sdk = self.root / "sdk"
        component = self.root / "hyper_multicore"
        for name in VENDOR.SCRIPTS:
            write(component / name)
        for name in VENDOR.SOURCE_TREES:
            write(component / name / "input.h")
        for name in ("include", "src/device", "lib"):
            write(sdk / name / "input")
        header = write(component / "shmem/ccsrc/data_plane/rma.h", "original")
        cann = self.root / "cann"
        write(cann / "opp/version.info", "Version=9.2.0")
        candidate = self.root / "work/vendor"
        write(candidate / "kernel.o")
        request = self.root / "request.json"
        with patch.dict(os.environ, ASCEND_HOME_PATH=str(cann), HP_VENDOR_ORIGIN='{"build_commit":"original"}'), \
                patch.object(VENDOR, "tool_inputs", return_value={"compiler": "fixture"}):
            VENDOR.prepare(self.root, self.root / "native", "arch22", sdk, request)
            VENDOR.built(request, candidate, time.time())
            VENDOR.publish(self.root, [json.loads(request.read_text(encoding="utf-8"))])
            VENDOR.prepare(self.root, self.root / "native", "arch22", sdk, request)
            hit = json.loads(request.read_text(encoding="utf-8"))
            self.assertTrue(hit["reused"])
            write(header, "changed")
            VENDOR.prepare(self.root, self.root / "native", "arch22", sdk, request)
            self.assertFalse(json.loads(request.read_text(encoding="utf-8"))["reused"])
            write(header, "original")
            write(Path(hit["directory"]) / "vendor/kernel.o", "corrupted")
            VENDOR.prepare(self.root, self.root / "native", "arch22", sdk, request)
            self.assertFalse(json.loads(request.read_text(encoding="utf-8"))["reused"])

    @contextmanager
    def staged_build(self):
        """Run actual publication with only expensive preparation and native workers replaced."""
        candidate = write(self.root / "staging/candidate/new.so").parent
        wheel = write(self.root / "staging/candidate.whl")
        outputs = WORKSPACE.BuildOutputs(self.root / "dist", self.root / "native/payload/hyper_multicore")
        write(outputs.payload / "old.so")
        write(outputs.distribution / "old.whl")
        replacements = {
            "ROOT": self.root, "NATIVE": self.root / "native", "validate_toolchain": lambda: None,
            "prepare_frameworks": lambda *_: ([], [Path(sys.executable)], [{}]),
            "build_common": lambda *_: (self.root / "common", self.root / "sdk"),
            "build_adapters": lambda *_: ([], [{"pta_version": "fixture"}]),
            "assemble_payload": lambda *_: (candidate, {}), "package_wheel": lambda *_: wheel,
        }
        with patch.multiple(BUILD, **replacements), patch.object(PUBLICATION.SourceGuard, "observe", return_value={}):
            yield SimpleNamespace(all=False, tests=False, native_only=False), outputs

    def test_precommit_failure_leaves_no_formal_outputs(self):
        """A payload switch failure also removes the already published wheel."""
        with self.staged_build() as (args, outputs), \
                patch.object(BUILD, "publish_payload", side_effect=OSError("publication interrupted")):
            with self.assertRaisesRegex(OSError, "publication interrupted"), outputs:
                BUILD.build(args, outputs)
            self.assertFalse(outputs.committed)
            self.assertFalse(outputs.payload.exists())
            self.assertFalse(list(outputs.distribution.iterdir()))

    def test_maintenance_failure_retains_committed_outputs(self):
        """Maintenance cannot undo the successfully committed artifact pair."""
        with self.staged_build() as (args, outputs), \
                patch.object(BUILD.VENDOR, "publish", side_effect=OSError("maintenance interrupted")):
            with outputs:
                BUILD.build(args, outputs)
            self.assertTrue(outputs.committed)
            self.assertTrue((outputs.payload / "new.so").is_file())
            self.assertEqual([path.name for path in outputs.distribution.iterdir()], ["candidate.whl"])

    def test_workspace_reset_preserves_unknown_and_redirected_data(self):
        """Only owned real workspaces may be erased for a rebuild."""
        workspace = self.root / "variants/target"
        manual = write(workspace / "manual.txt")
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            WORKSPACE.reset_workspace(workspace, self.root)
        self.assertTrue(manual.is_file())
        shutil.rmtree(workspace)
        outside = write(self.root / "outside/keep.txt")
        workspace.symlink_to(outside.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            WORKSPACE.reset_workspace(workspace, self.root)
        self.assertTrue(outside.is_file())
        ancestor = self.root / "redirected"
        ancestor.symlink_to(outside.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            WORKSPACE.reset_workspace(ancestor / "new/target", self.root)
        self.assertFalse((outside.parent / "new").exists())
        self.assertTrue(outside.is_file())

    def test_wheel_integrity_rejects_changed_and_missing_payload(self):
        """Payload membership and hashes reject invalid staged wheel contents."""
        content = b"native fixture"
        payload = self.root / "payload"
        write(payload / "build_info.json", "{}")
        write(payload / "adapters/manifest.json", "{}")
        relative = "adapters/fixture/libhyper_multicore_torch.so"
        info = {"payload_sha256": {relative: hashlib.sha256(content).hexdigest(),
                                  "adapters/manifest.json": hashlib.sha256(b"{}").hexdigest()}}
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("hyper_multicore/" + relative, content)
            archive.writestr("hyper_multicore/build_info.json", "{}")
            archive.writestr("hyper_multicore/adapters/manifest.json", "{}")
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            PUBLICATION.validate_wheel_payload(archive, members, payload, info)
            info["payload_sha256"][relative] = "0" * 64
            with self.assertRaisesRegex(ValueError, "payload mismatch"):
                PUBLICATION.validate_wheel_payload(archive, members, payload, info)
            with self.assertRaisesRegex(ValueError, "membership mismatch"):
                PUBLICATION.validate_wheel_payload(archive, [], payload, info)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)

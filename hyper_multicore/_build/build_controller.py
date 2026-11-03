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
"""Build common native resources once and package explicit framework adapter pairs."""

from __future__ import annotations
import argparse
import re
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import signal
import sys
import time
from . import artifacts as HELPERS, framework_environments as FRAMEWORK
from . import command_runner as CONSOLE, build_workspace as WORKSPACE
from . import preflight_cache as PREFLIGHT
from ._bootstrap import load_module
from .command_runner import message
from .build_utils import atomic_json as write_json, measured
from .build_state import BuildContext, BuildResult, PreparedFramework
from .native_inputs import compiler_identity, tool_inputs
from .native_cache import CacheTransaction, cache_valid, hashes, CACHE_SCHEMA_VERSION
from .build_publication import publish_wheel, publish_payload, validate_elf, validate_wheel

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "build/native"

MANIFEST = load_module("native_manifest")

COMMON_DIRECTORIES = (
    "hyper_multicore/ops",
    "hyper_multicore/cmake/vendor",
    "hyper_multicore/shmem/ccsrc/ops",
    "hyper_multicore/shmem/ccsrc/data_plane",
    "hyper_multicore/shmem/cmake",
)
COMMON_EXTRA_INPUTS = (
    "hyper_multicore/shmem/ccsrc/cann/device.h",
    "hyper_multicore/shmem/ccsrc/cann/signal.h",
    "hyper_multicore/build.sh",
    "hyper_multicore/set_env.bash",
    "hyper_multicore/shmem/build.sh",
    "hyper_multicore/shmem/_build/shmem_sdk.sh",
    "hyper_multicore/_build/assemble_multicore_source.py",
    "hyper_multicore/_build/merge_multicore_vendors.py",
    "hyper_multicore/_build/prepare_dependencies.py",
    "hyper_multicore/_build/dependencies.lock.json",
    "hyper_multicore/cmake/hardening.cmake",
    "scripts/native/cann_version.sh",
)


def common_source_fingerprint() -> str:
    """Include every consumed common source, including the packaged activation script."""
    return fingerprint_files(COMMON_DIRECTORIES, COMMON_EXTRA_INPUTS)


def fingerprint_files(directories, extra=()) -> str:
    """Hash only consumed inputs rather than the entire framework matrix/source tree."""
    paths = set()
    for relative in directories:
        path = ROOT / relative
        paths.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update(ROOT / p for p in extra)
    return FRAMEWORK.digest_json({p.relative_to(ROOT).as_posix(): MANIFEST.file_sha256(p) for p in sorted(paths)})


def probe_identity(python: Path, socs: str, run: BuildContext, *, source=None, expected=None) -> dict:
    """Collect identity, package checks and development paths in one target process."""
    code = (
        "import json,sys; from pathlib import Path; sys.path.insert(0,str(Path(sys.argv[1]).parent)); "
        "from _hyper_multicore_bootstrap import load_module; h=load_module('artifacts'); timings={}; "
        "info=h.build_identity(Path(sys.argv[2]),sys.argv[3],source=json.loads(sys.argv[4]),"
        "timings=timings,expected_versions=json.loads(sys.argv[5])); "
        "print(json.dumps({'identity':info,'timings':timings,'paths':h.probe_details()}))"
    )
    result = run.commands(
        [
            str(python),
            "-c",
            code,
            str(ROOT / "hyper_multicore/_build/artifacts.py"),
            str(ROOT),
            socs,
            json.dumps(source),
            json.dumps(expected or {}),
        ],
        label=f"probe-{python.parent.parent.name}",
        env=FRAMEWORK.isolated_environment(python),
        capture=True,
    )
    data = json.loads(result)
    run.frameworks[str(python)] = PreparedFramework(python, data["identity"], data["paths"])
    run.events[-1]["identity_timings"] = data["timings"]
    write_json(run.directory / "stages.json", run.events)
    identity = data["identity"]
    identity["torch_release"] = MANIFEST.release(identity["torch_version"])
    identity["pta_release"] = MANIFEST.release(identity["pta_version"])
    versions = [
        line.partition("=")[2].strip()
        for line in identity["cann_version_info"].splitlines()
        if line.partition("=")[0].strip() == "Version"
    ]
    if versions != ["9.2.0"]:
        raise ValueError(f"Builds require CANN 9.2.0, found {versions}")
    identity["cann_version"] = versions[0]
    return identity


def adapter_environment(python: Path) -> dict:
    """Allow newer framework compilers without changing the CANN vendor toolchain."""
    env = FRAMEWORK.isolated_environment(python)
    if "HYPER_MULTICORE_FRAMEWORK_CXX" in env:
        env["CXX"] = env["HYPER_MULTICORE_FRAMEWORK_CXX"]
    return env


def compiler_inputs(env: dict) -> dict:
    """Include effective compile, header, library and tool-Python search settings in task keys."""
    return {
        key: env.get(key)
        for key in (
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "CMAKE_BUILD_TYPE",
            "CMAKE_PREFIX_PATH",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
        )
    }


def preflight_fingerprint(identity: dict, python: Path, compiler: dict, env: dict) -> str:
    """Reuse interface probes when their actual inputs, rather than unrelated commits, match."""
    return FRAMEWORK.digest_json(
        {
            "framework": {
                k: identity[k]
                for k in (
                    "torch_version",
                    "pta_version",
                    "torch_cxx11_abi",
                    "python_abi",
                    "host_arch",
                    "torch_headers_sha256",
                    "pta_headers_sha256",
                    "torch_libraries_sha256",
                    "libtorch_npu_sha256",
                    "cann_root",
                    "cann_version_sha256",
                )
            },
            "source": fingerprint_files(
                ("hyper_multicore/cmake/framework_probe",),
                ("hyper_multicore/cmake/torch_compat.cmake", "hyper_multicore/torch/csrc/cached_op_api.h"),
            ),
            "python": str(python),
            "compiler": compiler,
            "flags": {
                k: env.get(k)
                for k in (
                    "CXX",
                    "CXXFLAGS",
                    "CPPFLAGS",
                    "LDFLAGS",
                    "CMAKE_PREFIX_PATH",
                    "CPATH",
                    "CPLUS_INCLUDE_PATH",
                    "LIBRARY_PATH",
                    "LD_LIBRARY_PATH",
                )
            },
        }
    )


def cmake_pair(
    python: Path,
    identity: dict,
    common: Path,
    sdk: Path,
    key: str,
    args,
    run: BuildContext,
    transaction=None,
    *,
    inputs=None,
    probe_inputs=None,
) -> tuple[Path, dict]:
    """Guard both standalone stage calls and the encompassing build transaction."""
    if transaction is not None:
        return _cmake_pair(python, identity, common, sdk, key, args, run, transaction, inputs, probe_inputs)
    with CacheTransaction() as guard:
        return _cmake_pair(python, identity, common, sdk, key, args, run, guard, inputs, probe_inputs)


def _cmake_pair(
    python: Path,
    identity: dict,
    common: Path,
    sdk: Path,
    key: str,
    args,
    run: BuildContext,
    transaction: CacheTransaction,
    inputs,
    probe_inputs,
) -> tuple[Path, dict]:
    """Build one binding/runtime and one unified adapter with exactly the target framework."""
    directory = NATIVE / "variants" / WORKSPACE.framework_slot(identity)
    WORKSPACE.check_workspace(directory, ROOT)
    if args.clean and directory.exists():
        shutil.rmtree(directory)
    identifier = "pt" + identity["pta_version"].replace("+", ".") + "-" + key[:12]
    paths = {
        "torch_so": f"adapters/{identifier}/libhyper_multicore_torch.so",
        "shmem_so": f"adapters/{identifier}/hyper_multicore_shmem_torch.so",
    }
    transaction.scopes[directory] = run.native_inputs.scope({"native": inputs, "probe": probe_inputs})
    with measured(run.events, "adapter-output-validation"):
        cached = cache_valid(directory, key)
    run.events.append({"stage": identifier, "cache_hit": cached, "fingerprint": key})
    if cached:
        previous = json.loads((directory / "identity.json").read_text())
        if previous.get("task_fingerprint") != key or previous.get("id") != identifier:
            raise ValueError(f"Changed cached adapter identity: {directory}")
        report_native_cache(f"adapter {identity['pta_version']}", True)
        return directory / "payload", {
            **identity,
            **paths,
            "id": identifier,
            "task_fingerprint": key,
            "native_build_identity": previous["native_build_identity"],
        }
    report_native_cache(f"adapter {identity['pta_version']}", False)
    transaction.prepare(directory)
    payload = directory / "payload"
    payload.mkdir(parents=True, exist_ok=True)
    env = adapter_environment(python)
    env.update(SHMEM_HOME_PATH=str(sdk), SHMEM_SOURCE_DIR=str(sdk / "shmem"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(common / "shmem/lib"), str(common / "shmem/lib/shmem"), env.get("LD_LIBRARY_PATH", "")]
    )
    for kind, source, extra in (
        (
            "shmem",
            "hyper_multicore/shmem/ccsrc",
            [f"-DSHMEM_OPS_LIB={common / 'shmem/lib/libhyper_multicore_shmem_ops.so'}"],
        ),
        ("torch", "hyper_multicore/torch", [f"-DHP_MULTICORE_VENDOR_ROOT={common / 'lib/vendors/hyper_multicore_nn'}"]),
    ):
        work = directory / kind
        run.commands(
            [
                "cmake",
                "-S",
                str(ROOT / source),
                "-B",
                str(work),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DPython3_EXECUTABLE={python}",
                f"-DCMAKE_INSTALL_PREFIX={payload}",
                *extra,
            ],
            label=f"configure-{identifier}-{kind}",
            env=env,
        )
        run.commands(
            ["cmake", "--build", str(work), "--parallel", str(args.jobs), "--verbose"],
            label=f"compile-{identifier}-{kind}",
            env=env,
        )
        if kind == "shmem":
            run.commands(
                ["ctest", "--test-dir", str(work), "--output-on-failure"], label=f"ctest-{identifier}", env=env
            )
        run.commands(["cmake", "--install", str(work)], label=f"install-{identifier}-{kind}", env=env)
    for name in ("libhyper_multicore_torch.so", "hyper_multicore_shmem_torch.so"):
        validate_elf(payload / name, identity["host_arch"], run.commands, env)
    record = {
        **identity,
        **paths,
        "id": identifier,
        "task_fingerprint": key,
        "native_build_identity": {k: identity[k] for k in ("build_commit", "source_sha256", "build_fingerprint")},
    }
    write_json(directory / "identity.json", record)
    completion = {"fingerprint": key, "hashes": hashes(payload), "inputs": inputs}
    transaction.complete(directory, completion)
    return payload, record


def validate_toolchain() -> None:
    """Reject unavailable common build inputs before preparing any framework environment."""
    cann = os.environ.get("ASCEND_HOME_PATH", "")
    if not cann:
        raise ValueError("ASCEND_HOME_PATH is missing; source the CANN 9.2.0 set_env.sh before building")
    version_file = Path(cann) / "opp/version.info"
    versions = [
        line.partition("=")[2].strip()
        for line in version_file.read_text().splitlines()
        if line.partition("=")[0].strip() == "Version"
    ]
    if versions != ["9.2.0"]:
        raise ValueError(f"Builds require CANN 9.2.0, found {versions}")
    required = ("cmake", "ctest", "gcc", "g++", "make", "git", "bisheng", "asc_opc", "readelf", "ldd", "ld")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise ValueError(f"Missing native build tools: {missing}")


def build(args, publication) -> Path:
    """Validate staged work, publish, then commit once before optional maintenance."""
    run = BuildContext(ROOT, NATIVE)
    with CacheTransaction() as transaction:
        result = _build(args, run, transaction)
    publication.committed = True
    run.events.extend(transaction.metrics.values())
    run.events.extend(run.native_inputs.metrics.values())
    finish_maintenance(args, run, result)
    return result.payload


def _build(args, run: BuildContext, transaction: CacheTransaction) -> BuildResult:
    """Coordinate task phases without duplicating cache, validation or publication policy."""
    started = time.monotonic()
    validate_toolchain()
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    transaction.input_checks.append(run.native_inputs.validate)
    with measured(run.events, "framework-preparation"):
        targets, pythons, identities, host_cxx = prepare_frameworks(args, run, transaction)
    with measured(run.events, "common-input-collection"):
        common_key, common_env = common_inputs(args, run, pythons[0], identities[0])
    common, sdk = build_common(args, run, pythons[0], common_key, common_env, transaction)
    outputs, records = build_adapters(args, run, pythons, identities, common_key, common, sdk, transaction)
    with measured(run.events, "payload-assembly-validation"):
        payload, info = assemble_payload(args, run_id, identities, outputs, records, common, common_key)
    wheel = None if args.native_only else package_wheel(pythons[0], run, payload, info, transaction)
    transaction.check()
    if wheel is not None:
        destination = ROOT / "dist" / wheel.name
        publish_wheel(wheel, destination)
        message(f"Wheel: {destination}")
        message(f"SHA256: {MANIFEST.file_sha256(destination)}")
    selected = publish_build_payload(payload, run_id, transaction)
    transaction.check_source()
    message(f"Published payload: {selected}; PTA adapters: {', '.join(r['pta_version'] for r in records)}")
    return BuildResult(selected, pythons, {t["pta_release"] for t in targets}, len(records), started)


def finish_maintenance(args, run: BuildContext, result: BuildResult):
    """The artifact transaction committed; reclamation can no longer invalidate its result."""
    pythons, branches, pairs, started = result.pythons, result.branches, result.pairs, result.started
    maintenance_started = time.monotonic()
    try:
        run.diagnostics.publish()
        if args.all:
            removed = FRAMEWORK.prune_environments(NATIVE, pythons, branches)
            run.events.append({"stage": "prune-framework-environments", "removed": removed})
        PREFLIGHT.finish(NATIVE, ROOT, "all" if args.all else "current", run.events)
        WORKSPACE.clear_temporary_outputs(NATIVE)
    except (Exception, KeyboardInterrupt) as error:
        detail = str(error) or type(error).__name__
        run.events.append({"stage": "build-maintenance", "status": "warning", "error": detail})
        message(f"Build committed; maintenance stopped: {detail}. The new artifact is valid and retained.", "WARN")
    run.events.append({"stage": "maintenance", "elapsed_seconds": time.monotonic() - maintenance_started})
    try:
        write_json(run.directory / "stages.json", run.events)
    except (Exception, KeyboardInterrupt) as error:
        message(f"Build committed; unable to finish maintenance records: {error}", "WARN")
    build_progress_summary(args, run, pairs, time.monotonic() - started)


def prepare_frameworks(args, run: BuildContext, transaction=None):
    """Resolve the target interpreters and validate their actual identities and interfaces."""
    targets = FRAMEWORK.load_targets(ROOT / "hyper_multicore/_build/frameworks.lock.json")
    base = Path(sys.executable).absolute()
    mode = f"full framework matrix: {len(targets)} targets" if args.all else "locally installed framework"
    message(
        f"Building {mode}; Python cp{sys.version_info.major}{sys.version_info.minor}, "
        f"{platform.machine()}, SoC {args.soc_list}, jobs {args.jobs}"
    )
    started = time.monotonic()
    source = HELPERS._source_identity(ROOT)
    run.events.append({"stage": "shared-source-identity", "elapsed_seconds": time.monotonic() - started})
    if args.all:
        for target in targets:
            FRAMEWORK.validate_target(target)
        pythons, identities = prepare_matrix_environments(targets, run, base, args.soc_list, source)
    else:
        message(f"Using local Python: {base}; no framework environment creation or installation")
        pythons = [base]
        message("Checking framework build identity, headers and libraries")
        identities = [probe_identity(base, args.soc_list, run, source=source)]
    attach_framework_environments(args, pythons, identities)
    host_cxx = compiler_identity(
        os.environ.get("HYPER_MULTICORE_FRAMEWORK_CXX", os.environ.get("CXX", "c++")),
        run.commands,
        "selected-framework-cxx-version",
    )
    for identity in identities:
        identity["host_cxx"] = host_cxx
    for identity in identities:
        if not any(
            t["torch_release"] == identity["torch_release"] and t["pta_release"] == identity["pta_release"]
            for t in targets
        ):
            raise ValueError(f"No supported build branch: {identity['torch_version']}/{identity['pta_version']}")
    if args.all:
        for target, identity in zip(targets, identities):
            if (
                identity["torch_version"] != target["torch_version"]
                or identity["pta_version"] != target["pta_build_version"]
            ):
                raise ValueError(f"Target framework identity changed: {target['id']}")
    preflight_frameworks(args, run, pythons, identities, host_cxx, transaction)
    return targets, pythons, identities, host_cxx


def prepare_matrix_environments(targets, run, base, socs, source):
    """Prepare and verify each target once, including its actual installed package versions."""
    pythons, identities = [], []

    def collect(python, expected):
        identities.append(probe_identity(python, socs, run, source=source, expected=expected))

    for index, target in enumerate(targets, 1):
        message(
            f"Environment {index}/{len(targets)}: Torch {target['torch_version']}, PTA {target['pta_build_version']}"
        )
        pythons.append(FRAMEWORK.prepare_environment(target, NATIVE, run.commands, base, identity_probe=collect))
    return pythons, identities


def build_progress_summary(args, run, pairs: int, elapsed: float):
    """Summarize completed work rather than requiring users to infer reuse from paths."""
    cache = [event for event in run.events if "cache_hit" in event]
    hits = sum(event["cache_hit"] for event in cache)
    message(
        f"Build succeeded in {elapsed:.1f}s; adapter pairs {pairs}; native cache hits {hits}, built {len(cache) - hits}"
    )
    if args.all:
        created = sum(event.get("stage", "").startswith("venv-") for event in run.events)
        message(f"Framework environments: created {created}, reused {pairs - created}")
    message(f"Build log: {run.directory / 'build.log'}")


def report_native_cache(kind: str, hit: bool) -> None:
    """Report native cache decisions while detailed identities stay in stage records."""
    work = (
        "cache hit; verified outputs reused, no compilation"
        if hit
        else "cache miss; no completed matching inputs, building"
    )
    message(f"{kind.capitalize()}: {work}")


def attach_framework_environments(args, pythons, identities):
    """Retain automatic dependency provenance separately from framework compatibility."""
    if args.all:
        for python, identity in zip(pythons, identities):
            environment = json.loads((python.parent.parent / "ready.json").read_text())
            identity["framework_environment"] = {"request": environment["request"], "packages": environment["packages"]}


def preflight_failure(identity: dict, compiler: dict, args, error: RuntimeError) -> str:
    """Explain compiler-standard failures without misdiagnosing unrelated linker errors."""
    version = compiler["version"].partition("\n")[0]
    detail = (
        f"Framework preflight failed: Torch={identity['torch_version']}, PTA={identity['pta_version']}\n"
        f"Compiler: {shlex.join(compiler['command'])} ({version})\n{error}"
    )
    if re.search(
        r"C\+\+20.*(?:required|support)|cxx_std_20|(?:gnu|c)\+\+20.*(?:invalid|unrecognized)", str(error), re.IGNORECASE
    ):
        command = [
            "bash",
            "build.sh",
            *(["--all"] if args.all else []),
            "--soc-list",
            args.soc_list,
            "--jobs",
            str(args.jobs),
        ]
        detail += (
            "\nThis Torch requires final C++20 (__cplusplus >= 202002L); accepting -std=c++20 alone is insufficient."
            "\nSelect a compiler compatible with the Torch/op-plugin headers; GCC 12.4 passed the recorded matrix build."
            "\nRetry from the source root, replacing the compiler path:"
            f"\n  HYPER_MULTICORE_FRAMEWORK_CXX=/path/to/compatible/g++ {shlex.join(command)}"
            "\nThe override affects framework preflight/adapters only; keep the CANN vendor/SDK/ops toolchain unchanged."
            "\nMatching framework environments and native caches are reused automatically."
        )
    return detail


def preflight_inputs(python, identity, host_cxx, env, run):
    """Describe all probe inputs separately from native cache keys."""
    digest = run.preflight_hashes
    development = PREFLIGHT.development_inputs(
        run.frameworks[str(python)].development, Path(identity["cann_root"]), digest
    )
    return {
        "contract": preflight_fingerprint(identity, python, host_cxx, env),
        "framework": {
            k: identity[k]
            for k in (
                "torch_version",
                "pta_version",
                "torch_headers_sha256",
                "pta_headers_sha256",
                "torch_libraries_sha256",
                "libtorch_npu_sha256",
            )
        },
        "compiler": tool_inputs(host_cxx, env, digest, run.commands),
        "development": development,
        "source": fingerprint_files(
            ("hyper_multicore/cmake/framework_probe",),
            ("hyper_multicore/cmake/torch_compat.cmake", "hyper_multicore/torch/csrc/cached_op_api.h"),
        ),
        "flags": compiler_inputs(env),
        "python": str(python),
        "schema": PREFLIGHT.SCHEMA_VERSION,
    }


def preflight_frameworks(args, run: BuildContext, pythons, identities, host_cxx, transaction=None):
    """Reuse only certified compile/link probes; never infer success from an existing SO."""
    for index, (python, identity) in enumerate(zip(pythons, identities), 1):
        env, version = adapter_environment(python), identity["pta_version"]
        env["CXX"] = shlex.join(host_cxx["command"])
        inputs = preflight_inputs(python, identity, host_cxx, env, run)
        key = FRAMEWORK.digest_json(inputs)
        work = NATIVE / "preflight" / WORKSPACE.framework_slot(identity)
        scope = f"preflight PTA {version}"
        run.frameworks[str(python)].tools = run.native_inputs.enrich_tools(inputs["compiler"], env, run.commands)
        run.diagnostics.observe(scope, key, inputs, work)
        message(f"Framework preflight {index}/{len(identities)}: PTA {version}")
        try:
            hit, reason = PREFLIGHT.ensure_probe(
                work,
                key,
                args,
                python,
                env,
                run.commands,
                version,
                run.preflight_hashes,
                transaction,
                ROOT,
                write_json,
                inputs.get("compiler"),
            )
        except RuntimeError as error:
            raise RuntimeError(preflight_failure(identity, host_cxx, args, error)) from error
        certificate = json.loads((work / "complete.json").read_text())
        run.frameworks[str(python)].preflight = {"inputs": inputs, "dependencies": certificate["extra_inputs"]}
        run.events.append(
            {
                "stage": f"preflight-{version}",
                "preflight_cache_hit": hit,
                "reason": reason,
                "fingerprint": key,
                "directory": work.name,
                "search_path": env.get("PATH"),
            }
        )
        message(f"Framework preflight PTA {version}: {'cache hit' if hit else 'passed'}; {reason}")


def cann_python_inputs(python, probe, env, run):
    """Retain raw help in tools.log and collect CANN modules from the real import probe."""
    code = (
        "import json,runpy,sys; from pathlib import Path; "
        "sys.path.insert(0,str(Path(sys.argv[1]).resolve().parent)); sys.argv=[sys.argv[1], '--help']\n"
        "try: runpy.run_path(sys.argv[0],run_name='__main__')\n"
        "except SystemExit as e:\n"
        " if e.code not in (None,0): raise\n"
        "files=sorted({str(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None)})\n"
        "print('\\nHP_CANN_IMPORT_INPUTS='+json.dumps(files))"
    )
    output = run([str(python), "-c", code, str(probe)], label="cann-python-import-probe", env=env, capture=True)
    return json.loads(output.rsplit("HP_CANN_IMPORT_INPUTS=", 1)[1])


def common_inputs(args, run: BuildContext, python: Path, identity: dict):
    """Collect only source, toolchain and compiler-Python inputs consumed by common tasks."""
    common_source = common_source_fingerprint()
    compiler = {
        name: {
            "path": str(Path(shutil.which(name)).resolve()),
            "version": run.commands([name, "--version"], label=f"toolchain-{name}", capture=True),
        }
        for name in ("gcc", "g++", "bisheng", "cmake", "ld", "make")
    }
    compiler["selected_cxx"] = compiler_identity(
        os.environ.get("CXX", "c++"), run.commands, "selected-common-cxx-version"
    )
    common_env = FRAMEWORK.cann_tool_environment(python)
    tool_packages = run.commands(
        [
            str(python),
            "-c",
            "from importlib.metadata import version; import json; print(json.dumps({n:version(n) for n in ('numpy','decorator','attrs','psutil')}))",
        ],
        label="cann-python-packages",
        env=FRAMEWORK.isolated_environment(python),
        capture=True,
    )
    cann = Path(common_env["ASCEND_HOME_PATH"])
    probe = cann / "compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py"
    if not probe.is_file():
        probe = cann / (
            platform.machine() + "-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py"
        )
    if not probe.is_file():
        raise ValueError(f"Missing CANN ASC compilation entry point: {probe}")
    python_files = cann_python_inputs(python, probe, common_env, run.commands)
    message("CANN Python compiler imports: passed")
    inputs = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "source": common_source,
        "socs": args.soc_list,
        "compiler_python_packages": json.loads(tool_packages),
        "cann_root": identity["cann_root"],
        "cann_sha": identity["cann_version_sha256"],
        "compiler": compiler,
        "asc_opc": MANIFEST.file_sha256(Path(shutil.which("asc_opc")).resolve()),
        "flags": compiler_inputs(common_env),
    }
    inputs["tools"] = run.native_inputs.common_tools(compiler, common_env, run.commands)
    inputs["cann_development"] = run.native_inputs.cann(cann, common_env, run.commands, python_files)
    common_key = FRAMEWORK.digest_json(inputs)
    run.common_inputs = inputs
    return common_key, common_env


def build_common(
    args, run: BuildContext, python: Path, common_key: str, common_env: dict, transaction: CacheTransaction
):
    """Prepare one validated common payload and its immutable SDK handoff."""
    common_dir = NATIVE / "common" / WORKSPACE.common_slot(args.soc_list)
    WORKSPACE.check_workspace(common_dir, ROOT)
    if args.clean and common_dir.exists():
        shutil.rmtree(common_dir)
    common = common_dir / "payload"
    run.diagnostics.observe("Shared native resources", common_key, run.common_inputs, common_dir)
    transaction.scopes[common_dir] = run.native_inputs.scope(run.common_inputs)
    with measured(run.events, "common-output-validation"):
        hit = cache_valid(common_dir, common_key)
    report_native_cache("common resources", hit)
    run.events.append(
        {
            "stage": "common",
            "cache_hit": hit,
            "fingerprint": common_key,
            "directory": common_dir.name,
            "socs": args.soc_list,
        }
    )
    if not hit:
        transaction.prepare(common_dir)
        env = dict(common_env)
        env.update(
            PYTHON_BIN=str(python),
            HP_NATIVE_STAGE="common",
            HP_COMPONENT_ROOT=str(common),
            HP_NATIVE_WORK_ROOT=str(common_dir / "work"),
            HP_SHMEM_WORK_ROOT=str(common_dir / "shmem-work"),
            HP_SHMEM_HANDOFF=str(common_dir / "sdk.json"),
        )
        run.commands(
            [
                "bash",
                str(ROOT / "hyper_multicore/build.sh"),
                "--stage-only",
                "--soc-list",
                args.soc_list,
                "--jobs",
                str(args.jobs),
                *(["--clean"] if args.clean else []),
            ],
            label="common-resources",
            env=env,
        )
        transaction.check(common_dir)
        run.native_inputs.verify_consumed(common_dir, common_env["ASCEND_HOME_PATH"])
        sdk_record = json.loads((common_dir / "sdk.json").read_text())
        transaction.complete(
            common_dir,
            {
                "fingerprint": common_key,
                "hashes": hashes(common),
                "inputs": run.common_inputs,
                "sdk_root": sdk_record["install_root"],
                "sdk_hashes": hashes(Path(sdk_record["install_root"])),
                "sdk_manifest_sha256": MANIFEST.file_sha256(common_dir / "sdk.json"),
            },
        )
    sdk = Path(json.loads((common_dir / "sdk.json").read_text())["install_root"])
    return common, sdk


def build_adapters(
    args, run: BuildContext, pythons, identities, common_key, common, sdk, transaction: CacheTransaction
):
    """Build or reuse exactly the requested framework pairs."""
    adapter_source = fingerprint_files(
        (
            "hyper_multicore/torch/csrc",
            "hyper_multicore/shmem/ccsrc/runtime",
            "hyper_multicore/shmem/ccsrc/cann",
            "hyper_multicore/shmem/ccsrc/bindings",
            "hyper_multicore/shmem/ccsrc/include",
            "hyper_multicore/shmem/ccsrc/device",
        ),
        (
            "hyper_multicore/torch/CMakeLists.txt",
            "hyper_multicore/shmem/ccsrc/CMakeLists.txt",
            "hyper_multicore/cmake/torch_compat.cmake",
            "hyper_multicore/cmake/hardening.cmake",
            "tests/ut/shmem/native_log_test.cpp",
        ),
    )
    records, outputs = [], []
    for index, (python, identity) in enumerate(zip(pythons, identities), 1):
        message(f"Adapter {index}/{len(identities)}: Torch {identity['torch_version']}, PTA {identity['pta_version']}")
        transaction.check_source()
        inputs = {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "common": common_key,
            "source": adapter_source,
            "python": str(python),
            "development": run.native_inputs.tree(
                Path(run.frameworks[str(python)].development["python_include"])
            ),
            "compiler": run.frameworks[str(python)].tools,
            "common_path": str(common),
            "flags": compiler_inputs(adapter_environment(python)),
            "framework": {
                k: identity[k]
                for k in (
                    "torch_version",
                    "pta_version",
                    "torch_commit",
                    "torch_cxx11_abi",
                    "python_abi",
                    "host_arch",
                    "op_plugin_headers_sha256",
                    "libtorch_npu_sha256",
                    "torch_headers_sha256",
                    "pta_headers_sha256",
                    "torch_libraries_sha256",
                )
            },
        }
        key = FRAMEWORK.digest_json(inputs)
        run.diagnostics.observe(
            f"Adapter PTA {identity['pta_release']}", key, inputs, NATIVE / "variants" / WORKSPACE.framework_slot(identity)
        )
        output, record = cmake_pair(
            python,
            identity,
            common,
            sdk,
            key,
            args,
            run,
            transaction,
            inputs=inputs,
            probe_inputs=run.frameworks[str(python)].preflight,
        )
        outputs.append(output)
        records.append(record)
    return outputs, records


def assemble_payload(args, run_id, identities, outputs, records, common, common_key):
    """Assemble and validate one selected payload without changing runtime registration policy."""
    staging = NATIVE / "staging" / run_id
    payload = staging / "hyper_multicore"
    shutil.copytree(common, payload)
    for output, record in zip(outputs, records):
        shutil.copytree(output, payload / "adapters" / record["id"])
    manifest = {
        "schema_version": 1,
        "python_abi": identities[0]["python_abi"],
        "host_arch": identities[0]["host_arch"],
        "variants": [
            {k: record[k] for k in ("id", "torch_release", "pta_release", "torch_cxx11_abi", "torch_so", "shmem_so")}
            for record in records
        ],
        "validated_runtime_pairs": [],
        "known_runtime_issues": [],
    }
    write_json(payload / "adapters/manifest.json", manifest)
    info = {
        **identities[0],
        "schema_version": 2,
        "build_mode": "all" if args.all else "current",
        "variants": records,
        "artifact_version": (
            HELPERS.COMPONENT_VERSION if args.all else HELPERS.artifact_version(identities[0]["pta_version"])
        ),
        "requested_target_ids": [v["id"] for v in records],
        "common_fingerprint": common_key,
        "framework_lock_sha256": (
            MANIFEST.file_sha256(ROOT / "hyper_multicore/_build/frameworks.lock.json") if args.all else None
        ),
    }
    info["native_sha256"] = {p.relative_to(payload).as_posix(): MANIFEST.file_sha256(p) for p in payload.rglob("*.so")}
    info["payload_sha256"] = HELPERS._payload_hashes(payload)
    HELPERS.validate_payload(payload, info)
    write_json(payload / "build_info.json", info)
    return payload, info


def package_wheel(python: Path, run: BuildContext, payload: Path, info: dict, transaction: CacheTransaction):
    """Validate the complete wheel and source state before atomically publishing its path."""
    env = FRAMEWORK.isolated_environment(python)
    env["HYPER_MULTICORE_NATIVE_OUTPUT_ROOT"] = str(payload)
    wheel_dir = payload.parent / "wheel"
    run.commands(
        [
            str(python),
            "setup.py",
            "-q",
            "build",
            "--build-base",
            str(payload.parent / "wheel-build"),
            "bdist_wheel",
            "--bdist-dir",
            str(payload.parent / "bdist"),
            "--dist-dir",
            str(wheel_dir),
        ],
        label="package-wheel",
        env=env,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one native wheel")
    with measured(run.events, "wheel-integrity-validation"):
        validate_wheel(wheels[0], payload, info)
    transaction.check_source()
    return wheels[0]


def publish_build_payload(payload: Path, run_id: str, transaction: CacheTransaction) -> Path:
    """Expose a complete selected payload after the final source check."""
    selected = NATIVE / "payload/hyper_multicore"
    selected.parent.mkdir(parents=True, exist_ok=True)
    candidate = selected.with_name(f".payload-{run_id}")
    shutil.copytree(payload, candidate)
    transaction.check_source()
    publish_payload(candidate, selected)
    return selected


def main() -> None:
    """Expose default local-framework builds and the explicit complete release matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Build every locked framework target; keep native caches")
    parser.add_argument("--soc-list", default="ascend910b,ascend910_93")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--native-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    socs = args.soc_list.split(",")
    if (
        args.jobs < 1
        or not socs
        or len(set(socs)) != len(socs)
        or any(s not in ("ascend910b", "ascend910_93") for s in socs)
    ):
        parser.error("--jobs must be positive and --soc-list must contain unique supported CANN SoCs")
    if platform.system() != "Linux":
        parser.error("Native builds require Linux/CANN/PTA")
    NATIVE.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    acquired = False
    publication = WORKSPACE.BuildOutputs(ROOT / "dist", NATIVE / "payload/hyper_multicore")
    try:
        with WORKSPACE.build_lock(NATIVE, "all" if args.all else "current"):
            acquired = True
            run_locked_build(args, publication=publication)
    except KeyboardInterrupt:
        if publication.committed:
            message("Build committed; final lock maintenance cancelled. The new artifact is retained.", "WARN")
            return
        notice = (
            "Build cancelled before commit; dist contains no installable output"
            if acquired
            else "Cancelled while waiting for the build lock; the active build and its outputs are untouched"
        )
        message(notice, "WARN")
        raise SystemExit(130) from None
    except OSError as error:
        if publication.committed:
            message(f"Build committed; final lock maintenance failed: {error}. The new artifact is retained.", "WARN")
            return
        message(f"Build lock or output preparation failed: {error}", "ERROR")
        raise SystemExit(1) from None


def run_locked_build(args, publication=None):
    """Only a locked real attempt owns output cleanup and its diagnostic transcript."""
    publication = publication or WORKSPACE.BuildOutputs(ROOT / "dist", NATIVE / "payload/hyper_multicore")
    with publication:
        directory = WORKSPACE.prepare_logs(NATIVE)
        with CONSOLE.capture_log(directory / "build.log"):
            message(f"Build log: {directory / 'build.log'}")
            try:
                build(args, publication)
            except BaseException as error:
                if publication.committed:
                    message(f"Build committed; maintenance stopped: {error}. The new artifact is retained.", "WARN")
                    return
                level = "WARN" if isinstance(error, KeyboardInterrupt) else "ERROR"
                message(
                    f"Build failed before commit: {error}; dist and formal payload are cleared. "
                    f"Build log: {directory / 'build.log'}",
                    level,
                )
                if isinstance(error, KeyboardInterrupt):
                    raise
                raise SystemExit(1) from error

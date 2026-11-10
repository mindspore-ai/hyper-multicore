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
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from queue import Queue, Empty
import re
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import signal
import sys
import threading
import time
from . import artifacts as HELPERS, framework_environments as FRAMEWORK
from . import vendor_cache as VENDOR
from . import command_runner as CONSOLE, build_workspace as WORKSPACE
from ._bootstrap import load_module
from .command_runner import message
from .build_utils import atomic_json as write_json, measured, version_label
from .build_state import BuildContext, BuildResult, PreparedFramework
from .build_publication import SourceGuard, publish_wheel, publish_payload, validate_elf, validate_wheel

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "build/native"

MANIFEST = load_module("native_manifest")

def probe_identity(python: Path, socs: str, run: BuildContext, *, source=None, expected=None) -> dict:
    """Collect framework identity and installed versions in the target process."""
    code = (
        "import json,sys; from pathlib import Path; sys.path.insert(0,str(Path(sys.argv[1]).parent)); "
        "from _hyper_multicore_bootstrap import load_module; h=load_module('artifacts'); timings={}; "
        "info=h.build_identity(Path(sys.argv[2]),sys.argv[3],source=json.loads(sys.argv[4]),"
        "timings=timings,expected_versions=json.loads(sys.argv[5])); "
        "print(json.dumps({'identity':info,'timings':timings,'packages':h.installed_packages()}))"
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
        env=FRAMEWORK.build_environment(python, isolated=run.isolated),
        capture=True,
    )
    data = json.loads(result)
    run.frameworks[str(python)] = PreparedFramework(python, data["identity"], data["packages"])
    run.events[-1]["identity_timings"] = data["timings"]
    write_json(run.directory / "stages.json", run.events)
    identity = data["identity"]
    identity["torch_release"] = MANIFEST.release(identity["torch_version"])
    identity["pta_release"] = MANIFEST.release(identity["pta_version"])
    return identity


def adapter_environment(python: Path, *, isolated=False) -> dict:
    """Allow newer framework compilers without changing the CANN vendor toolchain."""
    env = FRAMEWORK.build_environment(python, isolated=isolated)
    if "HYPER_MULTICORE_FRAMEWORK_CXX" in env:
        env["CXX"] = env["HYPER_MULTICORE_FRAMEWORK_CXX"]
    return env


def cmake_pair(python, identity, common, sdk, args, run):
    """Compile both framework libraries in fresh, attributed workspaces."""
    directory = NATIVE / "variants" / WORKSPACE.framework_slot(identity)
    WORKSPACE.reset_workspace(directory, ROOT)
    identifier = "pt" + version_label(identity["pta_version"]) + "-" + identity["build_fingerprint"][:12]
    paths = {"torch_so": f"adapters/{identifier}/libhyper_multicore_torch.so",
             "shmem_so": f"adapters/{identifier}/hyper_multicore_shmem_torch.so"}
    payload = directory / "payload"
    payload.mkdir()
    env = adapter_environment(python, isolated=run.isolated)
    env.update(SHMEM_HOME_PATH=str(sdk), SHMEM_SOURCE_DIR=str(sdk / "shmem"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(common / "shmem/lib"), str(common / "shmem/lib/shmem"), env.get("LD_LIBRARY_PATH", "")])
    compile_pair(python, directory, payload, common, args, run, env, identifier)
    with run.commands.scope("Load"):
        load_pair(python, payload, common, run, env, identifier)
    with run.commands.scope("ELF"):
        for name in ("libhyper_multicore_torch.so", "hyper_multicore_shmem_torch.so"):
            validate_elf(payload / name, identity["host_arch"], run.commands, env)
    record = {**identity, **paths, "id": identifier,
              "native_build_identity": {k: identity[k] for k in ("build_commit", "source_sha256", "build_fingerprint")}}
    write_json(directory / "identity.json", record)
    run.commands.add_event({"stage": "adapter", "pta_version": identity["pta_version"],
                            "compiled": True, "compiler_jobs": args.jobs})
    return payload, record


def compile_pair(python, directory, payload, common, args, run, env, identifier):
    """Build the real SHMEM binding/runtime and Torch targets, not separate probes."""
    for kind, source, extra in (
        ("shmem", "hyper_multicore/shmem/ccsrc",
         [f"-DSHMEM_OPS_LIB={common / 'shmem/lib/libhyper_multicore_shmem_ops.so'}",
          "-DHP_SHMEM_SOC_TYPE=" + ("Ascend950" if args.soc_list == "ascend950" else "Ascend910B")]),
        ("torch", "hyper_multicore/torch",
         [f"-DHP_MULTICORE_VENDOR_ROOT={common / 'lib/vendors/hyper_multicore_nn'}"]),
    ):
        with run.commands.scope(kind.title()):
            work = directory / kind
            run.commands(["cmake", "-S", str(ROOT / source), "-B", str(work),
                          "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                          f"-DPython3_EXECUTABLE={python}", f"-DCMAKE_INSTALL_PREFIX={payload}", *extra],
                         label=f"configure-{identifier}-{kind}", env=env)
            run.commands(["cmake", "--build", str(work), "--parallel", str(args.jobs), "--verbose"],
                         label=f"compile-{identifier}-{kind}", env=env)
            if kind == "shmem":
                run.commands(["ctest", "--test-dir", str(work), "--output-on-failure"],
                             label=f"ctest-{identifier}", env=env)
            run.commands(["cmake", "--install", str(work)], label=f"install-adapter-{identifier}-{kind}", env=env)


def load_pair(python, payload, common, run, env, identifier):
    """Resolve real adapter symbols using the selected interpreter, without device work."""
    code = (
        "import ctypes,sys,torch,torch_npu; "
        "ctypes.CDLL(sys.argv[1],mode=ctypes.RTLD_GLOBAL); "
        "torch.ops.load_library(sys.argv[2]); "
        "ctypes.CDLL(sys.argv[3],mode=ctypes.RTLD_GLOBAL)"
    )
    run.commands([str(python), "-c", code,
                  str(common / "shmem/lib/libhyper_multicore_shmem_ops.so"),
                  str(payload / "libhyper_multicore_torch.so"),
                  str(payload / "hyper_multicore_shmem_torch.so")], label=f"load-{identifier}", env=env)


def validate_toolchain() -> None:
    """Reject unavailable common build inputs before preparing any framework environment."""
    cann = os.environ.get("ASCEND_HOME_PATH", "")
    if not cann or not Path(cann).is_dir():
        raise ValueError("ASCEND_HOME_PATH must identify a toolkit directory; source the CANN set_env.sh before building")
    identity = HELPERS.cann_identity(Path(cann))
    message(f"CANN: {identity['cann_version']}; toolkit {identity['cann_root']}")
    required = ("cmake", "ctest", "gcc", "g++", "make", "git", "bisheng", "asc_opc", "readelf", "ldd", "ld")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise ValueError(f"Missing native build tools: {missing}")


def build(args, publication) -> Path:
    """Publish only unchanged source builds; maintenance cannot undo a commit."""
    run = BuildContext(ROOT, NATIVE)
    transaction = SourceGuard(ROOT)
    result = _build(args, run, transaction)
    transaction.check_source()
    publication.committed = True
    finish_maintenance(args, run, result)
    return result.payload


def _build(args, run: BuildContext, transaction: SourceGuard) -> BuildResult:
    """Compile common resources once and each requested pair from fresh workspaces."""
    started = time.monotonic()
    validate_toolchain()
    run_id = run.run_id
    with measured(run.events, "framework-preparation"):
        targets, pythons, identities = prepare_frameworks(args, run)
    with measured(run.events, "common-compilation"):
        common, sdk = build_common(args, run, Path(sys.executable).absolute())
    transaction.check_source()
    with measured(run.events, "adapter-phase"):
        outputs, records = build_adapters(args, run, pythons, identities, common, sdk, transaction)
    with measured(run.events, "payload-assembly-validation"):
        payload, info = assemble_payload(args, run_id, identities, outputs, records, common)
    if run.vendor_requests:
        info["vendor_sources"] = [{k: r[k] for k in ("soc", "origin", "reused")} for r in run.vendor_requests]
        write_json(payload / "build_info.json", info)
    wheel = None if args.native_only else package_wheel(pythons[0], run, payload, info, transaction)
    transaction.check_source()
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
        VENDOR.publish(ROOT, run.vendor_requests)
        run.events.extend({"stage": "vendor-cache-publication", "soc": r["soc"], "elapsed_seconds": r["publish_seconds"]}
                          for r in run.vendor_requests if "publish_seconds" in r)
        if args.all:
            removed = FRAMEWORK.prune_environments(NATIVE, pythons, branches)
            run.events.append({"stage": "prune-framework-environments", "removed": removed})
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


def prepare_frameworks(args, run: BuildContext):
    """Only matrix mode selects exact pairs; local builds use the caller's environment."""
    run.isolated = args.all
    config = ROOT / "hyper_multicore/_build/frameworks.lock.json"
    targets = FRAMEWORK.load_targets(config) if args.all else []
    base = Path(sys.executable).absolute()
    mode = f"full framework matrix: {len(targets)} targets" if args.all else "locally installed framework"
    message(f"Building {mode}; Python cp{sys.version_info.major}{sys.version_info.minor}, "
            f"{platform.machine()}, SoC {args.soc_list}, jobs {args.jobs}")
    source = HELPERS._source_identity(ROOT)
    if args.all:
        reference = json.loads(config.read_text()).get("reference_cann")
        report_cann_reference(reference, HELPERS.cann_identity(Path(os.environ["ASCEND_HOME_PATH"]))["cann_version"])
        pythons, identities = prepare_matrix_environments(targets, run, base, args.soc_list, source)
        for python, identity in zip(pythons, identities):
            identity["reference_cann"] = reference
            environment = json.loads((python.parent.parent / "ready.json").read_text())
            identity["framework_environment"] = {
                "request": environment["request"],
                "installed_packages": run.frameworks[str(python)].packages,
                "installation": environment.get("installation"),
            }
    else:
        report_local_framework(base)
        pythons = [base]
        identities = [probe_identity(base, args.soc_list, run, source=source)]
    return targets, pythons, identities


def report_cann_reference(reference, actual):
    """The matrix reference is informational, never an environment admission gate."""
    if reference:
        message(f"CANN reference: {reference}; actual: {actual}")
        if actual != reference:
            message("CANN differs from the matrix reference or is unknown; continuing with the installed toolkit", "WARN")
    else:
        message(f"CANN actual: {actual}")


def report_local_framework(python: Path):
    """Show installed framework facts before capability probes, without importing them."""
    message("Using locally installed framework; no environment creation or installation")
    message(f"  {'Python':<14}: {python}")
    for label, name in (("Torch", "torch"), ("torch_npu", "torch_npu")):
        package = MANIFEST.framework_package(name)
        message(f"  {label:<14}: {package['version']}")
        message(f"  {label + ' path':<14}: {package['root']}")


def prepare_matrix_environments(targets, run, base, socs, source):
    """Prepare and verify each target once, including its actual installed package versions."""
    message(f"Framework environment storage: {FRAMEWORK.environment_directory(NATIVE)}")
    pythons, identities = [], []

    def collect(python, expected):
        identities.append(probe_identity(python, socs, run, source=source, expected=expected))
        return run.frameworks[str(python)].packages

    for index, target in enumerate(targets, 1):
        message(
            f"Environment {index}/{len(targets)}: Torch {target['torch_version']}, PTA {target['pta_build_version']}"
        )
        pythons.append(FRAMEWORK.prepare_environment(target, NATIVE, run.commands, base, identity_probe=collect))
    return pythons, identities


def build_progress_summary(args, run, pairs: int, elapsed: float):
    """Report actual work instead of cross-build cache decisions."""
    message(f"Build succeeded in {elapsed:.1f}s; built common resources and {pairs} adapter pairs")
    if run.vendor_requests:
        reused = sum(r["reused"] for r in run.vendor_requests)
        message(f"Vendor snapshots: reused {reused}, compiled {len(run.vendor_requests) - reused}; SDK/ops rebuilt")
    if args.all:
        created = sum(event.get("stage", "").startswith("venv-") for event in run.events)
        message(f"Framework environments: created {created}, reused {pairs - created}")
    message(f"Build log: {run.directory / 'build.log'}")


def build_common(args, run: BuildContext, python: Path):
    """Use caller CANN tools for SDK/ops and cached vendor resources in either mode."""
    directory = NATIVE / "common" / WORKSPACE.common_slot(args.soc_list)
    WORKSPACE.reset_workspace(directory, ROOT)
    common = directory / "payload"
    env = FRAMEWORK.cann_tool_environment(python, isolated=False)
    message(f"Common CANN tools: Python {python}; using caller environment")
    env.update(PYTHON_BIN=str(python), HP_NATIVE_STAGE="common", HP_COMPONENT_ROOT=str(common),
               HP_NATIVE_WORK_ROOT=str(directory / "work"), HP_SHMEM_WORK_ROOT=str(directory / "shmem-work"),
               HP_SHMEM_HANDOFF=str(directory / "sdk.json"))
    origin = HELPERS._source_identity(ROOT)
    env.update(HP_VENDOR_CACHE_ROOT=str(NATIVE), HP_VENDOR_ORIGIN=json.dumps(origin))
    if getattr(args, "clean", False):
        env["HP_VENDOR_CACHE_CLEAN"] = "1"
    else:
        env.pop("HP_VENDOR_CACHE_CLEAN", None)
    run.commands(["bash", str(ROOT / "hyper_multicore/build.sh"), "--stage-only",
                  "--soc-list", args.soc_list, "--jobs", str(args.jobs)], label="common-resources", env=env)
    sdk = Path(json.loads((directory / "sdk.json").read_text())["install_root"])
    run.vendor_requests = [json.loads(p.read_text()) for p in sorted((directory / "work").glob("vendor-request-*.json"))]
    for request in run.vendor_requests:
        run.events.append({"stage": "vendor", **{k: request[k] for k in
                          ("soc", "reused", "lookup_seconds", "compile_seconds") if k in request}})
    run.events.append({"stage": "common", "compiled": True, "socs": args.soc_list})
    return common, sdk


def adapter_budgets(count, jobs):
    """Use serial, two-slot or all-target scheduling without exceeding the compile budget."""
    workers = count if jobs >= 2 * count else (2 if jobs >= 4 else 1)
    workers = min(workers, count)
    quotient, remainder = divmod(jobs, workers)
    return [quotient + (slot < remainder) for slot in range(workers)]


def build_adapter(index, count, python, identity, jobs, args, run, common, sdk, transaction):
    """Build one pair with a private argument copy and thread-local diagnostic labels."""
    selected = copy(args)
    selected.jobs = jobs
    started, wall_started = time.monotonic(), time.time()
    error = None
    try:
        with run.commands.scope(f"PTA {identity['pta_version']}"):
            message(f"Adapter {index + 1}/{count}: Torch {identity['torch_version']}, "
                    f"PTA {identity['pta_version']}; compiler jobs {jobs}")
            transaction.check_source()
            try:
                return cmake_pair(python, identity, common, sdk, selected, run)
            except CONSOLE.CommandCancelled:
                raise
            except RuntimeError as cause:
                raise RuntimeError(framework_failure(identity, args, python, cause)) from cause
    except BaseException as caught:
        error = caught
        raise
    finally:
        run.commands.add_event({"stage": f"adapter-compilation-{identity['pta_version']}",
                                "elapsed_seconds": time.monotonic() - started,
                                "started_at": wall_started, "ended_at": time.time(),
                                "status": CONSOLE.command_status(0, error), "compiler_jobs": jobs})


def parallel_adapters(args, run, pythons, identities, common, sdk, transaction, budgets):
    """Bind quotas to fixed workers so out-of-order completion cannot oversubscribe jobs."""
    targets = Queue()
    for index, (python, identity) in enumerate(zip(pythons, identities)):
        targets.put((index, python, identity))
    results = [None] * len(identities)
    progress = threading.Lock()
    completed = 0

    def slot(jobs):
        nonlocal completed
        while not run.commands.cancelled.is_set():
            try:
                index, python, identity = targets.get_nowait()
            except Empty:
                return
            try:
                result = build_adapter(index, len(identities), python, identity, jobs,
                                       args, run, common, sdk, transaction)
                with progress:
                    results[index] = result
                    completed += 1
                    message(f"Adapters completed {completed}/{len(identities)}; PTA {identity['pta_version']}")
            except BaseException as error:
                run.commands.cancel(error)
                raise
            finally:
                targets.task_done()

    pool = ThreadPoolExecutor(max_workers=len(budgets), thread_name_prefix="native-adapter")
    error = None
    try:
        futures = [pool.submit(slot, jobs) for jobs in budgets]
        for future in as_completed(futures):
            future.result()
    except BaseException as caught:
        run.commands.cancel(caught)
        error = run.commands.failure or caught
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if error is not None:
        raise error
    return results


def build_adapters(args, run: BuildContext, pythons, identities, common, sdk, transaction: SourceGuard):
    """Compile pairs with bounded concurrency, preserving matrix order for publication."""
    count = len(identities)
    budgets = adapter_budgets(count, args.jobs) if args.all else [args.jobs]
    message(f"Adapter schedule: {len(budgets)} concurrent pairs; "
            f"compiler jobs per slot {', '.join(map(str, budgets))}; total budget {args.jobs}")
    run.commands.add_event({"stage": "adapter-schedule", "workers": len(budgets),
                            "worker_jobs": budgets, "total_jobs": args.jobs, "targets": count})
    if len(budgets) == 1:
        results = [build_adapter(index, count, python, identity, args.jobs, args, run, common, sdk, transaction)
                   for index, (python, identity) in enumerate(zip(pythons, identities))]
    else:
        results = parallel_adapters(args, run, pythons, identities, common, sdk, transaction, budgets)
    outputs, records = zip(*results)
    return list(outputs), list(records)


def assemble_payload(args, run_id, identities, outputs, records, common):
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
        "reference_cann": identities[0].get("reference_cann"),
        "framework_lock_sha256": (
            MANIFEST.file_sha256(ROOT / "hyper_multicore/_build/frameworks.lock.json") if args.all else None
        ),
    }
    info["native_sha256"] = {p.relative_to(payload).as_posix(): MANIFEST.file_sha256(p) for p in payload.rglob("*.so")}
    info["payload_sha256"] = HELPERS._payload_hashes(payload)
    HELPERS.validate_payload(payload, info)
    write_json(payload / "build_info.json", info)
    return payload, info


def package_wheel(python: Path, run: BuildContext, payload: Path, info: dict, transaction: SourceGuard):
    """Validate the complete wheel and source state before atomically publishing its path."""
    env = FRAMEWORK.build_environment(python, isolated=run.isolated)
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


def publish_build_payload(payload: Path, run_id: str, transaction: SourceGuard) -> Path:
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
    parser.add_argument("--all", action="store_true",
                        help="Build every configured framework target; adapter concurrency follows --jobs")
    parser.add_argument("--soc-list", default="ascend910b,ascend910_93")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="Compiler job budget, shared by concurrent adapters")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--native-only", action="store_true",
                        help="Build a source-local native payload without a wheel; activate with source activate.bash")
    args = parser.parse_args()
    socs = args.soc_list.split(",")
    if (
        args.jobs < 1
        or not socs
        or len(set(socs)) != len(socs)
        or any(s not in ("ascend910b", "ascend910_93", "ascend950") for s in socs)
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


def framework_failure(identity: dict, args, python: Path, error: RuntimeError) -> str:
    """Keep the real compiler diagnostic and an actionable C++20 retry command."""
    compiler = os.environ.get("HYPER_MULTICORE_FRAMEWORK_CXX", os.environ.get("CXX", "c++"))
    detail = (f"Framework build/load failed: Torch={identity['torch_version']}, PTA={identity['pta_version']}"
              f"; Python={python}; compiler={compiler}\n{error}")
    if re.search(r"C\+\+20.*(?:required|support)|cxx_std_20|(?:gnu|c)\+\+20.*(?:invalid|unrecognized)",
                 str(error), re.IGNORECASE):
        command = ["bash", "build.sh", *(["--all"] if args.all else []),
                   "--soc-list", args.soc_list, "--jobs", str(args.jobs)]
        detail += (
            "\nThis Torch requires final C++20 (__cplusplus >= 202002L). Select a compiler compatible with its headers."
            "\nRetry from the source root, replacing the compiler path:"
            f"\n  HYPER_MULTICORE_FRAMEWORK_CXX=/path/to/compatible/g++ {shlex.join(command)}"
            "\nThe override affects framework adapters only; keep the CANN vendor/SDK/ops toolchain unchanged.")
    return detail

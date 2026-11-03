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
"""Use the same framework-free module identities as production build entrypoints."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[3]

BOOTSTRAP_NAME = "_hyper_multicore_bootstrap"
if BOOTSTRAP_NAME not in sys.modules:
    specification = importlib.util.spec_from_file_location(
        BOOTSTRAP_NAME, ROOT / "hyper_multicore/_build/_hyper_multicore_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(specification)
    sys.modules[BOOTSTRAP_NAME] = bootstrap
    specification.loader.exec_module(bootstrap)
load_module = sys.modules[BOOTSTRAP_NAME].load_module


def module_for(function):
    """Patch explicit module attributes rather than a function's hidden globals dictionary."""
    return sys.modules[function.__module__]


ENV = load_module("framework_environments")


def put(path: Path, text: str = "retained") -> Path:
    """Write a fixture file and create its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_target(cache: Path) -> dict:
    """Create version-only configuration and tiny local wheel fixtures."""
    for name in ("torch", "torch-npu", "numpy"):
        filename = name.replace("-", "_") + "-1.0.0-py3-none-any.whl"
        with zipfile.ZipFile(cache / filename, "w") as archive:
            archive.writestr(name.replace("-", "_") + "-1.0.0.dist-info/METADATA",
                             f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0.0\n")
    return {"id": "fixture", "torch_version": "1.0.0", "torch_release": "1.0.0", "pta_release": "1.0.0", "pta_build_version": "1.0.0",
            "indexes": dict.fromkeys(("torch", "pta", "dependencies"), "https://example.invalid/simple"),
            "build_requirements": ["numpy"]}


def cached_resolution(target: dict, native: Path, base: Path) -> tuple[Path, dict]:
    """Seed a fully resolved wheelhouse without giving the installer network access."""
    key = ENV.environment_key(target, base)
    resolution = native / "incoming"
    wheels = resolution / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    packages = []
    for path in (native / "framework-packages").glob("*.whl"):
        shutil.copy2(path, wheels / path.name)
        packages.append(ENV.wheel_record(path, "https://example.invalid/simple"))
    record = {"request": target, "packages": packages}
    (resolution / "pip-report.json").write_text(json.dumps({"install": []}))
    destination, record = ENV.STORE.publish_snapshot(resolution, native, record, target, key, ENV.verify_wheel)
    shutil.rmtree(resolution)
    return destination, record


def resolution_runner(cache, commands, interrupt=False):
    """Model pip's report and wheel downloads, including its transient local-root URLs."""
    def run(command, **_kwargs):
        commands.append(command)
        if "--dry-run" in command:
            entries = []
            for name in ("torch", "torch-npu", "numpy"):
                path = cache / (name.replace("-", "_") + "-1.0.0-py3-none-any.whl")
                entries.append({"metadata": {"name": name, "version": "1.0.0"}, "download_info": {
                    "url": path.as_uri(), "archive_info": {"hashes": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}})
            Path(command[command.index("--report") + 1]).write_text(json.dumps({"install": entries}))
        elif "--dest" in command:
            destination = Path(command[command.index("--dest") + 1])
            if "--require-hashes" in command:
                if interrupt:
                    raise KeyboardInterrupt()
                for line in Path(command[-1]).read_text().splitlines():
                    source = Path(urllib.parse.unquote(urllib.parse.urlparse(line.split()[0]).path))
                    shutil.copy2(source, destination / source.name)
            else:
                name = command[-1].split("==")[0].replace("-", "_")
                source = cache / (name + "-1.0.0-py3-none-any.whl")
                shutil.copy2(source, destination / source.name)
    return run

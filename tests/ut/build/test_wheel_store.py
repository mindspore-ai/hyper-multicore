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
"""Shared wheel snapshots, exact resolution closure and failed-download recovery."""

from pathlib import Path
import sys
import zipfile
import pytest
from tests.ut.build._build_test_support import load_module, make_target, cached_resolution, resolution_runner
ENV = load_module("framework_environments")


def flat_snapshot(native, identifier):
    cache = native / "framework-packages"
    cache.mkdir(exist_ok=True)
    target = make_target(cache)
    target["id"] = identifier
    directory, record = cached_resolution(target, native, Path(sys.executable))
    return target, directory, record


def no_network(*_args, **_kwargs):
    pytest.fail("a completed local snapshot must not access the network")


def test_two_snapshots_reuse_one_wheel_blob_per_digest(tmp_path):
    first = flat_snapshot(tmp_path, "first")
    second = flat_snapshot(tmp_path, "second")
    for target, directory, _record in (first, second):
        ENV.resolved_environment(target, tmp_path, Path(sys.executable), no_network, _record["environment_key"])
    expected = len(first[2]["packages"])
    assert len(list((tmp_path / "framework/wheels").glob("*.whl"))) == expected
    assert not list((tmp_path / "framework/resolutions").glob("*/wheels/*.whl"))
    assert first[1] != second[1] and first[1].exists() and second[1].exists()
    assert len(list((tmp_path / "framework/resolutions").glob("*.pip-report.json"))) == 2


def test_same_filename_different_content_cannot_replace_flat_wheel(tmp_path):
    _target, snapshot, record = flat_snapshot(tmp_path, "filename")
    package = record["packages"][0]
    source = tmp_path / package["filename"]
    store = ENV.STORE
    destination = store.wheel_path(tmp_path, package)
    original = destination.read_bytes()
    source.write_bytes(original)
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("different-build.txt", "same name, different build")
    changed = ENV.wheel_record(source, package["source"])
    with pytest.raises(ValueError, match="Conflicting shared wheel filename"):
        store.import_wheel(source, tmp_path, changed, ENV.verify_wheel)
    assert destination.read_bytes() == original


def test_new_storage_does_not_change_platform_rejection(tmp_path):
    _target, snapshot, original = flat_snapshot(tmp_path, "platform")
    package = dict(original["packages"][0], filename="torch-1.0.0-cp399-cp399-unavailable_platform.whl")
    source = tmp_path / package["filename"]
    source.write_bytes(ENV.STORE.wheel_path(tmp_path, original["packages"][0]).read_bytes())
    with pytest.raises(ValueError, match="platform"):
        ENV.STORE.import_wheel(source, tmp_path, package, ENV.verify_wheel)


def test_flat_report_integrity_is_checked_without_network(tmp_path):
    target, snapshot, record = flat_snapshot(tmp_path, "report")
    destination, _ = ENV.resolved_environment(target, tmp_path, Path(sys.executable), no_network, record["environment_key"])
    report = destination.with_suffix(".pip-report.json")
    report.write_text("{}")
    with pytest.raises((FileNotFoundError, ValueError)):
        ENV.resolved_environment(target, tmp_path, Path(sys.executable), no_network, record["environment_key"])


def test_new_request_reuses_only_source_matching_framework_roots(tmp_path):
    target, snapshot, record = flat_snapshot(tmp_path, "root-cache")
    ENV.resolved_environment(target, tmp_path, Path(sys.executable), no_network, record["environment_key"])
    changed = {**target, "id": "new-request", "build_requirements": ["numpy>=1"]}
    stage = tmp_path / "stage"
    stage.mkdir()
    roots = ENV.download_roots(changed, stage, tmp_path, Path(sys.executable), no_network)
    assert len(roots) == 2 and all(p.parent == tmp_path / "framework/wheels" for p in roots)
    changed["indexes"] = {**target["indexes"], "torch": "https://other.invalid/simple"}
    assert ENV.STORE.cached_root(tmp_path, changed, "torch", "1.0.0", ENV.verify_wheel) is None


def test_resolved_dependencies_reuse_shared_wheels_without_download(tmp_path):
    target, snapshot, record = flat_snapshot(tmp_path, "dependency-cache")
    _, record = ENV.resolved_environment(target, tmp_path, Path(sys.executable), no_network, record["environment_key"])
    entries = [{"metadata": {"name": p["name"], "version": p["version"]}, "download_info": {
        "url": "https://example.invalid/" + p["filename"], "archive_info": {"hashes": {"sha256": p["sha256"]}}}}
        for p in record["packages"]]
    stage = tmp_path / "download"
    stage.mkdir()
    assert ENV.reported_downloads(entries, tmp_path, stage) == []
    assert {p.name for p in stage.iterdir()} == {p["filename"] for p in record["packages"]}


@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_cached_resolution_corruption_does_not_download_or_mark_ready(tmp_path, damage):
    """A missing or changed cached wheel is an explicit failure, never an implicit refresh."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    resolution, record = cached_resolution(target, tmp_path, Path(sys.executable))
    victim = ENV.STORE.wheel_path(tmp_path, record["packages"][0])
    if damage == "missing":
        victim.unlink()
    else:
        victim.chmod(0o644)
        with zipfile.ZipFile(victim, "a") as archive:
            archive.writestr("changed", "unexpected bytes")
    commands = []
    def run(command, **_kwargs):
        commands.append(command)
    with pytest.raises((ValueError, FileNotFoundError)):
        ENV.prepare_environment(target, tmp_path, run, Path(sys.executable))
    assert not any(c[1:4] == ["-m", "pip", "download"] for c in commands)
    assert not list((tmp_path / "framework/envs").rglob("ready.json"))


def test_resolution_uses_explicit_sources_and_only_reported_dependencies(tmp_path):
    """Unrelated cached wheels never enter the generated installation closure."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    commands = []
    key = ENV.environment_key(target, Path(sys.executable))
    directory, record = ENV.resolved_environment(target, tmp_path, Path(sys.executable), resolution_runner(cache, commands), key)
    assert {p["name"] for p in record["packages"]} == {"torch", "torch-npu", "numpy"}
    assert len(list((tmp_path / "framework/wheels").glob("*.whl"))) == 3
    assert not (directory / "wheels").exists()
    roots = [c for c in commands if "--no-deps" in c and "--require-hashes" not in c]
    assert all("--index-url" in c and "--only-binary=:all:" in c for c in roots)
    assert "--ignore-installed" in next(c for c in commands if "--dry-run" in c)
    assert not (directory / "roots").exists()


def test_resolution_interruption_never_publishes_snapshot(tmp_path):
    """Incomplete downloads cannot survive as a reusable resolution."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    with pytest.raises(KeyboardInterrupt):
        ENV.resolved_environment(target, tmp_path, Path(sys.executable), resolution_runner(cache, [], interrupt=True), "a" * 64)
    assert not list((tmp_path / "framework/resolutions").iterdir())


def test_dotted_dependency_names_resolve_and_reuse_verified_wheels(tmp_path):
    """Report and wheel spellings share one canonical distribution identity."""
    target = make_target(tmp_path)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    wheel = incoming / "jaraco_context-1.0.0-py3-none-any.whl"
    name = "Jaraco..__Context"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("jaraco_context-1.0.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0.0\n")
    record = ENV.wheel_record(wheel, "https://example.invalid/" + wheel.name)
    entry = {"metadata": {"name": name, "version": record["version"]}, "download_info": {
        "url": record["source"], "archive_info": {"hashes": {"sha256": record["sha256"]}}}}
    packages = ENV.resolution_packages(target, [entry], incoming)
    assert packages[0]["name"] == "jaraco-context"
    native = tmp_path / "native"
    ENV.STORE.import_wheel(wheel, native, packages[0], ENV.verify_wheel)
    ENV.verify_wheel(ENV.STORE.wheel_path(native, record), {**record, "name": name})
    reused = tmp_path / "reused"
    reused.mkdir()
    assert ENV.reported_downloads([entry], native, reused) == []
    assert ENV.wheel_record(reused / wheel.name, record["source"]) == record
    target["build_requirements"] = ["Torch..__Npu"]
    with pytest.raises(ValueError, match="Invalid direct build requirement"):
        ENV.validate_sources_and_requirements(target)

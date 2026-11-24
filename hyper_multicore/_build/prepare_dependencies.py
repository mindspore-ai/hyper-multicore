# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Explicitly acquire or verify pinned native build dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = Path(__file__).with_name("dependencies.lock.json")
_DEFAULT_CACHE = _REPO_ROOT / "build" / "native" / "deps"


def _parse_args(dependencies: dict[str, Any]) -> argparse.Namespace:
    """Parse the explicit dependency acquisition contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dependency",
        action="append",
        required=True,
        choices=tuple(dependencies),
        help="Dependency to prepare. Repeat the option to prepare several dependencies.",
    )
    parser.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    parser.add_argument("--source-dir")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Acquire requested dependencies, then verify their locked identities."""
    dependencies = load_dependency_locks()
    args = _parse_args(dependencies)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _REPO_ROOT / cache_dir
    if args.source_dir and len(args.dependency) != 1:
        raise ValueError("--source-dir can only be used with one --dependency value.")
    prepared: dict[str, Any] = {}
    for dependency in dict.fromkeys(args.dependency):
        source_dir = Path(args.source_dir) if args.source_dir else cache_dir / dependency / "src"
        if not source_dir.is_absolute():
            source_dir = _REPO_ROOT / source_dir
        dependency_lock = dependencies[dependency]
        revisions = dependency_revisions(dependency_lock)
        external_source = bool(args.source_dir and source_dir.exists())
        if not source_dir.exists():
            if args.verify_only:
                raise ValueError(f"Dependency source directory does not exist: {source_dir}")
            _clone_dependency(next(iter(revisions.values())), source_dir)
        verified = {}
        for revision_name, revision in revisions.items():
            _ensure_commit(revision, source_dir, allow_fetch=not (args.verify_only or external_source))
            verified[revision_name] = verify_git_dependency(
                revision, source_dir, dependency_name=f"{dependency}.{revision_name}"
                if "revisions" in dependency_lock else dependency,
            )
        prepared[dependency] = (
            {"repository": dependency_lock["repository"], "source_dir": str(source_dir.resolve()),
             "revisions": verified} if "revisions" in dependency_lock else verified["default"]
        )
    print(json.dumps(prepared, sort_keys=True))
    return 0


def verify_git_dependency(
    dependency_lock: dict[str, Any],
    source_dir: str | Path,
    dependency_name: str | None = None,
) -> dict[str, Any]:
    """Verify a locked commit independently of the shared worktree's current HEAD.

    Args:
        dependency_lock: Dependency entry from ``dependencies.lock.json``.
        source_dir: Existing Git worktree to verify.
        dependency_name: Stable lock name; inferred from the repository when omitted.

    Returns:
        Verified source identity suitable for a run record.

    Raises:
        ValueError: If the source is missing, modified, or does not match the lock.
    """
    source_dir = Path(source_dir).resolve()
    if not (source_dir / ".git").exists():
        raise ValueError(f"Dependency source is not a Git worktree: {source_dir}")
    commit = dependency_lock["commit"]
    actual_commit = _git_output(source_dir, "rev-parse", f"{commit}^{{commit}}")
    actual_tree = _git_output(source_dir, "rev-parse", f"{commit}^{{tree}}")
    tracked_changes = _git_output(source_dir, "status", "--porcelain", "--untracked-files=no")
    if actual_commit != dependency_lock["commit"]:
        raise ValueError(f"Dependency commit mismatch: expected={dependency_lock['commit']}, actual={actual_commit}")
    if actual_tree != dependency_lock["tree"]:
        raise ValueError(f"Dependency tree mismatch: expected={dependency_lock['tree']}, actual={actual_tree}")
    if tracked_changes:
        raise ValueError("Dependency source contains tracked modifications; upstream patching is not allowed.")
    repository_name = Path(dependency_lock["repository"]).stem
    archive_hash = _git_archive_sha256(
        source_dir,
        dependency_lock["commit"],
        f"{repository_name}-{dependency_lock['version']}/",
    )
    expected_archive_hashes = {
        dependency_lock["git_archive_tar_sha256"],
        *dependency_lock.get("git_archive_tar_sha256_compat", []),
    }
    if archive_hash not in expected_archive_hashes:
        raise ValueError(
            "Dependency archive hash mismatch: "
            f"expected one of {sorted(expected_archive_hashes)}, actual={archive_hash}"
        )
    return {
        "dependency": dependency_name or repository_name.replace("-", "_"),
        "version": dependency_lock["version"],
        "repository": dependency_lock["repository"],
        "source_dir": str(source_dir),
        "commit": actual_commit,
        "tree": actual_tree,
        "git_archive_tar_sha256": archive_hash,
        "patched": False,
    }


def load_dependency_locks(lock_path: Path | None = None) -> dict[str, Any]:
    """Read schema 2 and reject repeated repository declarations before acquisition."""
    lock = json.loads((lock_path or _LOCK_PATH).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 2:
        raise ValueError("Native dependency lock must use schema_version 2")
    dependencies = lock["components"]["multicore"]
    repositories = {}
    for name, dependency in dependencies.items():
        repository = dependency["repository"]
        if repository in repositories:
            raise ValueError(
                f"Repository {repository} is declared by both {repositories[repository]} and {name}; "
                "declare additional versions under revisions instead."
            )
        repositories[repository] = name
        dependency_revisions(dependency)
    return dependencies


def dependency_revisions(dependency_lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve single-version or named revisions into complete verification entries."""
    if "revisions" not in dependency_lock:
        return {"default": dependency_lock}
    revisions = dependency_lock["revisions"]
    if not isinstance(revisions, dict) or not revisions:
        raise ValueError("Dependency revisions must be a nonempty mapping")
    if any(not isinstance(revision, dict) or "repository" in revision for revision in revisions.values()):
        raise ValueError("Each revision must be an object; repository belongs to the parent entry")
    return {name: {**revision, "repository": dependency_lock["repository"]}
            for name, revision in revisions.items()}


def _clone_dependency(dependency_lock: dict[str, Any], source_dir: Path) -> None:
    """Initialize one ordinary repository, then check out its first locked revision."""
    source_dir.mkdir(parents=True)
    _run_git(source_dir, "init")
    _run_git(source_dir, "remote", "add", "origin", dependency_lock["repository"])
    _ensure_commit(dependency_lock, source_dir, allow_fetch=True)
    _run_git(source_dir, "checkout", "--detach", dependency_lock["commit"])


def _ensure_commit(dependency_lock: dict[str, Any], source_dir: Path, allow_fetch: bool) -> None:
    """Fetch only missing commits and retain them without switching the shared HEAD."""
    if not (source_dir / ".git").exists():
        raise ValueError(f"Dependency source is not a Git worktree: {source_dir}")
    commit = dependency_lock["commit"]
    result = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=source_dir,
                            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        if not allow_fetch:
            raise ValueError(f"Locked commit {commit} is missing from {source_dir}")
        _run_git(source_dir, "fetch", "--depth", "1", "origin",
                 dependency_lock.get("fetch_ref") or dependency_lock["version"])
    if allow_fetch:
        _run_git(source_dir, "update-ref", f"refs/hyper_multicore/{commit}", commit)


def _run_git(source_dir: Path, *args: str) -> None:
    """Execute acquisition commands with diagnostics on stderr, leaving stdout as JSON."""
    result = subprocess.run(["git", *args], cwd=source_dir, check=False, stdout=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Git dependency command failed ({' '.join(args)}) in {source_dir}")


def _git_output(source_dir: Path, *args: str) -> str:
    """Run one read-only Git inspection and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=source_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Git inspection failed ({' '.join(args)}): {result.stderr.strip()}")
    return result.stdout.strip()


def _git_archive_sha256(source_dir: Path, commit: str, archive_prefix: str) -> str:
    """Hash a deterministic uncompressed Git archive without creating a tarball."""
    with subprocess.Popen(
        # Hash committed LFS pointers regardless of local LFS caches or filter settings.
        ["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
         "archive", "--format=tar", f"--prefix={archive_prefix}", commit],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Cannot read git archive output.")
        digest = hashlib.sha256()
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(chunk)
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise ValueError(f"Git archive failed: {stderr.decode(errors='replace').strip()}")
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        sys.stderr.write(f"[HP-NATIVE-DEPENDENCY-ERROR] {error}\n")
        raise SystemExit(1) from error

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
"""Exercise shared Git acquisition with real local repositories and locked archives."""

from contextlib import redirect_stdout
import copy
import hashlib
from importlib import util
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


from tests.ut.build._build_test_support import ROOT

HELPER_PATH = ROOT / "hyper_multicore/_build/prepare_dependencies.py"


class TestPrepareDependencies(unittest.TestCase):
    """One repository serves independently verified revisions without repeated downloads."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.git(self.upstream, "init")
        self.git(self.upstream, "config", "user.name", "Dependency Test")
        self.git(self.upstream, "config", "user.email", "dependency-test@example.invalid")
        self.git(self.upstream, "config", "commit.gpgsign", "false")
        self.revisions = {}
        for index, name in enumerate(("gmm", "mhc", "future"), 1):
            (self.upstream / "value.txt").write_text(name + "\n", encoding="utf-8")
            self.git(self.upstream, "add", "value.txt")
            self.git(self.upstream, "commit", "-m", name)
            version = f"v{index}"
            self.git(self.upstream, "tag", version)
            commit = self.git(self.upstream, "rev-parse", "HEAD")
            archive = subprocess.check_output(
                ["git", "archive", "--format=tar", f"--prefix=upstream-{version}/", commit], cwd=self.upstream,
            )
            self.revisions[name] = {"version": version, "commit": commit,
                                    "tree": self.git(self.upstream, "rev-parse", "HEAD^{tree}"),
                                    "git_archive_tar_sha256": hashlib.sha256(archive).hexdigest()}
        self.lock = {"schema_version": 2, "components": {"multicore": {"transformer": {
            "repository": self.upstream.as_uri(), "revisions": {
                name: self.revisions[name] for name in ("gmm", "mhc")},
        }}}}
        self.lock_path = self.root / "lock.json"
        self.cache = self.root / "cache"
        self.source = self.cache / "transformer/src"
        spec = util.spec_from_file_location("native_dependency_test", HELPER_PATH)
        self.helper = util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.helper._LOCK_PATH = self.lock_path

    @staticmethod
    def git(directory, *args):
        """Run fixture Git operations independently of the production helper."""
        return subprocess.check_output(["git", *args], cwd=directory, stderr=subprocess.PIPE, text=True).strip()

    def prepare(self, *arguments):
        """Use the actual CLI contract and require machine-readable stdout."""
        self.lock_path.write_text(json.dumps(self.lock), encoding="utf-8")
        stdout = io.StringIO()
        with patch("sys.argv", [str(HELPER_PATH), "--dependency", "transformer",
                                "--cache-dir", str(self.cache), *arguments]), redirect_stdout(stdout):
            self.assertEqual(self.helper.main(), 0)
        return json.loads(stdout.getvalue())["transformer"]

    def test_one_repository_contains_both_commits_and_exports_correct_sources(self):
        """Initialize once, fetch each missing revision and leave HEAD at the first."""
        with patch.object(self.helper, "_run_git", wraps=self.helper._run_git) as calls:
            result = self.prepare()
        commands = [call.args[1] for call in calls.call_args_list]
        self.assertEqual(commands.count("init"), 1)
        self.assertEqual(commands.count("fetch"), 2)
        self.assertEqual(list(self.cache.rglob(".git")), [self.source / ".git"])
        self.assertEqual(self.git(self.source, "rev-parse", "HEAD"), self.revisions["gmm"]["commit"])
        for name, revision in result["revisions"].items():
            self.assertEqual(revision["source_dir"], result["source_dir"])
            self.assertEqual(self.git(self.source, "show", f"{revision['commit']}:value.txt"), name)

    def test_warm_cache_works_without_upstream_or_fetch(self):
        """An unavailable remote must not affect cached verified revisions."""
        expected = self.prepare()
        self.upstream.rename(self.root / "offline")
        with patch.object(self.helper, "_run_git", wraps=self.helper._run_git) as calls:
            self.assertEqual(self.prepare(), expected)
            self.assertEqual(self.prepare("--verify-only"), expected)
        self.assertFalse(any(call.args[1] in {"init", "fetch", "checkout"} for call in calls.call_args_list))

    def test_new_revision_uses_existing_repository(self):
        """A third named revision only fetches its missing commit."""
        self.prepare()
        self.lock["components"]["multicore"]["transformer"]["revisions"]["future"] = self.revisions["future"]
        with patch.object(self.helper, "_run_git", wraps=self.helper._run_git) as calls:
            result = self.prepare()
        self.assertEqual(set(result["revisions"]), {"gmm", "mhc", "future"})
        self.assertEqual([call.args[1] for call in calls.call_args_list].count("fetch"), 1)
        self.assertFalse(any(call.args[1] == "init" for call in calls.call_args_list))

    def test_duplicate_repository_rejected_before_acquisition(self):
        """Additional versions cannot masquerade as another top-level dependency."""
        dependencies = self.lock["components"]["multicore"]
        dependencies["duplicate"] = copy.deepcopy(dependencies["transformer"])
        with patch.object(self.helper, "_run_git", side_effect=AssertionError("Unexpected acquisition")):
            with self.assertRaisesRegex(ValueError, "revisions instead"):
                self.prepare()
        self.assertFalse(self.cache.exists())

    def test_nested_tree_and_archive_mismatch_do_not_refresh_repository(self):
        """Wrong nested lock identities fail without deleting or redownloading sources."""
        self.prepare()
        revision = self.lock["components"]["multicore"]["transformer"]["revisions"]["mhc"]
        for field in ("tree", "git_archive_tar_sha256"):
            with self.subTest(field=field), patch.dict(revision, {field: "0" * len(revision[field])}):
                with patch.object(self.helper, "_run_git", wraps=self.helper._run_git) as calls:
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        self.prepare()
                self.assertFalse(any(call.args[1] in {"init", "fetch"} for call in calls.call_args_list))
                self.assertTrue((self.source / ".git").is_dir())

    def test_verify_only_missing_commit_does_not_fetch(self):
        """Read-only verification cannot populate a newly requested revision."""
        self.prepare()
        self.lock["components"]["multicore"]["transformer"]["revisions"]["future"] = self.revisions["future"]
        with patch.object(self.helper, "_run_git", side_effect=AssertionError("Unexpected mutation")):
            with self.assertRaisesRegex(ValueError, "missing"):
                self.prepare("--verify-only")

    def test_external_source_verification_preserves_head(self):
        """Explicit sources with another HEAD still verify the locked commit objects."""
        with patch.object(self.helper, "_run_git", side_effect=AssertionError("Unexpected mutation")):
            result = self.prepare("--source-dir", str(self.upstream))
        self.assertEqual(len(result["revisions"]), 2)
        self.assertEqual(self.git(self.upstream, "rev-parse", "HEAD"), self.revisions["future"]["commit"])

    def test_modified_cache_is_not_deleted(self):
        """Reject tracked edits while preserving the source for inspection."""
        self.prepare()
        (self.source / "value.txt").write_text("local edit\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tracked modifications"):
            self.prepare()
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "local edit\n")

    def test_lock_schema_and_nested_revision_shape(self):
        """Reject unsupported schema, empty revisions and hidden repository overrides."""
        with patch.dict(self.lock, {"schema_version": 1}):
            with self.assertRaisesRegex(ValueError, "schema_version 2"):
                self.prepare()
        dependency = self.lock["components"]["multicore"]["transformer"]
        for revisions in ({}, {"bad": "commit"}, {"bad": {"repository": "another"}}):
            with self.subTest(revisions=revisions), patch.dict(dependency, {"revisions": revisions}):
                with self.assertRaisesRegex(ValueError, "revision"):
                    self.prepare()

    def test_repository_check_verifies_nested_patches(self):
        """The existing engineering check must still reject corrupted nested patches."""
        project = self.root / "project"
        helper = project / "hyper_multicore/_build/prepare_dependencies.py"
        helper.parent.mkdir(parents=True)
        shutil.copy2(HELPER_PATH, helper)
        for name in ("scripts", "tests", "examples"):
            (project / name).mkdir()
        for name in ("setup.py", "build.sh", "MANIFEST.in", "requirements.txt"):
            (project / name).write_text("", encoding="utf-8")
        (project / "sample.patch").write_text("patch contents\n", encoding="utf-8")
        self.lock["components"]["multicore"]["transformer"]["revisions"]["mhc"]["patches"] = [
            {"path": "sample.patch", "sha256": "0" * 64}]
        helper.with_name("dependencies.lock.json").write_text(json.dumps(self.lock), encoding="utf-8")
        checker = runpy.run_path(str(ROOT / "scripts/check_repository.py"))["main"]
        with patch.dict(checker.__globals__, {"ROOT": project}):
            with self.assertRaisesRegex(SystemExit, "Locked patch missing or changed"):
                checker()


if __name__ == "__main__":
    unittest.main()

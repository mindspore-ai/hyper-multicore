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
"""Real publication orchestration has one commit boundary for both formal outputs."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.ut.build._build_test_support import ROOT, load_module, put

BUILD = load_module("build_controller")
CACHE = load_module("native_cache")
LAYOUT = BUILD.WORKSPACE


def test_wheel_metadata_does_not_pin_framework_versions():
    """Default wheels retain generic dependencies while recording the exact build separately."""
    publication = load_module("build_publication")
    info = {"build_mode": "current", "artifact_version": "0.1.0.dev0+pt.internal.42", "torch_version": "private torch/42"}
    prefix = f"Name: hyper-multicore\nVersion: {info['artifact_version']}\nRequires-Dist: "
    archive = SimpleNamespace(read=lambda _: (prefix + "torch\n").encode())
    publication.validate_wheel_metadata(archive, "METADATA", info)
    for requirement in ("torch==2.15.0.dev20261005+cpu", "torch @ https://example.invalid/torch.whl"):
        archive = SimpleNamespace(read=lambda _, value=requirement: (prefix + value + "\n").encode())
        with pytest.raises(ValueError, match="without a version constraint"):
            publication.validate_wheel_metadata(archive, "METADATA", info)


@pytest.fixture(name="staged_build")
def _staged_build(tmp_path, monkeypatch):
    """Replace expensive workers while running the real transaction and publishers."""
    candidate = tmp_path / "staging/candidate"
    put(candidate / "new.so", "new payload")
    output = BUILD.WORKSPACE.BuildOutputs(tmp_path / "dist", tmp_path / "native/payload/hyper_multicore")
    put(output.payload / "old.so", "old payload")
    put(output.distribution / "old.whl", "old wheel")
    state = {"checks": 0, "reject_on": None}
    def _validate():
        state["checks"] += 1
        if state["checks"] == state["reject_on"]:
            raise RuntimeError("final input changed")
    def _prepare(_args, _context, guard):
        guard.input_checks.append(_validate)
        return [{"pta_release": "2.12.0"}], [Path(sys.executable)], [{}], {}
    replacements = {"ROOT": tmp_path, "NATIVE": tmp_path / "native", "validate_toolchain": lambda **_: None,
        "prepare_frameworks": _prepare, "common_inputs": lambda *_: ("key", {}),
        "build_common": lambda *_: (tmp_path / "common", tmp_path / "sdk"),
        "build_adapters": lambda *_: ([], [{"pta_version": "2.12.0"}]),
        "assemble_payload": lambda *_: (candidate, {}),
        "package_wheel": lambda *_: put(tmp_path / "staging/candidate.whl", "new wheel")}
    for name, value in replacements.items():
        monkeypatch.setattr(BUILD, name, value)
    monkeypatch.setattr(CACHE, "source_observation", lambda: {})
    return SimpleNamespace(all=False, native_only=False), output, state


def test_success_publishes_only_the_new_wheel_and_payload(staged_build):
    args, output, _ = staged_build
    with output:
        assert BUILD.build(args, output) == output.payload
    assert output.committed
    assert [p.name for p in output.distribution.iterdir()] == ["candidate.whl"]
    assert [p.name for p in output.payload.iterdir()] == ["new.so"]


@pytest.mark.parametrize("reject_on", [1, 2])
def test_final_input_failure_clears_both_outputs(staged_build, reject_on):
    """Reject before publication and after the visible switch, exercising actual publishers."""
    args, output, state = staged_build
    state["reject_on"] = reject_on
    with pytest.raises(RuntimeError, match="final input changed"), output:
        BUILD.build(args, output)
    assert not output.committed and not output.payload.exists()
    assert not list(output.distribution.iterdir())


def test_mid_publication_failure_clears_both_outputs(staged_build, monkeypatch):
    args, output, _ = staged_build
    def _fail(*_args):
        assert (output.distribution / "candidate.whl").exists()
        raise OSError("payload exchange failed")
    monkeypatch.setattr(BUILD, "publish_payload", _fail)
    with pytest.raises(OSError, match="payload exchange failed"), output:
        BUILD.build(args, output)
    assert not output.committed and not output.payload.exists()
    assert not list(output.distribution.iterdir())


def test_precommit_cancellation_clears_both_outputs(staged_build, monkeypatch):
    args, output, _ = staged_build
    def _cancel(*_args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(BUILD, "prepare_frameworks", _cancel)
    with pytest.raises(KeyboardInterrupt), output:
        BUILD.build(args, output)
    assert not output.committed and not output.payload.exists()
    assert not list(output.distribution.iterdir())


@pytest.mark.parametrize("failure", [OSError("cannot reclaim"), KeyboardInterrupt()])
def test_postcommit_maintenance_failure_preserves_success(staged_build, monkeypatch, failure):
    args, output, _ = staged_build
    args.all = True
    def _fail(*_args):
        raise failure
    monkeypatch.setattr(BUILD.FRAMEWORK, "prune_environments", _fail)
    with output:
        assert BUILD.build(args, output) == output.payload
    assert output.committed and (output.payload / "new.so").exists()
    assert (output.distribution / "candidate.whl").exists()
    assert not (output.distribution / "old.whl").exists()


def test_interrupted_ownership_registration_can_retry(tmp_path, monkeypatch):
    directory = tmp_path / "variants/target"
    write = Path.write_text
    def _partial(path, text, *args, **kwargs):
        if path.name == ".workspace.json.tmp":
            write(path, text[:12])
            raise OSError("interrupted I/O")
        return write(path, text, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(Path, "write_text", _partial)
        with pytest.raises(OSError):
            LAYOUT.claim_workspace(directory, ROOT)
    LAYOUT.check_workspace(directory, ROOT)
    LAYOUT.claim_workspace(directory, ROOT)
    LAYOUT.check_workspace(directory, ROOT)
    assert json.loads((directory / ".workspace.json").read_text())["root"] == str(ROOT)
    assert not (directory / ".workspace.json.tmp").exists()


def test_build_bootstrap_never_imports_business_or_frameworks():
    script = f'''import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('hyper_multicore', 'torch', 'torch_npu'):
   raise RuntimeError('unexpected framework import: '+fullname)
sys.meta_path.insert(0,Block())
sys.path.insert(0,{str(ROOT / "hyper_multicore/_build")!r})
from _hyper_multicore_bootstrap import load_module
controller=load_module('build_controller')
assert controller.CONSOLE is load_module('command_runner')
assert controller.WORKSPACE is load_module('build_workspace')
assert controller.PREFLIGHT is load_module('preflight_cache')
assert controller.FRAMEWORK.STORE is load_module('framework_store')
assert controller.CacheTransaction is load_module('native_cache').CacheTransaction
assert controller is load_module('build_controller')
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

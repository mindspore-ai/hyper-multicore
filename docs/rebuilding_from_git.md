# Rebuild using only the public source repositories

The tracked source tree is the complete input owned by this component. A previous
developer machine, training-framework checkout, Git bundle, old runtime archive,
private SDK cache or experiment directory is not required. This component does not
import a downstream training framework.

## Publish and select the qualified source

The A5 BF16/MXFP8 integration is commit
`19ce41fb7d47430f12a69437af04fb06be8e0f61`. The immutable release ref is
`refs/tags/a5-mxfp8-20261109`. Maintainers must publish that commit and tag to
`https://github.com/mindspore-ai/hyper-multicore.git` before consumers can fetch it.
A public default branch that predates the integration does not contain this implementation.
No source-bundle fallback is part of the reconstruction contract.

```bash
git clone https://github.com/mindspore-ai/hyper-multicore.git
cd hyper-multicore
git fetch origin refs/tags/a5-mxfp8-20261109
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = 19ce41fb7d47430f12a69437af04fb06be8e0f61
```

## Prepare platform prerequisites on the new machine

Use aarch64 Linux, a qualified Ascend950 device/driver and the CANN9.2.0 developer
toolkit, including its compiler and operator headers. Install these platform packages
through their supported distribution channels. The source build does not install,
overwrite or reconstruct the system driver or CANN. The qualified host compiler is GCC12
with C++20 support; Git, Make, CMake>=3.18 and Python development headers are required.

The recorded qualification stack is Python3.11.10, torch2.12.0+cpu and
torch_npu2.12.0.post3. Those framework wheels are available on public package indexes;
no wheel from the former developer machine is required.

```bash
conda create -n multicore-a5 -c conda-forge python=3.11.10 -y
conda activate multicore-a5
source "${CANN_ROOT}/set_env.sh"
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.0+cpu'
python -m pip install --index-url https://pypi.org/simple 'torch_npu==2.12.0.post3'
python -m pip install 'numpy>=1.26,<2' setuptools wheel 'packaging>=22' attrs decorator psutil
bash build.sh --soc-list ascend950 --jobs 8
python -m pip install --no-deps --force-reinstall dist/hyper_multicore-*.whl
source "$(command -v hyper_multicore_set_env.bash)"
hyper_multicore_build_info --json
```

Run these commands from this source root in the selected Python environment. See
[build documentation](build.md) for other framework branches and matrix builds.
Changing those inputs requires its own qualification; it is not this recorded build.

## Source closure and reproducibility

- `hyper_multicore/_build/dependencies.lock.json` pins public GitCode commits,
  exported source hashes and every applied patch for ops-nn, ops-transformer,
  SHMEM and the JSON dependency. `frameworks.lock.json` records public Python indexes.
- Selected BF16 and MX Blaze headers and their transitive headers are committed.
  The source paths in their provenance JSON are historical metadata; compilation
  reads the committed files, not another local operator checkout or old installation path.
- Native dependency preparation fetches locked public sources and applies committed
  patches in an isolated build tree. No restored cache is required. GitCode download
  failures remain failures; fix connectivity and rerun without substituting another ref.
- SHMEM's device queue-publication fix is in the committed SDK adapter. A complete
  source build recompiles the consuming Mega entry as well as private host libraries.
- Generated `build/`, `dist/`, `.o`, `.so`, SDK exports and wheels are outputs, not
  required tracked inputs. The wheel carries native objects, vendor registration,
  private SHMEM and the Torch adapter; runtime does not compile them.
- `build_info.json` records source identity, framework/CANN ABI, native payload hashes
  and compiler provenance. Rebuilding in another absolute path need not produce
  a byte-identical wheel; compare the input contract and verify the new payload's
  own hashes rather than assuming an old wheel SHA is a universal release SHA.

An application repository should commit its selected source ref, commit, stack,
build/deployment instructions and any required adapter code. It can generate a wheel
and transport archive from those inputs on demand. A pre-existing experiment archive
must not be an implicit dependency of those instructions.

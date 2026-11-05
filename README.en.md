# hyper-multicore

[简体中文](README.md) | English

Ascend multicore operators and a PyTorch runtime for MegaMoE, MegaGate and MegaMHC.
The runtime overlaps communication and computation on AI Cores and manages its
private SHMEM communication resources.

- **MegaMoE**: expert-parallel forward and backward with push/pull dispatch.
- **MegaGate**: text/vision expert routing with autograd support.
- **MegaMHC**: residual-stream mixing with forward and backward operators.
- **Diagnostics**: kernel profiling and tensor dumps.

## Quick start

Use Linux with an Ascend NPU, Python/CPU architecture matching the wheel tags, and mutually compatible
Torch/torch-npu, CANN and driver installations. Source builds use the active CANN toolkit.
The optional matrix CANN reference is informational; a mismatch warns without stopping the build.
Runtime selection does not impose a PTA post-release floor or a CANN version pairing. See the
[build and installation guide](docs/build.md) for prerequisites and source builds.

Install a wheel built for that environment. Replace the paths below with your
CANN installation and the exact wheel printed by the build command:

```bash
source /path/to/cann/set_env.sh
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
hyper_multicore_build_info
```

To run the two-NPU MoE example against the installed package, copy the example
from the source checkout or extracted sdist into a separate directory:

```bash
# Start in the source root.
example_dir=$(mktemp -d)
cp examples/mega_moe_minimal.py "$example_dir/"
cd "$example_dir"
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc-per-node=2 mega_moe_minimal.py
```

The example runs forward and
backward and prints its shape, native build identity and elapsed time.
Use `--warmup` and `--iterations` to increase sampling. The [examples guide](examples/README.md)
provides complete [MoE](examples/moe/README.md) and [MHC](examples/mhc/README.md)
network integration and training comparisons.

Applications import the public components directly:

```python
from hyper_multicore import MegaMoeExperts, MegaGate, MegaMhc
from hyper_multicore.diagnostics import MegaKernelDiagnostics
```

See the [component guide](hyper_multicore/README.md) for configuration and usage.
Activate native paths before starting a Python process that uses the operators.
pip and Python imports do not compile native code.

## Development and tests

Python, device kernels and native bindings live under `hyper_multicore/`.
HyperParallel is a downstream consumer; building, testing and using this package
does not require it.

From the source root, install development dependencies and run the CPU tests:

```bash
python -m pip install -r requirements.txt setuptools wheel "packaging>=22"
python -m pip install pytest
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_repository.py
python -m pytest tests/ut -q
```

Select `tests/ut` for CPU tests or `tests/torch` for NPU tests explicitly;
without a path, pytest may collect both. The [test guide](tests/README.md) covers
CPU/NPU test placement, adding and running cases, and troubleshooting.

### Code checks

The three pre-commit hooks run Pylint, Lizard and Markdownlint.
Pylint uses version 3.3.7 and `.pylintrc`, loading the license, public function
annotations and docstrings, `sys.path` and subprocess rules from
`scripts/pylint_hypermulticore.py`. Lizard limits cyclomatic complexity to 15 and
function NLOC to 100, reporting only violations. Markdown rules are defined in
`.markdownlint.jsonc`.

Install pre-commit to check selected files manually. Developers who want checks
on commit can also enable the Git hook:

```bash
python -m pip install pre-commit
pre-commit run --files scripts/check_repository.py README.md
pre-commit install
```

pre-commit manages the hook dependencies. To run the Python wrappers directly,
install their tools in the current Python environment:

```bash
python -m pip install "pylint==3.3.7" lizard
python scripts/pre-commit/run_pylint.py scripts/check_repository.py
python scripts/pre-commit/run_lizard.py scripts/check_repository.py
```

Markdownlint needs a Node environment. With `markdownlint-cli2` installed, run:

```bash
markdownlint-cli2 --config .markdownlint.jsonc README.md README.en.md
```

Lizard, Markdownlint, Node and pre-commit are not version-pinned; the environment
must meet the selected tools' runtime requirements. `.clang-format`,
`hyper_multicore/CPPLINT.cfg` and `.codespellrc` remain available for their tools
and are not part of these three hooks.

## Documentation

| Guide | Contents |
| --- | --- |
| [Build and installation](docs/build.md) | Environment, build commands, artifacts, installation and activation |
| [Component usage](hyper_multicore/README.md) | Public APIs and configuration |
| [Testing](tests/README.md) | Test scope, contribution, execution and troubleshooting |
| [Architecture](docs/architecture.md) | Runtime, scheduling and operator extension |
| [SHMEM](docs/shmem.md) | Communication resources and lifecycle |
| [Diagnostics](docs/profiling.md) | Profiling and tensor dumps |

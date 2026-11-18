# HyperMulticore

[简体中文](README.md) | English

HyperMulticore is a megakernel programming framework for Ascend NPUs. It combines computation and
communication operations within a single kernel execution. The framework manages task dependencies
on the device to reduce separate kernel launches and enable finer-grained compute-communication overlap.

Developers can describe execution through computation graphs or direct task assembly, or integrate
existing components into models through PyTorch APIs. Current components include MegaMoE, MegaGate
and MegaMHC, alongside tools for task timelines and tensor snapshots.

[Python APIs](hyper_multicore/README.md) · [Build and installation](docs/build.md) ·
[Architecture](docs/architecture.md) · [Model integration](examples/README.md) · [Development entry](AGENTS.md)

Most detailed guides linked below are currently written in Chinese.

## Main capabilities

- **[MegaMoE](docs/mega_moe.md)**: expert-parallel dispatch, expert computation and result combination, with forward and backward support.
- **[MegaGate](docs/mega_gate.md)**: expert routing for text and vision tokens.
- **[MegaMHC](docs/mega_mhc.md)**: mixing and normalization of four residual streams.
- **[Diagnostics](docs/diagnostics.md)**: internal task timelines and tensor snapshots.

## Quick start

### Obtain and install

Running HyperMulticore requires Linux, Ascend NPUs, Python ≥3.10, and compatible Torch/torch-npu,
CANN and driver installations. Activate the target Python environment, install the
[runtime dependencies](requirements.txt), and obtain a wheel matching your environment.
With source only, first produce a wheel using the [default build](docs/build.md#默认构建).

Run the following commands in Bash, replacing the placeholder paths. `--no-deps` does not install runtime dependencies:

```bash
source /path/to/cann/set_env.sh
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
```

Complete activation before starting a Python process that imports Torch/torch-npu. For compatibility,
import-origin and package-integrity checks, see [installation and verification](docs/build.md#安装与验证).

### Run the two-NPU MoE example

The [minimal MoE example](examples/mega_moe_minimal.py) requires two available NPUs and a working HCCL environment.
Follow the [example setup](examples/README.md#运行单层-moe-示例) to obtain the script from the same version as your wheel.
Run it from a directory outside the source checkout containing only that script; adjust device IDs `0,1` for your environment:

```bash
(
    set -e
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    python -m torch.distributed.run --standalone --nproc-per-node=2 mega_moe_minimal.py
)
```

The example runs forward, backward and communication, and checks outputs and gradients.
On success, rank 0 prints JSON containing the run configuration and timings. This validates basic execution;
see the [test guide](tests/README.md) for accuracy and performance validation.

For single-NPU usage, start with [MegaGate](docs/mega_gate.md#基本用法) or [MegaMHC](docs/mega_mhc.md#基本用法).

## Documentation

| Goal | Reading |
| --- | --- |
| Select public APIs and configuration | [Python APIs](hyper_multicore/README.md) |
| Prepare the environment, build and install | [Build guide](docs/build.md) |
| Integrate existing models and preserve parameters | [Model integration and examples](examples/README.md) |
| Understand framework responsibilities and execution | [Architecture](docs/architecture.md) |
| Capture timelines and inspect tensor values | [Diagnostics](docs/diagnostics.md) |
| Prepare a development environment, run tests and static checks | [Test guide](tests/README.md) |

## Development and contributing

Before modifying the implementation, read the [development entry](AGENTS.md) for module responsibilities
and development conventions. Environment setup, test execution and static checks are covered in the
[test guide](tests/README.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

# hyper-multicore

简体中文 | [English](README.en.md)

HyperMulticore 为 MegaMoE、MegaGate 和 MegaMHC 提供 Ascend 多核算子及 PyTorch 运行时，
在 AI Core 上实现通信与计算重叠，并管理内部使用的 SHMEM 通信资源。

- **MegaMoE**：支持 push/pull 分发的专家并行前向与反向计算。
- **MegaGate**：支持自动求导的文本/视觉专家路由。
- **MegaMHC**：提供残差流混合的前向与反向算子。
- **诊断工具**：支持 kernel profiling 和 tensor dump。

## 快速开始

需要 Linux、Ascend NPU、与制品标签匹配的 Python/CPU 架构，以及彼此兼容的 Torch/torch-npu、CANN 和驱动。
源码构建使用当前激活的 CANN；矩阵中的参考 CANN 仅用于提示和追溯；运行时不强制 PTA post 下限或 CANN 版本配对。完整环境要求和源码构建方法见[构建与安装指南](docs/build.md)。

安装与当前环境匹配的 wheel。将下面的路径替换为实际 CANN 安装路径及构建命令打印的 wheel 路径：

```bash
source /path/to/cann/set_env.sh
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
hyper_multicore_build_info
```

将源码仓或解包后的 sdist 中的示例复制到独立目录，使用已安装包运行双卡 MoE 示例：

```bash
# 从源码根目录开始。
example_dir=$(mktemp -d)
cp examples/mega_moe_minimal.py "$example_dir/"
cd "$example_dir"
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc-per-node=2 mega_moe_minimal.py --dispatch-mode push
```

使用 `--dispatch-mode pull` 可切换为 pull 分发。示例执行前向与反向计算，
并输出形状、native 构建身份及耗时。通过 `--warmup` 和 `--iterations` 可增加采样次数。
完整网络接入与训练对照见 [examples 指南](examples/README.md)，包括
[MoE](examples/moe/README.md) 和 [MHC](examples/mhc/README.md)。

应用程序可直接导入公共组件：

```python
from hyper_multicore import MegaMoeExperts, MegaGate, MegaMhc
from hyper_multicore.diagnostics import MegaKernelDiagnostics
```

配置与使用方式见[组件指南](hyper_multicore/README.md)。使用算子的 Python 进程启动前，
需先激活 native 路径。pip 安装及 Python 导入均不会编译 native 代码。

## 开发与测试

Python 实现、设备 kernel 和 native binding 位于 `hyper_multicore/`。
HyperParallel 是下游消费方，构建、测试及使用本包均不依赖它。

从源码根目录安装开发所需工具并运行 CPU 单测：

```bash
python -m pip install -r requirements.txt setuptools wheel "packaging>=22"
python -m pip install pytest
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_repository.py
python -m pytest tests/ut -q
```

运行测试时，显式指定 `tests/ut`（CPU）或 `tests/torch`（NPU）；
不指定路径可能同时收集两者。[测试指南](tests/README.md)说明 CPU/NPU 用例的归属、
添加方式、执行命令及失败定位方法。

### 代码检查

三个 pre-commit hook 分别执行 Pylint、Lizard 和 Markdownlint。
Pylint 使用 3.3.7 及 `.pylintrc`，加载 `scripts/pylint_hypermulticore.py` 中的
许可证、公共函数类型注解与文档字符串、`sys.path` 和 subprocess 调用规则。
Lizard 要求圈复杂度不超过 15、函数有效行数不超过 100，并仅输出违规函数。
Markdown 规则见 `.markdownlint.jsonc`。

安装 pre-commit 后，可手动检查指定文件；需要提交时自动检查的开发者可启用 Git hook：

```bash
python -m pip install pre-commit
pre-commit run --files scripts/check_repository.py README.md
pre-commit install
```

pre-commit 为 hook 管理工具依赖。也可在当前 Python 环境中安装工具并直接运行包装器：

```bash
python -m pip install "pylint==3.3.7" lizard
python scripts/pre-commit/run_pylint.py scripts/check_repository.py
python scripts/pre-commit/run_lizard.py scripts/check_repository.py
```

Markdownlint 使用 Node 环境；已安装 `markdownlint-cli2` 时可直接运行：

```bash
markdownlint-cli2 --config .markdownlint.jsonc README.md README.en.md
```

Lizard、Markdownlint、Node 和 pre-commit 不固定版本；环境需满足所用工具的运行要求。
`.clang-format`、`hyper_multicore/CPPLINT.cfg` 和 `.codespellrc` 保留为相应工具的配置，
不属于上述三个 hook。

## 文档

| 指南 | 内容 |
| --- | --- |
| [构建与安装](docs/build.md) | 环境、构建命令、制品、安装与激活 |
| [组件使用](hyper_multicore/README.md) | 公共 API 与配置 |
| [测试](tests/README.md) | 用例范围、添加、执行与排错 |
| [架构](docs/architecture.md) | 运行时、调度与算子扩展 |
| [SHMEM](docs/shmem.md) | 通信资源与生命周期 |
| [诊断工具](docs/profiling.md) | Profiling 与 tensor dump |

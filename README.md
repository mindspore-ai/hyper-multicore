# HyperMulticore

简体中文 | [English](README.en.md)

HyperMulticore 是面向 Ascend NPU 的 megakernel 编程框架，用于将多个计算与通信操作组织到同一次
kernel 执行中。框架在设备侧管理任务依赖，减少独立 kernel 启动，并为计算与通信重叠提供更细粒度的调度。

开发者可以通过计算图或任务组装描述执行过程，也可以通过 PyTorch 接口将已有组件接入模型。
当前提供 MegaMoE、MegaGate、MegaMHC，以及任务时间线和张量快照诊断工具。

[Python 接口](hyper_multicore/README.md) · [构建与安装](docs/build.md) ·
[系统架构](docs/architecture.md) · [模型接入](examples/README.md) · [开发入口](AGENTS.md)

## 主要能力

- **[MegaMoE](docs/mega_moe.md)**：专家并行的分发、专家计算和结果汇合，支持前向与反向。
- **[MegaGate](docs/mega_gate.md)**：文本与视觉 token 的专家路由。
- **[MegaMHC](docs/mega_mhc.md)**：四路残差流的混合与归一化。
- **[诊断工具](docs/diagnostics.md)**：内部任务时间线与张量快照。

## 快速开始

### 获取并安装

运行需要 Linux、Ascend NPU、Python ≥3.10，以及配套的 Torch/torch-npu、CANN 和驱动。
先激活目标 Python 环境，安装[运行依赖](requirements.txt)，并准备与环境匹配的 wheel；
只有源码时，先按[默认构建](docs/build.md#默认构建)生成 wheel。

在 Bash 中执行以下命令，将占位路径替换为实际路径。`--no-deps` 不会补装运行依赖：

```bash
source /path/to/cann/set_env.sh
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
```

在启动导入 Torch/torch-npu 的 Python 进程前完成激活。环境兼容、包来源和完整性检查见
[安装与验证](docs/build.md#安装与验证)。

### 运行双卡 MoE 示例

[最小 MoE 示例](examples/mega_moe_minimal.py)需要两张可用 NPU 和可工作的 HCCL 环境。
按[示例运行说明](examples/README.md#运行单层-moe-示例)准备与 wheel 同版本的脚本，
在仅放置该脚本的源码外目录执行；设备编号 `0,1` 按实际环境调整：

```bash
(
    set -e
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    python -m torch.distributed.run --standalone --nproc-per-node=2 mega_moe_minimal.py
)
```

示例完成前向、反向和通信，并检查输出及梯度。成功时，rank 0 输出包含运行配置和耗时的 JSON。
这用于基础运行验证；精度与性能验证见[测试指南](tests/README.md)。

单卡调用可从 [MegaGate](docs/mega_gate.md#基本用法) 或 [MegaMHC](docs/mega_mhc.md#基本用法) 开始。

## 文档

| 需求 | 阅读位置 |
| --- | --- |
| 选择公开接口与配置 | [Python 接口](hyper_multicore/README.md) |
| 准备环境、构建和安装 | [构建指南](docs/build.md) |
| 接入已有模型与保留参数 | [模型接入与示例](examples/README.md) |
| 理解框架的职责与执行方式 | [系统架构](docs/architecture.md) |
| 了解 MegaMoE 的通算融合与任务调度 | [HyperParallel MegaMoE 技术博客](docs/blog/hyperparallel_megamoe.md) |
| 采集时间线与检查张量值 | [诊断指南](docs/diagnostics.md) |
| 准备开发环境、运行测试与静态检查 | [测试指南](tests/README.md) |

## 开发与贡献

修改实现前，请阅读[开发入口](AGENTS.md)，了解模块职责与开发约定。
开发环境、测试执行和静态检查见[测试指南](tests/README.md)。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。

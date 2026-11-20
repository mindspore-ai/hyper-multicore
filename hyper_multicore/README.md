# HyperMulticore Python 接口

HyperMulticore 是面向 Ascend NPU 的 megakernel 编程框架。本包提供可供应用组合使用的
PyTorch 接口；各特性的原理、用法和限制由对应文档说明。

## 公开接口

| 使用需求 | 主要入口 | 特性文档 |
| --- | --- | --- |
| 计算 token 的专家路由 | `MegaGate` | [MegaGate](../docs/mega_gate.md) |
| 执行四路残差流混合与归一化 | `MegaMhc` | [MegaMHC](../docs/mega_mhc.md) |
| 根据路由结果执行专家计算 | `MegaMoeExperts` | [MegaMoE](../docs/mega_moe.md) |
| 采集任务时间线 | `profiler.mega_kernel_profile()` | [诊断指南](../docs/diagnostics.md#采集任务时间线) |
| 采集调用或任务边界的张量 | `dump.tensor_dump()` | [诊断指南](../docs/diagnostics.md#采集张量快照) |

MegaMoE 的激活配置使用 `SwiGLUParameters` / `SiTUGLUParameters`，通用数值配置使用 `MegaMoeNumericsOptions`，字段与用法见[数值配置](../docs/mega_moe.md#模型数学与数值语义)。
各接口的精度支持和诊断能力见对应特性文档。

应用通过公共入口导入：

```python
from hyper_multicore import MegaGate, MegaMhc, MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.diagnostics import dump, profiler
```

使用 native 路径前需准备 CANN 并安装或构建、激活匹配的 payload，具体步骤见[构建与安装](../docs/build.md)。
设备、形状、精度与分布式要求见各特性文档；SHMEM 由需要通信的实现内部使用，不提供独立用户接口。

## 接口稳定性

`MegaMoeExperts`、`MegaMoeNumericsOptions`、`SwiGLUParameters` 与 `SiTUGLUParameters` 是公开稳定 API。
激活参数类型的公开不代表对应设备实现已经可用，支持范围见 MegaMoE 特性指南。

`MegaGate`、`MegaMhc` 和诊断采集接口是公开使用入口；当前文档未为其单独声明跨版本兼容性保证。
公开导出本身不等于稳定 API 承诺。

`MegaKernelDiagnostics` 仍从 `hyper_multicore.diagnostics` 公开导出，用于组件接入诊断运行时，
不是用户开启采集的入口，也未单独承诺跨版本兼容性。接入方式见[诊断设计](../docs/design/diagnostics_design.md#组件接入)。

`MegaMoeTuningOptions` 与 `MegaMoeSchedulerOptions` 从 `hyper_multicore.experimental` 导入，属于公开实验 API。
字段、可选值及默认执行策略不保证跨版本兼容，升级时需要重新验证。配置入口与支持范围见
[MegaMoE 调优配置](../docs/mega_moe.md#公开实验性调优配置)。

稳定与实验 API 都校验参数及支持范围；API 稳定性不代表跨硬件或跨版本逐 bit 数值一致。

## 阅读源码

阅读实现时，先从对应特性文档理解计算与调用方式；跨模块职责见[系统架构](../docs/architecture.md)，
源码职责地图与修改约定见[开发入口](../AGENTS.md)。

## 继续阅读

| 需求 | 文档 |
| --- | --- |
| 安装、构建与激活 | [构建指南](../docs/build.md) |
| 运行单层专家前反向 | [MegaMoE 独立示例](../examples/mega_moe_minimal.py) |
| 替换已有模块、保留参数和 checkpoint | [模型接入与示例](../examples/README.md) |
| 理解任务、执行配置与设备调度 | [系统架构](../docs/architecture.md) |
| 排查通信资源与重建问题 | [SHMEM](../docs/shmem.md) |
| 采集时间与中间张量 | [诊断指南](../docs/diagnostics.md) |

# HyperMulticore 开发入口

本项目提供 Ascend megakernel、PyTorch 接入和内部 SHMEM，供其他软件项目通过公共接口使用。本包独立构建、测试和运行，不依赖下游工程。

## 开发约定

代码应集中表达业务规则和状态，用当前需求足够的直接实现，让调用方少了解内部细节。目录与文件数量不是职责分离的尺度。修改代码时读取[架构规则](.agent/rules/architecture.md)和[代码质量](.agent/rules/code-quality.md)，据此选择落点与实现。

日常工作按[轻量变更流程](.agent/workflows/change.md)推进；只有需要解决设计取舍或恢复任务时才展开记录。修改文档或公开行为时使用[文档规则](.agent/rules/documentation.md)，测试与结果范围查[验证映射](.agent/references/validation-map.md)。

## 项目地图与任务导航

路径相对仓库根目录。按本次行为从公共入口、调用方和所属目录定位当前实现，缺少信息再扩展；无需遍历整张表，也不依赖固定的内部文件布局。

| 修改位置或行为 | 职责与参考入口 |
| --- | --- |
| `hyper_multicore/modules/`：修改组件接口、数学或梯度 | 从[Python 接口导航](hyper_multicore/README.md)找到对应特性指南，核对使用约定后定位实现与测试；新增组件按[既有边界的取舍](.agent/rules/architecture.md#修改既有边界时的取舍)判断落点 |
| 修改 MegaMoE 后端、调度或资源管理 | 先查 [MegaMoE 特性指南](docs/mega_moe.md)中的相关使用约定，再查 [MegaMoE 设计](docs/design/mega_moe_design.md)中的后端协作、任务生成或资源生命周期 |
| 理解框架，或修改 `scheduler/`、`tasks/`、`ops/runtime/`（包内） | 从[系统架构](docs/architecture.md)定位执行表示、运行时或状态说明，再按[执行边界规则](.agent/rules/architecture.md#数值任务编译与执行边界)核对实现与依赖计数 |
| `hyper_multicore/torch/`、`ops/hyper_*/`（包内） | 框架桥接与算子入口；职责边界查[架构规则](.agent/rules/architecture.md#职责与依赖方向) |
| `hyper_multicore/shmem/`：修改通信、分配或 runtime 重建 | 查 [SHMEM 说明](docs/shmem.md)中的接口与生命周期，按[通信与完成规则](.agent/rules/architecture.md#通信与完成)核对调用方和实现 |
| `hyper_multicore/_build/` | 锁源、构建与制品；查[构建参考](docs/build.md#制品与构建参考)，环境和加载查[环境准备](docs/build.md#环境准备)及[安装验证](docs/build.md#安装与验证) |
| 使用 profiler 或 tensor dump 排查问题 | 查[诊断指南](docs/diagnostics.md)中的采集、窗口、结果读取与排障说明 |
| `hyper_multicore/diagnostics/`：修改诊断实现或接入组件 | 查[诊断设计](docs/design/diagnostics_design.md)中的调用、存储与接入机制；采集行为变化时同时核对[诊断指南](docs/diagnostics.md) |
| `examples/`：修改模型替换、参数迁移或训练示例 | 查[模型接入指南](examples/README.md)及对应特性指南，核对参数归属、状态传递和训练条件 |
| `tests/`、`docs/` | 测试查[用例选择](tests/README.md#选择和执行用例)及[新增用例](tests/README.md#添加或扩展用例)；文档归属见[文档规则](.agent/rules/documentation.md#事实与维护位置)。测试、文档和示例不作为生产依赖 |

`.agent/rules/` 规定开发约束；系统架构和设计文档解释当前机制与职责；特性指南描述当前接口行为、使用条件和限制。

修改相关行为时，结合实现、调用方和测试核对，并同步受影响的说明。文档与实现不一致时应查明差异，不能仅凭其中一方改写既有约定。文档用途不代表接口稳定性承诺，兼容范围见[接口稳定性](hyper_multicore/README.md#接口稳定性)。

旧代码和规划报告不能自动成为新代码的模板。`.agent/` 通过本入口导航，不假设所有文件自动加载；指令本身不构成 CI 门禁。

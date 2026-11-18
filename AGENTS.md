# HyperMulticore 开发入口

本项目提供 Ascend megakernel、PyTorch 接入和内部 SHMEM，供其他软件项目通过公共接口使用。本包独立构建、测试和运行，不依赖下游工程。

## 开发约定

代码应集中表达业务规则和状态，用当前需求足够的直接实现，让调用方少了解内部细节。目录与文件数量不是职责分离的尺度。修改代码时读取[架构规则](.agent/rules/architecture.md)和[代码质量](.agent/rules/code-quality.md)，据此选择落点与实现。

日常工作按[轻量变更流程](.agent/workflows/change.md)推进；只有需要解决设计取舍或恢复任务时才展开记录。修改文档或公开行为时使用[文档规则](.agent/rules/documentation.md)，测试与结果范围查[验证映射](.agent/references/validation-map.md)。

## 项目地图与任务导航

路径相对仓库根目录。按本次行为从公共入口、调用方和所属目录定位当前实现，缺少信息再扩展；无需遍历整张表，也不依赖固定的内部文件布局。

| 修改位置或行为 | 职责与参考入口 |
| --- | --- |
| `hyper_multicore/modules/` | 组件接口、数学、调用和资源需求；从[组件使用指南](hyper_multicore/README.md)定位公共能力，新增组件按[既有边界的取舍](.agent/rules/architecture.md#修改既有边界时的取舍)核对接入方式 |
| `scheduler/`、`tasks/`、`ops/runtime/`（包内） | 图与任务编译、typed task、通用设备执行；沿组件调用链核对任务表示、编译与设备消费，事件变化按[执行边界](.agent/rules/architecture.md#数值任务编译与执行边界)检查 |
| `hyper_multicore/torch/`、`ops/hyper_*/`（包内） | 框架桥接与算子入口；职责边界查[架构规则](.agent/rules/architecture.md#职责与依赖方向) |
| `hyper_multicore/shmem/` | 通信与物理资源；从包的公共导出定位操作，沿调用方核对生命周期、重建和完成条件 |
| `hyper_multicore/_build/` | 锁源、构建与制品；查[构建参考](docs/build.md#制品与构建参考)，环境和加载查[环境准备](docs/build.md#环境准备)及[安装验证](docs/build.md#安装与验证) |
| `hyper_multicore/diagnostics/` | 通用观测；沿组件诊断接入查找执行与完成路径，检查范围见[验证映射](.agent/references/validation-map.md) |
| `tests/`、`docs/`、`examples/` | 验证、说明和用法，不作为生产依赖；查[测试选择](tests/README.md#选择和执行用例)及[新增用例](tests/README.md#添加或扩展用例) |

`.agent/rules/` 是当前开发规范。`docs/build.md` 是稳定的构建指南；`docs/` 中其他文档处于开发态，仅作背景参考，接口与示例须结合当前源码、调用方和测试核对。旧代码和规划报告不能自动成为新代码的模板。`.agent/` 通过本入口导航，不假设所有文件自动加载；指令本身不构成 CI 门禁。

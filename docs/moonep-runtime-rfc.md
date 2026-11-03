# [RFC]: 在 HyperMulticore 实现 MoonEP 动态专家借用

| 状态 | 设计评审稿，2026-11-03 |
| --- | --- |
| 实现范围 | hyper-multicore 的可选 MoonEP 运行时，Ascend A3 |
| 关联议题 | [HP 社区 MoonEP RFC #1128](https://github.com/mindspore-ai/hyper-parallel/issues/1128) |
| 评审决策 | 是否接受逐次前向规划、两种传输后端共用一份放置计划，以及由原参数所有者归并梯度的设计 |

## 动机（Motivation）

一次 Top-K 路由会为每个 token 选出若干专家。专家并行（EP）按 rank 分片持有专家权重。当同一次前向中的大量路由命中同一分片，耗时取决于最忙 rank 的专家计算。即使下一层的热点完全不同，本层其余 rank 也只能等待。社区议题提出在层执行期间规划 token 去向并预取临时专家副本。本设计具体定义何时值得复制、哪些 token 可以改派，以及训练时如何把梯度还给原参数。[社区 RFC #1128](https://github.com/mindspore-ai/hyper-parallel/issues/1128)

设计的约束来自训练语义。路由器选中的逻辑专家、Top-K 权重及有效路由条目必须保持一致。允许改变的是执行某个专家计算的物理 rank。原参数仍是 checkpoint 和优化器识别的对象，借用副本的生存期仅覆盖相关前向和反向。

## 目标设计

### 一次派发中的数据契约

设 EP 组有 R 个 rank、E 个全局专家，E 能被 R 整除，每个 rank 持有 B = E/R 个本地专家。下文的“行”指一个有效的 token–expert 配对；同一 token 的两个 Top-K 命中计作两行。

| 数据 | 形状与所有者 | 用途 |
| --- | --- | --- |
| 来源计数 C | [R, E]，按有序 EP 组排列 | C[s,e] 表示源 rank s 发给逻辑专家 e 的行数 |
| 借用表 copies | [R, B]，每次前向生成 | 每个借用槽存原全局专家 ID；负值表示空槽 |
| 虚拟编号 virtual_ids | 对应输入路由条目，保留 token 与 Top-K 位置 | 决定本次计算的目的 rank 和槽位 |
| 新计数 counts_by_source | [R, 2E] | 计算发送、接收分片长度及虚拟布局所需容量 |
| ExpertPlacement | 本次调用持有 | 一并保存虚拟编号、来源计数和设备端、主机端借用表 |

每个 rank 在虚拟编号空间内占用 2B 个槽。前 B 个对应本地专家，后 B 个容纳借来的专家。举例来说，R=2、E=4 时，rank 0 持有 e0、e1，rank 1 持有 e2、e3。假设路由行数为 [80, 0, 20, 0]，目标负载为每卡 50 行，且已通过收益判定。rank 1 可在第一个借用槽装入 e0，把 30 行 e0 的计算改派给自己。此时这些行使用虚拟编号 6，仍使用 e0 的权重和原路由分数。反向时，两卡对 e0 的贡献在 rank 0 汇总。数字仅用于演示映射。

### 收益判定与放置算法

1. **建立当前负载。** Native 后端对容量过滤后的专家计数做一次 all-gather，各 rank 得到相同的 C。MegaMoE 复用当前路由阶段已有的计数表。已知总行数时，目标负载取总行数除以 R 后向上取整；MegaMoE 缺少总数时，使用其计划 token 数乘 Top-K 作为保守目标。
2. **估算关键路径。** 令 Δ 为最忙 rank 超过目标的行数，H 为隐藏维度，I 为专家中间维度，N 为设备 cube 核数，W 为本地专家权重分片字节数。当前 A3 实现比较下列两个以秒为单位的估计量。系数 k 对带门控的三投影取 6，对双投影取 4。两项相等时沿用原路由。

   ~~~text
   T_saved    = k × Δ × H × I / (N × 4×10¹²)
   T_transfer = 0.0008 + 5 × W / (4×10¹¹)
   仅在 T_saved > T_transfer 时进入分配器
   ~~~

3. **分派超额工作。** 设备端分配器先统计各专家的全局行数，再按 rank 相对目标的盈亏配对。对每个输出配额，优先从剩余行数较多的本地专家划拨工作。每个接收 rank 至多选择 B 个远端专家副本，并生成渲染路由所需的累计边界。
4. **渲染并验证。** 路由渲染器根据原专家编号、专家内排列和累计边界生成虚拟编号。每个源 rank 的有效行数以及全组行数守恒；路由分数仍对应原 token 的 Top-K 位置。负载已均衡、估算收益不足或分配结果没有副本时，执行原后端的普通路由。

### 两种传输后端

| 阶段 | Native EP | MegaMoE |
| --- | --- | --- |
| 取得权重 | 在原专家模块进入 forward、FSDP 已完成 unshard 后，读取本次实际参数视图 | 执行器接收当前专家权重，按借用表向 SHMEM 专家缓冲区预取 |
| 运行专家 | 按 dtype 打包，通过变长 all-to-all 交换借用权重；按虚拟槽调用本地专家计算 | 按虚拟路由预留 token workspace 和专家缓冲区；沿用已有 push 或 pull 执行路径 |
| 返回梯度 | 自定义 autograd 按借用关系反向交换部分梯度，在 owner 侧以 FP32 累加后转换为原参数 dtype | 反向计划请求 FP32 权重梯度，SHMEM 归并远端贡献，返回原形状和 dtype |
| 释放或复用 | 反向释放本次调用的累加缓冲；保留图再次反向时重新累加 | 用租用状态、代次和 stream event 约束缓冲区复用；反向需要时重取权重 |

梯度归属遵循同一条关系：专家 e 的原参数接收所有执行 rank 对 e 的部分梯度之和。此处不再除以 EP 组大小。数据并行归约和优化器更新继续遵循上层框架已有的并行语义。

## 需要设计评审DFX建议

| 评审维度 | 设计取舍与验收依据 |
| --- | --- |
| 正确性 | 用相同权重、输入、路由 ID 和路由分数，对比普通执行与借用执行的输出、输入梯度、路由权重梯度和原参数梯度。覆盖空接收方、冻结参数、checkpoint 重算及 retained graph。 |
| 内存与并发 | 最大借用槽数受 B 限制。MegaMoE 按最终虚拟路由扩容；同一 SHMEM root 上不同缓冲区的传输用 event 排序。评审重点是延迟反向遇到后续前向时的租用与重取。 |
| 性能 | 固定专家布局没有规划及权重传输开销，适合均衡负载。逐次借用可以处理单层热点，代价包括计数同步、规划、权重传输和梯度返回。应同时测均衡、小负载和热点场景的完整前反向耗时。 |
| 替代方案 | 长期迁移热点专家能跨层次或跨步摊薄迁移开销，但需要管理持续变化的布局和参数状态。当前方案优先支持瞬时热点，保留所有者，以每次调用重新判断为代价。 |
| 可维护性 | 公共规划位于 hyper_multicore/moonep 的 placement、allocator、render 模块；native、precision、execution 管理 native 路径；mega_moe、shmem_transfer 和 csrc 管理融合路径。变更放置策略时，两后端须通过同一套路由守恒检查。 |
| 能力边界 | 当前构建包含面向 Ascend A3 的设备内核。INT32 路由范围、EP 拓扑和输入计数均需满足约束。MegaMoE 当前拒绝 stream capture。估算常数在其他设备上的适用性需要单独标定。 |

**评审待决事项。** 是否以当前 A3 代价模型作为首版启发式，并将其他设备的标定单独验收。

## 相关的RFCs和API

| 接口 | 契约 |
| --- | --- |
| build_placement(hidden, topk_ids, counts_by_source, capacity, rank) | 共享规划入口，输入当前调用的完整计数表，返回 ExpertPlacement 或 None。供后端内部使用。 |
| select_native_route(experts, states, expert_indices, local_counts, ep_group) | HP native 接入点，返回本次调用的 NativeDispatchSelection；返回对象负责排序、专家偏移和进入专家模块。 |
| run_local_experts(...) | 启用借用时，在原专家模块 forward 内使用 placement 提供的参数视图执行计算。 |
| MegaMoeExperts(..., enable_hot_replica=True) 的 forward(..., expert_weights=...) | 融合执行入口，接受外部参数视图；输出沿用已有 token 布局。 |

本篇细化社区议题中的运行时设计。配套《在 HyperParallel 接入 MoonEP》定义模型配置和 HP 侧调用边界。[社区 RFC #1128](https://github.com/mindspore-ai/hyper-parallel/issues/1128) · [社区 RFC 模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/ISSUE_TEMPLATE/003-RFC.yml)

## 完整的反馈期限

建议自公开征求意见之日起至少 7 天，具体截止日期由发布者填写。

## CC List

建议邀请 HyperMulticore MegaMoE、SHMEM、Ascend 算子以及 HP 专家并行维护者参与评审。

## 其他补充说明

本轮 A3 构建生成含分配器、SHMEM bridge 和传输内核的 hyper-multicore wheel。规划及 MegaMoE 相关单测 90/90 通过。已安装 wheel 的双卡 push 14/14、pull 7/7 通过。以上是迁移后的功能验证。原 PR 中每卡 4,096 token 热点场景约 2.45–2.48 倍的前反向加速来自迁移前的模块基准，本轮未重新测量训练吞吐。[原 PR 基准配置与结果](https://github.com/mindspore-ai/hyper-parallel/pull/1393)

### EP4 / EP8 连续反传的 BF16 权重梯度

在 CANN 9.1 上用 HP `06f3e9b7` 和 HyperMulticore `30ae689` 的 Python 实现配合已有原生二进制复测。保留 5 个不等长前向图并逆序反传时，MegaMoE + MoonEP 在 EP4、EP8 的 push 和 pull 模式下均有少量专家权重梯度元素超差。EIO、EOI 两种权重布局均可复现。相同输入下，普通 MegaMoE 通过。借用路径的前向输出、输入梯度和路由权重梯度也通过。

以 EP4 push 的 rank 0、第二层 gate/up 权重梯度坐标 `[1, 257, 74]` 为例，两条路径的 1154 对乘数逐项、逐序相同。在同一张 NPU 上用同一个 FP32 梯度算子 `npu_grouped_matmul_add_` 重放，只改变归约分组。

| 同一组乘数的归约方式 | FP32 结果 | 转为 BF16 |
| --- | ---: | ---: |
| 1154 行一次归约 | 2.195312738418579 | 2.203125 |
| 两组各 577 行分别归约，再以 FP32 相加 | 2.1953125 | 2.1875 |

分片结果恰好位于两个相邻 BF16 值的中点，按最近偶数舍入到较低值。整段归约结果稍高于中点，舍入到较高值。用实测单步梯度重放 BF16 累加，第 3 次反传时该元素在借用路径为 `1.03125`，参考路径为 `1.0625`。绝对差 `0.03125` 超过该元素的阈值 `0.002 + 0.02 × 1.0625 = 0.02325`。

本轮验证接受上述已定位的归约舍入差异，压力用例按原容差记录为失败。EP4 单次前向加反向对照的 36 个通过配置单独计数。[HP 接入与验证](https://github.com/mindspore-ai/hyper-parallel/pull/1393)

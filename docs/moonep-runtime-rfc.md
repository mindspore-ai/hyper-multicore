# [RFC]: HyperMulticore 的可选 MoonEP 专家放置与传输接口

| 项目 | 说明 |
| --- | --- |
| 状态 | 设计评审稿，2026-11-07 |
| 范围 | Ascend A3；规划、临时专家副本与梯度归还；不改变模型路由或参数所有权 |
| 关联 | [HP MoonEP RFC #1128](https://github.com/mindspore-ai/hyper-parallel/issues/1128) |

## 动机（Motivation）

固定 EP 分片下，一层 MoE 的耗时受最忙 rank 限制。MegaMoE 将激活传输和专家计算组织成细粒度任务以实现重叠，但不会自动消除专家所在 rank 的负载差异。MoonEP 根据本次路由将部分专家计算改派到较空闲的 rank：临时取得专家权重，处理原路由选中的 token，再将参数梯度还给原 owner。

MoonEP 不改变逻辑专家、选中的 Top-K 条目、路由概率、优化器或 checkpoint 的参数身份。均衡负载或预估收益不足时，执行原来的计算后端。

## 目标与边界

- MegaMoE 与 MoonEP 独立选择。普通专家计算与 MegaMoE 都可以使用同一官方规划实现；关闭 MoonEP 时保留原执行器。
- 调用方已有的计数协议提供 `[EP, E]` 计数表，不为判断是否搬运再执行一次计数集合通信。
- 调用方拥有进程组、SHMEM root、模型参数及 autograd；MoonEP provider 不创建第二套独立 SHMEM runtime。
- 当前官方 provider 要求各源 rank 的有效路由行数相同且非零。空源或不等行数沿用所选后端的无搬运路径（原生 MoE 或普通 MegaMoE）；不通过填充或改写数据假装支持。
- 官方 provider 与 HMC 原生 provider 是显式依赖选项，不通过修改全局函数切换其他执行器的行为。

## 数据契约

设 R 为 EP 大小，E 为全局专家数，B=E/R 为每个 owner 的本地专家数，b 为本次借用槽宽度，0≤b≤B。一条路由行对应一个 token–expert 配对。

| 对象 | 含义与生存期 |
| --- | --- |
| `counts_by_source[R,E]` | 当前调用的来源计数；所有 rank 使用同一有序 EP 组 |
| `copies[R,b]` | 每个借用槽对应的原全局专家 ID；负数为空槽 |
| `virtual_ids[tokens,K]` | 当前输入顺序下的执行位置，仍对应原 token 的 Top-K 槽 |
| 虚拟计数 `[R,R*(B+b)]` | 供后端计算派发分片、接收容量和专家偏移 |
| `ExpertPlacement` | 当前调用拥有的路由与副本元数据；反向保持其对应关系 |
| 对称专家 bank | 临时 home/borrowed 权重和梯度传输槽；不注册为模型参数 |

官方 provider 将借用宽度向上取整到受 B 限制的执行档位。工作区使用实际虚拟计数和实际宽度，不能把源 token 行数直接当作每个目的 rank 的接收上限。Top-K、热点、专家借用都会改变目的端的配对行数。

## 规划与执行流程

1. 原路由产生专家 ID、概率与计数。MegaMoE 在 `route.py` 的已有计数协议中调用选择回调；普通后端可复用其已收集的计数表。
2. `placement_is_profitable` 比较预估可减少的最忙 rank 计算量与规划、权重搬运和梯度归还成本。它是 A3 启发式模型，不是针对某个数据集的路由表。均衡或无收益时直接执行普通后端。
3. 使用官方 provider 时，`OfficialPlanner` 调用 `npu_patch._C.plan_from_counts`，再由 `render.py` 将专家内位置恢复到本次 token 顺序；缺少调用方逆排列信息时使用官方完整规划入口。HMC 自有 provider 的默认 `build_placement` 仍供其原调用方使用；TT 显式选择官方 provider，不串行运行两个分配器。
4. `OfficialPlanner` 的有界几何缓存只复用完全相同计数表下的专家配额。每次调用仍生成自身 token 顺序对应的虚拟 ID；缓存不持有模型参数或跨调用复用可变 autograd 状态。条目有版本、容量和跨流 event 检查。
5. 规划成功后一次预留虚拟执行工作区、planner scratch 和专家 bank。`heap_manager.py` 管理对称扩容及重绑，`PlanningWorkspace` 注册其重启分配契约。单纯增加源容量不能替代目的端行数检查。
6. 专家模块通过正常 forward/FSDP hook 提供本次真实参数视图。provider 预取借用权重；计算后端执行同一逻辑专家计算。原生 grouped GEMM 与 MegaMoE 共用规划语义，但计算调度各自保留。
7. 反向产生每个执行位置的参数梯度，provider 将远端部分送回原 owner，以 FP32 累加。回传可选择 FP32（默认）或 BF16；BF16 会额外量化部分梯度。最终梯度形状及 dtype 对应原参数，不除以 EP 大小。
8. bank 租用、代次与同 SHMEM root 的传输 event 防止后续前向或其他流提前覆盖；autograd 保存 owner 参数与本次计划，需要时重新预取，不保存可被后续调用覆写的临时权重视图。

## 模块职责和 API

| 模块 | 职责 |
| --- | --- |
| `modules/mega_moe/module.py` | `MegaMoeExperts`，接收外部权重及可选 planner/transfer provider |
| `moonep/mega_moe.py` | `MoonEPExecution`；普通/虚拟执行选择，保留 MegaMoE 调度 |
| `moonep/placement.py` | `ExpertPlacement`、共享收益判定、HMC 默认放置入口 |
| `moonep/official_planner.py` | `OfficialPlanner`、`PlanningScratch`；官方算法与计数/路由契约校验 |
| `moonep/official_execution.py` | `build_official_placement`、`OfficialTransferBackend`；将官方 API 接到现有 heap 和 lease |
| `moonep/official_transfer.py` | `ExpertTransfer`；普通 grouped GEMM 使用官方 bank/梯度归还 |
| `moonep/grouped_experts.py` | `TransferredExperts`；普通专家计算的可微参数与概率边界 |
| `moonep/shmem_transfer.py` | HMC bank pool、租用、同 root 事件排序和原 provider |
| 官方 `npu_patch` | 设备分配、prefetch、FP32/BF16 梯度归还 API 与原生内核 |

`MegaMoeExperts(..., enable_hot_replica=True, moonep_planner=..., moonep_transfer_backend=..., moonep_grad_reduce_dtype="fp32")` 是 HMC 执行器接入方式。上层 TT 的用户开关是 `mega_moe` 与 `enable_moonep`，不暴露 HMC 的内部字段名。传输 provider 保存在执行器及 pool key 中；其他执行器继续使用原 provider。

## DFX 与验收

| 维度 | 要求 |
| --- | --- |
| 正确性 | 相同输入、参数、逻辑 ID 和概率，对比输出、输入梯度、概率梯度及 owner 参数梯度；分别报告严格 BF16 和高精度参考结果 |
| 生命周期 | 覆盖空接收、冷专家、冻结参数、不同几何/流、连续 forward 后 backward、checkpoint、异常恢复与 CPU offload |
| 性能 | 计入规划、权重搬运与梯度归还；均衡、小负载、热点分别测试；不将强制搬运精度用例当生产收益 |
| 内存 | 对称扩容前校验全部计划的接收容量和存活 lease；持有的 autograd 图不能引用失效指针 |
| 兼容性 | provider 可选；关闭 MoonEP 保持原路径；不更换模型参数对象、优化器或 checkpoint 格式 |

当前验证针对 A3、私有 CANN 9.1 和匹配 Torch/PTA ABI；不代表 A2/A5、独立 CUDA/URMA 或全部官方 MoonEP 测试已通过。性能、功能与严格数值结果由对应版本测试报告提供，不在设计文档中合并不同版本的通过数。

## 反馈期限与评审参与者

建议公开征求意见至少 7 天；具体截止日期由发布者填写。建议邀请 HyperMulticore MegaMoE、SHMEM、Ascend 算子、官方 MoonEP 以及上层框架 EP 维护者评审。

## 已知数值边界

MegaMoE 通过 `apply_probs_before_fc2=True` 支持在 SwiGLU 后、FC2 前乘路由概率，TT 适配器显式选择此模式，与当前原生 `GroupedMLP` 一致。先保留 BF16 激活，再用 FP32 概率相乘并转回 BF16，输出归并不再乘概率；反向在该乘法位置计算概率梯度。概率与其梯度复用现有 dispatch/combine 完成事件，不增加集合通信。每份不可变 plan 按自己的 token 执行上界划分标量缓冲区，扩容后保留的旧 forward 仍使用原来的偏移。HyperMulticore 默认仍在 FC2 后乘概率，兼容原有调用方。乘法位置对齐与以下历史分片归约边界是两个独立问题，须分开验证。

### EP4 / EP8 连续反传的 BF16 权重梯度

在 CANN 9.1 上用 HP `06f3e9b7` 和 HyperMulticore `30ae689` 的 Python 实现配合已有原生二进制复测。保留 5 个不等长前向图并逆序反传时，MegaMoE + MoonEP 在 EP4、EP8 的 push 和 pull 模式下均有少量专家权重梯度元素超差。EIO、EOI 两种权重布局均可复现。相同输入下，普通 MegaMoE 通过。借用路径的前向输出、输入梯度和路由权重梯度也通过。

以 EP4 push 的 rank 0、第二层 gate/up 权重梯度坐标 `[1, 257, 74]` 为例，两条路径的 1154 对乘数逐项、逐序相同。在同一张 NPU 上用同一个 FP32 梯度算子 `npu_grouped_matmul_add_` 重放，只改变归约分组。

| 同一组乘数的归约方式 | FP32 结果 | 转为 BF16 |
| --- | ---: | ---: |
| 1154 行一次归约 | 2.195312738418579 | 2.203125 |
| 两组各 577 行分别归约，再以 FP32 相加 | 2.1953125 | 2.1875 |

分片结果恰好位于两个相邻 BF16 值的中点，按最近偶数舍入到较低值。整段归约结果稍高于中点，舍入到较高值。用实测单步梯度重放 BF16 累加，第 3 次反传时该元素在借用路径为 `1.03125`，参考路径为 `1.0625`。绝对差 `0.03125` 超过该元素的阈值 `0.002 + 0.02 × 1.0625 = 0.02325`。

本轮验证接受上述已定位的归约舍入差异，压力用例按原容差记录为失败。EP4 单次前向加反向对照的 36 个通过配置单独计数。[HP 接入与验证](https://github.com/mindspore-ai/hyper-parallel/pull/1393)

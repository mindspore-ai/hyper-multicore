# MoE 多核并行使用指南

Multicore 是独立的 Torch-only 组件，提供芯片内多核 MPMD 并行能力，结合核级内存语义单边通信，增强 MoE 通算掩盖和 MAC 利用率。

用户从独立组件 `hyper_multicore` 显式导入 `MegaGate` 或 `MegaMoeExperts`。
SHMEM 由 Multicore 在内部管理，不提供独立用户接口。

- [构建与交付](../docs/build.md)
- [架构设计与扩展](../docs/architecture.md)
- [MegaKernel Profiling 使用与设计](../docs/profiling.md)
- [私有 SHMEM 生命周期](../docs/shmem.md)
- [独立 MegaMoE 样例](../examples/mega_moe_minimal.py)
- [MegaMoe 路由端点与 shared expert 融合](../docs/mega_moe_route_fusion.md)

## 核心概念

多核并行是 HyperMPMD 的核心能力之一，从集群级 MPMD（Pipeline 并行）扩展到芯片内多核 MPMD：

- **O0**：通过框架层 host CPU 侧的调度，支持 cube、vector、单边通信算子分核执行
- **O1**：调度下沉到 AICore，支持 cube、vector、单边通信算子分核执行，进一步提升性能

本组件基于多核并行实现 MoE 通算掩盖（Multicore MoE-FFN）：将 MoE-FFN 的五个算子
（AllToAll-Dispatch、GMM1、SwiGLU、GMM2、AllToAll-Combine）融合为一个 kernel，由 AIC（AI Cube）和
AIV（AI Vector）核同时执行，实现通信与计算的细粒度重叠。

## 接口概览

多核并行模块位于 `hyper_multicore/`，包含以下组件：

| 组件 | 说明 |
|------|------|
| `modules/` | 多核并行模块实现 |
| `ops/` | 多核并行算子 |
| `scheduler/` | 多核并行调度器 |
| `tasks/` | 任务编排 |
| `torch/` | Torch native binding |
| `shmem/` | 仅供本组件使用的私有单边通信 |

源码构建生成一个同时包含正反向 kernel 的
`hyper_multicore_nn` vendor，并与构建环境对应的框架 adapter 一起进入 wheel 或本地 native payload。
Torch adapter 是通过 `torch.ops.load_library()` 加载的普通共享库。

---

## Torch managed API

### MegaGate

`MegaGate`提供可独立构造的Gate参数和返回值。NPU上的`sqrtsoftplus`路径使用native Route/RouteGrad，
投影MatMul由Torch和CANN选型；其他评分函数及非NPU环境使用Torch实现。使用native路径前须以
`bash build.sh`构建并激活生成的multicore payload。

```python
import torch

from hyper_multicore import MegaGate

gate = MegaGate(
    hidden_size=5120,
    num_experts=384,
    top_k=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    vision_enabled=True,
).to(device="npu", dtype=torch.bfloat16)

logits, routing_weights, expert_indices = gate(hidden_states, image_mask=image_mask)
```

`weight`默认使用标准差为0.02的Normal分布初始化，可通过`initializer_range`调整；文本和视觉correction
bias初始化为零。`hidden_states`形状为
`[batch, sequence, H]`。`logits`保持Projection输出dtype，`routing_weights`为FP32，专家索引为INT64。
传入`image_mask`时，mask形状须为`[batch, sequence]`；启用视觉bias时还须是同一NPU上的连续BOOL Tensor。
外层BF16 autocast开启时Projection产生BF16 logits；MegaGate将其转换为FP32后调用native Route。

### MegaMhc

`MegaMhc` 执行 shifted mHC 的 block 边界：将上一 block 的输出写入残差流，
预测供下一边界使用的混合系数，并产生当前 block 的 RMSNorm 输入。
所有输入和参数必须位于同一 NPU；使用前构建、安装并激活 native payload。

构造参数包括 `hidden_size`、`hc_eps=1e-6`、`norm_eps=1e-6`、`num_iters=20`、
`token_tile=32` 和 `device=None`。固定使用 4 路 residual streams、20 次 Sinkhorn；
`hidden_size` 必须是正的 128 倍数。模块构造始终要求 `H≤5760`，包括 `.eval()` 模式。
`token_tile` 必须是正整数。`device=None` 按 Torch 默认设备分配，实际执行前需迁移到 NPU。

设 `S` 为一个或多个 token 维度，展平 token 数为 `T=prod(S)`：

| 输入（按调用顺序） | 形状 | dtype |
| --- | --- | --- |
| `previous_output` | `[*S, H]` | BF16 |
| `residual` | `[*S, 4, H]` | BF16 |
| `previous_pre_mix` | `[*S, 4]` | FP32 |
| `previous_post_mix` | `[*S, 4]` | FP32 |
| `previous_residual_mix` | `[*S, 4, 4]` | FP32 |

返回值是固定顺序的五元组：

| 顺序 | 名称 | 含义 | 形状 / dtype |
| --- | --- | --- | --- |
| 1 | `new_residual` | 结合上一输出与残差混合矩阵后的残差流 | `[*S, 4, H]` / BF16 |
| 2 | `next_pre_mix` | 下一边界的输入混合系数 | `[*S, 4]` / FP32 |
| 3 | `next_post_mix` | 下一边界的输出分配系数 | `[*S, 4]` / FP32 |
| 4 | `next_residual_mix` | 下一边界的残差混合矩阵 | `[*S, 4, 4]` / FP32 |
| 5 | `block_input` | 用 **previous_pre_mix** 混合新残差并做 RMSNorm 的当前 block 输入 | `[*S, H]` / BF16 |

模块持有 FP32 的 `phi [24,4*H]`、`alpha [3]`、`bias [24]`，以及 BF16 的
`norm_weight [H]`。正常调用模块 `.bfloat16()` 或 `.to()` 时，`phi/alpha/bias` 及其
已有梯度会保留 FP32，并随模块迁移设备。直接手动修改这些参数的 dtype 不受该机制保护，
仍须满足上述混合 dtype 输入契约。

以下示例演示单卡前向和自动反向；示例 loss 仅用于展示五个输出参与 autograd，应用中使用真实网络 loss：

```python
import torch
import torch_npu

from hyper_multicore import MegaMhc

T, H = 128, 5120
device = torch.device("npu:0")
layer = MegaMhc(H, device=device)
previous_output = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
residual = torch.randn(T, 4, H, device=device, dtype=torch.bfloat16, requires_grad=True)
previous_pre = torch.full((T, 4), 0.25, device=device, dtype=torch.float32)
previous_post = torch.ones(T, 4, device=device, dtype=torch.float32)
previous_mix = torch.eye(4, device=device, dtype=torch.float32).expand(T, 4, 4).contiguous()
outputs = layer(previous_output, residual, previous_pre, previous_post, previous_mix)
new_residual, next_pre, next_post, next_mix, block_input = outputs
loss = sum(value.float().square().mean() for value in outputs)
loss.backward()
```

训练自动求导路径要求 `T≥max(40, 设备物理 AIV 核数)`；纯前向要求 `T≥1`。
纯前向函数入口为 `hyper_multicore.modules.mega_mhc.function.hyper_mega_mhc`，显式传入
五个输入及 `phi, alpha, bias, norm_weight`，设置 `need_backward=False` 后支持 `H≤12288`，
不建立匹配的反向图。超过模块构造上限时，直接按上述形状和 dtype 准备函数权重；
`.eval()` 或 `torch.no_grad()` 不会放宽模块构造限制。

形状、跨设备或尺寸错误抛 `ValueError`；非 Tensor、dtype 或非 NPU 输入错误抛 `TypeError`。
多 stream 调用前由调用方建立输入数据依赖。公共模块入口为 `MegaMhc`；
映射参数及其既有梯度在整网 dtype 转换时保持 FP32。

### MegaMoeExperts

Torch 模型通过 `MegaMoeExperts` 执行 Router 选出的专家，当前支持 Ascend NPU 上的 Torch BF16 训练。
先按[构建与交付](../docs/build.md)运行 `bash build.sh`，激活 CANN 和 native payload，
并为所有 EP rank 配置相同的 `HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT`（如 `tcp://<rank-zero-ip>:<port>`）。

在业务进程中选定当前 NPU、初始化 HCCL 进程组后，以下为两卡 EP 示例：

```python
import torch
import torch.distributed as dist
from hyper_multicore import MegaMoeExperts

experts = MegaMoeExperts(
    hidden_size=512,
    intermediate_size=128,
    num_experts=4,
    top_k=2,
    ep_size=dist.get_world_size(),
    ep_group=dist.group.WORLD,
).to(device="npu", dtype=torch.bfloat16)

output = experts(hidden_states, topk_ids, topk_weights)
```

permute / unpermute 与 shared expert 可按正反向独立选择，默认保留外部端点路径。
设置正的 `shared_intermediate_size` 后，可用 `fusion_mode="full"` 开启全部融合；
自动 token 接口不变。开关、权重布局、动态输入及最终相加的边界见
[融合使用说明](../docs/mega_moe_route_fusion.md)。

`ep_group` may also be a PP/DP-local EP subgroup; `ep_size` is inferred from it.
If supplied explicitly, `ep_size` must match the group size.
MegaMoE uses group-local expert ownership and does not take a TP-size argument.
Subgroups exchange independent CANN bootstrap IDs internally and do not use a
shared WORLD rendezvous port. See the [SHMEM lifecycle](../docs/shmem.md) for
cross-node interface selection and group-local shutdown requirements.

### 下游模型集成

DSV4.1、Qwen 和 TP/SP/CP/EP/FSDP 模型适配器及其集成测试由 HyperParallel 下游维护。
本仓只提供独立组件及其公共接口，不包含模型 YAML 或下游训练入口，也不依赖下游来完成测试。

### 输入与权重

- 分布式执行必须显式指定 `ep_group`，`E` 必须能被 EP 整除。
  token 数由每次实际输入决定，无需构造时指定长度或上限，各 rank 可不等长或为空。
  所有 EP rank 必须按相同顺序进入正反向，并使用一致的梯度参与策略。
  source/接收容量保留历史高水位，计划按几何桶缓存；待反向图保留原计划。
- `hidden_states` 扁平后的 token 数为 `T`，`topk_ids`、`topk_weights` 均为 `[T, K]`，
  ID 使用全局专家编号，同一个 token 的 K 个 ID 必须互不重复。
- 可选的 `tokens_per_expert` 是同 device 上 `[E]` 的精确本 rank histogram。
  省略时内部统计；传入时由调用方保证与当前 ID 一致。contiguous INT32 可直接使用，
  INT64 或非连续输入会转换。
- 默认 `expert_weight_layout="EIO"`：`gate_up_weight: [E/EP, H, 2I]`、
  `down_weight: [E/EP, I, H]`。EIO 非连续权重每次执行时转换为连续存储，梯度回到原参数。
  `expert_weight_layout="EOI"`：`gate_up_weight: [E/EP, 2I, H]`、`down_weight: [E/EP, H, I]`，
  必须使用连续存储，native 直接读取，梯度按相同物理布局返回。
  gate/up 顺序、SwiGLU 定义和 `swiglu_limit` 必须与来源模型一致。

Router 已有精确计数时，可直接传入：

```python
output = experts(
    hidden_states,
    topk_ids,
    topk_weights,
    tokens_per_expert=tokens_per_expert,
)
```

### 自动 token 与通信容量

MegaMoe 从实际 forward 输入自动维护容量，无需 token 上限或动态/静态模式开关。
构造时不分配通信 workspace，首次绑定从128行的最小 source token 容量开始，按实际需求增长并保留高水位。
调度计划按几何桶缓存，待反向图保留原计划；桶大小不改变真实 token 激活形状。
初次绑定、进入新桶及扩容有额外成本；稳定容量和已缓存计划可复用。

`dispatch_mode="push"` 为默认值，也可选择 `"pull"`；两者的 combine 都使用 PUT。
模式须在 EP rank 间一致，不能在同一 workspace 上切换。pull 在容量准备和 claim 后才写 source。

当前自动 token 模式下，push/pull 均要求 eager 执行，不支持 NPU graph capture；
即使 token 数和容量已稳定，检测到当前 stream 正在捕获时仍会抛出 `RuntimeError`。

| 参数 | push 默认值 | 作用 |
| --- | --- | --- |
| `initial_capacity_factor` | `1.25` | 首次接收容量相对预留 source 路由行数的倍数 |
| `capacity_growth_factor` | `1.25` | 接收容量增长倍数；`1.0` 仅满足当次需求 |

两个因子须是有限数且不小于1；省略或 `None` 采用默认值。pull 不接受显式容量因子。
即使 token 数固定，热点也可能触发 push 接收区扩容；路由不丢 token。
source 容量、调度行数与实际 owner 接收量分别维护：source/返回区按预留容量，
计算中间张量及待反向激活按实际接收量分配，空 owner 保留一行 ABI 占位。
pull 的接收区位于普通 HBM，不属于 SHMEM source/返回区。

扩容可能集体重启同一 SHMEM root 的 runtime，再恢复受管 workspace；外部专家参数不移动。
`HYPER_MULTICORE_SHMEM_HEAP_SIZE` 可设置固定字节预算；预算不足或不能安全重启时明确失败。
容量策略、固定预算和失败恢复的完整契约见 [SHMEM runtime restart](../docs/shmem.md#4-runtime-restart)。

### 资源共享与关闭

相同配置的串行层可在首次 forward 前共享 workspace，各层参数、梯度和待反向激活仍独立：

```python
MegaMoeExperts.share_execution_resources(layer.mlp.experts for layer in model.layers)
```

共享资源只允许串行提交；跨 stream 时调用方须建立输入 tensor 的依赖，不支持并发线程调用。
两条路径支持 non-reentrant checkpoint 和 `retain_graph=True`。默认由运行时托管 workspace，无需显式
关闭。模块 GC 仅登记使用权释放，不执行 collective；下一个资源组
首次 forward 时，所有 SHMEM Root rank 一起核对资源表，只复用各 rank 都无成员且无待反向图的兼容
workspace，其余可回收的孤儿 workspace 当场释放。此协调不在已绑定模型的每步 forward 中执行。
GC 时机不一致或仍有反向图会延后回收；持续保持活模型/图引用仍会占用内存，heap 受显式预算或物理 HBM 容量约束。

自动清理适配在首次绑定时安装：拦截 Torch `ProcessGroup.shutdown`（旧版本使用
`distributed_c10d._shutdown_backend`），在 WORLD 或 SHMEM Root 被关闭前先回收资源。
因此标准 `dist.destroy_process_group()` 及其提前导入的函数别名无需额外适配；不支持的 Torch
版本在 native 资源创建前报错。直接调用底层 HCCL/C++ 销毁接口不在此 Python 适配范围内。
另注册先于 torch_npu 退出钩子的 `atexit` 清理；所有 rank 正常退出时自动尝试回收。
这些边界仍要求所有 rank 同序进入且没有在途调用/待反向图。异常、Ctrl+C、kill 或通信故障
不承诺安全或及时退出，不从 signal handler 中执行 collective；SIGKILL 无法执行退出钩子。

需要提前释放时，可调用 `MegaMoeExperts.close()`，示例中的 `model.close()` 保持逐层关闭；
共享 workspace 在最后一个成员关闭时才释放，不释放普通模型参数。所有 SHMEM Root rank 须在无在途
调用及待反向图时同序关闭。关闭失败会保留句柄供安全条件下重试，部分关闭的模块不能再次执行。
只有所有 rank 均成功后才移除资源记录；对称内存部分释放状态不一致或原生 shutdown 失败须重启。
自动清理失败会阻止显式通信域销毁继续执行；进程退出时仅记录失败并继续框架退出流程，不能保证释放成功。

SHMEM Python层以进程级引用计数统一管理Runtime生命周期。整个MegaMoe子系统通过进程内私有Heap Manager
持有一份`shmem.acquire()`引用；execution resource只提交workspace reservation，不分别增减Runtime引用。
单个resource关闭时释放其workspace，资源池完成全部集体清理后再调用一次`shmem.release()`。
因此不同配置的孤儿workspace替换不会触发Runtime反复关闭和初始化；更大模型首次绑定可在这份引用内根据
实际物理heap容量执行Runtime restart。其他Multicore组件直接使用SHMEM时各自持有独立引用；存在第二个user
时MegaMoe不能独占并重启Runtime。
重新开启生命周期时，所有 rank 完成关闭后使用新的 `HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT`，
`HYPER_MULTICORE_SHMEM_HEAP_SIZE` 等堆配置在下一生命周期首次获取引用时重新生效。

独立运行方式见 [MegaMoe 示例](../examples/mega_moe_minimal.py)。Qwen 等模型集成留在下游。
MegaKernel 内部阶段采集方式见 [Profiling 指南](../docs/profiling.md)。

---

## 性能建议

1. **dispatch ↔ compute 掩盖**：MoE 的 AllToAll dispatch 与 expert compute 在不同核上并发，是最核心的掩盖收益
2. **单边通信**：基于内存语义的单边通信（Symmetric Memory）避免传统集合通信的同步开销
3. **RATR 通信重排**：通过 Rank-Aware Tile Reordering 将 AllToAll 流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标，降低尾延迟
4. **O0 vs O1**：O0 通过 host CPU 调度，O1 调度下沉到 AICore，性能更高但实现难度更大

### 从已有模块接入

`MegaMoeExperts.from_module(source, top_k=..., ep_group=...)` 接受已 materialize 的本地
SwiGLU `w1/w3/w2` 权重并复制为 EIO 或 EOI Parameters；不会迁移优化器状态。
`MegaGate.from_module(source, ...)` 绑定原 Router 的投影与修正偏置 Parameters，
`MegaMhc.from_module(source)` 绑定原四流 mHC 的参数与数学配置，不替换这些参数对象。
应在模块替换后创建优化器。Gate 修正偏置及 MHC 映射参数在 `.bfloat16()` / `.to()` 时保留 FP32。

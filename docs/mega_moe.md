# MegaMoE

`MegaMoeExperts` 接收 Router 选出的专家编号和概率，使用调用方持有的专家权重完成分发、专家计算与结果恢复。
MegaMoE 通过 megakernel 组织专家计算与通信，在任务依赖允许时重叠执行分发、计算和结果返回。
本文说明计算行为、调用方式、数值选择及资源使用。Router 评分和 TopK 选择由外部模型或 [MegaGate](mega_gate.md) 完成；
专家参数、优化器与参数分片也由模型管理。

| 阅读需求 | 位置 |
| --- | --- |
| 完成第一次调用 | [基本用法](#基本用法) |
| 检查输入、权重与返回值 | [接口说明](#接口说明) |
| 选择 BF16 或 MXFP8 | [精度支持](#精度支持) |
| 对齐模型数学与梯度 | [数值配置](#数值配置) |
| 管理容量、共享与关闭 | [训练与资源管理](#训练与资源管理) |
| 调整通信与任务执行 | [执行调优](#执行调优) |
| 修改任务、后端或资源实现 | [MegaMoE 设计](design/mega_moe_design.md) |

## 用途与计算关系

一次专家调用包含以下步骤：

```text
dispatch → up_proj → SwiGLU → down_proj → combine
```

在 EIO 权重语义下，单个专家的输入记为 X，两层权重为 W1、W2：

| 阶段 | 计算与数据关系 |
| --- | --- |
| dispatch | 按专家归属传输路由后的输入行 |
| up_proj | U = X · W1，得到 gate/up 两个分支 |
| SwiGLU | A = silu(gate) · up；裁剪及概率加权由数值选项决定 |
| down_proj | Y = A · W2 |
| combine | 将结果返回源 rank，再按组件选定的概率语义恢复 token 输出 |

这里的矩阵表达省略了 dtype 转换与存储布局。EOI 使用相应转置解释；数值选项决定裁剪位置和
路由概率在 FC2 前还是之后参与计算，调度策略不改变这些数学选择。
每个 token 最终得到一个 H 维输出，汇合所选 K 个专家的贡献。

## 基本用法

Torch 模型通过 `MegaMoeExperts` 执行 Router 选出的专家。`precision="bf16"` 为默认后端，
支持 `ascend910b`、`ascend910_93`（`arch22`）和 `ascend950`（`arch35`）。`precision="mxfp8"` 支持 `ascend950`。两种后端
使用相同的 forward 参数。先按[构建与交付](build.md)激活 CANN 和 native payload。

以下是两卡 EP 的调用片段。业务进程需先选定当前 NPU、初始化 HCCL 进程组，
并准备本 rank 的 `hidden_states` 及 Router 输出 `topk_ids`、`topk_weights`。
所有 EP rank 须按相同顺序参与调用，不能只在持有非空输入的 rank 上执行。
完整的初始化、前反向及退出流程见[独立运行示例](../examples/mega_moe_minimal.py)。

```python
import torch
import torch.distributed as dist
from hyper_multicore import MegaMoeExperts

experts = MegaMoeExperts(
    hidden_size=512,
    intermediate_size=128,
    num_experts=4,
    top_k=2,
    ep_group=dist.group.WORLD,
)

# 当前 EP rank 的模型参数，FSDP 场景需在专家模块的 unshard 后取得。
gate_up_weight = torch.nn.Parameter(torch.empty(2, 512, 256, device="npu", dtype=torch.bfloat16))
down_weight = torch.nn.Parameter(torch.empty(2, 128, 512, device="npu", dtype=torch.bfloat16))
torch.nn.init.normal_(gate_up_weight, std=0.02)
torch.nn.init.normal_(down_weight, std=0.02)
output = experts(hidden_states, topk_ids, topk_weights, expert_weights=(gate_up_weight, down_weight))
```

`ep_group` 也可以是 PP/DP 内的 EP 子组，本地专家数量及归属由该组的大小和组内 rank 顺序确定。
EP 大小从 `ep_group` 推导。独立单 rank 可省略 `ep_group`，
已初始化的多 rank 作业必须显式传入，包括使用单成员 EP 子组的情况。
MegaMoE 不接受 TP 大小参数。
跨节点网卡选择及组内关闭要求见 [SHMEM 生命周期](shmem.md)。

## 接口说明

模型执行和数值配置从 `hyper_multicore` 导入，执行调优配置从 `hyper_multicore.experimental` 导入：

| 公开接口 | 用途 | 使用约定 |
| --- | --- | --- |
| `MegaMoeExperts` | 配置模型尺寸和 EP 组，执行专家前反向并管理执行资源 | 公开稳定 API；专家参数由调用方持有，每次 forward 传入。 |
| `MegaMoeNumericsOptions` | 配置 SwiGLU 裁剪、路由概率的应用位置及 partial dW 精度 | 公开稳定 API；通过 `numerics_options=` 传入，按来源模型选择数值语义。 |
| `MegaMoePlacementOptions` | 配置专家放置策略 | 通过 `expert_placement=` 传入，默认使用原属 rank。 |
| `MegaMoeTuningOptions` | 配置通信模式、容量、分片和可选调度 | 公开实验 API；通过 `tuning_options=` 传入。字段、可选值及默认执行策略不保证跨版本兼容，升级时须重新验证。 |
| `MegaMoeSchedulerOptions` | 配置数学任务顺序、就绪扫描和 lowering | 公开实验 API；作为 `MegaMoeTuningOptions(scheduler=...)` 传入。 |

实验 API 同样校验参数和支持范围；API 稳定性不等于跨硬件或跨版本逐 bit 数值一致。
概率与 clamp 的位置由调用方明确选择，调度器不会自动改变这些数值语义。

构造 `MegaMoeExperts` 时指定模型尺寸和执行选项；构造本身不创建专家参数或分配通信 workspace。

| 构造参数 | 必填 / 默认值 | 含义 |
| --- | --- | --- |
| `hidden_size` | 必填 | 输入与输出的隐藏维度 H |
| `intermediate_size` | 必填 | 每个专家的 SwiGLU 中间维度 I |
| `num_experts` | 必填 | 全局路由专家数 E，须能被 EP 大小整除 |
| `top_k` | 必填 | 每个 token 选择的专家数 K |
| `ep_group` | `None` | 独立单 rank 可省略；分布式作业显式指定 EP 组 |
| `precision` | `"bf16"` | 可选 `"bf16"` 或 `"mxfp8"`，硬件与权重要求见精度支持 |
| `numerics_options` | `None` | 默认不裁剪、FC2 后加权，完整字段见数值配置 |
| `expert_placement` | `None` | 默认使用原属 rank，可选[自适应专家放置](#自适应专家放置) |
| `tuning_options` | `None` | 使用默认执行策略，公开实验配置见执行调优 |

本地专家数为 E/EP，专家归属按组内 rank 顺序确定。同一进程的 SHMEM 同时只能绑定一组有序的 EP 成员。

### 输入与权重

下文用 `E` 表示全局专家数、`EP` 表示专家并行组大小、`H` 表示隐藏维度、`I` 表示专家中间维度，
`K` 为每个 token 选择的专家数（`top_k`）。

设 `S` 为一个或多个 token 维度，`T=prod(S)`。本次输入、路由张量和本地专家权重位于同一 NPU。

| 参数或返回值 | 形状 | 类型与用途 |
| --- | --- | --- |
| `hidden_states` | `[*S, H]` | BF16；本 rank 的输入 token |
| `topk_ids` | `[T, K]` | INT32/INT64 专家编号；范围为 `[0, E)`，每行不重复 |
| `topk_weights` | `[T, K]` | 浮点路由概率；参与专家结果加权，可回传梯度 |
| `expert_weights` | 两个本地权重组成的 tuple | 每次调用必填；BF16 布局见下文，MXFP8 物理权重见精度支持 |
| `tokens_per_expert` | `[E]` | 可选的 INT32/INT64 精确本 rank histogram |
| 返回的 `output` | 与 `hidden_states` 相同 | 保持输入 shape 和 BF16 dtype，恢复到原 token 顺序 |

两种精度均按实际输入管理 token，各 rank 可不等长或为空。所有 EP rank 必须按相同顺序进入正反向，
并使用一致的梯度参与策略。

省略 `tokens_per_expert` 时内部统计；传入时由调用方保证与当前 ID 一致，不会重新统计来验证其内容。
连续 INT32 可直接使用，INT64 或非连续输入会转换。

BF16 的两个本地权重须采用相同布局：EIO 为 `([E/EP, H, 2I], [E/EP, I, H])`，
EOI 为 `([E/EP, 2I, H], [E/EP, H, I])`。首次使用时依据两组形状自动识别并绑定布局；
同一资源组和对应 EP rank 后续须保持相同布局。形状不能证明方阵是否被错误转置，
gate/up 顺序、SwiGLU 定义、clamp 位置和概率应用位置仍须由调用方与来源模型对齐。
EIO 非连续权重每次执行时转换为连续存储，梯度回到原参数；EOI 须连续，native 直接读取。
执行器不拥有参数或优化器状态；FSDP 应在参数所属模块的 forward 内传入当前 unshard 权重。

Router 已有精确计数时，可直接传入：

```python
output = experts(
    hidden_states,
    topk_ids,
    topk_weights,
    expert_weights=(gate_up_weight, down_weight),
    tokens_per_expert=tokens_per_expert,
)
```

### 自适应专家放置

为多 rank EP 的 Push 执行启用热点专家副本时，将下列配置传给 `MegaMoeExperts` 的 `expert_placement`。

```python
from hyper_multicore import MegaMoePlacementOptions
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner

placement = MegaMoePlacementOptions(strategy="adaptive", planner=HotReplicaPlanner())
```

- `ascend910_93` 支持 BF16 EIO 和 EOI。A3 MTE Push 在槽位和资源满足条件时使用分投影权重预取和融合梯度回传。
  带显式 `source_token_bound` 的内置策略按接收容量准入，使用相同的配额规划和路由描述。
  Reentrant checkpoint 重算沿用已保存的放置与权重快照协议。
- `ascend950` 支持 MXFP8 EIO，要求设备报告 32 个 Cube 核，使用内置 planner 的默认配置，
  并设置 `MegaMoeNumericsOptions(partial_weight_grad_dtype="float32")`。各 rank 启动前设置
  `HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable`，SHMEM 使用 `mte_udma`，
  `HYPER_MULTICORE_SHMEM_UDMA_QP_NUM` 设置为 8、16 或 32。
  每 rank 可借用的槽位上限为两个，且不超过本地专家数。

每次正向按实际路由决定是否启用副本。模型继续持有原属专家参数、优化器状态和 checkpoint，
反向把副本的梯度累加到对应原属专家。首次进入新容量或首次启用副本可能扩容 SHMEM，
测量稳态性能前应覆盖代表性的路由和容量。

## 精度支持

| 配置 | BF16 | MXFP8 |
|---|---|---|
| `precision` | `"bf16"`，默认 | `"mxfp8"`，`arch35` |
| `hidden_states` | BF16 | BF16；dispatch 后量化 |
| token 规格 | 按调用自动管理 | 按调用自动管理 |
| 权重 | 模型持有，每次 forward 传入 `expert_weights`；EIO/EOI | 模型持有的外部 MX root；EIO |
| H / I | 正整数；尺寸支持范围见下文 | 均须为 64 的正整数倍 |
| 激活与概率位置 | 支持 SiLU 前后裁剪、FC2 前或后加权 | 普通 SwiGLU、FC2 后加权；`swiglu_limit` 必须为 `None`，拒绝 FC2 前加权 |
| 调度选择 | arch22 支持 compact、nominal 和专家批处理；arch35 使用逐专家 dense | arch35 逐专家 dense；拒绝 compact、nominal 和 batching |
| 诊断 | task profiler 与适配范围内的 tensor dump | task profiler；不支持 tensor dump |
| `dispatch_mode` | `push` / `pull` | `push` / `pull` |

BF16 在 `arch22`、`arch35` 的公开前端均校验 H/I 为正整数，没有统一的 H/I 对齐倍数或上限校验。
这不意味着任意正整数尺寸均可执行：前反向还使用目标架构的 GMM、SwiGLU 及相应 tiling 路径，
当前尚未提供完整的尺寸支持表。基本示例使用 H=512、I=128；其他模型尺寸需按目标架构和权重布局验证，
不能套用 MXFP8 的 64 对齐条件作为 BF16 支持保证。
尺寸如何进入计算适配见[精度后端与计算适配](design/mega_moe_design.md#精度后端与计算适配)，
设备验证方式见[测试指南](../tests/README.md#npu-用例执行)。

### MXFP8 权重约定

`expert_weights` 传入两个原始 Tensor root，可以是 `torch.nn.Parameter` 或符合该约定的 Tensor 子类，
不能用普通字典或临时包装对象替代。root 代表模型持有的逻辑权重，并通过属性提供当前阶段的物理量化存储；
仅将 BF16 Tensor 转换 dtype 不会自动满足这个约定。

设本地专家数 `e=E/EP`。对任一权重，记逻辑形状为 `[e,M,N]`：

| 权重 | 逻辑 EIO 形状 | root 接受的形状 |
| --- | --- | --- |
| gate/up | `[e,H,2I]` | `[e,H,2I]` 或 `[e*H,2I]` |
| down | `[e,I,H]` | `[e,I,H]` 或 `[e*I,H]` |

M 和 N 必须均为 64 的倍数。root 上的物理字段按以下布局解释：

| 属性 | dtype | 消费时的视图 | 何时必须存在 |
| --- | --- | --- | --- |
| `col_data` | `torch.float8_e4m3fn` | `[e,M,N]` | 前向和反向 |
| `col_scale` | `torch.uint8`，物理 E8M0 编码 | `[e,M/64,N,2]` | 前向和反向 |
| `row_data` | `torch.float8_e4m3fn` | `[e,M,N]` | 反向 |
| `row_scale` | `torch.uint8`，物理 E8M0 编码 | `[e,M,N/64,2]` | 反向 |

当前阶段所需的物理 Tensor 必须连续，元素数量能按表中形状建立视图，并与 root、输入位于同一 NPU；
执行器不会通过隐式复制修复存储。scale 最后一维的两个值分别对应同一 64 元素组中的两个 32 元素量化块。
行向和列向数据分别与各自的 scale 配套，不能用同一份量化结果任意替代两种方向。

调用方负责量化并提供这些字段。FSDP 等参数管理机制应在前向、反向各自的 gather 完成后，
将当前物理字段更新到同一个原始 root 上；执行器保留 root，并在反向重新读取字段，
不保存临时 view 来替代当前权重。权重准备的完整回放入口见
[ascend950 独立回放](build.md#独立回放与队列诊断)。

### 通信与诊断支持

`ascend950` 的 SHMEM 默认启用 `mte_udma`，BF16 和 MXFP8 的专家通信随之使用 UDMA。
如需强制使用 MTE，在各 rank 启动前设置 `HYPER_MULTICORE_SHMEM_DATA_ENGINE=mte`。
`arch22` 默认使用 MTE，显式请求 UDMA 时会在设备绑定时报错。
`push` / `pull` 独立选择分发方向。堆预算和日志也沿用
[SHMEM 配置](shmem.md#6-配置与日志)。

调度可选项见 [MegaMoeSchedulerOptions](#调度选项)。完整构建、回放及免编译部署见
[ascend950 构建指南](build.md#ascend950-megamoe-源码构建)。

两种后端均支持无梯度执行及输入、权重和路由概率的梯度计算，精度选项不改变专家参数的归属。
具体诊断能力与采集方式见[诊断指南](diagnostics.md)。验证时分别检查实际精度、架构、Push/Pull 和采集模式；
某一种组合的结果不代表其他组合已通过。

## 数值配置

### 模型数学与数值语义

`MegaMoeNumericsOptions` 是公开稳定的数值配置接口，可直接从 `hyper_multicore` 导入。
类型定义位于 `hyper_multicore/modules/mega_moe/numerics.py`，由 MegaMoE 管理，与实验调优配置分开。
通过 `MegaMoeExperts(..., numerics_options=...)` 按模型逐层配置以下字段：

| 字段 | 默认值 | 语义 |
| --- | --- | --- |
| `partial_weight_grad_dtype` | `None` | `None` 和 `"native"` 使用后端默认的 BF16 partial dW；`"float32"` 使用 FP32 partial dW。最终梯度按调用方权重及梯度归并合同处理。 |
| `swiglu_limit` | `None` | 不进行 clamp；提供时须为正、有限且可由 FP32 表示的值。gate 分支只截上界，up 分支截到 `[-limit, limit]`。 |
| `clamp_after_silu` | `False` | `False` 在 SiLU 前截 gate；`True` 在 SiLU(gate) 后、乘 up 前截 gate 激活。仅在提供 `swiglu_limit` 时生效。 |
| `apply_probs_before_fc2` | `False` | `False` 在 FC2 后对专家输出加权；`True` 对 BF16 SwiGLU 激活加权，再送入 FC2。 |

```python
from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions

# 按来源模型选择数值语义；这里仅演示配置方式。
numerics = MegaMoeNumericsOptions(
    swiglu_limit=7.0,
    clamp_after_silu=True,
    apply_probs_before_fc2=True,
)
experts = MegaMoeExperts(
    hidden_size=512, intermediate_size=128, num_experts=4, top_k=2,
    ep_group=ep_group,
    numerics_options=numerics,
)
```

`clamp_after_silu` 和 `apply_probs_before_fc2` 必须是布尔值。
数值配置和调优配置都是不可变的 frozen dataclass；执行器在构造时校验并规范化配置，
不支持通过修改实例属性切换数值路径。省略 `numerics_options` 与传入
`MegaMoeNumericsOptions()` 使用相同的默认语义。
执行器保留 `swiglu_limit`、`clamp_after_silu` 和 `apply_probs_before_fc2` 三个只读查询属性；
改变语义应创建新的配置和执行器。

### 裁剪与概率加权

设 FC1 的 BF16 gate/up 输出为 `g`、`u`，clamp 上限为 `L`。转为 FP32 后，
两种 clamp 的计算顺序可写成以下数学示意，变量由前述定义给出：

```python
gate, up = g.float(), u.float()

# clamp_after_silu=False
activation = torch.nn.functional.silu(gate.clamp(max=L)) * up.clamp(-L, L)

# clamp_after_silu=True
activation = torch.nn.functional.silu(gate).clamp(max=L) * up.clamp(-L, L)
```

这里的 after 指 SiLU 之后，不是对完整 SwiGLU 乘积截断。两者定义不同的函数和梯度；
backward 和重算使用对应的 clamp 位置。`swiglu_limit=None` 时两种位置都走不截断路径。

`apply_probs_before_fc2=True` 保留两处 BF16 舍入：先将 SwiGLU 激活写为 BF16，
再以 FP32 乘对应路由概率，舍入为 BF16 后供 FC2 使用。其顺序可写成：

```python
activation_bf16 = activation.to(torch.bfloat16)
fc2_input = (activation_bf16.float() * probability.float()).to(torch.bfloat16)
```

`probability` 在此示意中按专家行广播。把概率直接融合进尚未舍入的 FP32 激活，再只转换一次，
会改变这个数值契约。FC2 前后乘概率即使在实数运算中等价，BF16 路径的舍入也可能不同。
默认的 `None`、`False`、`False` 表示不截断、FC2 后加权；需要其他语义时须显式设置，
并同时检查输出、输入梯度、权重梯度和路由概率梯度。

裁剪梯度在等号边界保留：gate 在裁剪变量不大于 `L` 时传递梯度，up 在
`-L <= u <= L` 时传递梯度；越过边界后相应分支梯度为零。SiLU 前模式的 gate
裁剪变量为 `g`，SiLU 后模式为 `SiLU(g)`。跨实现验收需同时核对这些边界和规约顺序。

公开的 `partial_weight_grad_dtype` 控制专家权重 partial dW 的写出精度。
最终 `.grad` 仍由叶子参数 dtype、autograd 和外层的分布式梯度归并决定。

### FC2 前加权的反向计算

BF16 路径的概率数学保留两次舍入：

```text
a0 = BF16(a)
a1 = BF16(FP32(a0) * p)
dp = sum(FP32(a0) * FP32(d_fc2_input))
d_a = BF16(FP32(d_fc2_input) * p)
```

其中 a 是 SwiGLU 激活，p 按专家行广播，dp 沿激活特征维规约。
反向使用未加权、已经舍入的 BF16 激活 a0 计算 dp，再将 FC2 输入梯度以 FP32 乘 p 并舍入为 BF16。
它通过重算恢复未加权激活，不执行除以 p；因此 p=0 时仍计算对应的 dp。
`dW2` 使用前向实际送入 FC2 的加权激活。checkpoint 重算执行相同的裁剪位置、概率位置和两处 BF16 舍入。

## 训练与资源管理

### 动态 token 与通信容量

执行器从实际 forward 输入维护容量，无需 token 上限或动态/静态模式开关。
初次绑定、生成新计划及扩容有额外成本；稳定容量和已缓存计划可复用。

默认使用 push 分发，接收容量初始及增长倍数均为 `1.25`，内部自动扩容，路由不丢 token。
通信模式、容量倍数和切分大小通过[公开实验性调优配置](#公开实验性调优配置)设置；省略 `tuning_options`
或传入默认的 `MegaMoeTuningOptions()` 使用相同的有效配置。

当前执行要求 eager 执行，不支持 NPU graph capture；
即使 token 数和容量已稳定，检测到当前 stream 正在捕获时仍会抛出 `RuntimeError`。

即使 token 数固定，热点也可能触发 push 接收区扩容。扩容可能集体重启同一 SHMEM root 的 runtime，
再恢复受管 workspace；外部专家参数不移动。
`HYPER_MULTICORE_SHMEM_HEAP_SIZE` 可设置固定字节预算；预算不足或不能安全重启时明确失败。
容量策略、固定预算和失败恢复的完整契约见 [SHMEM runtime restart](shmem.md#runtime-restart)。

### 资源共享与关闭

相同配置的串行层可在首次 forward 前共享 workspace，各层参数、梯度和待反向激活仍独立：

下面是模型内串行层的共享调用片段，`model` 由应用提供：

```python
MegaMoeExperts.share_execution_resources(layer.mlp.experts.executor for layer in model.layers)
```

共享资源只允许串行提交；跨 stream 时调用方须建立输入 tensor 的依赖，不支持并发线程调用。
共享组内的层必须具有相同的具体类型、规范化有效配置和权重布局；各 EP rank 对应层也须一致。
配置比较使用规范化后的字段值，不依赖 `MegaMoeNumericsOptions` 或 `MegaMoeTuningOptions`
的对象身份；省略配置与显式默认配置可以共享。
概率位置、clamp 位置和上限参与配置匹配，数值语义不同的层不能共享同一执行资源组。
支持 non-reentrant checkpoint 和 `retain_graph=True`。默认由运行时托管 workspace，无需显式
关闭。

GC 时机不一致或仍有反向图会延后回收；持续保持活模型/图引用仍会占用内存，heap 受显式预算或物理 HBM 容量约束。

#### 自动清理

标准 `dist.destroy_process_group()` 及其提前导入的函数别名会在相关通信组销毁前尝试回收资源，无需额外适配。
不支持的 Torch 版本在 native 资源创建前报错。直接调用底层 HCCL/C++ 销毁接口不在此 Python 适配范围内。
所有 rank 正常退出时自动尝试回收资源。
这些边界仍要求所有 rank 同序进入且没有在途调用/待反向图。异常、Ctrl+C、kill 或通信故障
不承诺安全或及时退出；SIGKILL 无法执行退出钩子。

#### 显式关闭与失败处理

需要提前释放时，可调用 `MegaMoeExperts.close()`，示例中的 `model.close()` 保持逐层关闭；
共享 workspace 在最后一个成员关闭时才释放，不释放普通模型参数。所有 SHMEM Root rank 须在无在途
调用及待反向图时同序关闭。关闭失败会保留句柄供安全条件下重试，部分关闭的模块不能再次执行。
只有所有 rank 均成功后才移除资源记录；对称内存部分释放状态不一致或原生 shutdown 失败须重启。
自动清理失败会阻止显式通信域销毁继续执行；进程退出时仅记录失败并继续框架退出流程，不能保证释放成功。

重新开启生命周期时，所有 rank 完成关闭后使用新的 `HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT`，
`HYPER_MULTICORE_SHMEM_HEAP_SIZE` 等堆配置在下一生命周期首次获取引用时重新生效。

## 执行调优

### 公开实验性调优配置

普通模型使用 `MegaMoeExperts` 的默认配置。需要调优时，从 `hyper_multicore.experimental`
显式导入 `MegaMoeTuningOptions`，仍使用同一个 `MegaMoeExperts` 入口。
配置对象是不可变的 frozen dataclass，包含以下字段：

| 参数 | 默认值 | 作用与约束 |
| --- | --- | --- |
| `dispatch_mode` | `"push"` | `"push"` 用 PUT 分发，`"pull"` 用 GET 分发；两者的 combine 均使用 PUT。 |
| `initial_capacity_factor` | `None`；push 实际为 `1.25` | push 初始接收行数相对预留 source 路由行数的倍率；须为有限数且不小于 `1.0`，pull 不接受显式值。 |
| `capacity_growth_factor` | `None`；push 实际为 `1.25` | push 中 source token 和接收容量的扩容倍率；`1.0` 按当次需求对齐扩容。须为有限数且不小于 `1.0`，pull 不接受显式值。 |
| `dispatch_split` | `128` | 正反向 dispatch 的通信分片行数，须为 `128` 的正整数倍。 |
| `combine_split` | `128` | 正反向 combine 的通信分片行数，须为 `128` 的正整数倍。 |
| `scheduler` | `None` | 可选 `MegaMoeSchedulerOptions`，保留硬件适用的默认调度。 |

push 初始接收容量按 `ceil(min(initial_capacity_factor, EP) × 预留 source token 数 × top_k)`
计算后向上对齐到 `128` 行，`EP` 为 `ep_group` 的成员数；首次绑定的预留 source token 数为 `128`。
两个容量因子不接受布尔值。
动态 token 调用中，pull 的 source 容量仍由内部按 `1.25` 增长；显式 heap 预算不足以支持几何扩容时，
运行时会尝试只满足本次最小需求，仍不足则报错。不同策略的性能取决于实际负载，须分别测量。

```python
import torch.distributed as dist
from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions

experts = MegaMoeExperts(
    hidden_size=512, intermediate_size=128, num_experts=4, top_k=2,
    ep_group=dist.group.WORLD,
    tuning_options=MegaMoeTuningOptions(
        dispatch_mode="pull", dispatch_split=128, combine_split=128,
    ),
)
```

执行器在构造时绑定配置，改变执行策略应创建新的配置和执行器。
同一个 EP 组的对应层必须使用一致的有效配置，同一个 workspace 不能切换通信模式。
参数、checkpoint 与优化器状态由外层模型管理，接入方式见
[Checkpoint 与参数归属](../examples/README.md#checkpoint与参数归属)。

### 调度选项

`MegaMoeSchedulerOptions` 从 `hyper_multicore.experimental` 导入，通过
`MegaMoeTuningOptions(scheduler=...)` 为单个 `MegaMoeExperts` 实例配置。
省略 `scheduler` 或传入空对象都使用设备和精度对应的默认策略。
各 EP rank 上同一层的配置须一致。

| 参数 | 可选值 | BF16 · `arch22` | BF16 / MXFP8 · `arch35`（`ascend950`） |
| --- | --- | --- | --- |
| `policy` | `None`（默认）、`"two_expert_pipeline"`、`"builder_order"` | 用 `None`，由 lowering 选择队列。指定策略名会报错。 | `None` 等同 `"two_expert_pipeline"`，交错相邻专家的数学阶段。`"builder_order"` 保留建图时的数学任务相对顺序。两者保持相同的任务覆盖、核归属和通信队列。 |
| `queue_ready_lookahead` | `None`（默认）、整数 `0`、`1`、`8` | compact 的窗口只作用于 Vector 队列，`None` 保留默认 `256`，Cube 始终 FIFO。dense 使用 FIFO，接受 `None` 或 `0`。 | `None` 等同 `0`。窗口作用于 Cube 和数学 Vector 队列，通信 Vector 始终 FIFO。 |
| `batch_experts` | `None`（默认）、`False`、`True` | `None` 等同 `False`。`True` 将每个 GMM 阶段的逐专家描述符合并为批处理，取消逐专家流水重排。dense 与 compact 均可用，compact 保留 Vector 就绪扫描。 | 接受 `None` 或 `False`。`True` 会报错。 |
| `nominal_schedule` | `None`（默认）、`False`、`True` | `None` 等同 `False`。`True` 按名义均分行数限制 compact 的 lane 数，并跳过描述符压缩倍率门槛。要求 Push、`intermediate_size < 2048` 且 lowering 为 `"auto"` 或 `"compact"`。 | 接受 `None` 或 `False`。`True` 会报错。 |
| `lowering` | `"auto"`（默认）、`"dense"`、`"compact"` | `"dense"` 保留固定分片。`"auto"` 在 Push、`intermediate_size < 2048` 且预计描述符数至少压缩四倍时选 compact，否则用 dense。显式 `"compact"` 要求满足这些条件，`nominal_schedule=True` 时免于倍率判定。 | `"auto"` 与 `"dense"` 都使用逐专家 dense。`"compact"` 会报错。 |

`0` 按队首顺序执行，`1` 通过就绪检查只看队首，`8` 在当前 worker 的前 8 个槽中取最靠前的就绪任务。
`256` 是 `arch22` compact 的内部默认值，使用 `None` 选择。窗口按单个 worker 队列中的任务槽计数，
参与执行的 AIV 核数由设备规格和任务放置决定。compact 使用持久 lane 遍历运行时有效分片，
`nominal_schedule` 调整参与分片的 lane 数，所有有效行仍会处理。

例如，在 `ascend950` 上为 BF16 或 MXFP8 使用同一个调度配置：

```python
from hyper_multicore.experimental import MegaMoeSchedulerOptions, MegaMoeTuningOptions

tuning = MegaMoeTuningOptions(
    dispatch_mode="pull",
    scheduler=MegaMoeSchedulerOptions(policy="builder_order", queue_ready_lookahead=8),
)
# 构造 MegaMoeExperts 时传入 tuning_options=tuning。
```

在 `arch22` 上明确要求 BF16 Push 使用 compact 批处理和 8 槽 Vector 扫描：

```python
tuning = MegaMoeTuningOptions(
    scheduler=MegaMoeSchedulerOptions(
        lowering="compact",
        nominal_schedule=True,
        batch_experts=True,
        queue_ready_lookahead=8,
    ),
)
```

构造阶段校验参数类型和值，并拒绝 MXFP8 的 compact、nominal 和 batching 请求。
设备绑定及构图阶段校验架构和实际分桶。显式请求 compact 或非零 Vector 窗口时，
若某个分桶只能使用 dense，会报错。`arch22` 使用策略名也会在构图时报错。
`arch35` 的具名策略和数学窗口对 MTE 与 UDMA 都生效。

## 模型接入

具体模型及 TP/SP/CP/EP/FSDP 适配器与集成测试由使用本包的各下游项目维护。
本仓只提供独立组件及其公共接口，不包含模型 YAML 或下游训练入口，也不依赖下游来完成测试。

`MegaMoeExperts` 不创建或转换专家参数。已有模型应在参数持有模块的 forward 内传入当前本地专家权重；
若 gate/up 分开保存，模型侧需按正确顺序合并或调整参数布局，并处理对应的 checkpoint 与优化器状态。
应在模块替换后创建优化器；完整接入及 checkpoint 处理见[模型接入指南](../examples/README.md)。

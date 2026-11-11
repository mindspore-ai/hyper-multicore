# MegaMoe permute / unpermute / shared expert 融合

`MegaMoeExperts` 可分别选择正反向的路由端点与 shared expert 融合。
公共接口仍从每次输入自动确定 token 数，不需要构造时指定长度、上限或动态模式。
先按[构建与安装](build.md)生成并激活匹配当前源码的 native payload；参数与分布式初始化的基础说明见
[MegaMoeExperts 使用指南](../hyper_multicore/README.md#megamoeexperts)。

## 开关与默认行为

| 构造参数 | 可选值 | 未设置环境变量时的默认值 |
| --- | --- | --- |
| `forward_fusion_mode` | `legacy`、`permute`、`unpermute`、`both` | `legacy` |
| `backward_fusion_mode` | `legacy`、`permute`、`unpermute`、`both` | `legacy` |
| `shared_intermediate_size` | 非负整数；`0` 不创建 shared 分支 | `0` |
| `forward_shared_expert` | `True`、`False` | `False` |
| `backward_shared_expert` | `True`、`False` | `False` |
| `fusion_mode` | `None`、`"full"` | `None` |

正向 `permute` 让 Dispatch 按标量路由索引直接读取原始输入，省去外部的展开激活张量；
正向 `unpermute` 将 weighted unpermute 放入 Combine 后的 kernel epilogue。
反向 `permute` 在 Dispatch 中读取 token-major dY 并应用路由权重，路由权重梯度也在同一个 kernel 中计算；
反向 `unpermute` 在 Combine 后归并 Top-K 路径的输入梯度。
`both` 同时开启对应方向的两个端点，`legacy` 保留外部端点实现。
这些开关不移除专家间 Dispatch / Combine 通信，也不保证训练所需的 saved tensors 随之消失。

shared 分支的正反向开关独立。正向关闭、反向开启时，shared 正向由框架计算，
其激活交给融合反向；正向开启、反向关闭时，反向通过框架重算 shared 分支。
两个开关均关闭且 shared 宽度非零时，该分支完整走外部框架 autograd。
shared 宽度为零时，不允许开启任一 shared 融合开关。

`fusion_mode="full"` 将路由正反向都设为 `both`，并同时开启 shared 正反向。
它要求 `shared_intermediate_size > 0`；显式传入相冲突的单独开关会报错。
默认值保持现有外部路径，是否启用融合应在目标形状、路由分布与训练调用方式下实测。

## 调用示例

在已选定当前 NPU、初始化 HCCL 并配置 SHMEM bootstrap 的进程中：

```python
import torch
import torch.distributed as dist

from hyper_multicore import MegaMoeExperts

experts = MegaMoeExperts(
    hidden_size=5120,
    intermediate_size=128,
    num_experts=4,
    top_k=2,
    ep_group=dist.group.WORLD,
    shared_intermediate_size=256,
    fusion_mode="full",
).to(device="npu", dtype=torch.bfloat16)

output = experts(hidden_states, topk_ids, topk_weights)
```

每次调用的 `hidden_states` 可以具有不同的 token 数；`topk_ids` 和 `topk_weights` 必须对应
本次展平后的 `[T, K]`，每个 token 的 K 个专家 ID 互不重复。要独立选择开关，将上述
`fusion_mode="full"` 替换为需要的四项，例如：

```python
forward_fusion_mode="both",
backward_fusion_mode="unpermute",
forward_shared_expert=False,
backward_shared_expert=True,
```

该片段是构造参数，不是在构造后修改已有模块。不同开关配置需要分别构造模块；
显式共享 execution resources 的模块须满足相同配置约束。

使用外部参数时，构造设置 `create_parameters=False`，每次 forward 传入：

```python
output = experts(
    hidden_states,
    topk_ids,
    topk_weights,
    expert_weights=(gate_up_weight, down_weight),
    shared_gate_up_weight=shared_gate_up_weight,
    shared_down_weight=shared_down_weight,
    tokens_per_expert=tokens_per_expert,
)
```

`tokens_per_expert` 可省略；提供时必须是本 rank 当前输入的 `topk_ids` 对全部 E 个全局专家的
精确 histogram，形状为 `[E]`、与输入同 device；不是跨 rank 汇总后的接收计数。
路由专家权重布局由 `expert_weight_layout` 决定，shared 权重始终使用以下 Torch linear 布局：

| 参数 | 形状 |
| --- | --- |
| `gate_up_weight`，`EIO`（默认） | `[E/EP, H, 2I]` |
| `down_weight`，`EIO`（默认） | `[E/EP, I, H]` |
| `gate_up_weight`，`EOI` | `[E/EP, 2I, H]` |
| `down_weight`，`EOI` | `[E/EP, H, I]` |
| `shared_gate_up_weight` | `[2S, H]` |
| `shared_down_weight` | `[H, S]` |

其中 `S=shared_intermediate_size`。权重与输入须位于同一 NPU、使用 BF16；shared 和 EOI 权重
必须连续。shared gate/up 按前后两半打包。`swiglu_limit` 只控制 routed experts；shared 分支保持
`linear(silu(gate) * up)`，不应用该 clamp。

## 环境变量与优先级

非 `None` 的单项显式构造参数优先于对应方向的环境变量，再优先于公共环境变量，最后使用上述默认值。
`fusion_mode="full"` 作为显式预设优先于环境变量；所有 EP rank 必须解析为相同配置。

| 环境变量 | 作用 |
| --- | --- |
| `HYPER_MULTICORE_MEGA_MOE_FUSION_MODE` | 未单独指定的路由正反向模式 |
| `HYPER_MULTICORE_MEGA_MOE_FORWARD_FUSION_MODE` | 路由正向模式 |
| `HYPER_MULTICORE_MEGA_MOE_BACKWARD_FUSION_MODE` | 路由反向模式 |
| `HYPER_MULTICORE_MEGA_MOE_SHARED_EXPERT` | 未单独指定的 shared 正反向开关 |
| `HYPER_MULTICORE_MEGA_MOE_FORWARD_SHARED_EXPERT` | shared 正向开关 |
| `HYPER_MULTICORE_MEGA_MOE_BACKWARD_SHARED_EXPERT` | shared 反向开关 |

shared 环境变量接受 `1/true/yes/on` 或 `0/false/no/off`，不区分大小写。
环境变量在模块构造时解析，修改环境后须重新构造模块。

## 动态输入、通信与整合边界

- push 和 pull 均沿用自动容量管理。各 rank 的实际 `T` 可以不同或为零；接收量为零的 rank
  仍参与通信与同步。空 expert 的权重梯度为零。
- source/接收缓冲区按需求增长并保留高水位。通信计划覆盖 EP 组的最大长度，融合端点和 shared
  计算使用本 rank 的实际长度，包含非 128 整除的尾块；返回值保持真实输入形状。
- 待反向图持有各自的计划与必要激活，后续变长调用和容量增长不会把旧图的实际长度改成新长度。
  `retain_graph=True` 的反向可重复执行；全部使用结束后释放图，再调用 `experts.close()`。
  所有 EP rank 仍须按相同顺序进入正反向，并采用一致的梯度参与策略。
- 路由端点融合要求 `K <= 16` 且每个 rank 的实际 `T*K <= 2**24`；同时保留自动容量路径的
  native INT32 尺寸校验。自动 push/pull 都只支持 eager 执行，不支持 NPU graph capture。
- 同一 workspace 不支持并发调用，push/pull 模式不能就地切换；具体容量因子与生命周期约束见
  [自动 token 与通信容量](../hyper_multicore/README.md#自动-token-与通信容量)。

正向开启 `unpermute` 且至少开启一个 shared 融合方向时，weighted unpermute epilogue
在 kernel 内完成 routed 输出的 BF16 舍入及与 shared 输出的相加。
这包括“shared 正向关闭、反向开启”的情况：此时 shared 正向先在 host 侧调度框架算子，
最终相加仍由 epilogue 完成。未开启正向 `unpermute`，或 shared 正反向都关闭时，最终相加在外部完成。
反向只有在 `unpermute` 与 shared 反向都开启时，才在 kernel 内合并 routed/shared 的 dX。
因此，开启 shared 投影融合与开启最终相加融合是两个不同的条件；`fusion_mode="full"` 同时满足两者。

融合路径在普通最终一阶反向中，可将自身保存的 BF16 routed `up_proj` 原位复用为
`swiglu_grad`，省去一个 `[实际接收行数, 2I]` 临时分配；无需新增开关。
保留图、启用高阶梯度、saved-tensor hooks（包括前向后注册）以及无法确认所有权时，
均使用独立缓冲，不改变原有开关、算子图和梯度计算顺序。

## 验证入口

从仓库根目录运行 CPU 调度、ABI 与参数检查：

```bash
python -m pytest tests/ut/modules/mega_moe tests/ut/scheduler tests/ut/torch/test_ops.py -q
```

激活当前源码构建的 native payload 后，选择空闲 NPU 执行真实设备测试。launcher 自行启动分布式
worker，不要再套一层 `torchrun`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest \
  tests/torch/test_mega_moe_route_fusion.py \
  tests/torch/test_mega_moe_shared.py::test_dynamic_shared_fusion \
  tests/torch/test_mega_moe_backward_memory.py -q

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m pytest \
  tests/torch/test_mega_moe_shared.py::test_shared_expert_numerics -q
```

这些入口分别覆盖 indexed pull 路由组合、push/pull 的变长与混合开关、零输入/零接收 rank、
扩容前保留图、重复反向，以及路由与 shared 的 64 种开关组合和输出/完整梯度对照。
测试环境、日志与证据目录说明见[测试指南](../tests/README.md)。

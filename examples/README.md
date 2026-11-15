# Multicore 模型接入指南与示例

本文说明如何在已有 Torch 模型中替换组件、迁移参数并接入训练。
组件能力、设备要求、动态 token 及资源生命周期见 [Multicore 使用指南](../hyper_multicore/README.md)，
安装环境见 [构建说明](../docs/build.md)。

示例由源码仓或 sdist 提供，不作为 wheel 运行资源。安装组件后，可将整个 `examples/` 复制到
源码外的空目录，从其父目录运行 `python -m examples.mhc.benchmark` 或对应 torchrun 命令。
`hyper_multicore` 应来自所选安装环境；示例不依赖 HyperParallel。

| 需求 | 阅读位置 |
| --- | --- |
| 安装后快速运行单层 MoE | [minimal](mega_moe_minimal.py)，支持 `--warmup` 和 `--iterations` |
| 替换已有 Router、专家或四流 mHC | 下方[模块与权重迁移](#已有模块与权重迁移) |
| 保留 checkpoint 和参数管理 | [Checkpoint与参数归属](#checkpoint与参数归属) |
| 运行仅含 MoE 的完整训练对照 | [MoE Transformer](moe/README.md) |
| 运行仅含 mHC 的完整训练对照 | [Dense mHC Transformer](mhc/README.md) |

以下片段使用已加载的原模块和真实模型输入，EP 组、模块替换及跨层状态由原模型提供。
三个组件可以分别接入；mHC 仅用于已有四流 shifted 连接结构。

## 已有模块与权重迁移

### Gate：保留评分函数与原参数

```python
from hyper_multicore import MegaGate

model.router = MegaGate.from_module(model.router)
logits, topk_weights, topk_ids = model.router(hidden_states)
```

源 Router 提供 `weight: [E,H]`、`top_k`、`scoring_func`、`routed_scaling_factor`，后三项也可显式传入。
可选 FP32 `bias: [E]` 是 TopK 修正偏置而非线性投影 bias，`bias_vl` 是视觉修正偏置。
参数直接绑定，梯度参与策略不变；缺少修正偏置时使用零偏置。
forward 返回三张量 tuple：`logits`、`topk_weights`、`topk_ids`。

支持 `sqrtsoftplus`、`softmax`、`sigmoid`。NPU `sqrtsoftplus` 可用原生 Route，其余支持场景使用 Torch；
实现选择保留原网络的评分函数，不能为了原生实现改变网络语义。
传入 `image_mask` 要求源模块已有视觉偏置，或构造时设置 `vision_enabled=True`，不支持时明确拒绝。

### MoE：只替换路由专家执行

下面的示例假设原专家已经将 gate/up 合并成 `gate_up_weight`，并以 `down_weight` 保存下投影。
权重由外层模块持有；原 Router 的辅助 loss 和负载均衡仍由模型处理。

```python
import torch
from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions

class RoutedExperts(torch.nn.Module):
    def __init__(self, source, *, hidden_size, intermediate_size, num_experts, top_k, ep_group,
                 numerics_options: MegaMoeNumericsOptions | None = None):
        super().__init__()
        self.gate_up_weight = source.gate_up_weight
        self.down_weight = source.down_weight
        self.executor = MegaMoeExperts(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_experts=num_experts, top_k=top_k, ep_group=ep_group,
            numerics_options=numerics_options,
        )

    def forward(self, hidden_states, topk_ids, topk_weights):
        return self.executor(hidden_states, topk_ids, topk_weights,
                             expert_weights=(self.gate_up_weight, self.down_weight))
```

`ep_group` 是实际 EP 组，组件从中推导 EP 大小。
未指定组仅适用于独立单 rank，分布式进程不会自动猜成 WORLD。token 容量由实际输入维护，无需上限或模式开关；
首次接入可省略 `tokens_per_expert`；复用 Router 的计数时，它须是当前本 rank、按全局专家编号的精确 `[E]` histogram。

示例中的模块应在 FSDP 包装和优化器创建前安装，调用时经过该模块的 `__call__`，使 FSDP hook
先 unshard 参数；EP DTensor 由模型 adapter 在此时取出本地专家片段。执行器按两组权重形状自动识别
EIO/EOI，契约见[输入与权重](../hyper_multicore/README.md#输入与权重)。如果原模型的 gate/up 是两个
独立参数，需要模型侧合并或调整存储；每次 forward 拼接可能产生额外复制。

按来源模型逐层构造 `MegaMoeNumericsOptions(swiglu_limit=..., clamp_after_silu=...,
apply_probs_before_fc2=...)`，再通过 `numerics_options=` 传入 adapter。
省略时使用不截断、FC2 后乘概率的默认语义。`clamp_after_silu=True` 指 SiLU 后、乘 up 前截断，
概率前移则保留激活 BF16 写出和乘概率后 BF16 写出两处舍入，详见
[数值语义](../hyper_multicore/README.md#模型数学与数值语义)。这些选项由模型语义决定。
需要调整 push/pull、容量或通信分片时，使用同一执行器的 `tuning_options=`，
传入公开实验性的 `MegaMoeTuningOptions`；示例与兼容性边界见
[调优配置](../hyper_multicore/README.md#公开实验性调优配置)。

### mHC：保留四流结构与显式状态

源模块须已采用四流 shifted mHC，提供 `hidden_size`、`hc_eps`、`norm_eps`、`num_iters` 及下列参数。
`from_module` 绑定原 Parameter 和配置，不把普通残差网络转换成 mHC：

| 参数 | 形状 | 精度 |
| --- | --- | --- |
| `phi` | `[24,4H]` | FP32 |
| `alpha` / `bias` | `[3]` / `[24]` | FP32 |
| `norm_weight` | `[H]` | BF16 |

```python
import torch
from hyper_multicore import MegaMhc

model.mhc = MegaMhc.from_module(model.mhc)
model.to(device=device, dtype=torch.bfloat16)
residual, pre, post, mix, block_input = model.mhc(
    previous_output, residual, previous_pre, previous_post, previous_mix,
)
state = (block(block_input), residual, pre, post, mix)  # 下一边界调用 model.mhc(*state)
```

组件接收五张量，返回更新后的 residual、三个映射与当前 block 输入，均按上述顺序使用普通 tuple。
原模型负责首层初始化、跨层状态传递，以及末层 pending block 输出的合并；
完整接线参考 [mHC 模型](mhc/model.py)，不在组件内部缓存网络状态。
模型侧模块统一使用 `MegaMhc`；底层 CANN/ACLNN 算子使用 `HyperMegaMhc` ABI 名称。

整网 dtype 转换保护 `phi`、`alpha`、`bias` 以及 Gate 修正偏置的 FP32 原值；普通权重和 `norm_weight` 可用 BF16。
已被转成 BF16 的控制参数应从原 FP32 权重重新加载，`.float()` 不能恢复丢失精度。
训练要求 H 为128的倍数且不超过5760，Sinkhorn 固定20次；本地 token 下限是 `max(40, 当前设备 AIV 核数)`，
执行前按实际设备校验。CP/SP 切分后的输入也须满足，不自动 padding 或改变数学定义。

## Checkpoint与参数归属

1. 构造原模型，加载 checkpoint，并按模型定义完成 materialize/初始化；`to_empty()` 后也须覆盖 norm、Router 和派生 buffer。
2. 在正确 EP 切片上替换模块并转换布局，再做设备/精度迁移和目标并行包装。
3. 对最终 Parameter 集合创建优化器；以相同输入检查输出、梯度和更新后，再扩大层数、重算与动态输入。

Gate/mHC 可绑定原 Parameter，MegaMoe 的参数由调用方持有。权重迁移不等于完整续训：
如模型侧合并/转置并创建新 Parameter，严格续训还需同步转换 FP32 master、`exp_avg`、`exp_avg_sq` 及 checkpoint 元数据；
gate/up 动量按相同次序转置拼接，down 动量转置，gate/up step 必须一致。不能直接加载旧优化器 state_dict 并宣称等价。

保留已有 FSDP/参数管理器时，forward 内通过 `expert_weights=(gate_up, down)`
传入当前已 unshard 的本地专家；形状和连续性契约见[输入与权重](../hyper_multicore/README.md#输入与权重)。
不要保留两套可训练参数或跨更新缓存转换权重；非连续/拼接转换可能逐步复制，autograd 连接和生命周期由 adapter 维护。

EP checkpoint 按组内 rank 顺序解释，避免框架和 adapter 重复应用 EP。
dense FSDP 与专家 EP 可共用设备；owner 独占的专家排除在 dense FSDP 外，有专家副本时仅在同一 owner 副本组分片。
共享 workspace 只适用于首次 forward 前配置兼容的串行层，不共享参数，也不消除各 autograd 图的普通 HBM 激活。

梯度归一化由目标 loss 决定：纯 EP16、无专家副本、目标为16个来源 loss 均值时，owner 专家 dW 只平均一次；
累积、token 加权及专家副本须重新推导，dense/Router/shared expert 沿用各自归约规则，clip 按唯一参数计数。
MoE fused 与 AdamW fused/foreach 独立；检查参数、grad、master、moment 的 dtype 和 DTensor layout，不能据普通 Tensor 推断分布式支持。

## 独立模型与验证范围

| 模型 | 网络结构 | 替换目标 | 运行规模 |
| --- | --- | --- | --- |
| [MoE Transformer](moe/README.md) | 普通 pre-norm/残差、Router、路由与共享专家 | `moe`、`gate` 的非空子集 | TP1/EP 可配置，默认 EP8 |
| [Dense mHC Transformer](mhc/README.md) | 四流 mHC、attention、dense SwiGLU | 仅 mHC 边界 | 单 NPU，无 EP/SHMEM |

两个模型不互相导入。`_transformer.py` 复用普通 attention/FFN/输出计算，
`_training.py` 复用优化器和 Tensor 对照；网络结构由各模型定义。
各 benchmark 对照初始权重、完整输出/梯度/更新，再测 optimizer step。
MoE 的组件选择只改变 Router/专家实现；已有模型接入无需采用示例的网络或运行规模。

```bash
python -m pytest -q tests/ut/examples
```

CPU 接线测试位于 `tests/ut/examples/`，参数迁移测试见
[test_frontend_migration.py](../tests/ut/modules/test_frontend_migration.py)。
真实设备组件检查分别见 [MegaGate](../tests/torch/test_mega_gate.py)、
[MegaMoe](../tests/torch/test_mega_moe.py)、[MegaMhc](../tests/torch/test_mega_mhc.py)。
CPU 接线、设备组件 ST 和完整模型 benchmark 各自验证不同范围。

普通网络层、all-to-all EP 和 FP32 master AdamW 位于 `_layers.py`、`_native.py`、`_optimizer.py`，
仅服务示例，不提供完整 DTensor、FSDP、模型替换或通用优化器框架。普通 EP 对照使用本地专家分片，
不使用收集全部权重的 correctness reference。DTensor 专属类型/全局布局测试由消费方框架维护。

# MegaGate

`MegaGate` 为每个 token 计算专家评分、选择 TopK 专家，并返回路由权重与专家编号。
它完成 Router 的投影与路由，可以独立使用，也可以将路由结果交给 [MegaMoE](mega_moe.md) 执行专家计算。
correction bias 用于调整专家选择，不作为线性投影 bias，也不加到返回的路由权重中。

## 基本用法

下面是文本路由的完整调用示例，需要可用 NPU；先按[构建与安装](build.md)准备并激活 native payload。
示例使用 `sqrtsoftplus` 评分，不启用视觉 bias：

```python
import torch
import torch_npu

from hyper_multicore import MegaGate

device = torch.device("npu:0")
gate = MegaGate(
    hidden_size=128,
    num_experts=33,
    top_k=4,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
).to(device=device, dtype=torch.bfloat16)
hidden_states = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

logits, routing_weights, expert_indices = gate(hidden_states)
print(logits.shape, routing_weights.shape, expert_indices.shape)
# torch.Size([128, 33]) torch.Size([128, 4]) torch.Size([128, 4])
```

三个输出分别用于读取专家 logits、对所选专家输出加权和定位专家。
输出的 token 维度已展平；传给 MegaMoE 时，`expert_indices` 对应 `topk_ids`，`routing_weights` 对应 `topk_weights`。

## 接口说明

设输入为 `[B, S, H]`，`T=B×S`，`E` 为专家数，`K` 为 `top_k`。
输入与模块参数应在同一设备；native 路径要求 NPU，其他路径见[执行路径](#自动微分与执行路径)。

| 输入或返回值 | 形状 | 含义与 dtype |
| --- | --- | --- |
| `hidden_states` | `[B, S, H]` | 浮点输入；投影按当前 autocast 策略执行 |
| `image_mask` | `[B, S]` | 可选 BOOL Tensor；`True` 使用视觉 bias，`False` 使用文本 bias |
| `logits` | `[T, E]` | 投影结果，保持投影输出 dtype |
| `routing_weights` | `[T, K]` | 所选专家的路由权重；native 路径为 FP32 |
| `expert_indices` | `[T, K]` | INT64 全局专家编号，与权重逐项对应 |

返回值按表中后三项的顺序组成 tuple。TopK 返回的专家不保证按评分排序。
Torch 路径的路由权重 dtype 由评分、归一化和 autocast 策略共同决定；关闭 autocast 时使用 FP32。
外层 BF16 autocast 开启时投影可产生 BF16 logits；native 路径会将 logits 转为 FP32 后执行 Route，
但返回的 `logits` 仍保持投影输出 dtype。

| 构造参数 | 必填 / 默认值 | 含义与约束 |
| --- | --- | --- |
| `hidden_size`、`num_experts` | 必填 | 正数，分别指定 H 和 E |
| `top_k` | 必填 | 位于 `[1, num_experts]` |
| `scoring_func` | 必填 | `"sqrtsoftplus"`、`"softmax"` 或 `"sigmoid"` |
| `routed_scaling_factor` | 必填 | 有限的路由权重倍率 |
| `vision_enabled` | `False` | 启用时创建独立的视觉 correction bias |
| `initializer_range` | `0.02` | 投影权重初始化的标准差，须为有限非负数 |

模块持有投影参数 `weight [E,H]`，以零均值 Normal 分布初始化；`bias [E]` 和可选的 `bias_vl [E]`
初始化为零。调用 `.bfloat16()` 或 `.to()` 时，correction bias 保持 FP32。

## 评分与专家选择

投影先生成 logits，再根据 `scoring_func` 计算评分：`sqrtsoftplus` 为 `sqrt(softplus(logits))`，
`softmax` 沿专家维归一化，`sigmoid` 逐元素计算。随后将 correction bias 加到评分上选择专家，
从未加 bias 的原始评分中取出对应权重。

以下为数学伪代码，省略 dtype 转换与设备实现：

```text
scores = scoring(logits)
indices = topk(scores + correction_bias, K)
weights = gather(scores, indices)
if K > 1:
    weights = weights / (sum(weights, expert_dim) + 1e-20)
routing_weights = weights * routed_scaling_factor
```

`top_k=1` 时不执行所选权重归一化，保留该专家评分再乘倍率；`top_k>1` 时先归一化再缩放。
因此不能把 K=1 理解成无论评分如何都返回倍率本身。correction bias 可以改变选中的专家，
但不会直接增加该专家的返回权重。

## 视觉路由

视觉和文本 token 共用投影权重与评分函数，只在专家选择时使用不同的 correction bias。
启用 `vision_enabled=True` 后，`image_mask=True` 的位置使用 `bias_vl`，其余位置使用 `bias`；
不传 mask 时所有 token 使用文本 bias。

以下片段沿用基本示例中的 `device` 和 `hidden_states`，将每个序列的后半部分标为视觉 token：

```python
visual_gate = MegaGate(
    hidden_size=128, num_experts=33, top_k=4,
    scoring_func="sqrtsoftplus", routed_scaling_factor=1.0,
    vision_enabled=True,
).to(device=device, dtype=torch.bfloat16)
image_mask = torch.zeros(2, 64, device=device, dtype=torch.bool)
image_mask[:, 32:] = True
visual_logits, visual_weights, visual_indices = visual_gate(hidden_states, image_mask=image_mask)
```

传入 mask 前必须启用视觉 bias，否则抛出 `ValueError`。mask 必须具有 `[B,S]` 形状；
native 视觉路由还要求它是同一 NPU 上的连续 BOOL Tensor。

## 自动微分与执行路径

loss 可以经 `logits` 或 `routing_weights` 回传到输入和投影权重。专家索引是离散选择，
correction bias 不通过这条选择路径获得梯度；native 路径也显式 detach 了 bias。
如模型需要更新 correction bias，应保留原模型相应的更新方法。

以下片段沿用基本示例，演示路由权重参与一阶反向；实际训练由网络 loss 决定梯度：

```python
loss = routing_weights.square().mean()
loss.backward()
print(hidden_states.grad.shape, gate.weight.grad.shape)
```

| 条件 | 路径 | 自动微分范围 |
| --- | --- | --- |
| NPU、非空输入、`sqrtsoftplus` | Torch 投影 + native Route/RouteGrad | native Route 仅支持一阶求导，不支持继续对 RouteGrad 求导 |
| `softmax`、`sigmoid`，或非 NPU、空输入 | Torch 实现 | 由所用 Torch 运算的 autograd 支持决定；TopK 索引仍是离散选择 |

投影 MatMul 由 Torch 和 CANN 选型。路径选择保持评分函数的定义，native 路径的求导限制不应推广到所有 Torch 路径。
诊断配置和采集方法见[诊断指南](diagnostics.md)。

## 模型接入

`MegaGate.from_module(source, ...)` 绑定原 Router 的投影与修正偏置 Parameters，不替换这些参数对象。
这里的 bias 是 TopK 修正偏置，不是线性投影 bias；缺少修正偏置时使用零偏置。
应在模块替换后创建优化器。源模块要求及替换示例见[Gate 模型接入](../examples/README.md#gate保留评分函数与原参数)。

# MegaMHC

`MegaMhc` 执行四路残差流的 shifted mHC block 边界计算：合入上一 block 的输出，
预测下一边界的混合系数，并为当前 block 生成经过 RMSNorm 的输入。所有输入和参数必须位于同一 NPU。

## 边界计算与状态传递

一次调用使用传入的 `previous_post_mix` 和 `previous_residual_mix`，将 `previous_output` 合入四路残差。
更新后的残差用于两个计算：预测供下一边界使用的 pre/post/residual mix；使用传入的
`previous_pre_mix` 混合残差并执行 RMSNorm，得到当前 `block_input`。

当前 block 不使用刚预测出的 `next_pre_mix`。执行 block 后，它的输出与本次返回的残差、混合系数一起进入下一边界：

```text
上一 block 输出 + residual + previous_post_mix/previous_residual_mix
    → new_residual
        → 预测 next_pre_mix、next_post_mix、next_residual_mix，留给下一边界
        → 用 previous_pre_mix 混合并执行 RMSNorm → block_input → 当前 block 输出

下一次输入 = (当前 block 输出, new_residual, next_pre_mix, next_post_mix, next_residual_mix)
```

调用方显式持有并传递这些状态。它们不是模块内部缓存，也不能用普通单路残差直接替代。
首层状态初始化、最后一个 block 输出的合入和四流合并由模型定义；完整示例见[MHC 模型接入](../examples/mhc/README.md)。

## 基本用法

先按[构建与安装](build.md)准备环境和 native payload。
训练要求展平 token 数 `T≥max(40, 当前设备配置的 AIV 核数)`，H 为 128 的正整数倍且不超过 5760。
下面的完整示例查询当前设备核数以选择 T，使用两个边界和一个简单线性 block 展示状态传递与反向：

```python
import torch
import torch_npu

from hyper_multicore import MegaMhc

device = torch.device("npu:0")
torch.npu.set_device(device)
T = max(128, torch.npu.get_device_limit(0)["vector_core_num"])
H = 512
first = MegaMhc(H, device=device)
second = MegaMhc(H, device=device)
block = torch.nn.Linear(H, H, bias=False, device=device, dtype=torch.bfloat16)

previous_output = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
residual = torch.randn(T, 4, H, device=device, dtype=torch.bfloat16, requires_grad=True)
previous_pre = torch.full((T, 4), 0.25, device=device, dtype=torch.float32)
previous_post = torch.ones(T, 4, device=device, dtype=torch.float32)
previous_mix = torch.eye(4, device=device, dtype=torch.float32).expand(T, 4, 4).contiguous()

new_residual, next_pre, next_post, next_mix, block_input = first(
    previous_output, residual, previous_pre, previous_post, previous_mix,
)
block_output = block(block_input)
state = (block_output, new_residual, next_pre, next_post, next_mix)
new_residual, next_pre, next_post, next_mix, next_block_input = second(*state)

loss = next_block_input.float().square().mean()
loss.backward()
print(next_block_input.shape, previous_output.grad.shape)
```

示例中的初值只用于演示已有状态进入两个边界，不定义任意模型的首层初始化。
loss 放在第二个 block 的输入上，用于展示梯度跨边界回传；真实网络继续执行下一个 block，最终按模型定义合并状态并计算 loss。

## 接口说明

| 构造参数 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `hidden_size` | 必填 | H 为 128 的正整数倍，模块构造要求 H≤5760 |
| `hc_eps` | `1e-6` | 混合系数计算使用的 epsilon |
| `norm_eps` | `1e-6` | RMSNorm 的 epsilon |
| `num_iters` | `20` | 当前固定为 20 次 Sinkhorn，不能作为普通调优参数改变 |
| `token_tile` | `32` | token 分块大小，须为正整数 |
| `device` | `None` | 默认按 Torch 默认设备分配，执行前需将模块迁移到 NPU |

固定使用四路 residual streams。`.eval()` 和 `torch.no_grad()` 都不会放宽模块构造时的 H 上限。

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

## 训练与纯前向

模块根据梯度上下文和输入、参数是否需要梯度选择自动微分路径；`.eval()` 不关闭 autograd。
需要反向时适用前述 token 下限，纯前向要求 `T≥1`。
形状、跨设备或尺寸错误抛 `ValueError`；非 Tensor、dtype 或非 NPU 输入错误抛 `TypeError`。
多 stream 调用前由调用方建立输入数据依赖。

### 底层纯前向入口

通常使用包顶层的 `MegaMhc`。需要显式提供全部权重、或使用大于模块上限的纯前向 H 时，
可从 `hyper_multicore.modules.mega_mhc` 导入 `hyper_mega_mhc`。
这是组件子包导出的底层函数，不是包顶层入口；当前没有单独的跨版本稳定性承诺，升级时须核对签名与支持范围。

函数接收五个状态输入及 `phi, alpha, bias, norm_weight`。设置 `need_backward=False` 时，
H 仍须为 128 的正整数倍，上限为 12288，不建立匹配的反向图。
超过模块构造上限时，由调用方按接口说明的形状和 dtype 准备权重，不能通过 `.eval()` 绕过构造检查。

以下片段沿用基本示例中的输入与 `first` 参数，展示如何显式选择纯前向：

```python
from hyper_multicore.modules.mega_mhc import hyper_mega_mhc

with torch.no_grad():
    forward_outputs = hyper_mega_mhc(
        previous_output, residual, previous_pre, previous_post, previous_mix,
        first.phi, first.alpha, first.bias, first.norm_weight,
        need_backward=False,
    )
```

## 模型接入

`MegaMhc.from_module(source)` 绑定原四流 shifted mHC 的参数与数学配置，不替换这些参数对象，
也不将普通残差网络转换成 mHC。应在模块替换后创建优化器；映射参数在 `.bfloat16()` / `.to()` 时保持 FP32。
参数要求与替换示例见[mHC 模型接入](../examples/README.md#mhc保留四流结构与显式状态)，
完整前反向运行与训练对照见[MHC 示例](../examples/mhc/README.md)。

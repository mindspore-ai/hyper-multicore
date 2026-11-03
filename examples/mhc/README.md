# Dense Transformer：四流 mHC 与普通 dense FFN

本例使用普通 attention、dense SwiGLU FFN 和四流 shifted mHC，不包含 Router、专家、EP 或共享专家。
只切换 mHC 边界的 Torch/原生实现，默认单张 NPU；不建立进程组，也不初始化 SHMEM。
MoE 与专家通信的示例独立放在 [moe/](../moe/README.md)。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| [model.py](model.py) | dense FFN 与四流状态串联的完整语言模型 |
| [mhc.py](mhc.py) | 同一参数与状态下的 Torch/MegaMhc 实现 |
| [benchmark.py](benchmark.py) | 边界五项输出/九组梯度检查、完整 dense 训练对照与单卡计时 |

`mhc.py` 中的 `MhcBoundary` 处理 attention/FFN 子层之间的四流状态：把上一子层的输出合回残差，
生成混合系数并准备下一子层的输入。attention、FFN 及其执行顺序由 `model.py` 组合。
组件只接收和返回五张量；首层初始化、跨层 tuple 传递及末层 pending 输出合并由本例 `model.py` 管理。

attention、dense FFN 和词表输出的普通数学操作复用 `../_transformer.py`；
优化器和 Tensor 对照工具复用 `../_training.py`。这些共享代码不选择模型架构，也不导入 MoE 实现。

## 运行

先按[构建说明](../../docs/build.md)安装当前 checkout，并激活匹配 CANN 和 native payload，再从仓库根目录执行：

```bash
source /path/to/cann/set_env.sh
source build/native/payload/hyper_multicore/lib/set_env.bash
python -m examples.mhc.benchmark \
  --output output/mhc_transformer.json
```

默认 B1×128、H512、两层、dense I2048、词表1024。普通权重使用 BF16，mHC 控制参数保持 FP32，
AdamW 使用 FP32 master 参数。无需 `torchrun`，多 rank 启动会在设备选择之前明确拒绝。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tokens` | 128 | 本地序列长度，须满足设备的 mHC 训练下限 |
| `--hidden-size` | 512 | 隐藏维度，须满足 mHC 的对齐和训练上限 |
| `--intermediate-size` | 2048 | 普通 dense SwiGLU 的中间维度 |
| `--num-layers` | 2 | 每层有 attention 和 dense FFN 两个 mHC 边界 |
| `--warmup-steps` / `--measured-steps` | 3 / 5 | 两个后端相同的预热和稳态样本数 |
| `--learning-rate` / `--weight-decay` | `1e-3` / `0.01` | 共同的 AdamW 配置 |
| `--seed` / `--device-index` | 42 / 0 | 初始权重/输入和可见 NPU 序号 |
| `--output` | `output/mhc_transformer_benchmark.json` | 结果路径 |

本例没有 `--mega-kernels`、`--dispatch-mode` 或 EP 参数。底层形状约束见
[mHC 参数与状态](../README.md#mhc保留四流结构与显式状态)。

## 已知精度限制

默认 BF16、两层、128 token、H=512、seed=42 的完整网络对照可能在 logits 检查处失败。
单边界输出/梯度检查通过不代表完整网络精度通过。benchmark 保留
`rtol=0.02, atol=0.002` 的逐元素检查并明确报错，不跳过失败或继续输出成功结论。

## 验证范围

先对照一个边界的五项输出和全部九组输入/参数梯度，再对照完整 dense 模型的初始参数、logits/loss、
全部梯度及 AdamW 更新。完整训练时原生 mHC 始终开启，不经过改为普通边界的中间模式。
不相等会明确报出 Tensor 名称及误差；没有离散路由可以掩盖或放大边界误差。

JSON 记录两个后端的完整本地 optimizer-step 样本、median 和比值。验证/预热不作稳态耗时；
性能顺序为 common → mega，正式比较需固定源/制品、形状与精度并排除设备污染。
精度及性能结果仅适用于对应的模型结构、形状、dtype 和运行环境。

CPU 训练及隔离测试见 [test_mhc_model.py](../../tests/ut/examples/mhc/test_mhc_model.py)。
CPU 用例用独立数学 oracle 替代原生边界，验证接线、梯度和优化器；真实 NPU 数值及性能仍需单独执行。

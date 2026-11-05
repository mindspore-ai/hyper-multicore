# MoE Transformer：普通残差连接与路由专家

本例只包含 Router、路由专家和共享专家，attention/FFN 使用普通 pre-norm 残差连接，模型中没有 mHC。
它是随机权重文本网络，用于比较相同架构下普通 EP 与 MegaMoe/MegaGate 的完整训练 step，不对应已发布模型。
四流 mHC 的独立 dense 模型见[另一个示例](../mhc/README.md)，两者不相互导入。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| [model.py](model.py) | 普通残差 attention/MoE decoder 与完整语言模型 |
| [gate.py](gate.py) | 同一 Router 参数下的 Torch/MegaGate 实现 |
| [experts.py](experts.py) | 路由/共享专家，以及普通 EP 专家的参数转换 |
| [benchmark.py](benchmark.py) | 等初始参数、输出/梯度/更新对照与完整 optimizer-step 计时 |
| [run_benchmark.sh](run_benchmark.sh) | 环境、安装态/源码态运行和可配置 rank 数的 torchrun 启动 |

公共组件接入从[示例导航](../README.md)开始，不需要采用本例的网络结构。

## 运行

按[构建说明](../../docs/build.md)准备匹配的 Torch/torch_npu、CANN9.1.0或更高版本及 native payload。
本例使用 **TP1，EP 等于 torchrun 的总 rank 数**；脚本默认单机8卡。
每 rank 默认 B1×4096、H5120、两层、专家与共享专家 I1792。
专家总数默认 `6 × EP`，TopK 默认 `min(8, 专家总数)`；EP16 对应 E96/K8，EP8 对应 E48/K8，EP4 对应 E24/K8。
EP1 时默认 E6/K6。
从仓库根目录执行：

```bash
CANN_ENV_FILE=/path/to/cann/set_env.sh \
PYTHON_BIN=/path/to/python \
bash examples/moe/run_benchmark.sh \
  --mega-kernels moe,gate --output output/moe_transformer.json
```

例如，单机4卡只需调整启动进程数和可见设备，默认每卡仍有6个专家：

```bash
CANN_ENV_FILE=/path/to/cann/set_env.sh \
PYTHON_BIN=/path/to/python \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash examples/moe/run_benchmark.sh \
  --output output/moe_ep4.json
```

如果需要固定全局专家总数，附加 `--num-experts 48 --top-k 8`；EP4 时这会改为每卡12个专家，负载不同于默认 E24。

### 910C 单机 EP16

使用适配 910C 的 Torch/torch_npu 与 CANN，并构建包含 `ascend910_93` 目标的 native payload：

```bash
source /path/to/cann/set_env.sh
bash build.sh --soc-list ascend910_93 --jobs 24
CANN_ENV_FILE=/path/to/cann/set_env.sh \
PYTHON_BIN=/path/to/python NPROC_PER_NODE=16 \
bash examples/moe/run_benchmark.sh \
  --output output/moe_910c_ep16.json
```

该命令启动16个 rank，使用 E96、每 rank 6个专家、TopK8。
未设置 `ASCEND_RT_VISIBLE_DEVICES` 时，脚本按 `NPROC_PER_NODE` 生成 `0,...,15`；
若已设置，保留用户指定的设备顺序，并检查设备 ID 唯一且数量不少于本机进程数。
逻辑 NPU 数须至少为16；不能在8个逻辑 NPU 上通过重复设备 ID 启动 EP16。
此基准通过内部调优配置支持 `--dispatch-mode pull`，默认 push；这些选项不属于 `MegaMoeExperts`
的公共接口。JSON 的 `topology` 应为 `world_size=16, tp=1, ep=16`。

### 参数

| 参数 | 默认值 / 可选项 | 含义 |
| --- | --- | --- |
| `--num-experts` | `6 × EP` | 全局专家数，须为正整数且能被 EP 整除 |
| `--top-k` | `min(8, num_experts)` | 每 token 选择的专家数，须在 `[1, num_experts]` 内 |
| `--mega-kernels` | `moe,gate`；两者的非空、无重复逗号子集 | 只选择 Router/专家实现，不接受 `mhc` |
| `--warmup-steps` / `--measured-steps` | 3 / 5，正整数 | 两个后端相同的预热和计时步数 |
| `--learning-rate` / `--weight-decay` | `1e-3` / `0.01` | FP32 master 参数 AdamW 的共同设置 |
| `--seed` | 42 | 参数和输入的共同 seed |
| `--output` | `output/moe_transformer_benchmark.json` | rank0 JSON 路径 |
| `--dispatch-mode` | `push` / `pull`，默认 `push` | 专家通信策略 |
| `--initial-capacity-factor` / `--capacity-growth-factor` | 省略，push 默认均为 `1.25` | 高级接收容量策略，pull 不接受显式数值 |

例如 `--mega-kernels gate` 仍运行相同 EP 大小的 MoE 网络，只改变 Router 的实现；
common 后端使用普通 Router/EP，两个后端保持相同的普通残差结构。
多机使用 `NNODES`、`NPROC_PER_NODE`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`，
EP 为 `NNODES × NPROC_PER_NODE`，由启动进程数确定。
也可直接用 `torchrun --nproc-per-node=4 -m examples.moe.benchmark` 启动，
运行前须自行激活环境并设置 SHMEM bootstrap endpoint。
`ASCEND_RT_VISIBLE_DEVICES` 选择本机设备，省略时默认 `0,...,NPROC_PER_NODE-1`；
`NATIVE_ENV_FILE` 选择匹配的激活脚本；启动器支持源码 payload 或已安装的组件。
JSON 的 `topology.world_size`、`topology.ep` 和专家形状记录实际配置。

## 验证范围

benchmark 的目标 loss 为各来源 rank 本地 loss 的均值：dense 梯度归约后平均，owner 专家梯度已汇总所有来源，
只在本地除以 EP 一次。对照初始参数、完整模型 logits/loss、全部梯度及 AdamW 更新后，记录最慢 rank 的整步耗时。
不存在切换到共同 mHC 后再做精度对照的流程；JSON 也不含 mHC 的实现或验证范围。
复制梯度的验证步不作为稳态计时，性能顺序为 common → mega；需固定源/制品和负载，排除外部设备进程。

CPU 接线测试位于 [test_moe_model.py](../../tests/ut/examples/moe/test_moe_model.py)，
覆盖普通残差、Router、专家梯度/更新和导入隔离。组件边界、动态 token 与 SHMEM 生命周期仍由
[NPU 用例](../../tests/torch/)验证，执行方式见[测试指南](../../tests/README.md)。
本例不恢复旧 checkpoint 的优化器状态，完整续训边界见[迁移指南](../README.md#checkpoint与参数归属)。

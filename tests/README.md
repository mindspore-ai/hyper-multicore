# 测试指南

`tests/` 包含 HyperMulticore 的 CPU 单测、真实 NPU 系统测试和共用测试工具。
本文说明用例归属、添加方式、执行命令以及失败定位。默认从仓库根目录执行命令；
仅 NPU 安装态验证在下文所述的源码外测试目录运行。

## 哪些用例放在这里

| 类型 | 放置位置 |
| --- | --- |
| MegaMoE、MegaGate、MegaMHC 的 native 正反向、梯度、动态输入及多 stream 行为 | [torch/](torch/) 中对应组件的 launcher/worker |
| SHMEM 实际设备上的分配、通信、同步和生命周期 | [torch/shmem/](torch/shmem/) |
| 需要真实 kernel 执行的 profiling、tensor dump 和计时 smoke | [torch/](torch/) 中对应诊断或性能用例 |
| 不需要 NPU 的调度、配置、序列化、参数校验和 mock 单测 | [ut/](ut/) |
| 多个测试共用的参考实现、golden 和启动工具 | [common/multicore/](common/multicore/) |

独立小网络示例的 CPU 接线与训练检查位于 `ut/examples/`，对应 [examples](../examples/README.md)。
真实模型、训练配方或下游并行框架的集成测试由对应消费方维护。本目录测试应能独立运行，
不依赖 HyperParallel 或外部模型工程。

### 目录约定

- `ut/test_*.py`、`ut/modules/`、`ut/torch/` 和 `ut/shmem/` 放置 CPU 单测，可直接由 pytest 执行。
- `ut/build/` 集中放置构建、打包、依赖准备、环境复用及适配矩阵相关单测和夹具。
- `common/` 放置共用工具和参考实现，不单独作为测试入口。

NPU 用例位于 `torch/`：

- `test_*.py` 是 pytest 收集入口，负责环境准备、设备需求和启动 worker。
- `_test_*.py` 是实际设备 worker，负责创建输入、执行算子和断言；由 launcher 指定运行。
- `shmem/` 按相同方式组织通信测试。
- `benchmark_mega_mhc.py` 和 `benchmark_mega_mhc_grad.py` 是独立精度对照脚本，
  不属于 pytest 自动收集的套件。运行 `pytest tests/torch` 不会执行它们。

launcher 及其顶层导入必须保持轻量，不导入 Torch、torch-npu 或 HyperMulticore。
框架和设备初始化放在 worker 中。复用现有组件的 launcher/helper，避免另建启动流程。

## 环境准备

### CPU 单测

使用配套的 Python/Torch/PTA 环境，从源码根目录安装开发依赖：

```bash
python -m pip install -r requirements.txt setuptools wheel "packaging>=22"
python -m pip install pytest
python -m pip install --no-deps --no-build-isolation -e .
```

CPU 单测不需要分配 NPU 卡；Torch 适配层的导入检查仍使用当前框架环境。
C++ runtime ABI reader 检查需要支持 C++17 的 G++。部分用例读取仓库内的源码、锁文件和构建资源，
因此 CPU 单测应在源码根目录运行，不使用下述只复制 `tests/` 的安装态目录。

### NPU 系统测试

需要 Linux、匹配制品标签的 Python/CPU 架构与 Torch/PTA 分支、彼此兼容的 CANN/驱动、pytest，以及已构建的 native payload。
使用 `hyper_multicore_build_info` 查看安装包构建矩阵；构建 PTA post 和 CANN 版本不作为运行时版本门禁。
安装和构建步骤见[构建与安装指南](../docs/build.md)。

先激活 CANN，然后选择一种运行方式：

**源码模式**：从仓库根目录使用本次构建的 payload。

```bash
source /path/to/cann/set_env.sh
source build/native/payload/hyper_multicore/lib/set_env.bash
```

**安装模式**：安装目标 wheel 后，将完整 `tests/` 复制到源码树之外。
清除指向源码的 `PYTHONPATH`，在新的测试目录激活已安装包，避免从 checkout 导入生产代码。

```bash
# 从源码根目录复制；测试目录不包含 hyper_multicore/ 源码包。
test_run_dir=$(mktemp -d)
cp -R tests "$test_run_dir/"
cd "$test_run_dir"
source /path/to/cann/set_env.sh
source "$(command -v hyper_multicore_set_env.bash)"
```

在启动 pytest 前完成上述激活。用 `ASCEND_RT_VISIBLE_DEVICES` 选择物理设备，按升序填写，
并确认参与设备有足够显存。launcher 从列表中取本用例所需的设备数；省略该变量时从卡 0 开始选择。
卡数不足会报错，不会自动缩小测试规模。

## 选择和执行用例

无参数 pytest 可能同时收集 CPU 和 NPU。应显式指定 `tests/ut` 或 `tests/torch`，
下面的命令均指定测试路径。

### CPU 单测执行

从源码根目录运行：

```bash
# 查看 CPU 用例，执行全部单测，或只运行一个文件。
python -m pytest tests/ut --collect-only -q
python -m pytest tests/ut -q
python -m pytest tests/ut/build/test_artifacts.py -q
```

可以用 `::test_name` 选择函数；对于 unittest 类，使用 `::ClassName::test_name`。
具体节点以 collect 输出为准。CPU 单测的 traceback 直接出现在当前 pytest 输出中。

### 构建系统定向回归

构建相关测试统一位于 `ut/build/`，按框架环境、native 编译调度、发布、工作区与命令、
适配加载和 CLI 等职责维护。它们使用临时目录、合成 wheel 和明确的 worker 替身，
不下载框架、不创建真实 venv、不编译 native 或启动 NPU 计算。

```bash
python -m pytest tests/ut/build tests/ut/test_loader.py -q
```

新增测试优先覆盖可观察行为：venv/vendor 复用、SDK/ops 和适配对重编、失败不发布、制品提交边界、目录保护和加载选择。
同一契约保留主要调用链上的代表性回归；不在多个层级重复模拟同一异常，
不为每个路径或异常类型展开笛卡尔积，也不固定内部计时字段、调用次数和日志措辞。
共享环境与依赖夹具放在 `ut/build/_build_test_support.py`；payload 布局夹具统一复用 `ut/build/test_artifacts.py`。
构建测试调整后集中运行这组 CPU 回归，失败修正仅重测受影响范围。
只有实际修改 native 输入、环境准备或运行行为时，才补对应的构建或设备验证。

### NPU 用例执行

launcher 已经负责启动分布式 worker，以下 pytest 命令不要再套一层 torchrun，也不要直接运行 `_test_*.py`。

### NPU 入口与设备数

| 入口 | 需要的 NPU | 主要覆盖 |
| --- | --- | --- |
| [test_mega_gate.py](torch/test_mega_gate.py) | 1 | text/vision 路由、正反向、数学顺序、动态形状、训练图和符号解析并发 |
| [test_mega_mhc.py](torch/test_mega_mhc.py) | 1 | 同形状前向/反向在多 stream 下与串行 native 执行结果一致 |
| [test_mega_moe.py](torch/test_mega_moe.py) | 按用例为 2、4 或 8 | 输出及完整梯度、SGD、push/pull、Router、容量、生命周期、subgroup 和动态 token |
| [test_swiglu_clamp_st.py](torch/test_swiglu_clamp_st.py) | 1 | clipped SwiGLU 数值 |
| [test_mega_moe_profiling.py](torch/test_mega_moe_profiling.py)、[test_mega_moe_tensor_dump.py](torch/test_mega_moe_tensor_dump.py) | 2 | kernel trace 和 tensor dump |
| [shmem/test_binding.py](torch/shmem/test_binding.py) | 按用例为 1 或 2 | 分配、释放、通信、stream 顺序、错误及原生日志 |
| [shmem/test_all_gather.py](torch/shmem/test_all_gather.py) | 按用例为 1、2、4 或 8 | all-gather 边界、数据正确性、stream 可见性和设备 guard |

MoE 基础精度用例需要 2 卡，Router、subgroup、group-list 和计时 smoke 需要 4 卡，
自动 token 用例需要 8 卡，包含变长/空 rank、容量增长及保留图回归。SHMEM 的设备 guard 用例虽然只有一个 worker，也需要两张可见卡。
`allcards` 标记不等于固定需要 8 卡，实际数量以 launcher 的 `num_proc`/`num_devices` 为准。

### 查看将要运行的用例

```bash
python -m pytest tests/torch --collect-only -q
python -m pytest tests/torch/test_mega_moe.py --collect-only -q
```

### 单用例

```bash
# 单卡 Gate 正反向。
export ASCEND_RT_VISIBLE_DEVICES=0
python -m pytest tests/torch/test_mega_gate.py::test_mega_gate_basic_parity -v -s

# 双卡 MoE 精度。
export ASCEND_RT_VISIBLE_DEVICES=0,1
python -m pytest tests/torch/test_mega_moe.py::test_mega_moe_level0_precision -v -s
```

参数化用例的具体 ID 以 collect 输出为准。复制该 ID 并加引号，可以只运行其中一项。

### 组件或完整套件

```bash
# 单卡组件测试。
export ASCEND_RT_VISIBLE_DEVICES=0
python -m pytest tests/torch/test_mega_gate.py tests/torch/test_mega_mhc.py -v

# 完整 NPU 套件，包含 8 卡用例。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs
python -m pytest tests/torch -v --junitxml=logs/npu.xml
```

也可以用 `-m level0` 选择基础检查，或 `-k` 按名称过滤。当前 MHC 多 stream 用例标记为 level1，
不会包含在 level0 选择中。默认 pytest 串行执行各 launcher；不要用 pytest-xdist 并行扩展整套测试。

## 添加或扩展用例

### CPU 用例

在 `tests/ut` 对应模块的 `test_*.py` 中新增测试函数或 unittest 方法，沿用相邻用例风格和 fixture。
使用 CPU 输入或显式 mock 检查 host 行为，不创建 NPU tensor 或启动分布式 worker。
需要 C++ 编译、文件或环境变量时，复用现有临时目录、编译 helper 和 mock，避免污染其他用例。
先 collect 新节点，再执行该用例及相关模块。

### NPU 用例

1. **确定测试归属。** 优先扩展 `tests/torch` 已有组件文件，共用参考实现放在 `tests/common/multicore`。
2. **编写 worker 断言。** 在对应 `_test_*.py` 中增加 `test_*` 函数，显式设置当前设备，
   固定随机输入，使用独立参考或既有 golden 检查结果。训练用例应检查相关梯度；按功能需要覆盖参数更新。
   沿用对应数值阈值，多卡用例应检查各 rank 的结果。
3. **注册 launcher。** 在对应 `test_*.py` 中增加入口，通过现有 helper 指定 worker 函数和实际卡数。
   添加平台、level、单卡/多卡及 essential 标记，不在 launcher 中执行算子。
4. **管理资源。** 复用环境准备、设备锁和有界超时 helper。通信用例使用独立 bootstrap endpoint；
   在 worker 中清理算子计划、SHMEM 引用和进程组，避免影响后续用例。
5. **定向验证。** 先 collect 新入口，再运行该用例及受影响的组件测试。新增入口改变覆盖或设备需求时，
   同步更新上表；不要通过放宽阈值或无条件 skip 掩盖失败。

例如，给 Gate 添加 worker 函数 `test_mega_gate_new_case` 后，可在已有
[test_mega_gate.py](torch/test_mega_gate.py) 中复用 `_run_worker` 注册入口：

```python
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
def test_mega_gate_new_case() -> None:
    _run_worker("test_mega_gate_new_case")
```

上例是新增入口模板，使用前必须在 [_test_mega_gate.py](torch/_test_mega_gate.py) 中实现对应 worker。
其他组件参考现有 [MoE launcher](torch/test_mega_moe.py) 和 [SHMEM launcher](torch/shmem/test_binding.py)。
新组件通过 [run_multicore_case](common/multicore/_case_launcher.py) 启动 worker；
`num_proc` 是进程数，`num_devices` 可声明比进程数更多的设备需求。

## 定位失败

CPU 单测失败时，先按 pytest 报告的文件、类和函数定位断言。确认使用的是预期源码及依赖环境，
检查 fixture/mock 是否隔离了环境和文件状态。ABI reader 的编译错误还需查看 G++ 输出。

NPU 用例失败时，按下面顺序定位。

### 先确认失败在哪一层

- **环境或加载失败**：检查是否激活了 CANN 和目标 payload，是否混用了源码及安装包，
  以及错误中的 Python/Torch/PTA/CANN ABI 信息。
- **一直显示 `waiting for devices`**：当前用例在等待设备锁；显示 `acquired devices` 后才开始 worker。
  默认 1200 秒超时包含锁等待。锁只协调使用同一临时目录的本用户测试，不代表外部任务未占用设备。
- **worker 非零退出或断言失败**：按失败节点找到各 rank 的 stderr/stdout，查看最先出现的 traceback、
  数值断言或设备错误。pytest 的 `CalledProcessError` 通常只是外层转报。

### 日志和输出位置

所有 NPU launcher 的 torchrun 日志位于启动目录下的 `logs/`，其结构为：

```text
logs/<worker-file>/<worker-case>/<run-id>/attempt_0/<rank>/stdout.log
logs/<worker-file>/<worker-case>/<run-id>/attempt_0/<rank>/stderr.log
```

`<worker-file>` 可能展开成源码绝对路径对应的多级子目录。可直接查找日志文件：

```bash
find logs -type f \( -name stdout.log -o -name stderr.log \) -print
```

`--junitxml=logs/npu.xml` 保存外层 pytest 结果。需要查看临时文件时，可在用例中使用 pytest 的
`tmp_path` 并在断言信息中打印路径；部分 SHMEM 用例的原生日志也保存在该目录中。

部分 MoE 用例还会保存结果 JSON、profiling 或 dump 文件。可选设置：

```bash
export HP_MEGA_MOE_EVIDENCE_DIR="$PWD/logs/moe-results"
```

该变量只控制使用它的 MoE 用例输出，不是 Gate/MHC/SHMEM 或全部 rank 日志的统一目录。
未设置时，这些 MoE 输出通常位于 pytest 的 `tmp_path`。归档日志存放在对应结果目录的
`worker_logs/` 下；每次运行使用不同结果目录，便于区分输出。

判断通过应同时检查 pytest 结果和所需 worker/rank 是否完整执行。skip、超时、进程崩溃或
缺失结果都不算通过；计时 smoke 通过只说明其断言满足，不代表性能达到某个发布门槛。

## 多框架制品回归

全量 wheel 在框架矩阵配置的每组环境分别安装同一个文件并核对 SHA256。
每组 CPU 从源码根目录运行完整 `tests/ut`；NPU 按上文复制完整 `tests/` 到源码外，使用安装态制品串行运行完整 `tests/torch`。
测试结果包括实际收集数量、退出码、JUnit 和 worker 日志。
默认单分支 wheel 另在实际构建环境完整回归，确认默认入口和功能保持可用。
验证环境差异和结果分别记录；构建矩阵描述构建输入，运行兼容性通过对应环境中的测试检查。

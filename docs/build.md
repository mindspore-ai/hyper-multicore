# 构建与安装

源码可直接构建到本仓 build 并激活运行；wheel 作为可选发布方式保留。CPU/NPU 用例的执行和排错见
[测试指南](../tests/README.md)。

## 源码目录内构建与激活

在已准备 CANN/Torch/PTA 的环境中，于本仓根目录执行：

```bash
bash build.sh --soc-list ascend950 --jobs 8 --native-only
source activate.bash
hyper_multicore_build_info --json
```

`--native-only` 不生成或安装 wheel，正式制品在 `build/native/payload/hyper_multicore`。
携带本仓源码及该目录到兼容环境后重新 source 即可；位置可以改变。
激活脚本不改变工作目录、不下载、不编译、不安装、不创建软链接。
后续关于 wheel 的章节是保留的可选发布能力，不是上述流程的前提。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Linux；CPU 架构需具备配套的 Torch/PTA wheel 和 CANN 工具链 |
| Python | >=3.10，具有开发头文件 |
| 框架 | 相互配套的 Torch、torch-npu/PTA；NumPy >=1.26,<2 |
| Ascend | 已配置的 CANN toolkit/ops；运行环境的 PTA、CANN 和驱动须彼此兼容 |
| 构建工具 | GCC/G++、CMake >=3.18、GNU Make、Git 和 CANN 编译工具；pybind11 头文件使用 Torch 提供的版本 |

激活调用者 Python 环境，配置 CANN 并安装构建依赖：

```bash
source /path/to/cann/set_env.sh
python -m pip install -r requirements.txt setuptools wheel "packaging>=22" decorator attrs psutil
```

默认构建还需在调用者环境安装 Torch/PTA。`--all` 使用独立框架 venv，调用者环境只需公共 CANN 编译依赖。
公共 SDK、ops 和 vendor 使用构建入口的 Python，保留调用者的 `PYTHONPATH`、user-site、搜索路径和编译参数。
框架适配对使用各自选定的框架环境。构建器不会自动安装或升级调用者的软件。

Torch adapter 和 SHMEM binding 的 C++ 标准由目标 TorchConfig 决定。Torch 2.14 要求正式 C++20，
仅支持早期草案的编译器不能满足其头文件检查。需要时指定兼容目标头文件的框架编译器：

```bash
HYPER_MULTICORE_FRAMEWORK_CXX=/path/to/g++ \
  bash build.sh --all --soc-list ascend910b --jobs 24
```

该变量只影响框架适配对；公共 CANN vendor/SDK/ops 使用 CANN 工具链。
未设置时沿用 `CXX` 或系统默认编译器。编译失败会保留原始诊断，C++20 错误附带编译器重试命令。

## 编译

```bash
# 使用当前安装的 Torch/PTA。
bash build.sh --soc-list ascend910b --jobs 24

# 为框架矩阵中的全部目标生成一个 wheel。
bash build.sh --all --soc-list ascend910b --jobs 24

# 构建双 SoC 产物。
bash build.sh --soc-list ascend910b,ascend910_93 --jobs 24

# 仅生成 native payload。
bash hyper_multicore/build.sh --soc-list ascend910b --jobs 24
```

| 行为 | 默认构建 | `--all` |
| --- | --- | --- |
| 框架来源 | 当前解释器实际导入的 Torch/PTA | [框架矩阵](../hyper_multicore/_build/frameworks.lock.json)中的准确配对 |
| 版本选择 | 接受正式版、开发版和自定义版本标识 | 使用配置指定的发行版本 |
| 框架安装 | 使用本机环境 | 在指定目录创建或复用 venv，默认 `build/native/envs/` |
| 公共资源 | 使用调用者 CANN 工具环境 | 使用调用者 CANN 工具环境 |
| 适配对 | 编译一组 | 按预算并行编译全部配置目标 |
| wheel | 包含当前 PTA 标签 | 不带单一 PTA 标签 |

默认路径不读取框架矩阵，也不按框架版本格式、分支或上下限拒绝构建。源码框架无需发行元数据，
版本不可得时记为 `unknown`。日志展示实际 Python、Torch/PTA 版本和安装位置。
两种模式均使用当前 CANN，不按其版本号拒绝构建；真实缺包、缺工具、头文件或符号问题由编译和加载检查报告。
矩阵中的可选 `reference_cann` 用于提示和追溯，与实际版本不同或实际版本未知时告警并继续。

每次构建准备一份公共资源。vendor 可复用，SHMEM SDK、ops 和框架适配对每次重新建立工作目录并编译。
每组适配对依次执行 SHMEM/Torch 配置、编译、安装及校验，并在目标 Python 中加载检查，不执行 NPU 计算。
最终清单始终按矩阵配置顺序排列。

`--soc-list` 设置编译目标；`--jobs` 为 J、矩阵目标数为 N 时，适配对并行度如下：

| 编译预算 | 并行适配对 |
| --- | ---: |
| J ≥ 2N | N |
| 4 ≤ J < 2N | 2 |
| J < 4 | 1 |

预算均分到固定执行槽，余数依次分配。七组目标使用 `--jobs 24` 时配额为 `4、4、4、3、3、3、3`；
`--jobs 8` 时两槽各用 4，`--jobs 2` 时串行使用 2。空闲槽领取下一目标时保留自己的配额。
公共阶段和默认单组构建使用完整 J 预算。该预算限制编译任务数，不是内存或全部系统线程的硬限制。

`--clean` 清除所选架构／SoC 的 vendor 缓存并重编；不删除框架 venv、依赖源码仓库或用户 pip 缓存。
`--all --clean` 强制完整重编全部配置目标。

### 输出与日志

| 内容 | 位置 |
| --- | --- |
| 最新成功 wheel | `dist/` |
| 正式 native payload | `build/native/payload/hyper_multicore/` |
| 公共资源工作区 | `build/native/common/<架构>-<SoC列表>/` |
| 适配对工作区 | `build/native/variants/<框架组合>/` |
| 终端记录 | `build/native/logs/build.log` |
| 工具原始输出 | `build/native/logs/tools.log` |
| 阶段、命令、耗时、退出码和日志字节区间 | `build/native/logs/stages.json` |

终端实时显示 pip/CMake 输出、环境新建或复用、安装包版本、编译进度和 vendor 缓存决策。
并行工具输出带 PTA 和阶段标识，静默阶段定期显示已用时间；内部 JSON 只写入工具日志。
失败摘要对应当前阶段，完整诊断可按 `stages.json` 中的区间定位。

构建通过 `build/native/.build.lock` 互斥。等待时显示持锁信息，Ctrl-C 可取消等待而不影响持锁构建。
获锁后替换上次日志；帮助命令、参数错误及等待锁不会清理日志。
日志仅保留最近一次，如需长期保存应在下次构建前复制到构建目录外。

组包成功后清理本次 staging；失败时保留日志和 staging，供排查使用，临时输出在下次构建开始时回收。
源码在构建和发布期间变化会拒绝发布。任一适配对失败或收到 Ctrl-C/SIGTERM 时，停止领取新任务，
终止活动进程组并归并日志。

wheel 和 payload 在 staging 完成校验后发布。提交前失败或取消会清空 `dist/` 和正式 payload；
提交后缓存维护或回收失败仅告警，保留本次有效制品。`dist/` 只保存本次成功制品，不用于历史归档。

## 安装与激活

### 使用 wheel

按 Python ABI、CPU 架构及 Torch/PTA 分支选择构建命令打印的 wheel，先准备运行依赖再安装：

```bash
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
```

激活脚本设置随包 vendor 和 SHMEM 库的搜索路径，应在导入 Torch/torch-npu 之前执行。
更换制品或环境后启动新的 Python 进程。检查安装来源时，在源码目录外移除源码 `PYTHONPATH` 后运行：

```bash
python -c 'import hyper_multicore; print(hyper_multicore.__file__)'
```

输出应指向安装位置。功能使用见[快速开始](../README.md#快速开始)。
预构建 wheel 的安装、导入和调用不会下载框架或 native 源码，也不会编译适配库。
wheel 不包含外部 Torch/PTA、CANN 和驱动。

### 使用源码 payload

构建 native 后，在源码根目录激活本地 payload：

```bash
source /path/to/cann/set_env.sh
source build/native/payload/hyper_multicore/lib/set_env.bash
```

此方式从当前源码导入 Python 实现、从本地 payload 加载 native 库。
同一进程使用一套激活环境。

## 制品内容与兼容性

默认 wheel 版本包含 PTA 的安全文件名标签，例如 `0.1.0.dev0+pt.2.12.0`。
开发版 `2.15.0.dev20261005+git2c8682d` 对应 `0.1.0.dev0+pt.2.15.0.dev20261005.git2c8682d`；
特殊字符或过长标识使用可读前缀和摘要，版本原文保存在 `build_info.json`。
全量 wheel 使用组件版本，例如 `hyper_multicore-0.1.0.dev0-cp310-cp310-linux_aarch64.whl`。
Python ABI 和 CPU 架构由 wheel 标签约束。

运行时按 Torch/PTA 数字分支、Python ABI、CPU 架构及 C++ ABI 唯一选择一组适配对。
开发版日期、post 和 local 后缀不作为相等门禁；无法提取数字分支的标识按不透明标识匹配。
运行 PTA 可早于构建 post，CANN 不必与构建版本相同；实际兼容性取决于运行环境的框架、CANN 和驱动。
内部文件集合、哈希及算子注册仍需匹配，加载失败不会跨分支尝试其他适配对。

vendor 位于 `lib/vendors/`，公共 SHMEM 库位于 `shmem/lib/`，适配对位于 `adapters/<ID>/`。
`build_info.json` 记录构建环境、来源和资源摘要；`adapters/manifest.json` 记录运行选择所需信息。
查看已安装制品的构建矩阵：

```bash
hyper_multicore_build_info
hyper_multicore_build_info --json
python -m pip show -f hyper-multicore
```

矩阵命令无需导入 Torch/PTA。文本输出展示框架配对和构建环境，`--json` 提供完整记录、适配对 ID 及 SO 路径。
构建矩阵用于描述构建输入，不是运行兼容性认证。

源码包通过 `python setup.py sdist` 生成，包含源码、CMake、锁文件、补丁和构建所需测试资源。

## 依赖与缓存

### 框架环境

[框架矩阵](../hyper_multicore/_build/frameworks.lock.json)维护准确 Torch/PTA 配对、下载索引和公共构建依赖。
开发者可直接增删目标或修改版本；可选参考 CANN 不参与环境身份。例如：

```json
{
  "reference_cann": "9.2.0",
  "targets": [
    {"id": "pt2.12.0.post3", "torch_version": "2.12.0+cpu", "pta_build_version": "2.12.0.post3"}
  ]
}
```

修改现有文件时保留 `schema_version`、`indexes` 和 `build_requirements`。
可通过 `indexes.dependencies_extra` 配置依赖安装的额外索引，例如：

```json
"dependencies_extra": ["https://mirrors.huaweicloud.com/repository/pypi/simple"]
```

每个地址对应一组 `pip install --extra-index-url`；缺失或空列表时不追加索引。
主源和额外索引共同参与选包，不按配置顺序作失败回退。该设置仅用于新建 venv 的依赖安装，
Torch/PTA wheel 仍从各自指定的主源下载。增删额外索引不改变环境身份，已有成功 venv 继续复用；
`ready.json` 保留创建时的请求和安装来源。默认开发构建不读取该配置。

pip 按当前 Python 和平台从各自索引选择准确框架 wheel，再解析必要依赖；缺少兼容 wheel 时失败，
不换版本或回退源码安装。构建不主动安装 pytest，也不需要独立的 pybind11 Python 包。
下载复用由 pip 自身缓存负责，尊重 `PIP_CACHE_DIR` 和 `PIP_NO_CACHE_DIR`，不清理用户全局缓存。

`HYPER_MULTICORE_VENV_PATH` 可指定 `--all` 的 venv 根目录，每个框架目标使用独立子目录：

```bash
export HYPER_MULTICORE_VENV_PATH=/data/hyper-multicore/venvs
bash build.sh --all --soc-list ascend910b --jobs 24
```

未设置或为空时使用 `build/native/envs/`。配置值必须是绝对路径，不接受相对路径或未展开的 `~/...`。
默认开发构建忽略此变量。环境创建、复用及回收使用同一根目录，启动日志显示实际位置。
更换根目录不会迁移已有环境，也不会回退查找旧目录。不同 checkout 不得并发使用同一 venv 根目录；
构建锁仅保护当前 checkout，不提供跨 checkout 的目录互斥。

`--clean` 不删除 venv。需要重建框架环境时，应停止相关构建后手动删除对应环境目录。
删除项目 `build/` 不会清理存放在外部的 venv。

存储位置不参与环境键。除额外索引外，请求、Python、架构或基础解释器变化会选择不同环境；
复用只严格核对 Torch/PTA，其他包的变化记录到实际安装版本，不自动升级或重装。
必要依赖缺失和接口问题由真实构建报告。`ready.json` 保存请求、安装来源和实际安装版本，作为追溯信息。
传递依赖由 pip 解析，不同时间创建的环境可能不同，不提供完整依赖离线重装快照。

成功全量构建后，保留当前环境及每个活动分支最近一套成功回退环境，回收更早闲置环境。
默认构建和提交前失败不触发该回收。

### vendor 与源码

默认和全量共用 `build/native/vendor-cache/<架构>-<SoC>/`，每个目标保留一份 vendor。
输入摘要覆盖锁定源码及补丁、vendor 构建脚本、SDK 头文件和链接库、CANN 实际路径及版本记录、
所选工具、调用者 Python 和有效环境参数；`--jobs` 不参与缓存身份。
仅切换构建模式而保持这些输入不变时可直接复用。
本仓通信源码按 `shmem/ccsrc/cann/` 和 `shmem/ccsrc/data_plane/` 目录整体取摘要，
目录内文件新增、删除或修改均触发 vendor 重编，包括未被当前 vendor 使用的文件。

命中时核对完整文件集合、SHA256 和 vendor 校验。输入失配、文件缺失、损坏或无法采集必要身份时重编。
新快照仅在制品提交成功并复核输入后原子发布。`build_info.json` 的 `vendor_sources` 区分原始 vendor 来源
与本次组包身份；激活脚本和最终制品每次重新生成。

该摘要不覆盖任意外部环境变化。同路径、同版本下修改未覆盖的 CANN 内部资源，或原地修改 Python 构建依赖的
内部实现后，执行 `--clean`。SDK、ops 和适配对不复用跨构建产物或 CMake 增量结果。

native 源码版本和补丁由[依赖锁](../hyper_multicore/_build/dependencies.lock.json)指定。
Git 仓库位于 `build/native/deps/`；只 fetch 缺失提交，已缓存提交不访问远端。
`ops_transformer` 的 `gmm` 和 `mhc` 共用仓库，按各自锁定提交导出片段；工作目录 HEAD 不参与版本选择。
提交、tree、archive 和补丁均按锁校验，补丁只应用于隔离源码副本。校验失败不删除源码或重新 clone。
准备器的 `--source-dir` 可校验已有源码，`--verify-only` 不获取提交或修改缓存。

### 固定工作目录

公共资源使用 `common/<架构>-<SoC列表>/`，适配对使用
`variants/<PythonABI>-<架构>-torch<Torch版本>-pt<PTA版本>-abi<0或1>/`。
这些是编译工作区，每次编译前重建对应目录；成功后保留，便于检查 CMake、编译参数及中间文件。
无归属内容或软链不会被当作可删除的工作区。

### 构建实现与故障恢复

构建入口通过标准库引导私有构建包，辅助模块无需导入业务包或 Torch/PTA。
`build_controller.py` 串联阶段，`framework_environments.py` 管理环境准备，`command_runner.py` 管理进程和日志，
`build_workspace.py` 管理工作区与锁，`build_publication.py` 校验和发布制品，`vendor_cache.py` 管理共享快照。
源码复核和输出提交状态分别由单一对象管理；提交后的维护错误不改变构建成功状态。

## CPU 开发

```bash
python -m pip install -r requirements.txt setuptools wheel "packaging>=22"
python -m pip install pytest
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_repository.py
python -m pytest tests/ut -q
```

CPU 单测包含 C++17 runtime ABI reader，需要 G++。无 native 的 editable 安装用于 Host 开发；
NPU 用例需先构建并激活 native payload。

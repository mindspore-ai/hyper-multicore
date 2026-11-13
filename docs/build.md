# 构建与安装

从源码构建 native wheel，或将已有 wheel 安装到配套环境。CPU/NPU 用例的执行和排错见
[测试指南](../tests/README.md)。

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
框架适配库使用各自选定的框架环境。构建器不会自动安装或升级调用者的软件。

Torch adapter 和 SHMEM binding 的 C++ 标准由目标 TorchConfig 决定。Torch 2.14 要求正式 C++20，
仅支持早期草案的编译器不能满足其头文件检查。需要时指定兼容目标头文件的框架编译器：

```bash
HYPER_MULTICORE_FRAMEWORK_CXX=/path/to/g++ \
  bash build.sh --all --jobs 24
```

该变量只影响框架适配库；公共 CANN vendor/SDK/ops 使用 CANN 工具链。
未设置时沿用 `CXX` 或系统默认编译器。编译失败会保留原始诊断，C++20 错误附带编译器重试命令。

## 编译

```bash
# 使用当前安装的 Torch/PTA，默认包含三个 SoC。
bash build.sh --jobs 24

# 为框架矩阵中的全部目标生成一个 wheel。
bash build.sh --all --jobs 24

# 构建双 SoC 产物。
bash build.sh --soc-list ascend910b,ascend910_93 --jobs 24

# 构建 Ascend950 产物。
bash build.sh --soc-list ascend950 --jobs 24

# 仅生成 native payload。
bash hyper_multicore/build.sh --soc-list ascend910b --jobs 24
```

| 行为 | 默认构建 | `--all` |
| --- | --- | --- |
| 框架来源 | 当前解释器实际导入的 Torch/PTA | [框架矩阵](../hyper_multicore/_build/frameworks.lock.json)中的准确配对 |
| 版本选择 | 接受正式版、开发版和自定义版本标识 | 使用配置指定的发行版本 |
| 框架安装 | 使用本机环境 | 在指定目录创建或复用 venv，默认 `build/native/envs/` |
| 公共资源 | 使用调用者 CANN 工具环境 | 使用调用者 CANN 工具环境 |
| 框架适配库 | 编译一组 | 按预算并行编译全部配置目标 |
| wheel | 包含当前 PTA 标签 | 不带单一 PTA 标签 |
| 构建源码检查 | 默认关闭，`--tests on` 开启 | 默认开启，`--tests off` 关闭 |

`--tests on/off` 控制[轻量构建源码检查](../tests/ut/build/README.md)，显式设置覆盖模式默认值。
检查使用入口 Python，在获得构建锁、初始化日志和输出事务后执行一次，先于环境准备及 native 编译。
它不安装测试框架、不执行真实编译或设备操作；失败时停止构建并沿用提交前清理规则。
关闭该检查不取消实际编译、加载检查或制品完整性校验。

```bash
bash build.sh --tests on
bash build.sh --all --tests off
```

默认路径不读取框架矩阵，也不按框架版本格式、分支或上下限拒绝构建。源码框架无需发行元数据，
版本不可得时记为 `unknown`。日志展示实际 Python、Torch/PTA 版本和安装位置。
两种模式均使用当前 CANN，不按其版本号拒绝构建；真实缺包、缺工具、头文件或符号问题由编译和加载检查报告。
矩阵中的可选 `reference_cann` 用于提示和追溯，与实际版本不同或实际版本未知时告警并继续。

每次构建准备一份公共资源。vendor 可复用，SHMEM SDK、ops 和框架适配库每次重新建立工作目录并编译。
910B/910C 共用 `arch22`，950 使用 `arch35`；每架构只编译一次 vendor kernel，注册信息按 SoC 生成；每组框架构建一份 Torch 适配库和所需的 SHMEM binding。
各后端在独立 Python 进程中完成加载检查，不执行 NPU 计算。
最终清单始终按矩阵配置顺序排列。

`--soc-list` 支持 `ascend910b`、`ascend910_93` 和 `ascend950`，保留列表顺序并去除各项两侧空白。
两种模式均默认包含三个 SoC，显式指定时支持任意非空子集，每次只生成一个 wheel。
重复项、空项和未知值会在获取构建锁及清理产物前被拒绝。

950 构建包含 MegaMoE、MegaGate、MegaMHC 前后向六个算子及 SHMEM SDK/ops 和框架适配库。
构建与加载校验不覆盖设备执行、数值正确性、通信及性能。

`--jobs` 为 J、矩阵目标数为 N 时，框架适配库并行度如下：

| 编译预算 | 并行框架适配库 |
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
| 公共资源工作区 | `build/native/common/<CPU架构>/` |
| SHMEM SDK/ops 工作区 | `build/native/sdk/<CPU架构>/<设备架构>/` |
| 框架适配库工作区 | `build/native/variants/<框架组合>/` |
| 依赖源码仓库 | `build/native/deps/<依赖>/` |
| 框架安装环境 | `build/native/envs/<安装身份>/` |
| vendor 缓存 | `build/native/vendor-cache/<CPU架构>-<设备架构>/` |
| 终端记录 | `build/native/logs/build.log` |
| 工具原始输出 | `build/native/logs/tools.log` |
| 阶段、命令、耗时、退出码和日志字节区间 | `build/native/logs/stages.json` |

终端实时显示 pip/CMake 输出、环境新建或复用、安装包版本、编译进度和 vendor 缓存决策。
并行工具输出带 PTA 和阶段标识，静默阶段定期显示已用时间；内部 JSON 只写入工具日志。
失败摘要对应当前阶段，完整诊断可按 `stages.json` 中的区间定位。

构建通过 `build/native/.build.lock` 互斥。等待时显示持锁信息，Ctrl-C 可取消等待而不影响持锁构建。
独立 SHMEM SDK/ops 入口使用同一把锁，并复用相同的公共资源与 SDK 工作区。
主构建入口获锁后替换上次日志；帮助命令、参数错误及等待锁不会清理日志。
日志仅保留最近一次，如需长期保存应在下次构建前复制到构建目录外。

组包成功后清理本次 staging；失败时保留日志和 staging，供排查使用，临时输出在下次构建开始时回收。
源码在构建和发布期间变化会拒绝发布。任一框架适配库失败或收到 Ctrl-C/SIGTERM 时，停止领取新任务，
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

激活脚本设置随包 vendor 的搜索路径，应在导入 Torch/torch-npu 之前执行。
SHMEM loader 按设备后端选择库并使用绝对路径加载。
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

运行时按 Torch/PTA 数字分支、Python ABI、CPU 架构及 C++ ABI 唯一选择一组框架适配库。
开发版日期、post 和 local 后缀不作为相等门禁；无法提取数字分支的标识按不透明标识匹配。
这些选择规则不代表二进制兼容保证。不同 post 的 PTA 仍可能存在执行桥符号差异，
运行环境须提供所选适配库实际依赖的接口。CANN 版本不要求相等，仍需与框架和驱动兼容。
内部文件集合、哈希及算子注册仍需匹配，加载失败不会跨分支尝试其他框架适配库。

vendor 位于 `lib/vendors/`，SHMEM 资源按 `arch22/arch35` 保存于 `shmem/lib/<arch>/`。
每个 `adapters/<ID>/` 包含一份 Torch 库，以及 `shmem/<arch>/` 下的各架构 binding。
ID 格式为 `pt<Torch版本>-pta<torch_npu版本>-<12位构建指纹>`，例如
`pt2.7.1-pta2.7.1.post11-e53991cd6b50`。Torch 的标准 CPU 分发后缀不进入可读标识，完整版本保存在构建记录中。
vendor kernel 位于 `op_impl/ai_core/tbe/kernel/<arch>/`；910B/910C 共用一份 `arch22` kernel，
950 使用 `arch35`。CANN 注册和查找索引继续按已选 SoC 保存，并指向对应架构的 kernel。
`build_info.json` 记录构建环境、来源和资源摘要；`adapters/manifest.json` 记录运行选择所需信息。
首次使用 SHMEM 时，查询可见逻辑设备 0 的 NPU 架构，按绝对路径加载一套后端；不创建 NPU 上下文。
同一进程可见设备须属于同一架构，SHMEM 实际初始化仍使用调用方设置的当前设备。
设备查询失败或对应后端未打包时直接报错，不回退至另一后端。
查看已安装制品的构建矩阵：

```bash
hyper_multicore_build_info
hyper_multicore_build_info --json
python -m pip show -f hyper-multicore
```

矩阵命令无需导入 Torch/PTA。文本输出展示框架配对和构建环境，`--json` 提供完整记录、框架适配库 ID 及 SO 路径。
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
    {"id": "pt2.12.0-pta2.12.0.post3", "torch_version": "2.12.0+cpu", "pta_build_version": "2.12.0.post3"}
  ]
}
```

修改现有文件时保留 `schema_version`、`indexes` 和 `build_requirements`。
可通过 `indexes.dependencies_extra` 配置框架下载及依赖安装共用的额外索引，例如：

```json
"dependencies_extra": ["https://mirrors.huaweicloud.com/repository/pypi/simple"]
```

每个地址在 Torch/PTA 的 `pip download` 和依赖的 `pip install` 中均对应一组
`--extra-index-url`；缺失或空列表时不追加索引。Torch、PTA 及依赖分别保留各自配置的主源，
主源和额外索引共同参与选包，不按配置顺序作失败回退，也不保证优先从额外索引下载。
准确框架版本要求保持不变。增删额外索引或修改展示 ID 不改变环境身份，已有成功 venv 继续复用；
每次复用仍校验实际框架版本。`ready.json` 更新当前请求和 ID，保留原安装来源。默认开发构建不读取该配置。

pip 按当前 Python 和平台从各自索引选择准确框架 wheel，再解析必要依赖；缺少兼容 wheel 时失败，
不换版本或回退源码安装。构建不主动安装 pytest，也不需要独立的 pybind11 Python 包。
下载复用由 pip 自身缓存负责，尊重 `PIP_CACHE_DIR` 和 `PIP_NO_CACHE_DIR`，不清理用户全局缓存。

`HYPER_MULTICORE_VENV_PATH` 可指定 `--all` 的 venv 根目录，每个框架目标使用独立子目录：

```bash
export HYPER_MULTICORE_VENV_PATH=/data/hyper-multicore/venvs
bash build.sh --all --jobs 24
```

未设置或为空时使用 `build/native/envs/`。配置值必须是绝对路径，不接受相对路径或未展开的 `~/...`。
默认开发构建忽略此变量。环境创建、复用及回收使用同一根目录，启动日志显示实际位置。
更换根目录不会迁移已有环境，也不会回退查找旧目录。不同 checkout 不得并发使用同一 venv 根目录；
构建锁仅保护当前 checkout，不提供跨 checkout 的目录互斥。

`--clean` 不删除 venv。需要重建框架环境时，应停止相关构建后手动删除对应环境目录。
删除项目 `build/` 不会清理存放在外部的 venv。

存储位置、展示 ID 和额外索引不参与环境键。实际版本、主要安装源、构建依赖、Python 或架构变化会选择不同环境；
复用只严格核对 Torch/PTA，其他包的变化记录到实际安装版本，不自动升级或重装。
必要依赖缺失和接口问题由真实构建报告。`ready.json` 保存请求、安装来源和实际安装版本，作为追溯信息。
环境键规则升级后，旧键环境需要重新准备，不自动迁移或回退查找。
传递依赖由 pip 解析，不同时间创建的环境可能不同，不提供完整依赖离线重装快照。

成功全量构建后，仅保留本次使用的框架环境，回收其余受管环境。
默认构建和提交前失败不触发该回收。

### vendor 与源码

默认和全量共用 `build/native/vendor-cache/<CPU架构>-<设备架构>/`，每个设备架构保留一份 vendor。
输入摘要覆盖锁定源码及补丁、vendor 构建脚本、SDK 头文件和链接库、CANN 实际路径及版本记录、
所选工具、调用者 Python 和有效环境参数；`--jobs` 不参与缓存身份。
仅切换构建模式而保持这些输入不变时可直接复用。
本仓通信源码按 `shmem/ccsrc/cann/` 和 `shmem/ccsrc/data_plane/` 目录整体取摘要，
目录内文件新增、删除或修改均触发 vendor 重编，包括未被当前 vendor 使用的文件。

命中时核对完整文件集合、SHA256 和 vendor 校验。输入失配、文件缺失、损坏或无法采集必要身份时重编。
新快照仅在制品提交成功并复核输入后原子发布。`build_info.json` 的 `vendor_sources` 区分原始 vendor 来源
与本次组包身份；激活脚本和最终制品每次重新生成。

该摘要不覆盖任意外部环境变化。同路径、同版本下修改未覆盖的 CANN 内部资源，或原地修改 Python 构建依赖的
内部实现后，执行 `--clean`。SDK、ops 和框架适配库不复用跨构建产物或 CMake 增量结果。

native 源码版本和补丁由[依赖锁](../hyper_multicore/_build/dependencies.lock.json)指定。
Git 仓库位于 `build/native/deps/`；只 fetch 缺失提交，已缓存提交不访问远端。
`ops_transformer` 的 `gmm` 和 `mhc` 共用仓库，按各自锁定提交导出片段；工作目录 HEAD 不参与版本选择。
950 SHMEM 使用的 JSON 头文件也从锁定提交导出，JSON 锁定记录参与其 SDK 工具链身份。
提交、tree、archive 和补丁均按锁校验，补丁只应用于隔离源码副本。校验失败不删除源码或重新 clone。
准备器的 `--source-dir` 可校验已有源码，`--verify-only` 不获取提交或修改缓存。

### 固定工作目录

公共资源使用 `common/<CPU架构>/`，所选 SoC 记录在构建身份与 SDK 交接记录中；框架适配库使用
`variants/<PythonABI>-<架构>-pt<Torch版本>-pta<PTA版本>-abi<0或1>/`。
SDK/ops、vendor 和 binding 的子工作区按 `arch22/arch35` 隔离。
SDK 工作区位于 `build/native/sdk/<CPU架构>/<设备架构>/`，不同 SoC 选择使用同一 SDK 路径。
目录中直接保存 `shmem-source/`、`shmem/`、`shmem-install/` 和 `ccsrc-ops/`；工具链指纹写入 SDK 交接记录。
这些是编译工作区，每次编译前重建对应目录；成功后保留，便于检查 CMake、编译参数及中间文件。
SDK 在准备入口按设备架构检查归属并重建，仅清理当前选择的架构。
无归属的非空目录、归属不匹配的目录，以及自身或祖先为软链的路径均拒绝清理；已有无标记目录需先移走。

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

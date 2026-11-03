# 构建与安装

从源码构建 native wheel，或将已有 wheel 安装到配套环境。CPU/NPU 用例的添加、执行和排错见
[测试指南](../tests/README.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Linux；CPU 架构需具备配套的 Torch/PTA wheel 和 CANN 工具链 |
| Python | 支持 >=3.10,<3.13；构建需要 Python 开发头文件和所选框架的兼容 wheel |
| 框架 | 相互配套的 Torch、torch-npu/PTA，NumPy >=1.26,<2 |
| Ascend | 源码构建要求 CANN toolkit/ops 9.2.0；运行环境的 PTA/CANN/driver 须彼此兼容 |
| 构建工具 | GCC/G++、CMake >=3.18、GNU Make、Git 和 CANN 编译工具（pybind11 头文件来自 Torch） |

先激活所选 Python 环境并安装配套框架，然后从源码根目录运行：

```bash
source /path/to/cann/set_env.sh
python -m pip install -r requirements.txt setuptools wheel "packaging>=22" decorator attrs psutil
```

CANN 路径按实际安装位置填写；GCC 版本由
[工具链检查脚本](../scripts/check_gcc_version.sh)检查。
Torch adapter 和 SHMEM binding 的 C++ 标准由目标 TorchConfig 决定。完整矩阵中的 Torch 2.14
要求正式 C++20；仅支持早期 C++20 草案的编译器不能满足其头文件检查。
需要时在构建前显式设置 `HYPER_MULTICORE_FRAMEWORK_CXX` 为兼容目标 op-plugin 头文件、且支持该标准的 G++。
例如 `HYPER_MULTICORE_FRAMEWORK_CXX=/path/to/g++ bash build.sh --all --soc-list ascend910b --jobs 24`。
该变量只影响框架适配对及其预检，公共 CANN vendor/SDK/ops 使用 CANN 工具链；
未设置时沿用普通 `CXX` 或系统默认编译器。框架与公共资源分别记录工具链并独立计算缓存身份。
仅满足语言标准仍不足以保证兼容；实际执行桥编译和链接预检是进入 native 构建的必要条件。
预检失败时，终端显示 Torch/PTA、所选编译器、首个编译/链接错误和完整日志路径。
若错误要求 C++20，会同时给出 `HYPER_MULTICORE_FRAMEWORK_CXX` 重试命令；
可使用兼容 op-plugin 的 GCC 12.4。部分编译器接受 `-std=c++20`，但 `__cplusplus` 低于 `202002L`，
因此仍不能通过 Torch 2.14 的头文件检查。选择满足要求且兼容 op-plugin 的编译器即可，
无需更换公共 CANN 工具链或清理有效缓存。
构建记录实际编译器路径和版本，不自动更换或安装工具链；默认构建使用当前选择的编译器。

## 编译

```bash
# 默认：只为当前安装的 Torch/PTA 构建一套适配对。
bash build.sh --soc-list ascend910b --jobs 24

# 发布：为框架锁中的全部组合构建适配对，汇总一个 wheel。
bash build.sh --all --soc-list ascend910b --jobs 24

# 为当前框架构建双 SoC 产物。
bash build.sh --soc-list ascend910b,ascend910_93 --jobs 24

# 仅构建 native payload，不生成 wheel。
bash hyper_multicore/build.sh --soc-list ascend910b --jobs 24
```

`--soc-list` 选择编译目标，`--jobs` 设置并行度。`--clean` 清理 native 工作及安装输出，
保留 Git/框架下载缓存和已准备的框架环境。`--all` 只扩大框架矩阵，
不隐含清缓存或增加 SoC。默认路径按实际头文件、执行桥和编译能力预检，不要求达到全量矩阵的 post 版本；
缺少能力时明确失败，不自动替换本机框架。SHMEM 为必需内部依赖。任一 native 构建或校验失败都会非零退出并停止组包。

| 输出 | 位置 |
| --- | --- |
| 完整 wheel | `dist/`，实际文件路径由根构建命令打印 |
| 公共资源及逐组合输出 | `build/native/common/<架构>-<SoC>/`、`build/native/variants/<框架组合>/` |
| 可运行 native payload | `build/native/payload/hyper_multicore/` |
| 构建及逐阶段日志 | `build/native/logs/`，包含完整终端记录 `build.log`、工具原始输出 `tools.log` 和 `stages.json` |

终端实时显示完整 pip 安装/下载及 CMake 编译输出，同时保存完整终端记录和工具原始输出。
控制器使用 `HH:MM:SS INFO/WARN/ERROR` 加阶段说明，不显示内部命令编号。
超过约 10 秒没有新输出时，提示当前阶段及已用时间；内部版本查询和哈希采集仅写入阶段记录。
环境准备显示 `Environment i/N`、新建或复用状态及实际包版本列表；
复用先核对版本和 `pip check`，通过后明确提示不下载、不重装。Python/架构/构建要求变化会使用不同环境键。
公共资源和适配对显示缓存命中或重建原因及 `Adapter i/N`；结束时汇总耗时、环境新建/复用、
native 缓存命中及构建数量、制品路径和日志位置。完整日志路径只在启动、失败及结束时显示。
这些是实时状态反馈，不需要确认输入，重定向输出或在 CI 中运行也保持普通文本。

构建锁被占用时显示等待状态，可以用 Ctrl-C 取消等待，不影响持锁构建。
构建锁位于 `build/native/.build.lock`；获锁后记录 PID、开始时间及模式，互斥由 `flock` 决定。
获得构建锁后初始化 `logs/`，仅保留最近一次构建日志。
失败或中断的日志和 staging 保留到下一次构建开始；`--help`、参数错误和等待锁不会清理它们。
wheel 和 payload 成功发布后清理本次 staging；下一次构建回收遗留的 staging 及未发布 payload 临时副本。
这些清理只针对自动生成的临时目录，保留正式 payload、框架环境、wheel 快照及 native/source 缓存；`dist/` 按下文的本次构建策略管理。
如需长期保存某次构建日志，请在下次构建前复制到构建目录外。

构建日志由上述三个文件组成。
`stages.json` 采用阶段列表，记录命令、耗时、状态、退出码及 `tools.log` 的起止字节位置；
失败和中断也保存当前阶段，错误摘要只读取该阶段区间。内部 JSON 原样返回构建器并写入工具日志。
CANN Python 导入检查只在终端显示结果，原始输出保存在 `tools.log`。

编译链接预检按输入复用成功记录：框架头文件/库、Torch CMake 配置、Python 开发资源、
CANN 头文件、实际链接依赖、编译器/生成器及有效选项均参与校验。
输入及预检产物匹配时跳过 CMake configure/build；缺少记录、输入变化、输出损坏或 `--clean` 时重新预检。
只有预检完成且输入复核通过后才保存成功记录；失败或中断不留下可复用的成功状态。
预检键只记录实际选择的编译器、CMake、生成器后端及编译器使用的汇编/链接等工具；
Unix Makefiles 不记录未使用的 Ninja。完整 `PATH` 作为诊断信息保留，不直接参与预检键，
无关 IDE 路径或其他未使用程序的变化不会触发重编。实际工具路径、内容或有效选项变化仍会失效。
预检配置会校对实际生成器、构建后端和编译器是否与指纹一致。

环境包版本与框架身份在目标解释器中采集，依赖一致性通过 `pip check` 检查。
单次构建复用源码身份、单个探针复用重复文件哈希，大文件流式计算 SHA256；不跨构建缓存文件哈希。
阶段记录分别提供框架导入、包版本、头文件/库哈希和源码身份耗时，不参与缓存键。
原生缓存记录实际编译器、辅助工具及其运行库的内容摘要；公共资源另外记录 CANN 的实际开发搜索根、
构建入口和链接输入。软链记录真实路径，编译依赖用于核对覆盖范围，不扫描整个 CANN。
这些输入在提交完成标记前复核；变化时拒绝认证。固定工作目录保存完整输入指纹；指纹变化时清理该目标输出后重建，
同一目标的不同输入使用相同工作目录。
公共资源、适配对和预检会对照最近一次成功构建的分类输入说明失配原因；
分类摘要缺失时不展示差异详情，缓存完整性仍按认证记录校验。
搜索路径参与输入比较，构建器不自动改写调用者环境。
`cache-inputs.json` 对重复的开发文件清单仅保存分类摘要和文件数量，保留可读版本、工具及选项；
它只解释缓存变化，缺失或损坏不影响缓存认证。预检认证中的实际依赖和输出哈希仍完整校验。

`preflight/index.json` 按 Python ABI、架构和默认／全量模式记录最近一次成功使用的目录，
保留两种模式的引用并集。缺少成功全量构建的引用记录时，默认构建不回收未引用目录；
完整构建及输入复核成功后才清理无引用、且 CMake 来源确认属于本构建器的预检目录。
失败构建不回收，索引损坏或目录归属不明时保守保留并提示。有效 CMake 目录保持原位。

失败时终端显示当前阶段的简短根因、命令解释器及日志位置，包含框架探针、缺包、缺头文件和下载失败；
完整工具输出仍写入日志，内部 JSON 不刷屏。底层脚本同时提供 `HP_NATIVE_REASON_CODE`。
日志包含依赖获取、源码组装、SHMEM、vendor、Torch adapter 和制品校验阶段。
源码及生成物分开保存，构建不会把 native 文件写回生产源码目录。

## 安装与激活

### 使用 wheel

按 Python ABI、CPU 架构和 Torch/PTA 分支选择制品，安装构建命令打印的唯一 wheel。
运行 PTA 可以早于构建 post，CANN 不必与构建版本相同；本包不强制该版本配对，
使用方须准备彼此兼容的框架、CANN 和驱动。
替换下面示意路径，不要用通配符一次安装多个制品：

```bash
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
```

`--no-deps` 使用当前环境预先配置的框架。激活脚本设置随包 vendor 和 SHMEM 库的路径；
必须在导入 Torch/torch-npu 之前激活。更换制品或环境后应启动新的 Python 进程。

检查安装来源时，在源码目录外运行，并移除指向源码的 `PYTHONPATH`：

```bash
python -c 'import hyper_multicore; print(hyper_multicore.__file__)'
```

输出应指向安装位置。然后按[快速开始](../README.md#快速开始)运行现有示例，
或按[测试指南](../tests/README.md)运行指定用例。

### 使用源码 payload

从源码根目录完成 native 构建后，在当前源码目录中激活本地 payload：

```bash
source /path/to/cann/set_env.sh
source build/native/payload/hyper_multicore/lib/set_env.bash
```

此模式从当前源码导入 Python 实现、从 `build/native/payload/` 定位 native 库。
不要同时激活另一安装版本的 payload。

## 制品内容与兼容性

默认 wheel 版本包含实际 `torch-npu` 发行版本，例如 `0.1.0.dev0+pt.2.12.0`。
`--all` 全量 wheel 使用组件版本，不带单一 PTA 后缀，例如
`hyper_multicore-0.1.0.dev0-cp310-cp310-linux_aarch64.whl`。
框架矩阵配置的 PTA 版本为 2.7.1.post11、2.9.0.post9、2.10.0.post7、2.11.0.post3、2.12.0.post3、2.13.0 和 2.14.0，
版本配对、下载索引和公共构建依赖见[框架矩阵](../hyper_multicore/_build/frameworks.lock.json)。
配置不绑定 Python ABI、CPU 架构、wheel 文件名或 SHA256；pip 按当前解释器选择兼容 wheel。
所选版本缺少兼容 wheel 时构建失败，不自动换版本或回退源码构建。Python ABI 和 CPU 架构由 wheel 标签约束。

`--all` 仅精确指定 Torch CPU/PTA 版本；传递依赖由 pip 按官方声明解析，构建工具和
CANN 的 NumPy/decorator/attrs/psutil 依赖集中声明，不手工固定具体版本。
构建不主动安装 pytest 及其专属依赖，也不需要独立的 pybind11 Python 包。
首次解析先下载框架，再解析必要依赖，使用独立暂存目录。结果保存在
`build/native/framework/resolutions/cp<ABI>-<架构>-pt<PTA版本>-<12位环境ID>.json` 及同名前缀的
`.pip-report.json` 中；完整环境键保存在 JSON 内，短 ID 冲突时拒绝覆盖。
wheel 本体平铺共享存储在 `build/native/framework/wheels/<原始文件名>.whl` 中，
同名同内容文件复用，同名异内容报错，不覆盖现有快照引用的数据。
自动记录实际版本、来源及 SHA256；安装仅使用本地带哈希清单，不联网补包。
解析快照与 pip 报告平铺存储，复用时保留有效的引用和安装清单。新请求优先从来源、
版本和平台匹配的快照复用框架 wheel；解析报告确定的依赖文件在共享池校验一致时也不重复下载。
来源或平台不匹配仍按指定索引获取，损坏和同名异内容明确报错。pip 来源配置保持隔离，
但尊重用户设置的 `PIP_CACHE_DIR` 和 `PIP_NO_CACHE_DIR`；不管理或清理用户的全局 pip 缓存。
成功 venv 保存在 `build/native/framework/envs/`，复用时核对已记录版本并运行 `pip check`，
不重新解析或自动升级；环境变化时明确报错。框架环境配置及实际依赖也记录在制品的
`build_info.json` 各变体 `framework_environment` 中。

相同配置在不同时间首次解析可能得到不同传递依赖；本地解析快照可用于重建，
但不承诺跨时间完整 Python 环境完全一致。已完成快照缺文件或哈希变化时明确失败，
不静默联网修复。Python/架构/配置变化创建独立环境。
完整身份保存在生成的 `build_info.json`，包括 Python/C++ ABI、Torch/PTA、CANN、
op-plugin 头文件、源码和 native 文件摘要。源码摘要包含算子 `.ini` 构建配置。
`native_sha256` 保留共享库摘要，`payload_sha256` 记录除 `build_info.json` 自身外的完整
payload 文件及 SHA256，包括 kernel、元数据、索引、配置和激活脚本。
打包前检查所选 SoC 的必需运行资源、文件集合及摘要；payload 缺少完整清单时会被拒绝，
应从可信构建输入重新生成并校验清单，不能通过跳过校验打包。
加载按 Torch/PTA 分支、Python/CPU/C++ ABI 选定唯一适配对，不比较运行 PTA post 与构建版本的大小，
也不以外部 PTA 库 SHA256、Torch commit、CANN 版本或版本文件哈希拦截。
缺少 PTA、缺库/符号或初始化失败时保留原始原因和选中变体；一个进程不换用其他适配对重试。
每组快照独立保存请求及准确解析结果，相同 wheel 本体只保存一份。
本地安装清单使用哈希、`--no-index`、`--no-deps` 和纯 wheel 模式；缺少或损坏的文件不会联网补齐。
共享文件按只读数据使用，不能直接修改；构建不回收仍被快照引用的共享 wheel。

`dist/` 只表示本次构建结果：获得构建锁后、开始实际构建前清空旧产物，成功后仅保留本次 wheel。
默认和 `--all` 共用该目录，构建结果直接存放在 `dist/` 下。
构建事务提交前失败或中断会清空 `dist/` 和正式 `build/native/payload/hyper_multicore/`，
该规则覆盖输出发布后、提交检查尚未通过的情况，失败时不恢复旧产物。
提交点位于全部制品校验、最终输入复核及事务成功退出之后。此后的 venv／预检回收及临时文件清理
属于维护操作：失败或被取消仅告警并停止后续清理，保留已提交的有效制品。
等待锁、提交前构建、提交后维护的取消状态分别提示；取消等待不影响持锁构建及其输出。
日志及失败 staging 仍按前述规则保留；有效 native 缓存不受正式产物清理影响。
`--help`、参数解析错误和等待锁尚未进入实际构建，不清理正在使用的输出。
需要长期保留的制品应在下一次构建前移到 `dist/` 外，安装时使用本次成功打印的准确路径。

每个变体在 `adapters/<variant-id>/` 中携带 `libhyper_multicore_torch.so` 与
`hyper_multicore_shmem_torch.so`；MegaMoE、MegaGate 和 MegaMHC 共用该变体的 Torch adapter，
通过统一入口在首次调用时加载。vendor 位于
`lib/vendors/hyper_multicore_nn/`，私有通信库位于 `shmem/lib/`。
wheel 包含运行所需 native payload 和 runtime 头文件，不包含完整 Torch、CANN 或 driver。
没有 `hyper_parallel` 兼容包，也不支持在同一进程中混用两套 native runtime。

查看当前安装包的实际构建矩阵，无需导入 Torch/PTA 或加载 SO：

```bash
hyper_multicore_build_info
hyper_multicore_build_info --json
python -m pip show -f hyper-multicore
```

矩阵命令读取当前安装位置的 `build_info.json` 和 `adapters/manifest.json`，展示构建模式、
框架版本、C++ ABI 及 CANN/SoC。适配对 ID 和具体 SO 路径通过 `--json` 查看。
默认包只显示一组；未安装 PTA 也能查看。
`pip show` 保留标准元数据输出，`-f` 可定位包内清单文件。
构建矩阵描述制品的构建输入，运行兼容性取决于安装环境的框架、CANN 和驱动。

生成源码包：

```bash
python setup.py sdist
```

sdist 使用通用源码版本，包含源码、CMake、锁文件、补丁及构建所需测试资源。
安装 wheel 和导入 Python 包均不会自动编译算子。

## 依赖与缓存

依赖固定在[锁文件](../hyper_multicore/_build/dependencies.lock.json)。一个顶层条目对应一个
Git 仓库；同仓库的多个版本放在 `revisions` 中。`ops_transformer` 的 `gmm` 和 `mhc`
共用 `build/native/deps/ops_transformer/src`，按锁定 commit 分别导出源码片段。
`ops_nn` 和 `shmem` 使用各自的缓存仓库，工作目录的 HEAD 不参与版本选择。

准备器只 fetch 缺失提交，完整热缓存不访问远端。提交、tree、archive 和补丁均按锁文件校验，
补丁只应用于隔离源码副本。依赖校验失败直接报错，不删除源码或重新 clone 掩盖失败。
构建缓存按输入指纹区分。`--source-dir` 校验已有依赖源码；`--verify-only` 不获取提交或修改缓存。

全量构建顺序使用项目内的 `build/native/framework/envs/`，通过各环境的绝对 Python 路径执行，
不激活或覆盖调用者环境。包哈希或基础解释器变化生成新环境；成功全量构建后保留当前环境及每条维护分支
最近一套成功回退环境，回收更早闲置环境。只有构建事务提交后才执行；提交前失败及默认路径不触发该回收。
成功时间通过临时文件原子更新，维护中断不会截断现有完成标记。

离线用户预先准备运行依赖后安装本地全量 wheel，安装、导入和首次调用不会下载框架、获取 native 源码或执行编译。
本仓不提供离线源码构建流程。


### 固定工作目录

工作目录相对于 `build/native/`。公共资源使用 `common/<架构>-<SoC列表>/`；
预检使用 `preflight/<PythonABI>-<架构>-torch<Torch版本>-pt<PTA版本>-abi<0或1>/`，
Adapter 使用 `variants/` 下的同名子目录。
名称包含 Python ABI、架构、完整框架版本和 C++ ABI；完整指纹保存在认证 JSON 中。
默认和全量构建使用同一命名规则，相同环境直接复用，不同 PTA 发行版本分别保留。

未知目录和软链不作为可删除缓存，固定槽位遇到无归属内容时明确拒绝替换。
框架环境与依赖数据位于 `framework/`，原生源码缓存位于 `deps/`。

阶段认证前后仅复核本阶段登记的外部输入，最终提交前统一复核全部登记输入；
源码观察独立执行，同一事务不在每个 Adapter 中重复进入、退出。
单次构建复用文件哈希和重复目录的输入枚举，最终仍检查文件代次、软链及目录成员变化。
`stages.json` 另外记录公共输入采集、输出完整性、源码与外部输入复核次数及耗时、组包和维护耗时。
这些计时用于解释热构建开销，不参与缓存键；日志仍只有 `build.log`、`tools.log`、`stages.json`。


### 构建实现与故障恢复

命令入口通过标准库引导独立构建包，构建内部按正常模块导入复用；辅助模块不导入业务包或 Torch/PTA。
实际框架身份只在选定环境的显式探针中采集。命令执行器只负责子进程、实时日志和退出状态；
框架证据、缓存认证与发布结果由明确的构建上下文和结果记录持有。

构建实现按职责集中维护：`command_runner.py` 管理命令、终端与日志；
`build_workspace.py` 管理目录定位、归属、锁和清理；`preflight_cache.py` 管理预检认证及回收；
`framework_store.py` 管理共享 wheel 和解析快照；`native_cache.py` 管理 native 缓存认证与失效诊断。
构建状态、输入采集和制品发布由独立模块管理，调度器负责串联这些步骤。

### 框架缓存目录布局

框架相关数据统一位于 `build/native/framework/`：`envs/` 保存隔离环境，
`resolutions/` 保存解析快照及 pip 报告，`wheels/` 平铺保存已校验的 wheel。
路径定位及归属校验由构建器的 `build_workspace.py` 管理。

## CPU 开发

```bash
python -m pip install -r requirements.txt setuptools wheel "packaging>=22"
python -m pip install pytest
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_repository.py
python -m pytest tests/ut -q
```

CPU 单测包含 C++17 runtime ABI reader，需要 G++。没有 native 的 editable 安装仅用于
Host 开发；执行 NPU 用例仍需先构建并激活 native payload。

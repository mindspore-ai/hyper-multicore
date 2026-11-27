# 诊断设计

本文面向维护诊断工具和接入新 megakernel 的开发者，解释采集如何与正常执行共用调用边界，
以及配置、buffer 和完成时机的归属。工具选择、采集参数与结果读取见[诊断指南](../diagnostics.md)。

外层 profiler 可以看到 kernel 的起止时间，但不能解释内部 task 的依赖等待与计算通信重叠。
内部 Profiler 因此在设备上记录 cycle 区间，Host 再恢复阶段和 owner 信息。Tensor dump 则在调用边界
或安全的 task 端口保存实际值，两种工具需要不同的数据和存储，但都依赖同一次 native 调用的参数与完成边界。

诊断层通过公共调用协议连接二者，避免让工具管理业务事件计数器、workspace 或反向租约。
设备只写固定格式的记录或快照，名称解析与文件生成放到 Host；关闭路径不承担逐 task 的诊断判断。
这保留了普通执行路径，也意味着诊断需要额外的配置、显存、同步和 Host 回读，采样数据不等同于无打点性能。

## 总体架构

### 并列工具与公共层

profiling 与 Kernel/Task dump 同属于 `diagnostics`，通过协议与组合接入，不相互继承或调用：

```text
diagnostics/
├── api.py                       # prepare / MegaKernelDiagnostics.launch
├── common/
│   ├── graph.py                 # 通用图契约与只读 task/stage/owner metadata
│   ├── runtime.py               # 普通 runtime 与惰性 metadata
│   ├── call.py                  # 完整 LaunchArguments 与完成/取消边界
│   ├── session.py               # DiagnosticSession 协议、进程互斥
│   └── schedule.py              # DiagnosticAction、schedule
├── profiler/
│   ├── api.py                   # profiling session、buffer 池、pending、窗口
│   ├── cycle_trace.py           # profiler 私有配置、容量、解析
│   └── trace_merger.py           # 离线融合
└── dumper/                      # 独立的 Kernel/Task dump 工具；不导入 profiler
    ├── __init__.py              # 对外导出
    ├── api.py                   # 配置、session、调用编排与导出
    ├── _kernel.py               # 调用边界快照与 stage 文件
    ├── _task.py                 # 任务端口视图、设备采集规划与解码
    └── _layout.py               # stage 目录名称规则与张量文件写出
```

依赖方向是 `profiler → common ← dumper`；公共层不导入具体工具或 MegaMoe。
业务接入提供 owner 规则与动态 Tensor 视图 resolver；这些解释由对应 megakernel 的主机侧维护。
通用 profiler 不认识 Expert、GMM 或 MoE 路由。

用户通过 `profiler.profile()` 或 `dumper.dump()` 创建 `ProfilerSession` 或 `DumperSession`。
组件适配器从 `hyper_multicore.diagnostics.dumper` 导入 `TaskDumpTensorView` 和 `TaskDumpTensorViewContext`。
带 `_` 前缀的实现模块仅供
dumper 内部使用；这是接口约定，Python 不禁止显式导入。内部单测可直接验证这些模块。

### 数据流

```text
plan：只创建标准 runtime_config_tensor
    │
每次调用检查 diagnostics.requires_diagnostic_launch
    ├─ False：直接调用普通 native wrapper → mode=0 → ProcessFast
    └─ True：通用 launch → 唯一活动 session.prepare_call(context)
                              │
                    选中调用后解析公共 metadata
                              │
                    ┌─────────┼──────────┐
                 Profiler   Kernel dump   Task dump
                 mode=1       mode=0       mode=2
                 cycle记录    调用边界    task端口
                    │         复制       快照
                 pending      │          │
                 同步回读     CPU快照    解码为CPU快照
                    │         └────┬─────┘
                    │          按调用写出 .pt
                 完成窗口        完成窗口
                 trace与回调     发布 manifest 与回调
```

每次只产生当前工具的一个句柄；不存在先准备 profile handle 再叠加或覆盖 dump 的过程。
共享的是 `diagnostic_buffer` ABI 参数位置，而不是同时写入同一内存。
两个工具的 session 当前互斥，均不改变业务 event counter、ready handshake、workspace 或 autograd 租约。

## 配置与调用生命周期

### 普通配置与工具私有派生配置

| 配置 | 准备时机 | 关键内容 |
|---|---|---|
| normal | plan 构造 | diagnostic_mode=0；普通 task/queue 数据 |
| profiler 私有配置 | 首次选中 WARMUP/RECORD/RECORD_AND_SAVE | diagnostic_mode=1；trace ID、owner 与每核容量 |
| Task dump 私有配置 | 每次选中 Task 调用 | diagnostic_mode=2；不使用 profiler 的 trace ID/容量 |

配置名称按表示形式区分：`runtime_config` 是可按字段访问的 `RuntimeConfigC` 结构化配置，
`runtime_config_tensor` 是序列化后的 `torch.uint8` Tensor，用于传给执行设备。二者描述的都是 Device 执行，
名称不以 Host/Device 表示生效边界；Tensor 也可以位于 CPU，例如 CPU 单测。
底层 native 接口和 Torch schema 的固定参数名沿用 ABI 合同；Python plan 和诊断调用对象按上述表示形式命名。

`prepare_mega_kernel_diagnostics()` 保存标准结构化配置 `_runtime_config`，并创建标准执行
`runtime_config_tensor`，不解析诊断 metadata、不执行 resolver、不计算 profile layout。
关闭或 action=NONE 时，业务直接走普通 wrapper，不进入 diagnostic launch，不创建诊断句柄。

首次选中 profiler 时，`_PreparedProfileRuntime` 复制诊断结构化配置，在副本中编码 trace 描述符、owner，
计算容量。该副本为 `profile_runtime_config`，序列化后的独立 Tensor 为 `profile_runtime_config_tensor`。
标准配置及其 Tensor 不受 profiler 私有配置影响。
session 按公共 runtime 对象缓存 profiler 配置，退出时释放该 session 的缓存。

`PreparedDiagnosticRuntime.diagnostic_runtime_config` 返回惰性适配后的诊断结构化配置。业务 Adapter 可在
私有副本中重排 Task 和队列，因此它不一定逐项对应标准 `runtime_config_tensor`。
`TaskDumpTensorViewContext.diagnostic_runtime_config` 指向该诊断配置。Profiler 和 Task dump 均通过
`copy_diagnostic_runtime_config()` 取得独立副本，再配置并序列化自己的执行版本，保证观测记录与实际执行的
Task 编号、描述和队列对应。Task dump 的结构化副本为 `task_dump_runtime_config`，对应 Tensor 为
`task_dump_runtime_config_tensor`。

Plan 中的 `diagnostics`（或 `fwd_diagnostics`、`route_diagnostics`）保存 `MegaKernelDiagnostics` 诊断句柄；
`MegaKernelDiagnostics.runtime_config_tensor` 和 plan 的对应属性提供标准执行 Tensor；
`DiagnosticCall.runtime_config_tensor` 提供本次调用最终选中的标准、profiling 或 Task dump 配置 Tensor。
`requires_diagnostic_launch` 只判断当前 session 和 schedule 是否需要
进入诊断准备流程，invocation 的 Kernel/rank/direction 过滤仍由 `prepare_call()` 完成。Kernel dump 也需要该
Host 调用路径，但 Device mode 为 DISABLED，不表示启用了 Device 插桩或最终一定会采集。
Session 的 `should_prepare_call()` 只判断当前 step 的 schedule：profiler 包含 WARMUP、RECORD 和
RECORD_AND_SAVE，WARMUP 的采样结果会被丢弃；dump 只包含 RECORD 和 RECORD_AND_SAVE，NONE 均不准备。
该方法不检查 session 是否处于活动作用域，也不执行具体调用过滤；活动 session 的判断由诊断入口负责。

公共 owner 是 `GraphDiagnosticMetadata.task_owner_ids`；`profile_owner_id` 只是 profiler 在私有配置中的
Device 编码，dump 不读取它。动态视图 resolver 和 dump 目录校验不会进入 profiler 准备路径。

### 调用与完成

`MegaKernelDiagnostics` 是计算实现的接入对象，不是用户开启采集的 context。
plan 持有它准备的普通执行配置；调用方提供本次真实 ABI 参数、方向、事件存储和可选业务标签。
活动 session 根据当前窗口与筛选条件返回一个 `DiagnosticCall`，其中包含完整 `LaunchArguments`：
执行配置、事件计数器、诊断 buffer 和应清零的计数器区域。

调用方通过 `with diagnostics.launch(...)` 执行 native wrapper。正常返回触发 `complete()`，
异常触发 `cancel()`；完成处理本身失败时也执行取消。工具通过各自的完成回调接收数据，
不互读另一工具的私有状态。这里的完成是工具接收本次调用的边界，不统一等同于设备已同步：
Profiler 可将 buffer 留在 pending 中稍后回读。Dumper 在 Host 快照可用后按输出配置写 Tensor 或生成摘要，
窗口保存 Tensor 文件的索引和所选摘要。

公共 session 保证进程内互斥。窗口动作与对外导出语义见
[控制采集窗口](../diagnostics.md#控制采集窗口)；Profiler 的 WARMUP 需要采样后丢弃，dump 的 WARMUP 不复制值。

### Profiler buffer 生命周期

业务执行持有自己的事件存储，诊断只取得本次调用需要的参数。例如 MoE 的 `all_event_counters`
前缀承载调度计数器，EP 大于 1 时尾部保存 ready generation；采集不改变计数器前缀的清零范围，
也不重置持续递增的 generation。

进入采集 action 后，内部 Profiler 按 `(device_id, buffer_size)` 从 session 私有池取得普通 NPU
`diagnostic_buffer`。每个尚未 drain 的 invocation 独占一块 buffer，因此同一 step 的多层正反向调用不会互相
覆盖；Device 直接把 record 写入最终 buffer。

D2H 和解析由以下时机触发：

1. warmup step 结束且数据需要丢弃；
2. active 窗口结束；
3. pending 调用数达到 `max_pending_calls`；
4. context 正常退出并仍有待完成数据。

drain 先同步 NPU；需要保留时直接把每块 buffer 搬到 Host 并解析，随后将 Device buffer 回收到当前 session
的池中，供后续 invocation 复用。异常退出时释放 pending 和池内显存，不生成容易误判的半成品 trace。
profiler context 退出后池随 session 销毁，不跨模型或下一次 context 长期持有显存。

## 设备采样与 ABI

### RuntimeConfig 字段

Python `RuntimeConfigC` 与 AscendC `RuntimeHeader` 必须保持 64 Byte 固定头的字段顺序和宽度一致。Task、event
等数组按图的实际容量紧随固定头序列化；profiling 复用固定头的保留字段，不改变动态数组的布局：

| 字段 | 类型 | 说明 |
|------|------|------|
| `diagnostic_mode` | `uint32` | 0=普通/Kernel dump，1=profiling，2=Task dump |
| `aic_profile_record_capacity` | `uint32` | 每个 AIC worker 的 record 容量 |
| `aiv_profile_record_capacity` | `uint32` | 每个 AIV worker 的 record 容量 |

`DeviceDiagnosticMode` 的成员为 `DISABLED=0`、`CYCLE_PROFILE=1`、`TASK_TENSOR_DUMP=2`。
它控制设备插桩；Host 的 `_ACTIVE_TOOL_NAME` 区分活动 session 是 `profiler` 还是 `dumper`。

每个 `TaskDesc` 额外包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `profile_desc_id` | `uint32` | 阶段语义；未配置时为 `0xFFFFFFFF` |
| `profile_owner_id` | `uint32` | 业务归属；未配置时为 `0xFFFFFFFF` |

公共 Device runtime 不理解 Expert、layer 等业务概念，只透传 ID。阶段名称和 owner label 只保存在 Host
metadata 中，不占用 Device record 的字符串空间。Host 按 `ComputeGraph.topological_sort()` 遍历节点，
使用每个节点的 `task_num` 还原其连续 TaskDesc 区间，并自动分配内部 descId；展示名优先取
`OperatorNode.diagnostic_stage_name`，未设置时由节点 `name` 转成驼峰形式。descId 是 RuntimeConfig 内部标识，
调用方不应依赖具体数值。

### Profiler buffer 布局

调度状态和 profiling buffer 的生命周期独立。以 MoE 为例：

```text
all_event_counters（此处以 MoE 对称内存为例）
├── scheduler event counters
└── ready generation tail（EP > 1）

diagnostic_buffer（普通 NPU 显存，仅采集 action 分配）
├── AIC slot 0
│   ├── 64-byte core header
│   └── N × 32-byte records
├── ...
└── AIV slots
```

每个 AIC/AIV worker 拥有独立 slot，避免全局 record allocator 和原子抢占。core header 保存 entry cycle、
有效/丢弃数量、core 类型、block ID 和容量；record 保存起止 cycle、`desc_id`、`task_id`、
`stage_task_index` 和 `owner_id`。profile slot 从 `diagnostic_buffer` 的 offset 0 开始，不占用 symmetric
heap。

`diagnostic_buffer` 是 profiling 与 Task dump Tensor dump 平等使用的公共诊断入参；当前两种工具的 session 互斥。
profiling 模式写入 cycle trace，Task dump 模式写入 Tensor 快照，普通执行与 Kernel dump 不读取其诊断内容。
它通过内部 native ABI 传递：正反向 OpProto、OpDef、ACLNN GetWorkspaceSize、L0 和 Device
kernel 的输入列表都显式传递它。Python 公共 `mega_moe()`、`mega_moe_grad()` 和
`MegaMoeExperts` 不要求调用者传入诊断 buffer，由私有 adapter 提供该参数。BF16 MoE 的普通路径使用
`all_event_counters` 作为未读取的占位 tensor；无事件计数器的路径可使用已有普通配置作占位，
不为此新增申请或数据搬运。

### 容量计算

Host 根据完整调度表复现每个 worker 可能执行的任务，分别计算最忙 AIC 和 AIV 的记录数：

```text
capacity = round_up(required_records, 16)
```

每个有依赖的普通 task 最多记录 WaitDependency、Compute 和 TriggerEvent 三段。
SHMEM put 自行发出通信通知，不额外生成公共 TriggerEvent 记录。显式 worker 队列也为就绪等待事件
预留记录。设备在记录槽满时累计 dropped count。每槽 stride 受设备 ABI 的 32 位范围约束。

profile buffer 依据运行时计划的槽位数分配，每个槽位包含 core header 和该类 worker 的记录区。
总字节数为 `AIC 槽位数 * AIC_stride + AIV 槽位数 * AIV_stride`，调度 event workspace 单独分配。
`max_pending_calls` 控制 Host session 的 buffer drain 频率。

### Cycle 与时间换算

Device 使用 `GetSystemCycle()` 读取 system counter：

```text
timestamp_us = (start_cycle - anchor_cycle) / cycle_frequency_mhz
duration_us  = (end_cycle - start_cycle) / cycle_frequency_mhz
```

系统计数器在 `ascend910b`、`ascend910_93` 使用 50 MHz，在 `ascend950` 使用 1000 MHz。SoC 名称来自
`torch_npu.npu.get_device_name()`；未知型号会在解析采集结果时显式失败，不接受用户从前端传入猜测值。

### 任务采样与关闭态快路径

采样路径按实际操作记录依赖等待、计算和完成发布；就绪扫描还可记录等待下一项可执行任务的区间。
具体区间的解释见[时间线说明](../diagnostics.md#查看与理解时间线)。同一 worker 的 record 保持写入顺序，
Host 导出时按 cycle 稳定排序。

worker 在 ready handshake 之后、`Process()` 的任务循环之前读取 `DeviceDiagnosticMode`，并在进入任务循环前选择快慢实现。关闭路径具有以下约束：

- 不读取 system cycle；
- 不执行 profiling barrier；
- 不写 core header 或 record；
- 不读取 `profile_desc_id` / `profile_owner_id`；
- 只清零 `all_event_counters` 的调度计数器前缀，保留 ready generation 尾部；
- 不申请普通 NPU `diagnostic_buffer`，也不创建 event-counter view；内部占位参数不会被 Device 读取。

这样可避免把 profiling 判断放进每个 task 的热循环。

## 张量快照的准备与存储

### Kernel dump 调用边界

Kernel dump 根据诊断图上的端口整理逻辑 Tensor：没有生产者的是外部输入，有生产者且被后续节点使用的是
阶段输出，其余是最终输出。调用前复制选中的外部输入，调用完成后复制选中的阶段及最终输出，
因此保存的是边界实际值，不是每个图节点执行时的历史值。

每个快照是独立 CPU Tensor，由调用句柄持有。一次调用完成后写出 `.pt` 并释放快照，文件保留原始 dtype。
同一逻辑 Tensor 的多个 stage 端口通过硬链接共享文件；stage 视图只描述图上的关系。
文件格式中影响用户读取的字段集中说明在[读取快照与 manifest](../diagnostics.md#读取快照与-manifest)。

### Task dump 视图解析

Task dump 先按调用、task、worker 与端口筛选，再为每个选中的端口解析实际存储范围。
`coreId` 使用与 cycle trace 的 `block_id` 相同的设备 worker 编号。轮转队列的配对 AIV 槽位
`s` 对应 worker `2*s+1`，显式队列以各类 worker 的队列视图确定归属。
静态连续切片使用默认视图；动态或硬编码地址语义由业务 resolver 在每次发射前返回
`TaskDumpTensorView(source_offset_bytes, shape)`。Host 校验真实 ABI Tensor 的 dtype、连续性和完整范围，
再处理可选截断及 payload 分配，Device 不执行 Python resolver。

静态切片根据源偏移裁剪尾块。跨行的内维切片不能按同等元素数直接解释成连续字节段；
这类视图需要业务 resolver 明确选择可复制的连续范围。只改变 shape 无法修正非连续地址。
任务协作产生的输出还必须在采样时已满足可见性，不能仅因它出现在图上就开放采集。

### Task dump buffer 与设备写入

Task dump Device buffer 使用版本 2：

```text
[0, 32)       Header：版本、有效条数、区域偏移和总大小
[32, 64)      对齐填充
[64, 262208)  Record table：4096 × 64 B，只有 record_count 条有效
[262208, ...) Payload：每条 record 对应独立 slot，slot 起点按 64 B 对齐
```

每条 record 的 10 个 uint32 字段描述 task、phase、ABI 源位置/偏移、目标 slot、字节数、状态、task index
和 worker ID，余下 24 B 对齐填充。record 和 payload slot 不共享写入者的数据缓存行；普通队列使用任意
worker 标记，广播队列使用具体 worker ID。buffer 随实际 payload 增长，记录表大小不等于快照字节容量。
在构造 Host bytearray 和设备分配前检查记录数、32 位 buffer ABI 上限和单次预算。合法的采集规划超出任一容量时，
session 告警并返回普通发射参数。发射前的诊断元数据解析、Tensor 筛选或采集分配抛出 `MemoryError` 或
`torch.OutOfMemoryError` 时，session 同样跳过本 rank 的本次采集。

Device 等待原依赖后采 before，执行计算并完成 barrier 后采 after，最后发布原业务完成事件。
采样使用通用字节复制和缓存同步。诊断关闭时仍进入 `ProcessFast`，不扫描 record、解析视图、分配快照区
或执行 D2H。调用句柄成功完成后解码并写出 CPU 快照，随后释放本次采集专用的配置 Tensor 和 buffer。

### Dumper 窗口发布

原始 Tensor 先写入调用的独立临时目录。文件全部写出后，session 将目录移入该 rank 的暂存窗口并登记 manifest 元数据。
采集自身抛出的 `MemoryError` 或 `torch.OutOfMemoryError` 由调用边界处理。文件错误在写出边界处理。
本 rank 告警并清理该调用的不完整文件。业务错误、采集协议错误和普通 `RuntimeError` 原样传播。
摘要在释放 Host 快照前计算，业务模块可提供实际调用的分组行范围。摘要记录的保留独立于原始 Tensor 的写入结果。
仅选择摘要时，窗口只保留可导出的小记录。Task 数据文件的索引和任务元数据统一落在窗口 manifest。
窗口结算时逐 rank 写 Tensor manifest 和所选 CSV。manifest 写入失败时清理该 rank 的原始快照，
使用新的暂存目录发布 CSV。每个 rank 通过一次目录重命名发布文件。发布失败或会话异常退出时清理未发布目录。
调用沿用准备时的窗口身份。该窗口结算或会话退出后返回的采集结果会被丢弃。

## Trace 对齐

### Invocation 分组

一个 active 窗口可以包含多层、多次正向和反向 launch。merger 先按 `args.invocation_id` 分组，再按每组
首次出现顺序与外层 kernel 一一配对。兼容没有 `invocation_id` 的旧单 invocation schema v1 trace，但拒绝
一部分事件有 ID、另一部分没有 ID 的歧义输入。

### 外层 Kernel 选择

默认流程：

1. 从内部 `kernelName` 生成允许下划线的大小写不敏感正则；
2. 只考虑持续时间为正的 complete event；
3. 优先选择具有 NPU/AICore/kernel 证据的 Device event；
4. 按外层时间排序，从 `kernel_index` 开始与 invocation 配对；
5. 自动名称未命中时，可按内部 span 与连续 Device kernel 的 duration 相似度降级选择并写 warning；
6. 显式 `kernel_pattern` 未命中时直接失败，不做名称降级；
7. 默认拒绝 Host wrapper，只有显式 `allow_host_wrapper=True` 才允许近似对齐。

CANN trace 中即使 category 为空，只要 `args["Task Type"]` 为 `MIX_AIC` 或 `MIX_AIV`，也可识别为
Device kernel。

### 时间与 Track 对齐

每组内部事件保持相对时间和持续时长不变，将最早内部事件平移到对应外层 kernel 的 `ts`。融合后的事件使用
外层 process ID，并为每个 `(core_type, block_id)` 分配不冲突的新 thread ID，名称为：

```text
<kernelName>/<coreType>/<blockId>
```

同时写入更靠前的 `thread_sort_index`，使 AIC/AIV 内部 track 靠近外层 NPU kernel 显示。

对齐诊断与接口参数见[与外层 Trace 融合](../diagnostics.md#与外层-trace-融合)。
独立 trace 的可读字段由使用指南维护，设备记录格式由前文 ABI 维护。

## 组件接入

### 默认图节点名称

接入 `ComputeGraph` 的新 MegaKernel 默认以 `OperatorNode.name` 生成阶段名，例如 `qkv_proj` 显示为
`QkvProj`。需要保留缩写或使用更易读的业务术语时，在节点定义旁设置 `diagnostic_stage_name`。图外 task 仍可
显示 WaitDependency、TriggerEvent、TerminateTask 和 task-type fallback。

Profiling 允许多个节点使用相同展示名，仍按拓扑位置分配不同的内部 descId。Kernel dump 和 Task dump
要求同一图中的 stage 展示名唯一，且净化后的目录名不能冲突或使用保留名称；这些目录约束不影响 profiling。

### 组件接入流程

推荐按以下顺序接入：

1. 使用同一份 `ComputeGraph` 按拓扑顺序构造 RuntimeConfig task，并保证每个节点的 `task_num` 与实际连续
   TaskDesc 区间一致；
2. 在需要自定义名称的 `OperatorNode` 上填写 `diagnostic_stage_name`；其余节点直接复用 `name`；
3. 需要 owner 时，由业务接入实现接收节点、TaskDesc 和 topology 的 resolver；
   不属于任何 owner 的节点返回 `None`；
4. RuntimeConfig builder 只生成普通执行的 task/queue，不写诊断字段；
5. 在 plan 中调用 `prepare_mega_kernel_diagnostics(graph, runtime_config, ...)`，可显式调用
   `validate_mega_kernel_diagnostic_contract()` 检查图/task/ABI；诊断 metadata 与 profile image 自动惰性准备；
6. 在内部 OpProto、OpDef、ACLNN、L0、Device kernel 和框架 adapter 中贯通 `diagnostic_buffer` 输入，但保持
   模型用户接口不变；
7. 正反向先检查 `diagnostics.requires_diagnostic_launch`；关闭时直接走普通 wrapper，开启时通过
   `with diagnostics.launch(...)` 提供完整参数并管理完成/取消；
8. 增加布局、metadata、Torch session、导出、merge 和真实 NPU 测试。

图节点是阶段身份的唯一来源，无需额外的 TaskDesc 字段匹配表。若某个 builder 不能保证“按图拓扑顺序、每个
节点连续写入 task”，应先在通用构图/填充层显式记录每个节点的 task ID 区间，再应用图 metadata，不能退回
依赖 tiling 或输入槽位猜测。公共 task builder 不得写具体模型的 owner 语义。

### ABI 演进

修改 header、record、容量或 core 类型时，必须同步更新：

- Python ctypes 与 AscendC struct；
- OpProto、OpDef、算子 JSON、ACLNN GetWorkspaceSize、L0 和框架 adapter 的输入顺序；
- Host buffer size 和 slot offset 计算；
- Device recorder；
- Host parser 和 schema 版本；
- 单元测试、NPU 完整性断言和本文档。

不兼容的结构性变更必须升级相应 schema，不得让旧解析器静默接受不兼容布局。

### Native ABI 与广播队列

只有调度依赖、在 C++ 内硬编码 Tensor 地址的 MegaKernel，可通过 `graph_adapter` 惰性补充真实端口。
回调由对应 megakernel 的主机侧提供，接收原图和私有 Host 配置，返回诊断图；不能修改原图或普通
Device 配置。`bind_graph_tensor_ports()` 在图副本上设置 `diagnostic_host_ports`，此时端口来自 Host 描述，
不受 `TaskDesc.inputs/outputs` 各 4 项的序列化容量限制，也不会改写计算使用的 TaskDesc。
这些端口可用 `shape=[]`、`is_dynamic=True` 表示形状在调用时解析；导出文件的实际维度以 `actualShape`
和 `storedShape` 为准。导出 rank 使用 job-global rank，MoE 的 Expert ID/路由计算仍使用 EP 组内 rank，
二者通过业务 `owner_context` 区分，避免多个并行组都写入 `rank_0`。

普通队列以 `execution_mode="event_driven"` 接入；MoE 配对 AIV worker 使用 `vector_worker_stride=2`，
MHC 全 AIV worker 使用 `vector_worker_stride=1`。MegaGate 使用 `execution_mode="aiv_pipeline"`：每个
worker 在自己的 token 行分片上执行同一组 task ID，没有调度事件。Profiler 按完整阶段序列计算每核容量；
Task dump 按 `(task_id, worker_id)` 分配 record/slot，并由 Device 检查 worker ID，防止不同核写入同一份快照。
广播目录使用 `task_<id>_worker_<id>`，普通队列仍使用 `task_<id>`。

MHC 在 Torch wrapper 分配输出之后、发射之前绑定真实 ABI Tensor。MegaGate 正向在诊断开启时通过
`mega_gate_route_out` 预分配同样的 5 个输出；关闭时调用普通 native wrapper。地址列表只在 Device Task dump 分支
临时构造，不给普通 worker 增加诊断地址数组或初始化工作。

RouteGrad 的后处理由一个 `task_num=0` 的 Host 边界节点描述。它可参加 Kernel dump 的来源/去向整理，
但没有设备 task ID，不生成 Task dump record，也不会在 timeline 中伪造 megakernel task。
各组件当前可采集的端口范围见[支持范围](../diagnostics.md#支持范围与使用限制)。

### MegaMoE 接入示例

MegaMoe 的阶段顺序来自生成普通 RuntimeConfig 的正反向 `ComputeGraph`。诊断 graph adapter 在副本中
保留实际入队的 task，更新节点的 task 数量，并同步重映射描述与队列中的 task ID；普通配置不被修改。
因此诊断输出的 task ID 应在本次采集内解释，不能直接当作普通配置中的序号。每个
`OperatorNode` 可就地设置 `diagnostic_stage_name`；GMM1、GMM2、DispatchGrad 等业务名称与对应算子定义放在
一起。公共 metadata 在首次采集时按拓扑顺序解析；profiler 将 descId 写入自己的配置副本，阶段信息直接来自图节点。

业务接入提供 `Expert` owner label。`metadata.task_owner_ids` 和 Task 视图 resolver 的 `context.owner_id`
记录用于计算张量偏移的调度槽。HotReplica 按本次 placement 把槽映射为筛选和产物使用的全局专家编号。
内置 planner 复用规划时读取的 Host 借入表。自定义 planner 可随 placement 提供同形状的 Host 副本。
仅提供设备表时，在诊断工具请求专家映射时回读。构图时的 owner resolver 接收
`OperatorNode` 与其 TaskDesc，因此 Dispatch/Combine 等同类型算子可按节点身份区分。公共 GMM、SwiGLU 和
AllToAll task builder 不写 MegaMoe owner，模型用户也无需传阶段名、rank、device 或 RuntimeConfig。

以下以 Profiler session 为例，展示 MegaMoE 的正反向接入路径：

```text
_MegaMoeFunction.forward / backward
    ├─ diagnostics.requires_diagnostic_launch=False → 原 mega_moe/mega_moe_grad wrapper
    └─ True → diagnostics.launch(runtime ABI parameters, direction, events)
                → profiler.prepare_call → 完整 profile 参数
                → native wrapper_with_diagnostic_buffer
                → complete / 异常 cancel
```

`direction` 区分正反向 invocation。Profiler 路径中，成功 launch 后句柄标记完成，Device 直接写入本次独占的 buffer，
无需 D2D snapshot；drain 才同步与回读。异常取消只撤销本次 profiling 状态，模型用户无需传诊断参数。

## 验证重点

测试环境、命令和 NPU launcher 入口统一见[测试指南](../../tests/README.md)。验证按以下层次覆盖：

- Host 单测：布局、容量、metadata、schedule、session 互斥、窗口导出、取消和离线对齐。
- 调用隔离：关闭态不新增诊断 buffer；pending 调用独占 buffer，drain 后按 device/size 复用；
  诊断配置不修改普通 plan、事件计数器或其他调用的状态。
- 设备结果：各组件正反向正确，trace 的 cycle、方向、关键阶段和记录数完整，dropped 为零；
  Kernel dump 保留 dtype 与端口关系，Task dump task/worker 隔离且 payload 数值正确，失败路径能清理资源。
- 对齐结果：各 rank 独立输出，invocation 数量与外层设备 kernel 对应，降级与越界告警可以解释。

关闭态性能需比较“不含诊断 runtime 的基线”和“包含能力但 action 为 NONE 的产物”，
固定 workload、数值配置、预热、采样轮数和 rank 聚合方式，使用既定回退门槛，不能只比较一次均值。
CPU 模拟的规划与解码测试不替代设备执行，功能回归也不替代关闭态性能对照。

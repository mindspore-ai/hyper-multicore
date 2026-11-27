# HyperMulticore 诊断指南

HyperMulticore 提供 megakernel 内部任务时间线和张量快照采集。Profiler 用于分析任务执行、依赖等待及
计算通信重叠；tensor dump 用于检查调用边界或 task 执行前后的张量值。两类工具通过 Python context
启用，只采集已接入诊断能力的调用，不会自动采集模型中的所有 Torch 算子或任意中间张量。

使用前按[构建与安装](build.md)准备并激活与 Python 包匹配的 native payload。组件接入、采样与存储实现见
[诊断设计](design/diagnostics_design.md)；组件的输入和数值约束仍由各自的特性指南说明。

## 选择诊断工具

| 要解决的问题 | 工具 | 主要产物 |
| --- | --- | --- |
| 哪些任务在等待，计算与通信是否重叠？ | Profiler | Chrome Trace JSON |
| 一次 native 调用的输入输出是否异常？ | Kernel dump | 张量 `.pt` 文件和 manifest |
| 哪个任务阶段开始产生数值偏差？ | Task dump | 按 task/worker 区分的快照和 manifest |
| 内部任务如何对应外层 kernel？ | Trace 融合 | 合并后的时间线与对齐诊断 |

Profiler 与 dump 在同一进程中互斥，需要分次采集，不能嵌套启用。Task dump 只支持组件已提供的安全端口，
MXFP8 当前只支持 Profiler；具体范围见[支持范围与使用限制](#支持范围与使用限制)。

## 采集任务时间线

### 基本用法

下面是单卡 MegaGate native Route 的完整前向采集示例。投影仍是外层 Torch 计算，内部时间线记录的是
Route 的 task。示例先在 context 外完成一次真实 warmup，再采集一次调用：

```python
from pathlib import Path

import torch
import torch_npu

from hyper_multicore import MegaGate
from hyper_multicore.diagnostics import profiler

torch.npu.set_device(0)
gate = MegaGate(
    hidden_size=128, num_experts=33, top_k=4,
    scoring_func="sqrtsoftplus", routed_scaling_factor=1.0,
).to(device="npu:0", dtype=torch.bfloat16)
hidden_states = torch.randn(2, 64, 128, device="npu:0", dtype=torch.bfloat16)

with torch.no_grad():
    gate(hidden_states)
    torch.npu.synchronize()
    with profiler.profile() as capture:
        logits, routing_weights, expert_indices = gate(hidden_states)

trace_path = Path("./traces/gate.json")
trace = capture.export_chrome_trace(trace_path)
print(trace_path.resolve())
print(trace["megaKernelCycleTrace"])
```

省略 schedule 时，context 中的调用属于同一个采集窗口；正常退出后即可导出。包含反向的 context 也会
记录已接入的反向调用。`export_chrome_trace()` 创建父目录并写入指定文件，同名文件会被覆盖。

### 查看与理解时间线

将导出的 JSON 在支持 Chrome Trace 格式的查看器中打开，例如 Perfetto。每条 AIC/AIV worker track
展示该 worker 的内部区间；一个窗口可以包含多层、多次正反向调用，通过事件详情中的
`invocation_id`、`step`、`direction` 区分。`owner_id` 是可选的业务归属，例如 MoE 的全局专家编号。

| 区间 | 表示什么 |
| --- | --- |
| `WaitDependency` | 已选中 task 对自身依赖的检查与等待；无依赖的 task 不产生该区间 |
| `ReadyQueueWait` | 就绪窗口内没有可执行任务时，worker 等到选中下一任务的时间；关联随后选中的 task，可能涵盖多个候选任务的等待 |
| 计算阶段名，如 `GMM1`、`Softplus` | 相应 task 的计算区间；图外 task 可使用 task type 兜底名称 |
| `TriggerEvent` | 公共 runtime 发布 task 完成事件的区间；由通信计算核自行发出的通知不另记这一公共区间 |

FIFO 队列直接选择队首，只有 task 级依赖等待；启用就绪扫描的队列可以同时出现两种等待区间。
阶段信息存在时，Wait/Trigger 名称会带上阶段名，例如 `GMM1_WaitDependency`。

事件的 `args` 保留以下定位信息，不依赖展示名称是否详细：

| 字段 | 含义 |
| --- | --- |
| `rank`、`device_id` | 数据来源 |
| `core_type`、`block_id` | worker 类型和编号 |
| `task_id`、`desc_id` | 当前诊断执行配置中的任务与阶段标识 |
| `task_stage` / `task_type` | 可解析的图阶段；只有 task-type 兜底记录带 `task_type` |
| `stage_task_index`、`task_number` | 阶段内零起始索引和一起始展示序号 |
| `invocation_id`、`step`、`direction` | 调用、逻辑 step 和正反向归属 |
| `owner_id` | 可选业务归属 |
| `start_cycle`、`end_cycle`、`core_entry_cycle` | 原始设备计数与 worker 入口计数 |

先检查顶层 `megaKernelCycleTrace`：当前 `schemaVersion` 为 1，`invocationCount` 和 `recordCount`
应与实际采集相符，`droppedRecordCount` 应为 0，并逐项检查 `warnings`。`invocations` 提供每次调用的
kernel、direction、rank、device、时间锚点及记录数。记录丢失或对齐告警尚未解释清楚时，时间线不足以支持
完整的调度结论。

采集会增加执行和显存开销。时间线适合解释内部行为，性能比较应另行关闭诊断，按
[测试指南](../tests/README.md)使用相同输入与数值配置测量。

### 采集配置

`profiler.profile()` 返回 `ProfilerSession`，接受以下关键字参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `schedule` | `None` | 采集整个 context；需要控制窗口时传入公共 `schedule()` 的返回值 |
| `on_trace_ready` | `None` | 每个窗口完成后调用 `callback(capture)` |
| `detailed_task_names` | `False` | 改为包含 owner/task 的展示名，如 `Expert3_GMM1_task72`；不改变原始 ID |
| `max_pending_calls` | `16` | 正整数；限制同步回读之前保留的调用 buffer 数量，不改变每个 worker 的记录容量 |

接口不接受 rank、device、cycle 频率或 RuntimeConfig；这些信息由实际调用确定。降低
`max_pending_calls` 可以减少待回读 buffer 的占用，但会更频繁地同步；它不能修复 dropped records。

## 采集张量快照

`DumpLevel` 定义 `KERNEL="kernel"` 和 `TASK="task"`；`level` 默认 KERNEL，也接受大小写不敏感的
`"kernel"` / `"task"` 字符串并规范化为枚举。`dumper.dump()` 返回 `DumperSession`，管理 step、调用记录和输出窗口。

### Kernel dump：调用边界快照

Kernel dump 在选中的 native 调用前复制外部输入，调用完成后复制中间和最终结果，并保留原始 dtype。
它保存的是这些边界上的实际存储值，不能恢复原地更新或 workspace 复用前的历史中间值。
目录中的 stage 表示端口归属，不表示该 stage 执行结束时曾单独采样。
各组件通过诊断图提供 stage 和输入输出端口。Kernel dump 默认采集图中所有 stage 的输入和输出。
用 `stages` 和 `io` 缩小采集范围时，只导出同时命中两个条件的端口。
多个选中端口引用同一张量时，只采集和统计一次。

以下片段沿用上例创建的 `gate`、`hidden_states`，在 Profiler context 结束后独立执行。
`mkdtemp()` 为本次采集创建新目录，便于重复运行示例：

```python
import tempfile

from hyper_multicore.diagnostics import dumper

dump_dir = tempfile.mkdtemp(prefix="hyper-multicore-kernel-dump-")
config = dumper.DumpConfig(dump_dir, level=dumper.DumpLevel.KERNEL, directions="forward")
with torch.no_grad(), dumper.dump(config) as dump_session:
    logits, routing_weights, expert_indices = gate(hidden_states)

print(dump_session.last_manifest_paths)
```

每次调用完成后，Dumper 同步写入本次快照并释放 Tensor。Host 快照内存取决于单次采集规模。
窗口结束时发布该窗口的文件，`last_manifest_paths` 提供最近一次结算中成功发布的各 rank manifest 路径。
需要减小采集量时，可以筛选 Tensor、方向或 rank。

### 作业内生成摘要

用 `DumpConfig.outputs` 选择产物。`tensor` 保存现有的 `.pt` 和 manifest，`csv` 为每个 rank 和窗口写
一份 `summary.csv`，`log` 在窗口结算时向标准输出打印最多八行预览。默认值是 `("tensor",)`。
只做跨作业初筛时选 `("csv", "log")`，需要随后逐元素对照时选 `("tensor", "csv")`。

```python
config = dumper.DumpConfig(
    "./megamoe_summary",
    kernel_names="mega_moe",
    directions="forward",
    ranks=0,
    stages=("GMM1", "SwiGLU", "GMM2"),
    io="output",
    outputs=("csv", "log"),
    tags={"global_step": 30, "microbatch": 0, "module": "layers.2.moe"},
)
```

将此配置用于目标调用所在的采集会话。配置内的 `tags` 对会话内所有调用生效，接入方提供的调用级标签可覆盖同名值。
在两次作业中使用相同的业务标签标记对照点。`step` 和 `invocation` 帮助定位各自产物内的调用。

摘要在 Host 快照释放前计算，窗口结算时导出。摘要模式保留设备到 Host 的采样开销，可省去原始 Tensor 文件的写入、
回读和传输。CSV 每个逻辑快照一行，`ports` 列列出该快照对应的 stage、输入或输出及端口序号。
`min`、`max`、`mean` 统计有限元素，空范围或无有限元素时留空。`nonfinite` 给出非有限元素数量，
`first_values` 是统计范围按存储顺序展平后的前四个值。

终端预览把同一快照的选中端口列在一行。`showing=8/30` 表示该 rank 在本窗口采集了 30 个快照，终端展示其中 8 行。
单独选择 `log` 或同时选择 `log` 和 `tensor` 时，只为预览行计算统计值。启用 `csv` 时，为全部快照计算统计值。以下是一条 GMM1 输出记录。

```text
Dump summary rank=0 window=0 showing=1/1
  step=0 invocation=0 GMM1.output[0] phase=after range=expert_rows shape=[256, 256] bfloat16 min=-0.231445 max=0.213867 mean=0.000286252 nonfinite=0 values=[0.0119019, 0.0664062, 0.00260925, 0.0456543]
```

Task 预览还显示 task、owner、worker 和截断状态。

`range=stored` 表示统计覆盖实际保存的形状，逻辑有效范围可能需要业务元数据解释。
MegaMoE BF16 forward 接入的专家激活使用 `range=expert_rows`，`expert_rows` 记录各专家及其在本 rank
Tensor 中从第 0 行起连续覆盖有效数据的左闭右开行范围，格式为 `[全局专家编号, 起始行, 结束行]`。
HotReplica 借入的专家也使用其全局编号。按同一次调用对齐时，可用这个编号筛选 Task 的 `owner_ids`，并与 profiler trace 中的 `owner_id` 对照。
`owner_ids` 只匹配有单一归属的任务。启用专家批处理时，跨专家的 GMM 任务没有单一 owner，按专家编号筛选该阶段会得到空结果。
批量 GMM 的静态权重可以单独采集。整组动态激活须在 Task 输入采样时全部就绪。
例如 BF16 前向在本 rank 批量执行多个专家时，`GMM2.input[0]` 采样前只等待首个专家的完成事件。
选中该端口时，本 rank 告警 `unsupported_port`，跳过本次调用的全部采集，照常发射 Kernel。
要看整组 SwiGLU 激活，选择 Kernel dump 的 `stages="SwiGLU", io="output"`。要看 task 写出的 tile，用同一筛选选择 Task dump。
Tensor 的 shape 仍是实际存储形状。Task 摘要包含 task、owner、worker、源偏移和截断标记。
HotReplica 的逐专家任务跳过本次未分配的槽。已分配专家即使收到零个 token，其权重仍可采集。
批量 GMM 权重快照保留物理布局。存在空槽时，
`range=valid_elements` 的统计只覆盖 CSV `valid_ranges`（manifest 中为 `validRanges`）列出的有效区间。
每个区间按保存张量展平后的元素下标表示，左闭右开，截断后仅包含保存前缀中的元素。
`range=first_tile` 和 `range=range_unknown` 分别表示首个 tile 和与实际执行范围对应关系未知。

原始快照和摘要独立处理写入失败。Tensor 文件或 manifest 写入失败时，已计算的摘要仍可发布。
摘要因资源不足、输入范围无效或写入失败而跳过时，原始快照仍可发布。告警包含当前 rank 和出错位置。
只发布摘要的窗口通过 CSV 路径或日志读取，`last_manifest_paths` 继续报告 Tensor manifest。

### Task dump：任务边界快照

Task dump 在 task 依赖满足后采集所选输入，在计算及所需同步完成后采集所选输出，再发布原业务完成事件。
它只读取已适配的 task 端口，不是任意时刻的全量内存快照。

必须指定 `stages`、`task_ids`、`stage_task_indices`、`owner_ids` 或 `tensor_names` 中至少一种有效筛选。
仅提供 `ranks`、`core_types`、`core_ids` 或 `io` 不满足这一要求。下面继续使用同一 MegaGate，
只采集 Softplus 的 logits 输入；该端口按实际 AIV worker 的 token 行分片导出：

```python
task_dump_dir = tempfile.mkdtemp(prefix="hyper-multicore-task-dump-")
config = dumper.DumpConfig(
    task_dump_dir, level=dumper.DumpLevel.TASK, directions="forward",
    stages="Softplus", io="input",
)
with torch.no_grad(), dumper.dump(config) as task_dumper:
    logits, routing_weights, expert_indices = gate(hidden_states)

print(task_dumper.last_manifest_paths)
```

`device_memory_budget_mb` 默认为 512 MiB，可配置为正整数。它限制单次 Task dump 的设备采集 buffer，包含记录表、对齐与快照数据。
Task dump 每次调用最多支持 4,096 条记录，设备采集 buffer 的大小也必须能用 32 位字段表示。
选中的快照超出记录数、32 位大小上限或配置预算时，本 rank 告警并跳过整次采集，使用标准参数发射 Kernel。
对应告警原因为 `task_record_limit_exceeded`、`device_buffer_abi_limit_exceeded` 和 `device_buffer_budget_exceeded`。
`max_bytes_per_tensor` 默认为
`None`，设置正整数后限制每条快照的字节数；截断结果保存为展平前缀，不能当作完整张量比较。
如果上限存不下某个选中 Tensor 的一个元素，发射前会抛出 `ValueError`，给出 Tensor 名、dtype 和所需字节数。
Kernel dump 通过 Tensor 和角色筛选控制单次采集的数据量。

### 筛选采集对象

字符串或整数筛选可传单个值或集合，多个筛选条件同时生效。名称按精确值匹配，不是正则表达式。

| 配置 | 适用范围 | 值从哪里取得 |
| --- | --- | --- |
| `kernel_names`、`directions`、`ranks` | Kernel dump / Task dump | manifest 的 `kernelName`、`direction` 和 job-global `rank`；方向为 `forward` / `backward` |
| `tensor_names` | Kernel dump / Task dump | 可选的精确张量筛选，名称来自 manifest 中 Tensor 的 `name` |
| `roles` | Kernel dump | `external_input`、`stage_output`、`final_output`；默认全部 |
| `stages` | Kernel dump / Task dump | 组件诊断图的 stage 名，见 manifest 的 `ports[].stage` 或 trace 的 `args.task_stage` |
| `task_ids`、`stage_task_indices` | Task dump | 当前 trace 的 `task_id`、`stage_task_index`；或 Task dump manifest 的 `taskId`、`stageTaskIndex` |
| `owner_ids` | Task dump | trace 的 `owner_id` 或 Task dump manifest 的 `owner.id` |
| `core_types`、`core_ids` | Task dump | 同一 rank 和设备下 trace 的 `args.core_type`、`args.block_id`，对应 manifest 的 `coreType`、`coreId` |
| `io` | Kernel dump / Task dump | `input`、`output` 或二者的集合，默认两者 |

`DumpConfig` 在创建时检查参数的适用级别。Kernel 拒绝 Task 专用筛选和 `max_bytes_per_tensor`，Task 拒绝 `roles`。
Kernel 可保留共用配置中的 `device_memory_budget_mb=512`，此值在 Kernel dump 中不生效。修改预算只适用于 Task dump。

可以先用一次小范围 Kernel dump 或 Profiler 采集确认名称，再配置 Task dump。图中的 tensor 有名称并不意味着它在所有
task 上都能安全采样；远端地址、协作累加或非连续存储等限制见下文的支持表。task ID 和阶段内索引对应
当前诊断执行配置，改变输入、调度或实现后应重新确认，不应作为跨版本标识保存。
部分路径会在诊断配置中重映射 task ID，应取本次采集输出的值，而非普通配置中的序号。

核号在 `AIC` 和 `AIV` 内分别编号。若 trace 的核为 `AIV/9`，可用
`core_types="AIV", core_ids=9` 选择该核。旧版配对 AIV dump 记录的是调度槽位，例如槽位 `4`
对应设备 worker `9`；使用新版本时请以 trace 的 `block_id` 为准。

如果调用通过了 kernel 名、方向和 rank 筛选，但 Tensor、角色或 task 端口筛选没有选出可导出的快照，
Dumper 会记录 `empty_selection` 并继续正常执行 kernel。调用级筛选未命中时直接正常发射，无需告警。
空采集、超出 Task dump 容量、采集自身抛出 `MemoryError` 或 `torch.OutOfMemoryError`，以及调用文件写入失败，
都会使本 rank 跳过此次采集，保留业务结果和此前成功的采集。
发射前解析诊断元数据或筛选 Kernel Tensor 时遇到上述内存不足，Dumper 会以标准参数发射且不分配采集编号。
普通 `RuntimeError` 原样传播，包括 Torch CPU 分配器以该类型报告的内存不足。
每个 session 按 rank 和原因记录首次 WARNING，注明 kernel、rank、方向和 step。已分配采集编号时还会注明 invocation 标识。
只包含跳过调用的窗口在 `step()` 结算或正常退出上下文后返回空的 `last_manifest_paths`，也没有 manifest 产物。
若最后一次 `step()` 结算后直接退出上下文，`last_manifest_paths` 仍返回该次结算的结果。

### 读取快照与 manifest

输出按 `rank_<rank>/window_<index>/` 组织，每个窗口的 `manifest.json` 索引各 invocation。
相对输出路径以进入 Dumper 会话时的工作目录为基准。
结算前，本次采集的文件位于同级临时目录。Dumper 写好 manifest 后发布窗口目录。
窗口发布失败时，Dumper 记录该 rank 的 WARNING 并清理未发布的结果。发生同名冲突时保留已有窗口，重新采集请换目录。
异常退出会清理尚未发布的窗口。
Kernel dump 的多个 stage 端口可以通过硬链接引用同一快照，因此输出文件系统需要支持硬链接。

下面的读取片段适用于前面的 Kernel dump 和 Task dump 示例。`file` 相对 manifest 所在窗口目录，不应从当前工作目录拼接：

```python
import json

for session in (dump_session, task_dumper):
    for manifest_path in session.last_manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for invocation in manifest["invocations"]:
            groups = [invocation] if manifest["level"] == "kernel" else invocation["tasks"]
            for group in groups:
                for item in group["tensors"]:
                    tensor_path = manifest_path.parent / item["file"]
                    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
                    print(item["name"], item["capturePhase"], tensor.shape, tensor.dtype)
```

只加载可信的采集文件。新产物的 `schemaVersion` 在 Kernel dump 中为 5，在 Task dump 中为 6。
以下字段用于解释结果。

| 位置 / 字段 | 含义 |
| --- | --- |
| 顶层 `level` | `kernel` 或 `task` |
| 顶层 `capturePhaseSemantics` | Kernel dump 为 `mega_kernel_boundary`，Task dump 为 `task_boundary` |
| invocation 的 `invocationId`、`step`、`direction`、`kernelName` | 将文件对应到一次调用 |
| Kernel dump tensor 的 `role`、`producers`、`consumers`、`ports` | 图上的来源、去向与 stage 端口关系 |
| Task dump task 的 `taskId`、`stageTaskIndex`、`stage`、`owner`、`coreType`、`coreId` | task 与执行 worker；MegaGate 广播 task 的多个 worker 共用 task ID，`stageTaskIndex` 和 `owner.id` 均为 trace 中的 worker 编号 |
| tensor 的 `capturePhase`、`actualShape` | 采样在 `before` 还是 `after`；Kernel dump 是调用边界实际张量形状，Task dump 是所选端口视图的完整形状 |
| tensor 的 `originalDtype`、`storedDtype`、`file` | 原始与存储 dtype，以及 `.pt` 文件位置 |
| Kernel dump 的 `declaredShape` | 图上声明的形状；动态端口应以 `actualShape` 为准 |
| Task dump 的 `storedShape`、`fullBytes`、`storedBytes`、`truncated` | 实际保存形状、完整及保存字节数、是否截断；截断时不能直接 reshape 为 `actualShape` |
| Task dump 的 `sourceOffsetBytes`、`note` | 所选视图相对源 Tensor 的字节偏移，以及组件补充的解释 |

需要逐元素比较时，一次采集同时保存 Tensor 和 CSV。下面在所有 EP rank 执行同一次 MegaMoE 调用，
在 rank 0 取出窗口路径。示例中的 `layer` 是公开接口 `MegaMoeExperts`，路由参数和本地专家权重由模型提供。

```python
import tempfile

from hyper_multicore.diagnostics import dumper

dump_dir = tempfile.mkdtemp(prefix="megamoe_compare_")
config = dumper.DumpConfig(dump_dir, kernel_names="mega_moe", directions="forward",
                           ranks=0, stages="SwiGLU", io="output", outputs=("tensor", "csv"))
with dumper.dump(config) as session:
    output = layer(hidden_states, topk_ids, topk_weights, expert_weights=(gate_up_weight, down_weight),
                   tokens_per_expert=tokens_per_expert)
if session.last_manifest_paths:
    manifest_path, = session.last_manifest_paths
    print(manifest_path)
```

离线分析时，将 `manifest_path` 设为上面输出的路径（`pathlib.Path`）。从同一窗口的 CSV 选取调用和端口，
按 `expert_rows` 去掉容量填充。从 Native 作业取同一批 token 的 SwiGLU 输出，
按相同的全局专家和专家内 token 顺序组成 CPU Tensor `reference`。HotReplica 下同一专家可分布在多个 rank，
对齐时还需确认当前 rank 收到的 token。

```python
import csv
import json

invocation_id = 0
port = ("SwiGLU", "output", 0)
with manifest_path.with_name("summary.csv").open(encoding="utf-8", newline="") as stream:
    row = next(row for row in csv.DictReader(stream)
               if int(row["invocation"]) == invocation_id and list(port) in json.loads(row["ports"]))
expert_rows = json.loads(row["expert_rows"])
actual = dumper.load_tensor(manifest_path, invocation_id=invocation_id, port=port)
actual = actual[:expert_rows[-1][2] if expert_rows else 0]
result = dumper.compare_tensors(reference, actual, rtol=0.02, atol=0.002)
print(result["over_tolerance"], result["first_mismatch_index"],
      result["first_mismatch_reference_value"], result["first_mismatch_actual_value"])
```

`first_mismatch_index` 是按张量展平顺序找到的首个超差坐标，全部满足容差时为 `None`。
`max_abs_error` 和 `max_error_index` 给出最大绝对误差及其位置。所有坐标都相对于传入的张量。
MegaGate 的 `expert_indices` 等整数张量可用 `reference.dtype == actual.dtype and torch.equal(reference, actual)`
精确核对 dtype、形状和元素。加载 Task 快照时可用 `task_id` 和 `worker=("AIV", 9)` 指定同一端口的具体记录。

Task schema 6 将任务元数据集中在 manifest 的 `invocations[].tasks[]`，原始 Tensor 仍存放在各 task 目录。
原先通过 `metadataFile` 读取 `task.json` 的脚本从 manifest 的任务记录取相同字段。`load_tensor()` 继续支持
Task schema 5。对照时用 manifest 中的调用、task、worker 和端口标识确定各快照的采样范围。

### 接口与格式更新

manifest schema 5 和 6 的 `level` 使用 `kernel` / `task`，采集时机使用 `capturePhase` 和
`capturePhaseSemantics` 字段。旧接口及解析器需要按下表调整，不提供兼容别名：

| 原接口或字段 | 当前接口或字段 |
| --- | --- |
| `hyper_multicore.diagnostics.dump`、`hyper_multicore.diagnostics.tensor_dump` | `hyper_multicore.diagnostics.dumper` |
| `TensorDumpConfig`、`TensorDumpLevel` | `DumpConfig`、`DumpLevel`，级别使用 `kernel` 或 `task`，原值为 `L0` 或 `L1` |
| `TorchMegaKernelTensorDumper`、`TensorDumpSession`、`tensor_dump()` | `DumperSession`，通过 `dumper.dump()` 创建 |
| `TorchMegaKernelProfiler`、`mega_kernel_profile()` | `ProfilerSession`，通过 `profiler.profile()` 创建 |
| `L1TensorView`、`L1TensorViewContext` | `TaskDumpTensorView`、`TaskDumpTensorViewContext` |
| `MegaKernelDiagnostics.is_active` | `requires_diagnostic_launch`，表示活动 session 和 schedule 是否要求进入诊断准备路径 |
| session 的 `is_capture_enabled()` | `should_prepare_call()`，表示当前 step 是否需要诊断调用准备 |
| 诊断或 plan 的配置 Tensor 属性，如 `runtime_config`、`fwd_runtime_config`、`profile_runtime_config` | 对应的 `runtime_config_tensor`、`fwd_runtime_config_tensor`、`profile_runtime_config_tensor` |
| `diagnostic_host_runtime_config`、view context 的 `host_runtime_config` | `diagnostic_runtime_config`，均表示结构化诊断配置 |

## 控制采集窗口

### schedule 与 step

两种工具共用 `from hyper_multicore.diagnostics import schedule`，但采样行为不同：

| Action | Profiler | Dump |
| --- | --- | --- |
| `NONE` | 不采集 | 不复制张量 |
| `WARMUP` | 采样，在 step 结束后同步并丢弃 | 不复制张量 |
| `RECORD` | 保留调用数据 | 按调用写出快照 |
| `RECORD_AND_SAVE` | 完成窗口并回调 | 按调用写出快照，结算时发布窗口并回调 |

`schedule(wait=..., warmup=..., active=..., repeat=0, skip_first=0)` 的参数均为非负整数，且
`active > 0`。先跳过 `skip_first` 个 step，再重复 wait、warmup、active；`repeat=0` 表示持续重复。
逻辑 step 从 0 开始，每次训练迭代的调用全部返回后，在 context 内调用一次 `step()`。
退出 context 前也应等待本轮调用全部返回。正常退出时发布尚未保存的有效窗口，异常退出时丢弃未发布的窗口。

### 训练循环与窗口回调

以下是接入已有训练循环的片段：`data_loader`、`optimizer` 和 `train_step(batch)` 由应用提供，
`train_step` 返回待反向的 loss。每个窗口单独导出，避免只保留最后一个窗口：

```python
import torch.distributed as dist

from hyper_multicore.diagnostics import profiler, schedule

rank = dist.get_rank() if dist.is_initialized() else 0

def save_trace(capture):
    capture.export_chrome_trace(f"./traces/rank{rank}_step{capture.step_num}.json")

with profiler.profile(
    schedule=schedule(skip_first=5, wait=1, warmup=1, active=2, repeat=3),
    on_trace_ready=save_trace,
) as capture:
    for batch in data_loader:
        optimizer.zero_grad()
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        capture.step()
```

`export_chrome_trace()` 导出最近完成的窗口；尚无已完成窗口时会报错。当前窗口仍在采集时，
导出的仍是上一窗口。需要保存每个窗口时，使用 `on_trace_ready` 回调，在窗口完成后接收 capture 并导出。

dump 会自动写文件。窗口成功发布 Tensor manifest 后，`on_dump_ready` 回调可通过参数的
`last_manifest_paths` 读取产物。纯摘要窗口直接读取 CSV 或日志。两种回调都在数据已就绪时处理结果，
无需在回调中再次调用 `step()`。

分布式采集需区分诊断选择与业务参与。dump 的 `ranks` 只过滤采集结果；Profiler 没有 rank 筛选参数，
可由应用只在目标进程启用 context。无论是否采集，所有相关 rank 都必须继续参与同一业务通信序列。
每个 rank 使用独立的 trace 文件。稳态分析应先在 context 外用等价 shape 做真实 warmup，
同步 NPU，并在需要比较 rank 启动时序时让所有参与 rank 完成 barrier。

例如，下列是已初始化 EP 的 MegaMoE 正向调用片段；对象、输入和权重准备见
[MegaMoE 基本用法](mega_moe.md#基本用法)。两次调用都传入当前权重：

```python
with torch.no_grad():
    experts(hidden_states, topk_ids, topk_weights,
            expert_weights=expert_weights, tokens_per_expert=tokens_per_expert)
    torch.npu.synchronize()
    dist.barrier()
    with profiler.profile(detailed_task_names=True) as capture:
        output = experts(hidden_states, topk_ids, topk_weights,
                         expert_weights=expert_weights, tokens_per_expert=tokens_per_expert)

capture.export_chrome_trace(f"./traces/rank{dist.get_rank()}_moe.json")
```

## 与外层 Trace 融合

内部 Profiler 不启动或包装 `torch.profiler`、CANN `msprof`。外层 trace 需单独采集，并与内部 trace 覆盖
同一次运行的调用。取得对应 rank/device 的外层 Chrome Trace JSON 后，可离线融合。

外层 trace 若包含内部采集窗口之前的预热或其他同名调用，需要通过 `kernel_index` 指向本窗口对应的
第一次调用，或让外层采集范围与内部窗口一致。下面假设外层记录了基本示例中恰好一次预热和一次正式
Route，因此使用 `kernel_index=1` 跳过预热；若外层只记录正式采集，应使用 `0`。
其他程序应根据实际匹配到的调用顺序确定起点，不能仅凭文件来自同一次运行判断配对正确。

```python
from hyper_multicore.diagnostics import profiler

# 替换成同一次运行中实际生成的文件。
merged = profiler.merge_chrome_traces(
    framework_trace="./traces/framework.json",
    mega_kernel_trace="./traces/gate.json",
    output="./traces/merged.json",
    kernel_index=1,  # 本例外层 trace 包含一次未纳入内部采集的 Route 预热。
)
print(merged["megaKernelTraceMerge"])
```

前两个参数可传路径或 trace 对象；外层 trace 也支持顶层 event list。结果写入 `output` 并作为 dict 返回。
`kernel_pattern` 可显式指定外层 kernel 的正则，`kernel_index` 指定匹配起点（默认 `0`），`outer_pid` 可限制进程。
默认只选择设备 kernel；`allow_host_wrapper=True` 才允许 Host wrapper 的近似对齐。

融合按 invocation 顺序匹配外层 kernel，保持内部区间的持续时间与相对时间，再平移到外层时间轴；
它不是独立时钟的精确校准。多 rank 结果应逐 rank/device 融合，不混用不同运行或设备的数据。
自动名称匹配失败时可能按时长降级选择；显式 `kernel_pattern` 未命中则直接失败。

检查结果中的 `megaKernelTraceMerge`，不能只看是否成功写文件：

| 字段 | 用途 |
| --- | --- |
| `mergedInvocationCount`、`mergedEventCount` | 核对融合规模 |
| `usedFallbackKernelSelection` | 是否发生名称匹配降级 |
| `alignments` | 每次调用选中的外层事件、时间及评分 |
| `exceedsOuterKernelUs`（各 alignment 内） | 内部 span 超出外层 kernel 的时长 |
| `warnings` | 降级、Host 对齐及边界异常 |

## 支持范围与使用限制

采集范围以 native 调用和已接入端口为边界，同一模块中的其他 Torch 算子不自动进入内部时间线。

| 路径 | Profiler | Kernel dump / Task dump |
| --- | --- | --- |
| MegaMoE BF16 正反向 | 内部 task 时间线 | Kernel dump 调用边界；Task dump 按动态专家/token 视图采集。Pull 不将远端源地址当本地输入；协作 GMM 输出需选已满足依赖的下游输入 |
| MegaMoE MXFP8 正反向 | 支持，外层为 `HyperMegaMoeMxfp8` / `HyperMegaMoeMxfp8Grad` | 继续普通发射；若录制中的 Dumper 选中此调用，告警 `unsupported_kernel` 并跳过本次采集 |
| MegaMHC 正向 | 内部 task 时间线 | Kernel dump 调用输入、输出及已接入的工作张量；Task dump 连续 token 分块。不包含环形 X-cast workspace 和非连续 Sinkhorn cache |
| MegaMHC 反向 | 内部 task 时间线 | Kernel dump 包括归约后的参数梯度；Task dump 仅安全的 token 局部端口，协作累加的参数梯度只允许 Kernel dump |
| MegaGate native Route | AIV 流水 task | Kernel dump logits、bias、mask 和 native 结果；Task dump 已接入的 task 输入输出，按 worker 的 token 行分片采集，不包含内部 CANN workspace |
| MegaGate native RouteGrad | megakernel 内 task | Kernel dump 包括 ACLNN 后处理后的 `grad_logits`；Task dump 仅内部 task 的已接入输入，scatter/sqrt/softplus 后处理不属于该 task 序列 |
| MegaGate Torch fallback | 无内部 task 采集 | 不通过 native dump 采集，使用外层 Torch 诊断工具 |

MegaMoE BF16 的 compact Task dump 仍会导出快照。通信端口的快照范围可能与实际分片不一致，
SwiGLU 的持久 task 只采集首个 tile。每个进程首次解析 compact 任务的采集视图时会提示此限制，
并在相关快照标注覆盖范围。选到协作 GMM 输出、远端通信端口或 mHC 反向协作归约时，Dumper 会提示
替代采集点，当前 rank 跳过该次采集并照常执行。

MXFP8 重计算执行正向算子，因此 `direction` 本身不能区分普通 forward 与 recompute。
区分二者需要训练集成层附加阶段标签，内部接入方式见 [MegaMoE 辅助计算与诊断](design/mega_moe_design.md#辅助计算与诊断)。
所有模式都需要兼容的 Python/native 制品；新增组件的接入方式见
[诊断设计](design/diagnostics_design.md#组件接入)。

## 常见问题

| 现象 | 检查项 |
| --- | --- |
| `no completed ... window` | 是否完成 active 窗口、调用了边界 `step()`，或正常退出 context |
| trace 或快照为空 | 是否实际调用已接入路径；是否处于 NONE/WARMUP；筛选值是否匹配 |
| 时间线零散或启动有大空洞 | 是否使用业务等价 shape、完成真实 warmup，以及必要的同步与 rank barrier |
| 未知 SoC | 当前 device name 是否有已知 system counter 频率映射；不要猜测频率换算 |
| dropped records | 记录容量由当前队列计算；需核对配置与设备执行是否一致，不能通过 `max_pending_calls` 修复 |
| Task dump 缺少筛选或不支持端口 | 检查必需筛选及组件支持范围；Kernel dump 能读取不等于 Task dump 能安全采集 |
| 单次 Task dump 超预算或采集分配 OOM | 减少单次调用的采集对象，检查 WARNING 中的 rank、step 和 invocation |
| 输出窗口已存在 | 使用新目录重新采集 |
| 写盘失败 | 检查输出权限和空间。Kernel dump 需要输出文件系统支持同目录树内硬链接 |
| merge 无匹配或告警 | 核对同次运行、rank/device、pattern/index/pid、invocation 顺序及是否误选 Host wrapper |
| context 冲突 | 同一进程只允许一个活动诊断 session，Profiler 与 dump 分次运行 |

排查实现或补充接入时阅读[诊断设计](design/diagnostics_design.md)；执行验证的环境与入口集中维护在
[测试指南](../tests/README.md)。

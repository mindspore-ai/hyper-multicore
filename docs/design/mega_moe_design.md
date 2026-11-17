# MegaMoE 设计

本文面向需要修改或扩展 MegaMoE 的开发者，解释分布式计算、精度后端、任务调度和资源管理如何协作。
接口、数值选项和调用方责任见[MegaMoE 特性指南](../mega_moe.md)；task、执行配置和设备 runtime 的公共模型见
[系统架构](../architecture.md)。这里描述当前设计，不将 MegaMoE 的专家拓扑或资源策略作为其他特性的接入要求。

## 设计目标与整体结构

MegaMoE 将专家分发、计算和结果恢复结合起来，使不同专家或数据块的工作可以在依赖满足后并行执行。
路由和通信资源由公共前端管理，精度后端解释具体计算；这样两种精度可以共用分布式调用流程，
同时保留各自的量化、参数布局和反向保存方式。

| 部分 | 负责的内容 | 与其他部分的关系 |
| --- | --- | --- |
| 公共前端与调用准备 | 检查输入、解释 EP 归属、准备路由并收敛跨 rank 错误 | 将路由结果和当前权重交给所选后端 |
| BF16 / MXFP8 后端 | 数学图、参数绑定、plan 与自动微分 | 使用共享通信资源，保存各自反向需要的状态 |
| 共享资源管理 | 容量、workspace 绑定、共享、扩容与关闭 | 管理存储使用期，不持有模型专家参数 |
| 任务编译与设备执行 | 将计算展开为任务，并按队列与依赖执行 | 消费后端生成的描述，与组件通信协议协作 |

当前实现可从[组件入口](../../hyper_multicore/modules/mega_moe/module.py)、
[调用准备](../../hyper_multicore/modules/mega_moe/invocation.py)和
[共享资源管理](../../hyper_multicore/modules/mega_moe/resources.py)沿调用链阅读。

## 分布式计算与数据流

EP 子组内部交换各自的 CANN bootstrap ID，建立对应的通信域，不使用共享 WORLD rendezvous 端口。
这使 PP/DP 下的独立 EP 子组能够分别初始化；同一进程的活动 SHMEM 成员仍受特性指南中的 EP 绑定约定约束。

Router 在源 rank 上为每个 token 选择全局专家编号。分发把输入行交给专家所在的 owner rank，
owner 使用本地专家权重执行 FC1、SwiGLU 和 FC2，再把结果送回源 rank 恢复 token 输出。
路由准备得到实际接收计数与映射，使源 token 数、owner 接收行数和返回顺序能够分别处理。

Push 由源 rank 发起 PUT，Pull 由 owner 发起 GET；两者的 combine 都使用 PUT。
因此改变分发方向也会改变源数据何时可以复用，调用结束必须覆盖相应的通信完成协议。
数值选择独立于传输方向，裁剪和概率加权的精确定义由[数值配置](../mega_moe.md#数值配置)维护。

AIV 执行传输及激活任务，AIC 执行 grouped matmul。不同专家或数据块在依赖允许时可以重叠执行，
从而减少串行 kernel 发射和计算、通信相互等待的边界。这是执行组织的目的，不代表特定负载的实测收益。

### BF16 反向

以下沿用[前向计算关系](../mega_moe.md#用途与计算关系)中的符号和 EIO 权重表达，记专家输出梯度为 dY。
下面以不在 SwiGLU 内额外施加概率的链式求导为骨架：

| 阶段 | 数学关系 | 主要依赖 |
| --- | --- | --- |
| act_grad | dA = dY · W2ᵀ | 分发后的 dY 与当前 W2 |
| w2_grad | dW2 = Aᵀ · dY | 保存的 A 与分发后的 dY |
| swiglu_grad | dU = SwiGLUBackward(dA, U) | dA 与保存的激活输入 |
| gate_grad | dX = dU · W1ᵀ | dU 与当前 W1 |
| w1_grad | dW1 = Xᵀ · dU | 保存或恢复的 X 与 dU |
| combine | 返回 dX，恢复输入梯度的 token 顺序 | gate_grad 的输出 |

act_grad 与 w2_grad 有共同的 dY 来源，但不存在彼此的数学依赖。
gate_grad 与 w1_grad 都消费 dU；w1_grad 并不以 gate_grad 的计算结果作为输入。
combine 需要 dX。实际允许的并行度还取决于任务填充、任务依赖事件（下文简称依赖事件）和核放置，不能仅由数学图推断执行时间线。

反向图的任务填充使用不同计数推进方式，将激活梯度、权重梯度与返回传输纳入相应依赖事件和终止条件。
具体输入布局及依赖计数增量由 [BF16 backward graph](../../hyper_multicore/modules/mega_moe/backends/bf16/backward/graph.py)
与任务 filler 共同确定，不将某种计数推进模式当成所有反向图的规则。

## 精度后端与计算适配

BF16 的 plan 将实际 H/I 传给 GMM 与 SwiGLU tiling 准备逻辑。
当前[前向 tiling](../../hyper_multicore/modules/mega_moe/backends/bf16/forward/tiling_tables.py)和
[反向 tiling](../../hyper_multicore/modules/mega_moe/backends/bf16/backward/tiling_tables.py)
按任务分块选择模板并填入尺寸；它们不是完整的模型尺寸支持矩阵。
`arch35` 还通过相应计算适配器解释实际 GMM 问题与 SwiGLUGrad 参数，
不能仅凭公共前端的正整数校验推断两种架构支持完全相同的尺寸集合。

两种精度共用公开调用入口、外部参数持有方式、路由准备、资源生命周期及传输机制，
数学图、物理参数与保存方式仍各自实现。共享队列协议不意味着相同任务顺序、缓存策略或数值。

| 内部状态 | BF16 | MXFP8 |
| --- | --- | --- |
| 梯度存储 | 按 counts 处理冷专家 dW；已消费的 dispatch 暂存可以用于 dX | 预先清零 partial dW，dX 单独分配 |
| 反向保存 | 保存对应调用的激活、映射和 plan | 保存反向所需结果及原始权重 root，反向重新解析物理存储 |

BF16 前向的任务图由组件生成。FC2 前概率加权的反向需要恢复未加权激活，数值关系见
[FC2 前加权的反向计算](../mega_moe.md#fc2-前加权的反向计算)。
这些计算在激活及其反向 task 内使用局部存储和局部同步信号，不为每个 task 单独发射概率算子。
实现见[概率数学 helper](../../hyper_multicore/ops/common/mega_moe/routing_probabilities.h)。

### MXFP8 计算与保存

MXFP8 在分发后的专家输入及中间激活上准备量化数据，GMM 使用相应数据与 scale 描述符。
模型提供当前权重 root 的前向列布局和反向行布局；反向重新解析当前物理存储，不能依赖之前的临时权重 view。
后端单独管理量化张量、计算中间结果、partial dW 与反向需要的状态。

这一后端仍使用公共的路由、传输与结果恢复机制。量化步骤和 native 算子的 buffer-list 由
[MXFP8 graph](../../hyper_multicore/modules/mega_moe/backends/mxfp8/graph.py)、
[execution](../../hyper_multicore/modules/mega_moe/backends/mxfp8/execution.py)及
[权重绑定](../../hyper_multicore/modules/mega_moe/backends/mxfp8/weights.py)解释，
不能直接套用 BF16 的 task 数、tiling 或 native 算子参数表。

### 辅助计算与诊断

| 功能 | 执行边界 |
| --- | --- |
| 冷专家梯度清零 | 独立辅助 kernel，按 counts 选择需要清零的专家；不同精度的 partial dW 准备方式不同 |
| 路由 metadata | 独立辅助 kernel 生成 row/element 描述符，并有标准张量操作 fallback |
| saved mapping 搬运 | 独立辅助 kernel 复用前向映射，展开 token 并重排概率 |
| BF16 概率数学 | 在对应激活 task 内执行 |
| BF16 SwiGLU 裁剪 | 适配锁定计算源码及相应 fusion、clamp 补丁 |

辅助 kernel 由 Host 调用，不属于融合 kernel 内的 task；其实现入口见
[MoE utilities](../../hyper_multicore/ops/common/mega_moe/utilities.asc)。依赖 revision、补丁与目标架构由
[构建指南](../build.md)及依赖锁维护。

BF16 与 MXFP8 均接入任务 profiler，MXFP8 当前拒绝 tensor dump 捕获；
BF16 dump 的支持范围由其端口与视图适配决定。具体采集方法和限制见[诊断指南](../diagnostics.md)。
测试存在或源码完成接入不等于所有模式已验证，检查范围与运行方式见[测试指南](../../tests/README.md)。

MXFP8 训练集成层通过内部标签接口 `mx_profile_scope(layer=..., phase=...)` 标记层及
`forward`、`recompute`、`backward` 阶段。重计算仍执行正向算子，因此需要这些标签补充调用方向信息。
该接口由 [MXFP8 后端诊断实现](../../hyper_multicore/modules/mega_moe/backends/mxfp8/diagnostics.py)提供，
不属于公共采集入口；普通用户通过 Profiler context 采集，step、micro 和采样 rank 的选择由训练集成层负责。

## 任务生成与设备调度

BF16 后端先建立由算子和张量依赖组成的数学图。算子的 `SplitSpec` 描述从哪些输入取得分块信息、
如何切分输出以及如何计算任务数量。选择 compact 时，lowering 先将相应算子的任务数量改为持久 lane 的数量；
随后图按依赖顺序传播分块信息，让每个算子得到本次需要处理的分块与任务数。

分块信息确定后，任务填充器把每个算子展开为 task，写入参数位置、数据偏移及等待和发布的依赖事件。
组件生成路径补齐终止与通信协议信息，队列编排再安排各 worker 负责的任务及其候选顺序。
这些描述形成执行配置，设备 runtime 按队列和就绪条件消费；MXFP8 后端生成自己的计算任务与量化描述，
再接入共享的专家队列策略。下面说明各步骤中必须保持的关系。

### 从组件图到任务

公共的图、task 与执行配置关系见[任务与执行配置](../architecture.md#任务与执行配置)。
BF16 的图传播根据 `SplitSpec` 和上游张量的分块信息决定任务数量；
若消费者的切分条件依赖上游结果，需要引用图内真实生产的张量，而不是未获得切分信息的外部叶子输入。
例如权重梯度同时读取保存的输入与本轮梯度时，不能仅因两者形状相近就交换切分传播来源。

`TaskSplitValue` 在这条路径中承担专家拓扑与填充计数。
任务数量受架构、形状、lowering 和批处理影响；固定的 tile 大小或某一版任务数公式不代表所有执行模式。

分发可以按专家建立下游依赖，权重梯度和返回传输还需纳入本次 megakernel 执行的完成条件。
专家批处理改变任务集合或粒度时，依赖事件的计数增量也必须调整；空专家、尾块和终止任务需要一并计入。

### 队列与通信进展

worker 的队列选择、依赖等待和完成发布见[运行时模型](../architecture.md#运行时模型)。
normal、profile 和支持 Task dump 的路径复用队列执行逻辑；采集开销仍可能改变运行中实际观察到的就绪顺序。

arch22 BF16 支持 compact、nominal 和专家批处理；compact 的粒度变化先于队列编排。
arch35 BF16 与 MXFP8 将各自数学阶段映射到共享专家调度策略，通信采用固定 owner 和 FIFO 队列，
数学队列可以进行独立就绪选择。

BF16 离线生成路径还使用 RATR（Rank-Aware Tile Reordering），按目标 rank 交错排列 dispatch/combine
传输任务，并按本地 rank 错开目标顺序。它调整队列索引，不改变数学计算；当前实现见
[revise_task_queue](../../hyper_multicore/scheduler/scheduler.py)。这类通信重排旨在减少热点和等待，
效果取决于实际负载与网络，不能将离线重排策略推广为所有执行模式的固定顺序。

MoE 跨 rank 的 source-ready/read-complete 协议由
[组件调用协议](../../hyper_multicore/ops/common/mega_moe/invocation_protocol.h)持有；
设备 runtime 通过协议 hook 处理本地 task、依赖事件及队列执行结束后的通信协议。
通信方向由 `dispatch_mode` 选择，引擎选择独立于分发方向。
有效调优入口见[调度选项](../mega_moe.md#调度选项)，
队列导出和比较见[独立回放与队列诊断](../build.md#独立回放与队列诊断)。

计算与通信重叠依靠不同核上的任务在依赖满足后并行执行。单边通信通过对称存储地址直接发起远端读写，
仍需依赖通知与完成协议保证数据可见性和存储复用，不能理解为消除了同步。
主机侧准备配置，NPU 侧执行任务调度；实际收益取决于任务粒度、路由分布及硬件和网络资源。

### 离线工具

以下工具用于生成 BF16 组件的二进制调度与 tiling 数据，不是公共组件 API 的必需步骤，
也不是 MXFP8 或新模型的通用构图入口。在线执行由组件在需要时准备 plan。

在已准备依赖的仓库根目录执行，核数和形状须按目标配置选择；下面保留一组显式示例参数：

```bash
python -m hyper_multicore.modules.mega_moe.backends.bf16.forward.gen_runtime_data \
    --tp 4 --ep 4 --seq_size 8192 --all_expert_num 32 --top_k 8 \
    --hidden_size 7168 --intermediate_size 2048 --num_cube_cores 24 \
    --output_dir /tmp/mega-moe-forward

python -m hyper_multicore.modules.mega_moe.backends.bf16.backward.gen_runtime_data \
    --tp 4 --ep 4 --seq_size 8192 --all_expert_num 32 --top_k 8 \
    --hidden_size 7168 --intermediate_size 2048 --num_cube_cores 24 \
    --output_dir /tmp/mega-moe-backward
```

这里的 `tp` 属于离线工具的拓扑参数，不是 `MegaMoeExperts` 的公共构造选项。
每个 rank 有独立的 `runtime_config_input_rank_<i>.bin`；前向输出 up_proj、SwiGLU、down_proj tiling，
反向输出 act_grad、gate_grad、w1_grad、w2_grad、SwiGLUGrad tiling。
工具还写出依赖计数和 GMM workspace 文件，其大小由当前生成逻辑决定，不代表所有运行模式的固定预算。
生成文件本身不证明配置能够在目标设备正确执行。

需要在 Python 内复用时，从
[前向生成器](../../hyper_multicore/modules/mega_moe/backends/bf16/forward/gen_runtime_data.py)或
[反向生成器](../../hyper_multicore/modules/mega_moe/backends/bf16/backward/gen_runtime_data.py)
核对图传播、`build_config_for_rank` 和序列化顺序。
这些函数识别具体 MoE 节点，并据此补终止、调整队列和配置通信握手，不能直接用于 attention 等其他组件。

## 参数绑定与执行接口

### 概率数据与任务绑定

BF16 的 FC2 前加权路径在逻辑图中声明概率读写，再进行 lowering、任务填充和队列编排。
`ProbabilityTaskBinding` 关联辅助 operand、激活/传输选项与数据完成条件；
`ProbabilityFillConfig` 在既有任务基础上增加概率 operand，诊断也从同一声明取得语义。
定义见[概率图绑定](../../hyper_multicore/modules/mega_moe/probability.py)。

`source_p`、`received_p`、`received_dp`、`returned_dp` 四个逻辑区域共用 FP32 对称存储，
其布局由 [ProbabilityLayout](../../hyper_multicore/modules/mega_moe/probability_abi.py)描述。
Push 在发布 token 通信通知之前完成概率 PUT，Pull 在更新本地 dispatch 依赖计数之前完成 token 与概率 GET。
消费者等待条件涵盖 token 与概率数据可消费，不能只以其中一种传输的完成作为依据。

### 任务选项与参数校验

数学、遍历、传输内容和通信引擎分别表达；依赖轮询间隔由执行配置的独立字段承载。
数值选项不能借通信引擎或轮询字段传递。

| 定义入口 | 表达的选择 |
| --- | --- |
| [GMM task](../../hyper_multicore/tasks/gmm.py) | GMM 工作覆盖方式 |
| [SwiGLU task](../../hyper_multicore/tasks/swiglu.py) | 裁剪位置及 tile 遍历方式 |
| [传输 task](../../hyper_multicore/tasks/alltoall.py) | 传输分块及引擎选择 |
| [MoE task options](../../hyper_multicore/modules/mega_moe/task_options.py) | 概率应用位置、token 与概率或概率梯度的传输组合 |
| [Device MoE options](../../hyper_multicore/ops/common/mega_moe/task_options.h) | 设备对组件选项的解释 |

物理 bit、端口及 descriptor 数量由对应定义维护。Host 与 Device 分别在其边界校验编码和存储，
未知或冲突选项、缺失 operand、错误单位与方向不能被默认值掩盖。
概率存储与传输要求属于这条 BF16 数值路径，MXFP8 当前使用 FC2 后加权。

### 精度后端的 native 算子接入

任务、队列、依赖事件和序列化的公共关系见[任务与执行配置](../architecture.md#任务与执行配置)，
调用绑定及完成边界见[运行时模型](../architecture.md#运行时模型)。

BF16 使用组件的张量参数布局，MXFP8 还使用量化/GMM 描述符和展开后的 buffer-list。
两种路径分别将当前权重、激活、通信存储和数学参数绑定到自身 CANN 算子入口；共享执行配置格式并不统一这些参数位置。
完整调用签名见[PyTorch 接口](../../hyper_multicore/torch/ops.py)与
[MXFP8 launch](../../hyper_multicore/modules/mega_moe/backends/mxfp8/launch.py)。

## 状态与资源生命周期

执行描述可以复用，但通信存储仍会被每次调用读写。资源管理需要同时保证描述适用、
本地执行结束和远端访问结束，并保留待反向图仍需使用的状态。以下机制共同维护这些条件。

### 计划与存储使用期

构造时不分配通信 workspace，首次绑定从128行的最小 source token 容量开始，按实际需求增长并保留高水位。
调度计划按几何桶缓存，待反向图保留原计划；桶大小不改变真实 token 激活形状。
进入尚未缓存的新桶时需要准备相应计划。

source 容量、调度行数与实际 owner 接收量分别维护：source/返回区按预留容量，
计算中间张量及待反向激活按实际接收量分配，空 owner 保留一行 ABI 占位。

共享资源管理协调通信容量、分配、绑定、扩容和清理。一次调用准备路由后，由所选精度后端取得匹配 plan，
借用所需通信存储并组织数学计算。Pull 的源发布与消费处于同一租约内，自动扩容发生在占用租约之前。
各后端独立管理 plan 缓存；反向保留原调用需要的 plan 和状态，不使用模块后来一次调用的“最新结果”。

BF16 缓存考虑 token 桶和 partial dW 精度；MXFP8 按覆盖范围与接收容量组织缓存，Pull 使用容量上界复用 plan。
这些缓存粒度是内部策略，不改变实际输出形状，也不代表两个后端具有相同命中率或存储布局。

通信存储中的依赖事件计数与跨调用握手状态具有不同生命周期。每次 kernel 执行前清理依赖计数不能破坏仍用于
跨 rank 协调的 generation；源存储复用还要等待远端读取完成。
租约由 workspace 协调：`claim()` 取得逻辑占用，`release()` 在执行流（stream）上记录 stream event 后归还占用。
同一 stream 的后续使用依靠顺序关系，其他 stream 使用前等待该 stream event；归还租约不表示设备已经完成，也不释放存储。
关闭时资源管理检查活跃调用与仍需存储的反向图，并完成设备同步、跨 rank 协调和物理回收。
这些机制由[组件 workspace](../../hyper_multicore/modules/mega_moe/workspace.py)维护，SHMEM 不解释专家容量公式。
资源共享、显式关闭和自动清理的用户约定见[资源共享与关闭](../mega_moe.md#资源共享与关闭)，
物理重建见 [SHMEM restart](../shmem.md#runtime-restart)。

### 自动回收与清理接入

自动清理适配在首次绑定时安装：拦截 Torch `ProcessGroup.shutdown`（旧版本使用
`distributed_c10d._shutdown_backend`），在 WORLD 或 SHMEM Root 被关闭前先回收资源。
另注册先于 torch_npu 退出钩子的 `atexit` 清理，不从 signal handler 中执行 collective。

模块 GC 仅登记使用权释放，不执行 collective；下一个资源组
首次 forward 时，所有 SHMEM Root rank 一起核对资源表，只复用各 rank 都无成员且无待反向图的兼容
workspace，其余可回收的孤儿 workspace 当场释放。此协调不在已绑定模型的每步 forward 中执行。

### 通信 runtime 的持有与重建

SHMEM Python 层通过进程级引用计数管理通信 runtime 的生命周期。MegaMoE 通过进程内的 Heap Manager
持有一份 `shmem.acquire()` 引用，各执行资源组只提交 workspace 预留，不分别增减 runtime 引用。
单个资源组关闭时释放自己的 workspace，资源池完成全部集体清理后再调用一次 `shmem.release()`。

因此，替换不同配置的孤儿 workspace 不会触发通信 runtime 反复关闭和初始化。
更大模型首次绑定时，可以在这份独占引用内按实际 heap 容量重启 runtime。
其他组件直接使用 SHMEM 时持有各自的引用；存在其他使用者时，MegaMoE 不能独占并重启 runtime。

## 扩展与验证

新增精度后端时，后端负责自己的权重表示、计算参数与反向状态，继续通过公共前端取得路由及通信资源。
改变任务粒度时，要同时核对有效行覆盖、空专家、尾块、生产者计数和终止条件；仅调整队列顺序则保持任务集合及数据依赖。
资源生命周期修改还需覆盖连续调用、共享层、跨 stream、待反向图与关闭失败后的状态。

构建和制品准备见[构建指南](../build.md)，用例选择与设备执行见[测试指南](../../tests/README.md)。
CPU 检查不能证明设备端数值或通信进展；验证结果应限定到实际覆盖的架构、精度与通信模式。

### 问题定位

| 现象 | 优先核对 |
| --- | --- |
| 消费者持续等待 | 生产者是否存在、依赖计数是否达到阈值、队列能否推进，通信数据是否完成 |
| 任务覆盖或输出异常 | 图中切分传播来源、lowering 后工作量、尾块与空专家处理、实际核放置 |
| scratch 或依赖计数越界 | 本次容量、物理分配字节数、每核存储及 Host/Device 对布局的解释 |
| native 算子加载或组装失败 | [构建与安装](../build.md)中的制品身份、依赖与日志 |
| 离线脚本导入失败 | 使用当前后端的完整模块路径，从正确的包环境通过 `python -m` 运行 |

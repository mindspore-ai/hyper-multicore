// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#define HYPER_MULTICORE_MX_ROW_ONLY
#include "mxfp8/mx_task_graph_worker.h"
#include "hyper_mega_moe_mxfp8_grad_tiling_key.h"
template <int8_t MX_MIX>
__global__ __aicore__ void hyper_mega_moe_mxfp8_grad(
    GM_ADDR buffers, GM_ADDR runtime, GM_ADDR events, GM_ADDR diagnostic,
    GM_ADDR launch_metadata, GM_ADDR buffers_ref, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  RunMxTaskGraph<MulticoreRuntime::VectorWorkerPolicy::LEGACY_ODD>(buffers, runtime, events, diagnostic, launch_metadata);
}

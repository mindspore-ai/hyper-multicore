/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "kernel_operator.h"
#include "worker_kernel.cpp"
#include "kda_cp1_fast.h"

extern "C" __global__ __aicore__ void hyper_mega_kda(
    GM_ADDR hidden_states, GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v,
    GM_ADDR w_fa, GM_ADDR w_fb, GM_ADDR w_b, GM_ADDR w_ga, GM_ADDR w_gb,
    GM_ADDR w_conv_q, GM_ADDR w_conv_k, GM_ADDR w_conv_v,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR o_norm_weight, GM_ADDR w_o,
    GM_ADDR comm_send_buf, GM_ADDR comm_recv_buf,
    GM_ADDR runtime_config, GM_ADDR all_event_counters, GM_ADDR diagnostic_buffer,
    GM_ADDR bwd_cache, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR userWorkspace = GetUserWorkspace(workspace);
  GM_ADDR inputList[25] = {
      hidden_states, w_q, w_k, w_v, w_fa, w_fb, w_b, w_ga, w_gb,
      w_conv_q, w_conv_k, w_conv_v, a_log, dt_bias, o_norm_weight, w_o,
      comm_send_buf, comm_recv_buf, runtime_config, all_event_counters, diagnostic_buffer,
      bwd_cache, out, userWorkspace, tiling};
  if (tilingData.projMTile == 0U && ((tilingData.numHeads & 1U) == 0U)) {
    if (tilingData.cpSize == 1U) {
      MulticoreKda::RunCp1FastPath(GetBlockIdx(), inputList, &tilingData);
      return;
    }
    if (tilingData.cpMode == 1U || tilingData.cpMode == 2U) {
      MulticoreKda::RunStatePassingCpFastPath(GetBlockIdx(), inputList, &tilingData);
      return;
    }
  }
  MulticoreKda::RunHyperMegaKdaWorker(GetBlockIdx(), reinterpret_cast<__gm__ uint8_t *>(runtime_config),
                                      inputList, &tilingData);
}

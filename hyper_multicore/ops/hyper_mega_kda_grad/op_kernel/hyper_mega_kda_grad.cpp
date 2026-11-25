/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "kernel_operator.h"
#include "kda_bwd_fast.h"

extern "C" __global__ __aicore__ void hyper_mega_kda_grad(
    GM_ADDR d_o_normed, GM_ADDR q_in, GM_ADDR k_in, GM_ADDR v_in,
    GM_ADDR g_in, GM_ADDR beta_in, GM_ADDR g_out,
    GM_ADDR w_conv_q, GM_ADDR w_conv_k, GM_ADDR w_conv_v,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR o_norm_weight,
    GM_ADDR bwd_cache, GM_ADDR comm_send_buf, GM_ADDR comm_recv_buf,
    GM_ADDR all_event_counters,
    GM_ADDR d_q_in, GM_ADDR d_k_in, GM_ADDR d_v_in,
    GM_ADDR d_g_in, GM_ADDR d_beta_in, GM_ADDR d_g_out,
    GM_ADDR d_w_conv, GM_ADDR d_a_log, GM_ADDR d_dt_bias, GM_ADDR d_o_norm_weight,
    GM_ADDR workspace, GM_ADDR tiling) {
  using namespace AscendC;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR userWorkspace = GetUserWorkspace(workspace);
  GM_ADDR io[29] = {
      d_o_normed, q_in, k_in, v_in, g_in, beta_in, g_out,
      w_conv_q, w_conv_k, w_conv_v, a_log, dt_bias, o_norm_weight,
      bwd_cache, comm_send_buf, comm_recv_buf, all_event_counters,
      d_q_in, d_k_in, d_v_in, d_g_in, d_beta_in, d_g_out,
      d_w_conv, d_a_log, d_dt_bias, d_o_norm_weight,
      userWorkspace, tiling};
  MulticoreKdaBwd::RunHyperMegaKdaGrad(GetBlockIdx(), io, &tilingData);
}

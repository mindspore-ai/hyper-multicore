/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include <tuple>
#include <torch/library.h>
#include "csrc/cached_op_api.h"

namespace {

at::Tensor &mega_kda_npu(
    const at::Tensor &hidden_states,
    const at::Tensor &w_q, const at::Tensor &w_k, const at::Tensor &w_v,
    const at::Tensor &w_fa, const at::Tensor &w_fb, const at::Tensor &w_b,
    const at::Tensor &w_ga, const at::Tensor &w_gb,
    const at::Tensor &w_conv_q, const at::Tensor &w_conv_k, const at::Tensor &w_conv_v,
    const at::Tensor &a_log, const at::Tensor &dt_bias,
    const at::Tensor &o_norm_weight, const at::Tensor &w_o,
    const at::Tensor &comm_send_buf, const at::Tensor &comm_recv_buf,
    const at::Tensor &runtime_config, const at::Tensor &all_event_counters,
    const at::Tensor &diagnostic_buffer, const at::Tensor &bwd_cache,
    at::Tensor &out,
    int64_t num_heads, int64_t head_dim, int64_t chunk_size, int64_t cp_size, int64_t cp_rank,
    int64_t head_groups, int64_t num_segs, int64_t proj_m_tile, int64_t o_proj_m_tile,
    bool use_full_rank_gate, bool has_lower_bound,
    double lower_bound, double norm_eps, double scale, int64_t cp_mode, bool save_for_bwd) {
  HYPER_EXEC_NPU_CMD_EXT(
      aclnnHyperMegaKda,
      hidden_states, w_q, w_k, w_v, w_fa, w_fb, w_b, w_ga, w_gb,
      w_conv_q, w_conv_k, w_conv_v, a_log, dt_bias, o_norm_weight, w_o,
      comm_send_buf, comm_recv_buf, runtime_config, all_event_counters, diagnostic_buffer, bwd_cache,
      num_heads, head_dim, chunk_size, cp_size, cp_rank,
      head_groups, num_segs, proj_m_tile, o_proj_m_tile,
      use_full_rank_gate, has_lower_bound,
      lower_bound, norm_eps, scale, cp_mode, save_for_bwd,
      out);
  return out;
}

using GradOutputs = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

GradOutputs mega_kda_grad_npu(
    const at::Tensor &d_o_normed,
    const at::Tensor &q_in, const at::Tensor &k_in, const at::Tensor &v_in,
    const at::Tensor &g_in, const at::Tensor &beta_in, const at::Tensor &g_out,
    const at::Tensor &w_conv_q, const at::Tensor &w_conv_k, const at::Tensor &w_conv_v,
    const at::Tensor &a_log, const at::Tensor &dt_bias, const at::Tensor &o_norm_weight,
    const at::Tensor &bwd_cache, const at::Tensor &comm_send_buf, const at::Tensor &comm_recv_buf,
    const at::Tensor &all_event_counters,
    int64_t num_heads, int64_t head_dim, int64_t chunk_size, int64_t cp_size, int64_t cp_rank, int64_t cp_mode,
    bool has_lower_bound, double lower_bound, double norm_eps, double scale) {
  auto bf_opts = q_in.options();
  auto fp_opts = q_in.options().dtype(at::kFloat);
  const int64_t S = q_in.size(0);
  const int64_t D = num_heads * head_dim;

  auto d_q_in          = at::empty({S, D}, bf_opts);
  auto d_k_in          = at::empty({S, D}, bf_opts);
  auto d_v_in          = at::empty({S, D}, bf_opts);
  auto d_g_in          = at::empty({S, D}, bf_opts);
  auto d_beta_in       = at::empty({S, num_heads}, bf_opts);
  auto d_g_out         = at::empty({S, D}, bf_opts);
  auto d_w_conv        = at::empty({3, D, 4}, fp_opts);
  auto d_a_log         = at::empty({num_heads}, fp_opts);
  auto d_dt_bias       = at::empty({D}, fp_opts);
  auto d_o_norm_weight = at::empty({head_dim}, fp_opts);

  HYPER_EXEC_NPU_CMD_EXT(
      aclnnHyperMegaKdaGrad,
      d_o_normed, q_in, k_in, v_in, g_in, beta_in, g_out,
      w_conv_q, w_conv_k, w_conv_v, a_log, dt_bias, o_norm_weight,
      bwd_cache, comm_send_buf, comm_recv_buf, all_event_counters,
      num_heads, head_dim, chunk_size, cp_size, cp_rank, cp_mode,
      has_lower_bound, lower_bound, norm_eps, scale,
      d_q_in, d_k_in, d_v_in, d_g_in, d_beta_in, d_g_out,
      d_w_conv, d_a_log, d_dt_bias, d_o_norm_weight);

  return std::make_tuple(
      d_q_in, d_k_in, d_v_in, d_g_in, d_beta_in, d_g_out,
      d_w_conv, d_a_log, d_dt_bias, d_o_norm_weight);
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("mega_kda", TORCH_FN(mega_kda_npu));
  m.impl("mega_kda_grad", TORCH_FN(mega_kda_grad_npu));
}

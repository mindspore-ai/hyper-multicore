/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def(
    "mega_kda("
    "  Tensor hidden_states,"
    "  Tensor w_q, Tensor w_k, Tensor w_v,"
    "  Tensor w_fa, Tensor w_fb, Tensor w_b,"
    "  Tensor w_ga, Tensor w_gb,"
    "  Tensor w_conv_q, Tensor w_conv_k, Tensor w_conv_v,"
    "  Tensor a_log, Tensor dt_bias, Tensor o_norm_weight, Tensor w_o,"
    "  Tensor comm_send_buf, Tensor comm_recv_buf,"
    "  Tensor runtime_config, Tensor all_event_counters, Tensor diagnostic_buffer,"
    "  Tensor bwd_cache,"
    "  Tensor(a!) out,"
    "  int num_heads, int head_dim, int chunk_size, int cp_size, int cp_rank,"
    "  int head_groups, int num_segs, int proj_m_tile, int o_proj_m_tile,"
    "  bool use_full_rank_gate, bool has_lower_bound,"
    "  float lower_bound, float norm_eps, float scale, int cp_mode, bool save_for_bwd"
    ") -> Tensor(a!)");

  m.def(
    "mega_kda_grad("
    "  Tensor d_o_normed,"
    "  Tensor q_in, Tensor k_in, Tensor v_in, Tensor g_in, Tensor beta_in, Tensor g_out,"
    "  Tensor w_conv_q, Tensor w_conv_k, Tensor w_conv_v,"
    "  Tensor a_log, Tensor dt_bias, Tensor o_norm_weight,"
    "  Tensor bwd_cache, Tensor comm_send_buf, Tensor comm_recv_buf, Tensor all_event_counters,"
    "  int num_heads, int head_dim, int chunk_size, int cp_size, int cp_rank, int cp_mode,"
    "  bool has_lower_bound, float lower_bound, float norm_eps, float scale"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

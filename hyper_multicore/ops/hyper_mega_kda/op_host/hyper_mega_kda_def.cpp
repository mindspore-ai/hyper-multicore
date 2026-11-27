/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops {

class HyperMegaKda : public OpDef {
 public:
  explicit HyperMegaKda(const char *name) : OpDef(name) {
    AddInput("hidden_states", {ge::DT_BF16});
    AddInput("w_q", {ge::DT_BF16});
    AddInput("w_k", {ge::DT_BF16});
    AddInput("w_v", {ge::DT_BF16});
    AddInput("w_fa", {ge::DT_BF16});
    AddInput("w_fb", {ge::DT_BF16});
    AddInput("w_b", {ge::DT_BF16});
    AddInput("w_ga", {ge::DT_BF16});
    AddInput("w_gb", {ge::DT_BF16});
    AddInput("w_conv_q", {ge::DT_BF16});
    AddInput("w_conv_k", {ge::DT_BF16});
    AddInput("w_conv_v", {ge::DT_BF16});
    AddInput("a_log", {ge::DT_FLOAT});
    AddInput("dt_bias", {ge::DT_FLOAT});
    AddInput("o_norm_weight", {ge::DT_BF16});
    AddInput("w_o", {ge::DT_BF16});
    AddInput("comm_send_buf", {ge::DT_UINT8});
    AddInput("comm_recv_buf", {ge::DT_UINT8});
    AddInput("runtime_config", {ge::DT_UINT8});
    AddInput("all_event_counters", {ge::DT_INT32});
    AddInput("diagnostic_buffer", {ge::DT_INT32});
    AddInput("bwd_cache", {ge::DT_UINT8});

    this->Output("out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_BF16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});

    this->Attr("num_heads").AttrType(OPTIONAL).Int(16);
    this->Attr("head_dim").AttrType(OPTIONAL).Int(128);
    this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
    this->Attr("cp_size").AttrType(OPTIONAL).Int(1);
    this->Attr("cp_rank").AttrType(OPTIONAL).Int(0);
    this->Attr("head_groups").AttrType(OPTIONAL).Int(2);
    this->Attr("num_segs").AttrType(OPTIONAL).Int(1);
    this->Attr("proj_m_tile").AttrType(OPTIONAL).Int(256);
    this->Attr("o_proj_m_tile").AttrType(OPTIONAL).Int(256);
    this->Attr("use_full_rank_gate").AttrType(OPTIONAL).Bool(false);
    this->Attr("has_lower_bound").AttrType(OPTIONAL).Bool(true);
    this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0f);
    this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
    this->Attr("scale").AttrType(OPTIONAL).Float(0.0883883476f);
    this->Attr("cp_mode").AttrType(OPTIONAL).Int(0);
    this->Attr("save_for_bwd").AttrType(OPTIONAL).Bool(false);

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(false)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(false)
      .ExtendCfgInfo("prebuildPattern.value", "Opaque")
      .ExtendCfgInfo("coreType.value", "AiCore")
      .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    this->AICore().AddConfig("ascend910b", config);
    this->AICore().AddConfig("ascend910_93", config);
    this->AICore().AddConfig("ascend950", config);
  }

 private:
  void AddInput(const char *name, std::initializer_list<ge::DataType> types) {
    this->Input(name).ParamType(REQUIRED).DataType(types).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  }
};

OP_ADD(HyperMegaKda);

}  // namespace ops

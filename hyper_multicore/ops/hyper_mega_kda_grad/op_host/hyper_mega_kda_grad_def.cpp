/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops {

class HyperMegaKdaGrad : public OpDef {
 public:
  explicit HyperMegaKdaGrad(const char *name) : OpDef(name) {
    AddInput("d_o_normed", {ge::DT_BF16});
    AddInput("q_in", {ge::DT_BF16});
    AddInput("k_in", {ge::DT_BF16});
    AddInput("v_in", {ge::DT_BF16});
    AddInput("g_in", {ge::DT_BF16});
    AddInput("beta_in", {ge::DT_BF16});
    AddInput("g_out", {ge::DT_BF16});
    AddInput("w_conv_q", {ge::DT_BF16});
    AddInput("w_conv_k", {ge::DT_BF16});
    AddInput("w_conv_v", {ge::DT_BF16});
    AddInput("a_log", {ge::DT_FLOAT});
    AddInput("dt_bias", {ge::DT_FLOAT});
    AddInput("o_norm_weight", {ge::DT_BF16});
    AddInput("bwd_cache", {ge::DT_UINT8});
    AddInput("comm_send_buf", {ge::DT_UINT8});
    AddInput("comm_recv_buf", {ge::DT_UINT8});
    AddInput("all_event_counters", {ge::DT_INT32});

    AddOutput("d_q_in", {ge::DT_BF16});
    AddOutput("d_k_in", {ge::DT_BF16});
    AddOutput("d_v_in", {ge::DT_BF16});
    AddOutput("d_g_in", {ge::DT_BF16});
    AddOutput("d_beta_in", {ge::DT_BF16});
    AddOutput("d_g_out", {ge::DT_BF16});
    AddOutput("d_w_conv", {ge::DT_FLOAT});
    AddOutput("d_a_log", {ge::DT_FLOAT});
    AddOutput("d_dt_bias", {ge::DT_FLOAT});
    AddOutput("d_o_norm_weight", {ge::DT_FLOAT});

    this->Attr("num_heads").AttrType(OPTIONAL).Int(16);
    this->Attr("head_dim").AttrType(OPTIONAL).Int(128);
    this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
    this->Attr("cp_size").AttrType(OPTIONAL).Int(1);
    this->Attr("cp_rank").AttrType(OPTIONAL).Int(0);
    this->Attr("cp_mode").AttrType(OPTIONAL).Int(0);
    this->Attr("has_lower_bound").AttrType(OPTIONAL).Bool(true);
    this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0f);
    this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
    this->Attr("scale").AttrType(OPTIONAL).Float(0.0883883476f);

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
  void AddOutput(const char *name, std::initializer_list<ge::DataType> types) {
    this->Output(name).ParamType(REQUIRED).DataType(types).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  }
};

OP_ADD(HyperMegaKdaGrad);

}  // namespace ops

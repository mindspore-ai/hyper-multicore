// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include "register/op_def_registry.h"
namespace ops {
class HyperMegaMoeMxfp8Grad : public OpDef {
 public:
  explicit HyperMegaMoeMxfp8Grad(const char* name) : OpDef(name) {
    Input("buffers").ParamType(DYNAMIC).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
    for (const auto* resource : {"runtime_config", "events", "diagnostic", "launch_metadata"}) {
      Input(resource).ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
    }
    Output("buffers_ref").ParamType(DYNAMIC).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
    Attr("buffer_count").AttrType(REQUIRED).Int();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
      .ExtendCfgInfo("prebuildPattern.value", "Opaque").ExtendCfgInfo("coreType.value", "AiCore")
      .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    AICore().AddConfig("ascend950", config);
  }
};
OP_ADD(HyperMegaMoeMxfp8Grad);
}  // namespace ops

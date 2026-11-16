// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#pragma once
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
namespace ops {
class MegaMoeMxfp8Schema : public OpDef {
 public:
  explicit MegaMoeMxfp8Schema(const char* name) : OpDef(name) {
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
namespace mega_moe_mxfp8_detail {
inline ge::graphStatus InferShape(gert::InferShapeContext* context) {
  auto count = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (count == nullptr || *count <= 0 || *count > 64) return ge::GRAPH_FAILED;
  for (int64_t i = 0; i < *count; ++i) {
    *context->GetOutputShape(i) = *context->GetDynamicInputShape(0, i);
  }
  return ge::GRAPH_SUCCESS;
}
inline ge::graphStatus InferType(gert::InferDataTypeContext* context) {
  auto count = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (count == nullptr || *count <= 0 || *count > 64) return ge::GRAPH_FAILED;
  for (int64_t i = 0; i < *count; ++i) context->SetOutputDataType(i, ge::DT_UINT8);
  return ge::GRAPH_SUCCESS;
}
}
}  // namespace ops

// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include "register/op_impl_registry.h"
namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context) {
  auto count = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (count == nullptr || *count <= 0 || *count > 64) return ge::GRAPH_FAILED;
  for (int64_t i = 0; i < *count; ++i) {
    *context->GetOutputShape(i) = *context->GetDynamicInputShape(0, i);
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext* context) {
  auto count = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (count == nullptr || *count <= 0 || *count > 64) return ge::GRAPH_FAILED;
  for (int64_t i = 0; i < *count; ++i) context->SetOutputDataType(i, ge::DT_UINT8);
  return ge::GRAPH_SUCCESS;
}
}
IMPL_OP_INFERSHAPE(HyperMegaMoeMxfp8Grad).InferShape(InferShape).InferDataType(InferType);
}

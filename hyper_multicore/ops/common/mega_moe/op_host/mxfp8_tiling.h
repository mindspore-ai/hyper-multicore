// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#pragma once
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling::mega_moe_mxfp8_detail {
template <class TilingData>
inline ge::graphStatus Tiling(gert::TilingContext* context) {
  auto count = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (count == nullptr || *count <= 0 || *count > 64) return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint32_t cores = platform.GetCoreNumAic();
  if (cores == 0) return ge::GRAPH_FAILED;
  TilingData tiling;
  tiling.set_bufferCount(*count);
  tiling.set_cubeCount(cores);
  auto raw = context->GetRawTilingData();
  tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(cores);
  context->SetTilingKey(GET_TPL_TILING_KEY(1));
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling::mega_moe_mxfp8_detail

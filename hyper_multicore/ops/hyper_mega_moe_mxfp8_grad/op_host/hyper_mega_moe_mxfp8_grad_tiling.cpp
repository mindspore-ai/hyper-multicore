// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include "hyper_mega_moe_mxfp8_grad_tiling.h"
#include "../op_kernel/hyper_mega_moe_mxfp8_grad_tiling_key.h"
#include "common/mega_moe/op_host/mxfp8_tiling.h"
namespace optiling {
IMPL_OP_OPTILING(HyperMegaMoeMxfp8Grad).Tiling(mega_moe_mxfp8_detail::Tiling<HyperMegaMoeMxfp8GradTilingData>);
}  // namespace optiling

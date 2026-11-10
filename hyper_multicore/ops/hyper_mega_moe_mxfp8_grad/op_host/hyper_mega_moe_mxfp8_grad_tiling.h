// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MEGA_MOE_MXFP8_GRAD_TILING_H_
#define HYPER_MEGA_MOE_MXFP8_GRAD_TILING_H_
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HyperMegaMoeMxfp8GradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bufferCount);
TILING_DATA_FIELD_DEF(uint32_t, cubeCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HyperMegaMoeMxfp8Grad, HyperMegaMoeMxfp8GradTilingData)
}
#endif

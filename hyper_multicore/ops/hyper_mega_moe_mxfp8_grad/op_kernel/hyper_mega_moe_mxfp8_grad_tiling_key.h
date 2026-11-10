// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
// A single selector makes the registered CANN build retain the MIX launch contract.
ASCENDC_TPL_ARGS_DECL(HyperMegaMoeMxfp8Grad, ASCENDC_TPL_UINT_DECL(MX_MIX, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                  ASCENDC_TPL_UINT_SEL(MX_MIX, ASCENDC_TPL_UI_LIST, 1)));

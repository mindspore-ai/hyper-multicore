// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include "common/mega_moe/op_host/mxfp8_schema.h"
namespace ops {
class HyperMegaMoeMxfp8 : public MegaMoeMxfp8Schema { public: using MegaMoeMxfp8Schema::MegaMoeMxfp8Schema; };
OP_ADD(HyperMegaMoeMxfp8);
IMPL_OP_INFERSHAPE(HyperMegaMoeMxfp8).InferShape(mega_moe_mxfp8_detail::InferShape)
    .InferDataType(mega_moe_mxfp8_detail::InferType);
}  // namespace ops

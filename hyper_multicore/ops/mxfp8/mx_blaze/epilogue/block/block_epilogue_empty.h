/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_empty.h
 * \brief
 */

#pragma once
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

class BlockEpilogueEmpty {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        Params() = default;
    };

    __aicore__ inline BlockEpilogueEmpty() {}

    __aicore__ inline void Run() { return; }

    __aicore__ inline void operator()(Params const& params) { Run(); }

    __aicore__ inline void operator()(BlockShape const& blockShape, BlockCoord const& blockCoord,
                                      int64_t dstStartOffset = 0, int64_t srcStartOffset = 0)
    {
        return;
    }
};
} // namespace Block
} // namespace Epilogue

namespace Gemm::Block {
using Epilogue::Block::BlockEpilogueEmpty;
}

} // namespace Blaze

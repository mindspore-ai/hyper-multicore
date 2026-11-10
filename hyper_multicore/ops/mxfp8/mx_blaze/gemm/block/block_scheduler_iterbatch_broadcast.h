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
 * \file block_scheduler_iterbatch_broadcast.h
 * \brief Scheduler for IterBatch-Broadcast path: batch grouping + broadcast axis info
 */

#pragma once

#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerIterBatchBroadcast {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        uint32_t baseM = 1;
        uint32_t baseN = 1;
        uint32_t baseK = 1;
        uint32_t iterBatchL1 = 1;
        uint32_t iterBatchL0 = 1;
        uint32_t broadcastAxisA = 1;
        uint32_t broadcastAxisB = 1;
        uint32_t aBatchDim0 = 1;
        uint32_t aBatchDim1 = 1;
        uint32_t aBatchDim2 = 1;
        uint32_t aBatchDim3 = 1;
        uint32_t bBatchDim0 = 1;
        uint32_t bBatchDim1 = 1;
        uint32_t bBatchDim2 = 1;
        uint32_t bBatchDim3 = 1;
        uint32_t cBatchDim0 = 1;
        uint32_t cBatchDim1 = 1;
        uint32_t cBatchDim2 = 1;
        uint32_t cBatchDim3 = 1;
        uint8_t isHf32 = 0;
    };

public:
    __aicore__ inline BlockSchedulerIterBatchBroadcast(const ProblemShape& shape, const Params& params)
    {
        m_ = asc::te::get<MNK_M>(shape);
        n_ = asc::te::get<MNK_N>(shape);
        k_ = asc::te::get<MNK_K>(shape);
        b_ = asc::te::get<MNK_B>(shape);
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        iterBatchL1_ = params.iterBatchL1;
        iterBatchL0_ = params.iterBatchL0;
        broadcastAxisA_ = params.broadcastAxisA;
        broadcastAxisB_ = params.broadcastAxisB;
        aBatchDim0_ = params.aBatchDim0;
        aBatchDim1_ = params.aBatchDim1;
        aBatchDim2_ = params.aBatchDim2;
        aBatchDim3_ = params.aBatchDim3;
        bBatchDim0_ = params.bBatchDim0;
        bBatchDim1_ = params.bBatchDim1;
        bBatchDim2_ = params.bBatchDim2;
        bBatchDim3_ = params.bBatchDim3;
        cBatchDim0_ = params.cBatchDim0;
        cBatchDim1_ = params.cBatchDim1;
        cBatchDim2_ = params.cBatchDim2;
        cBatchDim3_ = params.cBatchDim3;
        InitBatchUnit();
    }

    __aicore__ inline void InitBatchUnit()
    {
        bool isBroadcastA = broadcastAxisA_ < static_cast<int64_t>(MAX_BATCH_DIM);
        int64_t bcAxis = isBroadcastA ? broadcastAxisA_ : broadcastAxisB_;
        const int64_t cBatchDims[MAX_BATCH_DIM] = {cBatchDim0_, cBatchDim1_, cBatchDim2_, cBatchDim3_};
        if (bcAxis >= static_cast<int64_t>(MAX_BATCH_DIM)) {
            batchBroadcast_ = 1;
            innerProduct_ = 1;
            outerProduct_ = 1;
            batchUnit_ = b_;
            batchUnitCnt_ = CeilDiv(b_, iterBatchL1_);
            return;
        }
        batchBroadcast_ = cBatchDims[bcAxis];
        innerProduct_ = 1;
        for (int64_t i = bcAxis + 1; i < static_cast<int64_t>(MAX_BATCH_DIM); ++i) {
            innerProduct_ *= cBatchDims[i];
        }
        outerProduct_ = 1;
        for (int64_t i = 0; i < bcAxis; ++i) {
            outerProduct_ *= cBatchDims[i];
        }
        batchUnit_ = (innerProduct_ == 1) ? batchBroadcast_ : innerProduct_;
        batchUnitCnt_ = CeilDiv(batchUnit_, iterBatchL1_);
    }

    __aicore__ inline int64_t GetBlockNums()
    {
        // tile never crosses batchUnit_ boundary: (b_ / batchUnit_) units, each split into batchUnitCnt_ segments
        return (b_ / batchUnit_) * batchUnitCnt_;
    }

    __aicore__ inline int64_t GetCoreNums(int64_t blockNum)
    {
        int64_t tileNum = GetBlockNums();
        return (tileNum < blockNum) ? tileNum : blockNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx, int64_t tileNum)
    {
        int64_t segIdx = tileIdx % batchUnitCnt_;
        uint64_t curIterBatchL1 = (segIdx + 1 == batchUnitCnt_) ? (batchUnit_ - segIdx * iterBatchL1_) : iterBatchL1_;
        return {m_, n_, k_, curIterBatchL1};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
    {
        int64_t batchUnitIdx = tileIdx / batchUnitCnt_;
        int64_t segIdx = tileIdx % batchUnitCnt_;
        return {0, 0, 0, batchUnitIdx * batchUnit_ + segIdx * iterBatchL1_};
    }

    __aicore__ inline int64_t ComputeABroadcastIndex(int64_t cBatchIdx) const
    {
        int64_t cDim123 = cBatchDim1_ * cBatchDim2_ * cBatchDim3_;
        int64_t cDim23 = cBatchDim2_ * cBatchDim3_;
        int64_t batchC0 = cBatchIdx / cDim123;
        int64_t batchC1 = (cBatchIdx % cDim123) / cDim23;
        int64_t batchC2 = (cBatchIdx % cDim23) / cBatchDim3_;
        int64_t batchC3 = cBatchIdx % cBatchDim3_;
        int64_t batchA0 = batchC0 % aBatchDim0_;
        int64_t batchA1 = batchC1 % aBatchDim1_;
        int64_t batchA2 = batchC2 % aBatchDim2_;
        int64_t batchA3 = batchC3 % aBatchDim3_;
        return batchA0 * (aBatchDim1_ * aBatchDim2_ * aBatchDim3_) + batchA1 * (aBatchDim2_ * aBatchDim3_) +
               batchA2 * aBatchDim3_ + batchA3;
    }

    __aicore__ inline int64_t ComputeBBroadcastIndex(int64_t cBatchIdx) const
    {
        int64_t cDim123 = cBatchDim1_ * cBatchDim2_ * cBatchDim3_;
        int64_t cDim23 = cBatchDim2_ * cBatchDim3_;
        int64_t batchC0 = cBatchIdx / cDim123;
        int64_t batchC1 = (cBatchIdx % cDim123) / cDim23;
        int64_t batchC2 = (cBatchIdx % cDim23) / cBatchDim3_;
        int64_t batchC3 = cBatchIdx % cBatchDim3_;
        int64_t batchB0 = batchC0 % bBatchDim0_;
        int64_t batchB1 = batchC1 % bBatchDim1_;
        int64_t batchB2 = batchC2 % bBatchDim2_;
        int64_t batchB3 = batchC3 % bBatchDim3_;
        return batchB0 * (bBatchDim1_ * bBatchDim2_ * bBatchDim3_) + batchB1 * (bBatchDim2_ * bBatchDim3_) +
               batchB2 * bBatchDim3_ + batchB3;
    }

    __aicore__ inline bool IsBroadcastSideSingleBatch() const
    {
        bool isBroadcastA = broadcastAxisA_ < static_cast<int64_t>(MAX_BATCH_DIM);
        int64_t bcAxis = isBroadcastA ? broadcastAxisA_ : broadcastAxisB_;
        if (bcAxis >= static_cast<int64_t>(MAX_BATCH_DIM)) {
            return false;
        }
        return innerProduct_ == 1;
    }

private:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t b_{0};
    int64_t iterBatchL1_{1};
    int64_t iterBatchL0_{1};
    int64_t baseM_{1};
    int64_t baseN_{1};
    int64_t baseK_{1};
    int64_t broadcastAxisA_{1};
    int64_t broadcastAxisB_{1};
    int64_t aBatchDim0_{1};
    int64_t aBatchDim1_{1};
    int64_t aBatchDim2_{1};
    int64_t aBatchDim3_{1};
    int64_t bBatchDim0_{1};
    int64_t bBatchDim1_{1};
    int64_t bBatchDim2_{1};
    int64_t bBatchDim3_{1};
    int64_t cBatchDim0_{1};
    int64_t cBatchDim1_{1};
    int64_t cBatchDim2_{1};
    int64_t cBatchDim3_{1};
    int64_t batchUnit_{1};
    int64_t batchUnitCnt_{1};
    int64_t batchBroadcast_{1};
    int64_t innerProduct_{1};
    int64_t outerProduct_{1};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze

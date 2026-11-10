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
 * \file block_epilogue_fixpipe.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "mxfp8/mx_blaze/epilogue/fusion/default_fusion_op.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "mxfp8/mx_blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_,
          typename FusionOp_ = Gemm::Block::DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFixpipe {
public:
    __aicore__ inline BlockEpilogueFixpipe() {}

    struct Params {
        GM_ADDR outGmAddr{nullptr};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using DispatchPolicy = DispatchPolicy_;

    // block shape
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr uint32_t DATA_BLOCK = 32;
    static constexpr uint32_t OUT_ALIGN = DATA_BLOCK / sizeof(DataTypeOut);

    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr int64_t SPLIT_M_ALIGN = 2;

    // input ub tensor and output global tensor

    AscendC::LocalTensor<DataTypeIn> ubLocalTmp_;
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;

    // attribute
    ProblemShape problemShape_;
    uint64_t cvPingPong_{0};

    __aicore__ inline void Init(Params const& params, ProblemShape& problemShape)
    {
        // init output global
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));
        problemShape_ = problemShape;
        static_assert(sizeof(DataTypeIn) >= sizeof(DataTypeOut), "DataTypeIn size must be >= DataTypeOut size");
    }

    __aicore__ inline void Run(BlockShape const& blockShape, int64_t dstOffset, bool splitM, int64_t baseM,
                               int64_t baseN, uint64_t ubDB = 1)
    {
        int64_t mL1 = asc::te::get<Gemm::MNK_M>(blockShape);
        int64_t curM = mL1;
        if (baseM != 0) {
            curM = Blaze::Gemm::Min(curM, baseM);
        }
        int64_t halfBlockShapeM = Blaze::Gemm::CeilDiv(curM, AscendC::GetTaskRation());
        int64_t blockShapeM = curM;
        if (splitM) {
            blockShapeM = (static_cast<uint64_t>(curM) & 1UL) > 0UL ? (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                                                                      halfBlockShapeM;
        }
        int64_t nL1 = asc::te::get<Gemm::MNK_N>(blockShape);
        int64_t curBaseN = (baseN != 0) ? Blaze::Gemm::Min(nL1, baseN) : nL1;
        int64_t nL1Iter = Blaze::Gemm::CeilDiv(nL1, curBaseN);
        int64_t N = asc::te::get<Gemm::MNK_N>(problemShape_);
        constexpr int64_t c0Size = static_cast<int64_t>(asc::te::c0_element<DataTypeOut>);
        constexpr int64_t ubHalfElems = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn) /
                                                             Gemm::DOUBLE_BUFFER_COUNT);
        bool enablePp = (ubDB > 1);

        for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
            int64_t tileN = (nIdx + 1 == nL1Iter) ? (nL1 - curBaseN * nIdx) : curBaseN;
            int64_t blockShapeNAlign = Blaze::Gemm::CeilAlign(tileN, c0Size);
            int64_t inputSize = blockShapeM * blockShapeNAlign;
            uint16_t slot = enablePp ? static_cast<uint16_t>(cvPingPong_ & 1UL) : 0U;

            // wait for AIC fixpipe (chunk ready) on the pipe that consumes UB first
            if constexpr (DispatchPolicy::FUSED_OP_TYPE == Gemm::OP_TYPE_RELU) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + slot);
            } else {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + slot);
            }
            AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::VECIN, 0,
                                                      AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
            // point UB source to this chunk's ping-pong slot
            ubLocalTmp_ = ubLocal_[slot * ubHalfElems];

            if (inputSize > 0) {
                // copyOut dstOffset along N advances per chunk; subBlock M split preserved
                int64_t offset = dstOffset + nIdx * curBaseN + halfBlockShapeM * N * (AscendC::GetSubBlockIdx() & 0x1);
                AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(blockShapeM),
                                                      static_cast<uint32_t>(tileN * sizeof(DataTypeOut)), 0,
                                                      static_cast<int64_t>((N - tileN) * sizeof(DataTypeOut)), 0};
                if constexpr (DispatchPolicy::FUSED_OP_TYPE == Gemm::OP_TYPE_RELU &&
                              !AscendC::IsSameType<DataTypeOut, bfloat16_t>::value) {
                    AscendC::Relu(ubLocalTmp_, ubLocalTmp_, blockShapeM * tileN);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
                }
                AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], ubLocalTmp_, copyParams);
            }

            // notify AIC the UB slot is free
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot);
            cvPingPong_++;
        }
    }

    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset = 0, bool splitM = false,
                                      int64_t baseM = 0, int64_t baseN = 0, uint64_t ubDB = 1)
    {
        Run(blockShape, dstOffset, splitM, baseM, baseN, ubDB);
        return;
    }
};
} // namespace Block
} // namespace Epilogue
} // namespace Blaze

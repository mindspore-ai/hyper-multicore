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
 * \file kernel_matmul_with_scale_add.h
 * \brief Batched matmul kernel with a dedicated scale-add epilogue.
 */

#pragma once

#include "kernel_basic_intf.h"

#include "mxfp8/mx_blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadFmmWithScaleAdd, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockShape = typename BlockScheduler::BlockShape;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline GemmUniversal()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1);
        }
    }

    __aicore__ inline ~GemmUniversal()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1);
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1 + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void operator()(const Params& params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        Init(params);
        if ASCEND_IS_AIV {
            if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }

        BlockScheduler bs(params.problemShape, params.schParams);
        if (curBlockIdx >= bs.GetCoreNums()) {
            return;
        }

        BlockEpilogue epilogueOp;
        BlockMmad blockMmad;
        Blaze::Gemm::SetHF32(params.schParams.isHf32);
        epilogueOp.Init(params.epilogueParams, problemShape_);
        if ASCEND_IS_AIC {
            auto mmParams = params.mmadParams;
            // Keep one accumulator buffer at the beginning of UB. The epilogue places its shared x3/output buffer
            // immediately after the actual accumulator data.
            mmParams.ubDB = 1;
            blockMmad.Init(mmParams);
        }
        MatmulProcess(params, epilogueOp, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());
        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline void MatmulProcess(const Params& params, BlockEpilogue& epilogueOp, BlockMmad& blockMmad,
                                         BlockScheduler& bs, int64_t curBlockIdx, int64_t coreNums,
                                         int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA{}(m_, k_);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = NormalizeBlockShape(bs.template GetBlockShape<TRANS_B, BType>(blockIdx));
            auto blockCoord = bs.GetBlockCoord(blockIdx);
            const int64_t coordM = asc::te::get<MNK_M>(blockCoord);
            const int64_t coordN = asc::te::get<MNK_N>(blockCoord);
            const int64_t batchIdx = asc::te::get<MNK_B>(blockCoord);
            const int64_t shapeM = asc::te::get<MNK_M>(blockShape);
            int64_t shapeN = asc::te::get<MNK_N>(blockShape);
            const int64_t shapeK = asc::te::get<MNK_K>(blockShape);
            shapeN = AscendC::Std::min(shapeN, static_cast<int64_t>(n_) - coordN);
            if (shapeM <= 0 || shapeN <= 0) {
                continue;
            }
            TupleShape validBlockShape{shapeM, shapeN, shapeK, 1};

            const uint64_t batchOffsetA = static_cast<uint64_t>(batchIdx) * m_ * k_;
            const uint64_t batchOffsetB = static_cast<uint64_t>(batchIdx) * k_ * n_;
            const int64_t offsetC = (batchIdx * static_cast<int64_t>(m_) + coordM) * static_cast<int64_t>(n_) + coordN;
            auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_ + batchOffsetA),
                                            layoutA);
            auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_ + batchOffsetB),
                                            layoutB);
            SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
            auto gmBlockA = gmA.slice(AscendC::MakeCoord(coordM, 0L), AscendC::MakeShape(shapeM, shapeK));
            auto gmBlockB = gmB.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(shapeK, shapeN));
            auto gmBlockBias = gmBias.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));

            // UB Tensor ownership stays in the kernel, matching the other GemmUniversal implementations.
            auto layoutUb = MakeLayoutC{}(shapeM, shapeN);
            auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), layoutUb);
            if ASCEND_IS_AIC {
                blockMmad(gmBlockA, gmBlockB, gmBlockBias, ubTensor, validBlockShape);
            }
            if ASCEND_IS_AIV {
                epilogueOp(ubTensor, validBlockShape, offsetC, params.mmadParams.splitM, params.schParams.baseM,
                           params.schParams.baseN);
            }
        }
    }

    __aicore__ inline BlockShape NormalizeBlockShape(const BlockShape& blockShape) const
    {
        return {asc::te::get<MNK_M>(blockShape), asc::te::get<MNK_N>(blockShape), asc::te::get<MNK_K>(blockShape), 1};
    }

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = params.problemShape;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode) const
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

private:
    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t FLAG_ID_MAX = 16;

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    TupleShape problemShape_{};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

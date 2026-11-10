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
 * \file kernel_matmul_fixpipe_opti.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "mxfp8/mx_blaze/epilogue/block/block_epilogue_fixpipe.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad_matmul_bl1_full_load.h"
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
                        KernelMmadMultiBlockFixpipeOpti, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);     // ping
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1); // pong
        }
    }
    __aicore__ inline ~GemmUniversal()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);                   // ping
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);     // ping
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1);               // pong
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1 + FLAG_ID_MAX); // pong
        }
    }

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    // mmad
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
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    static_assert(
        AscendC::Std::is_one_of_v<
            AscendC::Std::tuple<AType, BType, CType, BiasType>, AscendC::Std::tuple<half, half, half, half>,
            AscendC::Std::tuple<half, half, half, float>, AscendC::Std::tuple<half, half, float, half>,
            AscendC::Std::tuple<half, half, float, float>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, float>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, float, bfloat16_t>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, float, float>, AscendC::Std::tuple<float, float, float, float>>,
        "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be nz_layout_ptn or zn_layout_ptn");

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        // 初始化epilogue和mmad
        BlockEpilogue epilogueOp;
        BlockMmad blockMmad;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        Init(params);
        if ASCEND_IS_AIV {
            if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        // 初始化blockScheduler
        BlockScheduler bs(params.problemShape, params.schParams);
        int64_t realCoreNums = bs.GetCoreNums();
        if (curBlockIdx >= realCoreNums) {
            return;
        }
        Blaze::Gemm::SetHF32(params.schParams.isHf32);
        epilogueOp.Init(params.epilogueParams, problemShape_);
        if ASCEND_IS_AIC {
            blockMmad.Init(params.mmadParams);
        }
        MatmulProcess(params, epilogueOp, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());
        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline auto MakeLayoutA2D(Params const& params)
    {
        auto layoutA = MakeLayoutA{}(m_, k_);
        if constexpr (!TRANS_A) {
            // 连续场景下rowStride表示k或1, 非连续场景下表示m轴的stride
            uint64_t rowStride = params.mmadParams.rowStride == 0 ? k_ : params.mmadParams.rowStride;
            layoutA = asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, m_), asc::te::make_shape(asc::te::_1{}, k_)),
                asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, rowStride),
                                     asc::te::make_stride(asc::te::_0{}, asc::te::_1{})));
        }
        return layoutA;
    }

    __aicore__ inline void MatmulProcess(Params const& params, BlockEpilogue& epilogueOp, BlockMmad& blockMmad,
                                         BlockScheduler& bs, int64_t curBlockIdx, int64_t coreNums,
                                         int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA2D(params);
        auto layoutB = MakeLayoutB{}(k_, n_);       // ND layout for B
        auto layoutC = MakeLayoutC{}(m_, n_);       // ND layout for C
        auto layoutBias = MakeLayoutBias{}(1L, n_); // ND layout for Bias
        // A,B,C Gm Tensor
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto blockCoord = bs.GetBlockCoord(blockIdx); // (m_, n_, k_, b)
            auto coordM = asc::te::get<MNK_M>(blockCoord);
            auto coordN = asc::te::get<MNK_N>(blockCoord);
            int64_t offsetC = coordM * n_ + coordN;
            auto shapeM = asc::te::get<MNK_M>(blockShape);
            auto shapeN = asc::te::get<MNK_N>(blockShape);
            auto shapeK = asc::te::get<MNK_K>(blockShape);
            shapeN = AscendC::Std::min(shapeN, static_cast<int64_t>(n_) - coordN);
            TupleShape validBlockShape{shapeM, shapeN, shapeK, asc::te::get<MNK_B>(blockShape)};

            auto gmBlockA = gmA.slice(AscendC::MakeCoord(coordM, 0L), AscendC::MakeShape(shapeM, shapeK));
            auto gmBlockB = gmB.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(shapeK, shapeN));

            int64_t ubOffsetElems = 0;
            auto layoutUB = MakeLayoutC{}(shapeM, shapeN);
            auto ubLocal = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, CType>(ubOffsetElems * sizeof(CType)), layoutUB);
            auto gmBlockBias = gmBias.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));
            if ASCEND_IS_AIC {
                if constexpr (BlockMmad::DispatchPolicy::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
                    blockMmad(gmBlockA, gmB, gmBias, ubLocal, validBlockShape);
                } else {
                    blockMmad(gmBlockA, gmBlockB, gmBlockBias, ubLocal, validBlockShape);
                }
            }
            if ASCEND_IS_AIV {
                // Calculate epilogue (internally loops N with per-chunk cv sync)
                epilogueOp(validBlockShape, offsetC, params.mmadParams.splitM, params.schParams.baseM,
                           params.schParams.baseN, params.mmadParams.ubDB);
            }
        }
    }
    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

private:
    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化

    TupleShape problemShape_{};
    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

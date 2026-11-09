/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_tqbmm_mx.h
 * \brief
 */

#pragma once

#include "kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "mxfp8/mx_blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "mxfp8/mx_blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockTQBMM, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static_assert(AscendC::Std::is_one_of_v<AscendC::Std::tuple<AType, BType, CType>,
                                            AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t>,
                                            AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e4m3fn_t, half>,
                                            AscendC::Std::tuple<fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t>,
                                            AscendC::Std::tuple<fp4x2_e2m1_t, fp4x2_e2m1_t, half>>,
                  "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be NZLayoutPtn or ZNLayoutPtn");

    struct TQBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        TQBMMTiling tqbmmParams;
        Params() = default;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicAdd<float>();
        }
        Init(params);

        const auto& problemShape = params.problemShape;
        const auto& tqbmmParams = params.tqbmmParams;

        BlockScheduler bs(problemShape, params.schParams);

        const BlockShape l0BlockShape{static_cast<int64_t>(tqbmmParams.baseM), static_cast<int64_t>(tqbmmParams.baseN),
                                      static_cast<int64_t>(tqbmmParams.baseK), 0};
        mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias_, tqbmmParams.dbL0C > 1);

        const uint64_t scaleKLen = Blaze::Gemm::CeilDiv(k_, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                                   MXFP_MULTI_BASE_SIZE;

        const uint64_t aBatchStride = isPermX1_ ? k_ : (m_ * k_);
        const uint64_t aMStride = isPermX1_ ? (batch_ * k_) : k_;

        auto layoutA = MakeBatchLayout<LayoutA, AType, AscendC::Std::Int<C0_SIZE>>(batch_, m_, k_, aBatchStride,
                                                                                   aMStride);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);
        auto layoutC = MakeBatchLayout<LayoutC, CType, AscendC::Std::Int<asc::te::c0_element<CType>>>(batch_, m_, n_,
                                                                                                      n_, batch_ * n_);

        const uint64_t scaleABatchStride = isPermX1_ ? scaleKLen : m_ * scaleKLen;
        const uint64_t scaleAMStride = isPermX1_ ? batch_ * scaleKLen : scaleKLen;
        auto layoutScaleA = MakeBatchLayout<ScaleALayoutPtn, AscendC::fp8_e8m0_t, AscendC::Std::Int<SCALE_C0>>(
            batch_, m_, scaleKLen, scaleABatchStride, scaleAMStride);
        auto layoutScaleB = MakeLayoutScaleB{}(batch_, scaleKLen, n_);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
        auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(n_));

        const int64_t singleBatchBlockCnt = bs.GetTotalCnt();
        const int64_t coreNums = AscendC::GetBlockNum();
        const uint64_t tailRoundStart = coreNums > 0 ?
                                            (static_cast<uint64_t>(singleBatchBlockCnt) * batch_ / coreNums) *
                                                coreNums :
                                            0UL;

        for (uint64_t batchIdx = 0; batchIdx < batch_; ++batchIdx) {
            const bool isTailRound = (batchIdx + 1) * singleBatchBlockCnt > tailRoundStart;
            if (needUpdateTail_ ||
                (isTailRound && ((bs.GetEndBlockIdx() + 1) + (batch_ - batchIdx - 1) * bs.GetTotalCnt()) *
                                        params.schParams.mTailTile * params.schParams.nTailTile <=
                                    coreNums)) {
                needUpdateTail_ = true;
                bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
            }
            if constexpr (IS_ATOMIC_ADD) {
                gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
            }
            auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

            BlockCoord blockCoord;
            int64_t mPos = 0L;
            int64_t nPos = 0L;
            constexpr int64_t kPos = 0L;
            while (bs.GetTileIdx(blockCoord)) {
                BlockShape singleShape = bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE,
                                                                   QuantMode::MX_PERGROUP_MODE, WEIGHT_NZ>(blockCoord);
                const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
                const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
                if (baseM <= 0 || baseN <= 0) {
                    break;
                }
                SetBL2Cache(problemShape, baseM, baseN, tqbmmParams.bMustHitL2, gmB);
                bs.GetTileCoord(blockCoord, mPos, nPos);

                auto subTensorA = gmA.slice(
                    asc::te::make_coord(batchIdx, asc::te::make_coord(mPos, kPos)),
                    asc::te::make_shape(1L, asc::te::make_shape(baseM, static_cast<int64_t>(k_))));
                auto gmBlockA = asc::te::squeeze<0>(subTensorA);

                auto subScaleA = gmScaleA.slice(
                    asc::te::make_coord(batchIdx, asc::te::make_coord(mPos, kPos)),
                    asc::te::make_shape(1L, asc::te::make_shape(baseM, static_cast<int64_t>(scaleKLen))));
                auto gmBlockScaleA = asc::te::squeeze<0>(subScaleA);

                auto subTensorB = gmB.slice(
                    asc::te::make_coord(batchIdx, asc::te::make_coord(kPos, nPos)),
                    asc::te::make_shape(1L, asc::te::make_shape(static_cast<int64_t>(k_), baseN)));
                auto gmBlockB = asc::te::squeeze<0>(subTensorB);

                auto subScaleB = gmScaleB.slice(
                    asc::te::make_coord(batchIdx, asc::te::make_coord(kPos, nPos)),
                    asc::te::make_shape(1L, asc::te::make_shape(static_cast<int64_t>(scaleKLen), baseN)));
                auto gmBlockScaleB = asc::te::squeeze<0>(subScaleB);

                auto subTensorC = gmC.slice(asc::te::make_coord(batchIdx, asc::te::make_coord(mPos, nPos)),
                                            asc::te::make_shape(1L, asc::te::make_shape(baseM, baseN)));
                auto gmBlockC = asc::te::squeeze<0>(subTensorC);

                auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));

                mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
            }
            bs.UpdateNextBatchBlockRoundParams();
        }
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicNone();
        }
    }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;

    using ScaleALayoutPtn = asc::te::scalea_nd_layout_ptn;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;

    template <typename Ptn, typename T, typename IntType>
    __aicore__ inline auto MakeBatchLayout(uint64_t batch, uint64_t row, uint64_t col, uint64_t batchStride,
                                           uint64_t rowStride)
    {
        return asc::te::make_pattern_layout<Ptn, asc::te::layout_trait<T, IntType>>(
            asc::te::make_shape(batch, asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, row),
                                                           asc::te::make_shape(asc::te::_1{}, col))),
            asc::te::make_stride(batchStride,
                                 asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, rowStride),
                                                      asc::te::make_stride(asc::te::_0{}, asc::te::_1{}))));
    }

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB)
    {
        constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
        const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
        const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                     isCurrentNAligned;
        gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    }

    __aicore__ inline void Init(const Params& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        isBias_ = false;
        isPermX1_ = (BlockMmad::NON_CONTIGUOUS_TYPE ==
                     static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    }

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;

    BlockMmad mmadOp_;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
    bool isBias_{false};
    bool isPermX1_{false};
    bool needUpdateTail_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

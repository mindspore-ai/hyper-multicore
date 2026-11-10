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
 * \file kernel_grouped_matmul.h
 * \brief Non-quant grouped matmul kernel implemented with Tensor API.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "tensor_api/tensor.h"

#include "mxfp8/mx_blaze/epilogue/block/block_epilogue_empty.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad_matmul_basic.h"
#include "mxfp8/mx_blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "mxfp8/mx_blaze/gemm/kernel/kernel_universal.h"
#include "mxfp8/mx_blaze/gemm/tile/compute.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "mxfp8/mx_blaze/gemm/utils/layout_utils.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelGroupedMmadNoQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
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
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using GroupCoord = typename BlockScheduler::GroupCoord;

    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr bool WEIGHT_NZ = BlockMmad::WEIGHT_NZ_FORMAT;
    static constexpr bool ENABLE_INPLACE_ADD = DispatchPolicy::OUTPUT_MODE == MatmulOutputMode::INPLACE_ADD;

    struct GMMTiling {
        uint32_t groupNum{0};
        int32_t groupType{0};
        uint32_t groupListType{0};
        uint64_t singleX{0};
        uint64_t singleWeight{0};
        uint64_t singleY{0};
        uint32_t hasBias{0};
        uint32_t weightNoL2Cache{0};
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schedulerParams;
        GMMTiling gmmParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        auto blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockNum == 0 || params.gmmParams.groupNum == 0 || !IsValidGroupParams(params)) {
            return;
        }

        Init(params);
        auto groupListLayout = asc::te::make_layout(
            asc::te::make_shape(static_cast<int64_t>(params.gmmParams.groupNum)), asc::te::make_stride(1L));
        auto gmGroupList = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                    reinterpret_cast<__gm__ int64_t*>(params.mmParams.groupListGmAddr)),
                                                groupListLayout);
        BlockScheduler bs(params.schedulerParams);

        if ASCEND_IS_AIV {
            ProcessEmptyGroups(params, bs, gmGroupList);
            return;
        }

        EnableAtomicAdd();
        for (uint32_t groupIdx = 0; groupIdx < params.gmmParams.groupNum; ++groupIdx) {
            auto groupCoord = PrepareGroup(params, bs, gmGroupList, groupIdx);

            auto problemM = asc::te::get<MNK_M>(problemShape_);
            auto problemN = asc::te::get<MNK_N>(problemShape_);
            auto problemK = asc::te::get<MNK_K>(problemShape_);
            if (problemM > 0 && problemN > 0 && problemK > 0) {
                ProcessSingleGroup(params, bs, groupCoord, groupIdx);
            }
        }
        DisableAtomicAdd();
    }

private:
    static constexpr int32_t GROUP_TYPE_SPLIT_M = 0;
    static constexpr int32_t GROUP_TYPE_SPLIT_K = 2;
    static constexpr uint32_t GROUP_LIST_TYPE_OFFSET = 0;
    static constexpr uint32_t GROUP_LIST_TYPE_COUNT = 1;
    static constexpr uint64_t DIM_NUM = 2;
    static constexpr uint64_t NZ_DESC_DIM_NUM = 4;
    static constexpr uint64_t SHAPE_BUF_SIZE = 8;
    static constexpr uint64_t OUTER_BLOCK_SIZE = 16;
    static constexpr uint64_t A_C0_SIZE = asc::te::c0_element<AType>;
    static constexpr uint64_t B_C0_SIZE = asc::te::c0_element<BType>;
    static constexpr uint64_t OUTPUT_BLOCK_ELEMENTS = BLOCK_BYTE_SIZE / sizeof(CType);
    static constexpr uint8_t FILL_ZERO_EVENT_ID = 0;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<A_C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<B_C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
    using MakeLinearLayout = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn,
                                                          asc::te::layout_trait_default<CType>>;

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = params.problemShape;
        if ASCEND_IS_AIC {
            blockMmad_.Init(params.mmParams);
        }
    }

    template <typename T>
    __aicore__ inline __gm__ T* GetTensorAddr(uint64_t tensorIdx, GM_ADDR tensorListAddr) const
    {
        AscendC::ListTensorDesc tensorList(reinterpret_cast<__gm__ void*>(tensorListAddr));
        return tensorList.GetDataPtr<T>(tensorIdx);
    }

    template <typename T>
    __aicore__ inline __gm__ T* ResolveTensorAddr(uint64_t tensorIdx, GM_ADDR tensorAddr) const
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            return reinterpret_cast<__gm__ T*>(tensorAddr);
        } else {
            return GetTensorAddr<T>(tensorIdx, tensorAddr);
        }
    }

    __aicore__ inline void GetTensorShape(uint64_t tensorIdx, GM_ADDR tensorListAddr, uint64_t* shape, bool isWeight)
    {
        AscendC::ListTensorDesc tensorList(reinterpret_cast<__gm__ void*>(tensorListAddr));
        AscendC::TensorDesc<int32_t> desc;
        desc.SetShapeAddr(shapeBuf_);
        tensorList.GetDesc(desc, tensorIdx);
        auto dim = desc.GetDim();
        if (dim < DIM_NUM || dim > SHAPE_BUF_SIZE) {
            return;
        }

        if (isWeight && WEIGHT_NZ && dim >= NZ_DESC_DIM_NUM) {
            auto logicalK = TRANS_B ? desc.GetShape(dim - 4) * B_C0_SIZE : desc.GetShape(dim - 3) * OUTER_BLOCK_SIZE;
            auto logicalN = TRANS_B ? desc.GetShape(dim - 3) * OUTER_BLOCK_SIZE : desc.GetShape(dim - 4) * B_C0_SIZE;
            shape[0] = TRANS_B ? logicalN : logicalK;
            shape[1] = TRANS_B ? logicalK : logicalN;
            return;
        }
        for (uint64_t index = 0, count = 0; index < dim; ++index) {
            if (dim - index <= DIM_NUM) {
                shape[count++] = static_cast<uint64_t>(desc.GetShape(index));
            }
        }
    }

    __aicore__ inline bool IsSingleTensor(const GMMTiling& gmmParams) const
    {
        return gmmParams.singleX == 1 && gmmParams.singleWeight == 1 && gmmParams.singleY == 1;
    }

    __aicore__ inline bool IsValidGroupParams(const Params& params) const
    {
        auto validGroupListType = params.gmmParams.groupListType == GROUP_LIST_TYPE_OFFSET ||
                                  params.gmmParams.groupListType == GROUP_LIST_TYPE_COUNT;
        if constexpr (ENABLE_INPLACE_ADD) {
            return params.gmmParams.groupType == GROUP_TYPE_SPLIT_K && validGroupListType &&
                   IsSingleTensor(params.gmmParams) && params.gmmParams.hasBias == 0 &&
                   params.mmParams.groupListGmAddr != nullptr;
        }
        auto validGroupType = params.gmmParams.groupType == -1 || params.gmmParams.groupType == GROUP_TYPE_SPLIT_M ||
                              params.gmmParams.groupType == GROUP_TYPE_SPLIT_K;
        return validGroupType && validGroupListType &&
               (params.gmmParams.groupType == -1 || params.mmParams.groupListGmAddr != nullptr);
    }

    __aicore__ inline void SetProblemShape(const Params& params, uint32_t groupIdx, int64_t splitValue)
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            auto problemM = asc::te::get<MNK_M>(params.problemShape);
            auto problemN = asc::te::get<MNK_N>(params.problemShape);
            problemShape_ = ProblemShape{problemM, problemN, splitValue, 1};
            return;
        }

        // Preserve the shape resolved by previous groups. Mixed or multi tensors resolve it per group.
        const bool isSingleTensor = IsSingleTensor(params.gmmParams);
        auto problemM = asc::te::get<MNK_M>(problemShape_);
        auto problemN = asc::te::get<MNK_N>(problemShape_);
        auto problemK = asc::te::get<MNK_K>(problemShape_);
        if (!isSingleTensor) {
            uint64_t xShape[DIM_NUM] = {0, 0};
            uint64_t weightShape[DIM_NUM] = {0, 0};
            GetTensorShape(params.gmmParams.singleX == 0 ? groupIdx : 0, params.mmParams.aGmAddr, xShape, false);
            GetTensorShape(params.gmmParams.singleWeight == 0 ? groupIdx : 0, params.mmParams.bGmAddr, weightShape,
                           true);
            problemM = static_cast<int64_t>(TRANS_A ? xShape[1] : xShape[0]);
            problemK = static_cast<int64_t>(TRANS_B ? weightShape[1] : weightShape[0]);
            problemN = static_cast<int64_t>(TRANS_B ? weightShape[0] : weightShape[1]);
        } else if (params.gmmParams.groupType == GROUP_TYPE_SPLIT_K && groupIdx == 0U) {
            // For split-K single tensors, derive M from the X descriptor instead of trusting the tiling shape.
            uint64_t xShape[DIM_NUM] = {0, 0};
            GetTensorShape(0U, params.mmParams.aGmAddr, xShape, false);
            problemM = static_cast<int64_t>(TRANS_A ? xShape[1] : xShape[0]);
        }
        if (params.gmmParams.groupType == GROUP_TYPE_SPLIT_M) {
            problemM = splitValue;
        } else if (params.gmmParams.groupType == GROUP_TYPE_SPLIT_K) {
            problemK = splitValue;
        }
        problemShape_ = ProblemShape{problemM, problemN, problemK, 1};
    }

    template <typename GroupListTensor>
    __aicore__ inline void PrepareGroupShape(const Params& params, BlockScheduler& bs,
                                             const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        auto groupValue = params.gmmParams.groupType == -1 ? 0 : gmGroupList[groupIdx];
        auto splitValue = bs.GetSplitValue(groupValue, params.gmmParams.groupListType);
        SetProblemShape(params, groupIdx, splitValue);
    }

    template <typename GroupListTensor>
    __aicore__ inline GroupCoord PrepareGroup(const Params& params, BlockScheduler& bs,
                                              const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        PrepareGroupShape(params, bs, gmGroupList, groupIdx);
        return bs.UpdateNextGroup(problemShape_);
    }

    // The kernel owns zero-template lifetime, pipeline synchronization and UB-to-GM orchestration.
    // The Tile layer fills the contiguous UB tensor with the requested value.
    template <typename ZeroUbTensor>
    __aicore__ inline void PrepareZeroUb(const ZeroUbTensor& zeroUb, uint32_t zeroTileElements, bool& zeroUbReady) const
    {
        if (zeroUbReady || zeroTileElements == 0) {
            return;
        }

        Blaze::Gemm::Tile::FillUb<CType>::FillWithValue(zeroUb, static_cast<CType>(0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FILL_ZERO_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FILL_ZERO_EVENT_ID);
        zeroUbReady = true;
    }

    template <typename GmTensor, typename ZeroUbTensor>
    __aicore__ inline bool CopyZeroUbToGm(const GmTensor& gmDst, const ZeroUbTensor& zeroUb, uint32_t zeroTileElements,
                                          bool& zeroUbReady) const
    {
        using GmElementType = asc::te::get_attribute_element_type<typename GmTensor::element_type*>;
        using GmLayoutPattern = asc::te::get_layout_pattern<typename GmTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<GmElementType, CType>,
                      "CopyZeroUbToGm requires a GM tensor with the output scalar type");
        static_assert(AscendC::Std::is_same_v<GmLayoutPattern, asc::te::nd_ext_layout_ptn>,
                      "CopyZeroUbToGm requires a contiguous NDExt GM tensor");

        auto outputElements = static_cast<uint64_t>(asc::te::get_total_column_shape(gmDst.layout()));
        if (outputElements == 0 || zeroTileElements == 0) {
            return false;
        }

        PrepareZeroUb(zeroUb, zeroTileElements, zeroUbReady);
        auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        auto copiedElements = static_cast<uint64_t>(0);
        while (copiedElements < outputElements) {
            auto currentElements = static_cast<int64_t>(
                Min(static_cast<uint64_t>(zeroTileElements), outputElements - copiedElements));
            auto gmOutput = gmDst.slice(
                asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(copiedElements)),
                asc::te::make_shape(static_cast<int64_t>(1), currentElements));
            auto ubOutput = zeroUb.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                         asc::te::make_shape(static_cast<int64_t>(1), currentElements));
            asc::te::copy(copyUB2GM, gmOutput, ubOutput);
            copiedElements += static_cast<uint64_t>(currentElements);
        }
        return true;
    }

    __aicore__ inline void WaitZeroGmCopy(bool hasOutputCopy) const
    {
        if (!hasOutputCopy) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(FILL_ZERO_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(FILL_ZERO_EVENT_ID);
    }

    template <typename GroupListTensor>
    __aicore__ inline void ProcessEmptyGroups(const Params& params, BlockScheduler& bs,
                                              const GroupListTensor& gmGroupList)
    {
        if (params.gmmParams.groupType != GROUP_TYPE_SPLIT_K || params.mmParams.groupListGmAddr == nullptr) {
            return;
        }

        auto taskRatio = AscendC::GetTaskRation();
        auto logicalCoreNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (taskRatio == 0 || logicalCoreNum == 0) {
            return;
        }

        // In a MIX_AIC_1_2 launch, both AIV sub-blocks participate. Give each AIV a disjoint
        // contiguous element range so no two vector cores write the same GM address.
        auto logicalCoreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx()) / taskRatio;
        auto workerIdx = logicalCoreIdx * taskRatio + AscendC::GetSubBlockIdx();
        auto workerNum = logicalCoreNum * taskRatio;
        if (workerIdx >= workerNum) {
            return;
        }

        constexpr int64_t emptyOutputUbElements = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(CType));
        auto zeroUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0),
                                           MakeLinearLayout{}(static_cast<int64_t>(1), emptyOutputUbElements));
        auto zeroTileElements = static_cast<uint32_t>(asc::te::get_total_column_shape(zeroUb.layout()));
        auto zeroUbReady = false;
        auto hasOutputCopy = false;

        for (uint32_t groupIdx = 0; groupIdx < params.gmmParams.groupNum; ++groupIdx) {
            // Empty output only needs group shape parsing and C offset progression.
            PrepareGroupShape(params, bs, gmGroupList, groupIdx);

            auto m = asc::te::get<MNK_M>(problemShape_);
            auto n = asc::te::get<MNK_N>(problemShape_);
            auto k = asc::te::get<MNK_K>(problemShape_);
            auto currentCOffset = bs.UpdateNextOutputOffset(problemShape_);
            if (m <= 0 || n <= 0 || k != 0) {
                continue;
            }

            // [M, 0] * [0, N] is an M-by-N zero tensor. The output is contiguous, so split its
            // flattened element range rather than imposing an unnecessary row-alignment constraint.
            auto totalElements = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
            auto elementsPerWorker = CeilAlign(CeilDiv(totalElements, workerNum), OUTPUT_BLOCK_ELEMENTS);
            auto workerOffset = workerIdx * elementsPerWorker;
            if (workerOffset >= totalElements) {
                continue;
            }
            auto workerElements = Min(elementsPerWorker, totalElements - workerOffset);
            auto cPtr = GetTensorAddr<CType>(params.gmmParams.singleY == 0 ? groupIdx : 0, params.mmParams.cGmAddr) +
                        currentCOffset;
            auto gmGroupOutput = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(cPtr),
                MakeLinearLayout{}(static_cast<int64_t>(1), static_cast<int64_t>(totalElements)));
            auto gmWorkerOutput = gmGroupOutput.slice(
                asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(workerOffset)),
                asc::te::make_shape(static_cast<int64_t>(1), static_cast<int64_t>(workerElements)));
            if (CopyZeroUbToGm(gmWorkerOutput, zeroUb, zeroTileElements, zeroUbReady)) {
                hasOutputCopy = true;
            }
        }
        WaitZeroGmCopy(hasOutputCopy);
    }

    template <typename TensorB>
    __aicore__ inline void SetWeightL2CacheHint(TensorB& gmB, const Params& params, int64_t problemM) const
    {
        if (params.gmmParams.weightNoL2Cache == 1 && static_cast<int64_t>(params.mmParams.mL1) > problemM) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        } else {
            gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
        }
    }

    __aicore__ inline void EnableAtomicAdd() const
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            static_assert(AscendC::Std::is_same_v<CType, float>, "GMM inplace add only supports FP32 output");
            AscendC::SetAtomicAdd<float>();
        }
    }

    __aicore__ inline void DisableAtomicAdd() const
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            AscendC::SetAtomicNone();
        }
    }

    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockScheduler& bs, const GroupCoord& groupCoord,
                                              uint32_t groupIdx)
    {
        auto m = asc::te::get<MNK_M>(problemShape_);
        auto n = asc::te::get<MNK_N>(problemShape_);
        auto k = asc::te::get<MNK_K>(problemShape_);
        auto aOffset = asc::te::get<MNK_M>(groupCoord);
        auto bOffset = asc::te::get<MNK_N>(groupCoord);
        auto biasOffset = asc::te::get<MNK_K>(groupCoord);
        auto cOffset = asc::te::get<MNK_B>(groupCoord);
        auto aPtr = ResolveTensorAddr<AType>(params.gmmParams.singleX == 0 ? groupIdx : 0, params.mmParams.aGmAddr) +
                    aOffset;
        auto bPtr = ResolveTensorAddr<BType>(params.gmmParams.singleWeight == 0 ? groupIdx : 0,
                                             params.mmParams.bGmAddr) +
                    bOffset;
        auto cPtr = ResolveTensorAddr<CType>(params.gmmParams.singleY == 0 ? groupIdx : 0, params.mmParams.cGmAddr) +
                    cOffset;

        __gm__ BiasType* biasPtr = nullptr;
        if (params.gmmParams.hasBias != 0) {
            biasPtr = ResolveTensorAddr<BiasType>(params.gmmParams.singleWeight == 0 ? groupIdx : 0,
                                                  params.mmParams.biasGmAddr) +
                      biasOffset;
        }

        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutB = MakeLayoutB{}(k, n);
        auto layoutC = MakeLayoutC{}(m, n);
        auto layoutBias = MakeLayoutBias{}(static_cast<int64_t>(1), n);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aPtr), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bPtr), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cPtr), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);
        SetWeightL2CacheHint(gmB, params, m);
        if constexpr (ENABLE_INPLACE_ADD) {
            gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
        }

        auto blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (blockIdx >= bs.GetCoreNums()) {
            return;
        }
        for (int64_t taskIdx = bs.GetFirstBlockIdx(blockIdx); taskIdx < bs.GetBlockNums();
             taskIdx += AscendC::GetBlockNum()) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(taskIdx);
            auto blockCoord = bs.GetBlockCoord(taskIdx);
            auto blockM = asc::te::get<MNK_M>(blockShape);
            auto blockN = asc::te::get<MNK_N>(blockShape);
            auto blockMOffset = asc::te::get<MNK_M>(blockCoord);
            auto blockNOffset = asc::te::get<MNK_N>(blockCoord);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            auto gmBlockA = gmA.slice(asc::te::make_coord(blockMOffset, static_cast<int64_t>(0)),
                                      asc::te::make_shape(blockM, k));
            auto gmBlockB = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), blockNOffset),
                                      asc::te::make_shape(k, blockN));
            auto gmBlockC = gmC.slice(asc::te::make_coord(blockMOffset, blockNOffset),
                                      asc::te::make_shape(blockM, blockN));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(static_cast<int64_t>(0), blockNOffset),
                                            asc::te::make_shape(static_cast<int64_t>(1), blockN));
            blockMmad_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
        }
    }

    BlockMmad blockMmad_;
    ProblemShape problemShape_{};
    uint64_t shapeBuf_[SHAPE_BUF_SIZE] = {0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

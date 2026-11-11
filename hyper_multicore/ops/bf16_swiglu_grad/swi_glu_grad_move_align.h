/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu_grad_move_align.h
 * \brief
 */

#ifndef SWI_GLU_GRAD_MOVE_ALIGN_H
#define SWI_GLU_GRAD_MOVE_ALIGN_H

#include "swi_glu_grad_base.h"

namespace SwiGluGrad {
using namespace AscendC;

template <typename T>
class SwiGluGradMoveAlignKernel : public SwiGluGradBaseKernel<T> {
public:
    __aicore__ inline SwiGluGradMoveAlignKernel(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR out, const GluBaseTilingData& tilingData, int32_t logicalIndex = -1);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t offset, int64_t gradOffset, uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void ComputePerLoop(int32_t dataCount);
    __aicore__ inline void ProcessNotSplitCol();
    __aicore__ inline void ProcessSplitCol();

protected:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueA_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueB_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueGrad_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueA_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueB_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<T> outGm_;

    // mode
    bool needSplitCol_{true};

    // ub内循环处理大小
    uint32_t rowUbNorm_{0};
    uint32_t rowUbTail_{0};
    int64_t rowLoopByUb_{0};
    uint32_t rowPerLoopNorm_{0};
    uint32_t colUbNorm_{0};
    uint32_t colUbTail_{0};
    int64_t colLoopByUb_{0};

    static constexpr int32_t BLOCK_NUM = ONE_BLK_SIZE / sizeof(T);
};

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::Init(GM_ADDR grad, GM_ADDR x, GM_ADDR out,
                                                          const GluBaseTilingData& tilingData, int32_t logicalIndex)
{
    if (!this->InitBase(tilingData, logicalIndex)) {
        return;
    }

    uint64_t existNodeSize = this->ubSize_ / BUFFER_NUMBER / DOUBLE_BUFFER;
    existNodeSize = RoundDownToVL(existNodeSize);

    needSplitCol_ = this->colProcess_ * sizeof(T) > existNodeSize ? true : false;
    if (!needSplitCol_) {
        rowPerLoopNorm_ = existNodeSize / (((this->colProcess_ + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM) * sizeof(T));
        rowLoopByUb_ = (this->rowProcess_ + rowPerLoopNorm_ - 1) / rowPerLoopNorm_;
        rowUbTail_ = this->rowProcess_ - rowPerLoopNorm_ * (rowLoopByUb_ - 1);
    } else {
        // N轴较大，需要对N轴切分
        colUbNorm_ = existNodeSize / sizeof(T);
        colLoopByUb_ = (this->colProcess_ + colUbNorm_ - 1) / colUbNorm_;
        colUbTail_ = this->colProcess_ - colUbNorm_ * (colLoopByUb_ - 1);

        rowPerLoopNorm_ = 1;
        rowLoopByUb_ = this->rowProcess_;
    }

    // init params
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    gradYGm_.SetGlobalBuffer((__gm__ T*)grad);
    outGm_.SetGlobalBuffer((__gm__ T*)out);

    pipe_.InitBuffer(inQueA_, DOUBLE_BUFFER, existNodeSize);
    pipe_.InitBuffer(inQueB_, DOUBLE_BUFFER, existNodeSize);
    pipe_.InitBuffer(inQueGrad_, DOUBLE_BUFFER, existNodeSize);
    pipe_.InitBuffer(outQueA_, DOUBLE_BUFFER, existNodeSize);
    pipe_.InitBuffer(outQueB_, DOUBLE_BUFFER, existNodeSize);
}

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::CopyIn(int64_t offset, int64_t gradOffset, uint32_t rowLen,
                                                            uint32_t colLen)
{
    LocalTensor<T> xATensor = inQueA_.AllocTensor<T>();
    LocalTensor<T> xBTensor = inQueB_.AllocTensor<T>();
    LocalTensor<T> gradTensor = inQueGrad_.AllocTensor<T>();

    uint64_t partA = this->partAStart_ + offset;
    uint64_t partB = this->partBStart_ + offset;
    uint64_t grad = this->gradStart_ + gradOffset;

    uint8_t rightPadNum = static_cast<uint8_t>((colLen + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM - colLen);

    DataCopyExtParams copyParamForX{static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                    static_cast<uint32_t>((this->colTotal_ - colLen) * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadNum),
                                      static_cast<uint8_t>(0)};

    DataCopyPad(xATensor, xGm_[partA], copyParamForX, padParams);
    DataCopyPad(xBTensor, xGm_[partB], copyParamForX, padParams);

    DataCopyExtParams copyParamsForGrad{static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                        static_cast<uint32_t>((this->colTotalA_ - colLen) * sizeof(T)),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(gradTensor, gradYGm_[grad], copyParamsForGrad, padParams);

    inQueA_.EnQue<T>(xATensor);
    inQueB_.EnQue<T>(xBTensor);
    inQueGrad_.EnQue<T>(gradTensor);
}

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::CopyOut(int64_t offset, uint32_t rowLen, uint32_t colLen)
{
    LocalTensor<T> xATensor = outQueA_.DeQue<T>();
    LocalTensor<T> xBTensor = outQueB_.DeQue<T>();

    uint64_t partA = this->partAStart_ + offset;
    uint64_t partB = this->partBStart_ + offset;

    DataCopyExtParams copyParams{
        static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((this->colTotal_ - colLen) * sizeof(T)), static_cast<uint32_t>(0)};

    DataCopyPad(outGm_[partA], xATensor, copyParams);
    DataCopyPad(outGm_[partB], xBTensor, copyParams);

    outQueA_.FreeTensor(xATensor);
    outQueB_.FreeTensor(xBTensor);
}

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::Process()
{
    if (!needSplitCol_) {
        ProcessNotSplitCol();
    } else {
        ProcessSplitCol();
    }
}

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::ProcessNotSplitCol()
{
    int64_t offset = 0;
    int64_t gradOffset = 0;
    int32_t colProcessAlign = ((this->colProcess_ + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
    int64_t stride = rowPerLoopNorm_ * this->colTotal_;
    int64_t gradStride = rowPerLoopNorm_ * this->colTotalA_;
    for (int64_t i = 0; i < rowLoopByUb_ - 1; i++) {
        CopyIn(offset, gradOffset, rowPerLoopNorm_, this->colProcess_);
        ComputePerLoop(rowPerLoopNorm_ * colProcessAlign);
        CopyOut(offset, rowPerLoopNorm_, this->colProcess_);
        offset = offset + stride;
        gradOffset = gradOffset + gradStride;
    }

    CopyIn(offset, gradOffset, rowUbTail_, this->colProcess_);
    ComputePerLoop(rowUbTail_ * colProcessAlign);
    CopyOut(offset, rowUbTail_, this->colProcess_);
}

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::ProcessSplitCol()
{
    int64_t offset = 0;
    int64_t gradOffset = 0;
    int32_t colUbNormAlign = (colUbNorm_ + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM;
    int32_t colUbTailAlign = (colUbTail_ + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM;
    int64_t colLoopStride = rowPerLoopNorm_ * colUbNorm_;
    int64_t rowLoopStride = rowPerLoopNorm_ * colUbTail_ + this->colTotal_ - this->colProcess_;
    int64_t gradRowLoopStride = rowPerLoopNorm_ * colUbTail_ + this->colTotalA_ - this->colProcess_;

    for (int64_t i = 0; i < rowLoopByUb_; i++) {
        for (int64_t j = 0; j < colLoopByUb_ - 1; j++) {
            CopyIn(offset, gradOffset, rowPerLoopNorm_, colUbNorm_);
            ComputePerLoop(rowPerLoopNorm_ * colUbNormAlign);
            CopyOut(offset, rowPerLoopNorm_, colUbNorm_);
            offset = offset + colLoopStride;
            gradOffset = gradOffset + colLoopStride;
        }

        CopyIn(offset, gradOffset, rowPerLoopNorm_, colUbTail_);
        ComputePerLoop(rowPerLoopNorm_ * colUbTailAlign);
        CopyOut(offset, rowPerLoopNorm_, colUbTail_);
        offset = offset + rowLoopStride;
        gradOffset = gradOffset + gradRowLoopStride;
    }
}

template <typename T>
__aicore__ inline void SwiGluGradMoveAlignKernel<T>::ComputePerLoop(int32_t dataCount)
{
    LocalTensor<T> xATensor = inQueA_.DeQue<T>();
    LocalTensor<T> xBTensor = inQueB_.DeQue<T>();
    LocalTensor<T> gradTensor = inQueGrad_.DeQue<T>();
    LocalTensor<T> outATensor = outQueA_.AllocTensor<T>();
    LocalTensor<T> outBTensor = outQueB_.AllocTensor<T>();

    this->Compute(xATensor, xBTensor, gradTensor, outATensor, outBTensor, dataCount);

    outQueA_.EnQue<T>(outATensor);
    outQueB_.EnQue<T>(outBTensor);
    inQueA_.FreeTensor(xATensor);
    inQueB_.FreeTensor(xBTensor);
    inQueGrad_.FreeTensor(gradTensor);
}

} // namespace SwiGluGrad
#endif
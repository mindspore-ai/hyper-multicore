/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 * Next-gen HyperMegaKda compute primitives:
 * - Coarse-grained BF16 Linear projections (reused MatmulImpl)
 * - Vectorized BT=64 ShortConv1d + SiLU + L2Norm (with 3-row Halo for AllGather/PrefixSum CP)
 * - FP32 PreGate + 6-step Hillis-Steele cumsum -> BF16 Cube operands
 * - Raw BF16 Cube IntraCube + 64x64 L1-resident Block-Schur TriInv + WY
 * - 1:2 AIC<->2*AIV CrossCoreSetFlag<0x2> FwdH & InterSummary (UB-resident FP32 State)
 * - Fused FP32-L0C FinalizeO (O = qg @ h + Aqk @ v_new -> BF16)
 * - Unified SHMEM PutMemSignal (Ulysses A2A #1/#2, Halo P2P, State AllGather & Tree-Scan)
 */
#ifndef HYPER_MEGA_KDA_COMPUTE_H
#define HYPER_MEGA_KDA_COMPUTE_H

#include "kda_cube_raw.h"
#include "lib/matmul_intf.h"
#ifndef DTYPE_DISPATCH_TARGET
#define DTYPE_DISPATCH_TARGET uint16_t
#endif
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "runtime/runtime_config.hpp"

#ifndef ASCEND_IS_AIV
#define ASCEND_IS_AIV constexpr(g_coreType == AscendC::AIV)
#endif
#ifndef ASCEND_IS_AIC
#define ASCEND_IS_AIC constexpr(g_coreType == AscendC::AIC)
#endif

namespace MulticoreKda {

using namespace AscendC;
using MulticoreRuntime::TaskDesc;

constexpr uint32_t IDX_HIDDEN = 0;
constexpr uint32_t IDX_WQ = 1;
constexpr uint32_t IDX_WK = 2;
constexpr uint32_t IDX_WV = 3;
constexpr uint32_t IDX_WFA = 4;
constexpr uint32_t IDX_WFB = 5;
constexpr uint32_t IDX_WB = 6;
constexpr uint32_t IDX_WGA = 7;
constexpr uint32_t IDX_WGB = 8;
constexpr uint32_t IDX_WCONV_Q = 9;
constexpr uint32_t IDX_WCONV_K = 10;
constexpr uint32_t IDX_WCONV_V = 11;
constexpr uint32_t IDX_ALOG = 12;
constexpr uint32_t IDX_DTBIAS = 13;
constexpr uint32_t IDX_ONORMW = 14;
constexpr uint32_t IDX_WO = 15;
constexpr uint32_t IDX_SEND_BUF = 16;
constexpr uint32_t IDX_RECV_BUF = 17;
constexpr uint32_t IDX_RUNTIME_CFG = 18;
constexpr uint32_t IDX_EVENT = 19;
constexpr uint32_t IDX_DIAG = 20;
constexpr uint32_t IDX_BWD_CACHE = 21;
constexpr uint32_t IDX_OUT = 22;
constexpr uint32_t IDX_WORKSPACE = 23;
constexpr uint32_t IDX_TILING = 24;

// Fine-grained parallel BF16 Linear projection kinds (extra0 of TASK_KDA_PROJ_BATCH)
constexpr uint32_t PROJ_KIND_FA_GA_B = 0;  // f_a, g_a (if low-rank), b_proj (all H heads -> wsBetaAll)
constexpr uint32_t PROJ_KIND_Q = 1;        // q_proj for (g, dst, mt)
constexpr uint32_t PROJ_KIND_K = 2;        // k_proj for (g, dst, mt)
constexpr uint32_t PROJ_KIND_V = 3;        // v_proj for (g, dst, mt)
constexpr uint32_t PROJ_KIND_FB = 4;       // f_b_proj for (g, dst, mt)
constexpr uint32_t PROJ_KIND_GOUT = 5;     // g_out (g_b or full-rank g_proj) for (g, srcRank, mt)
constexpr uint32_t PROJ_KIND_O = 6;        // o_proj for (ot)

// CP modes
constexpr uint32_t CP_MODE_ULYSSES = 0;
constexpr uint32_t CP_MODE_ALLGATHER = 1;
constexpr uint32_t CP_MODE_PREFIX_SUM = 2;

__aicore__ inline float BitsToFloat(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = bits;
  return v.f;
}

__aicore__ inline GM_ADDR GetProjBase(GM_ADDR *inputList, const HyperMegaKdaLaunchTilingData *t) {
  return (t->cpMode == CP_MODE_ULYSSES) ? inputList[IDX_RECV_BUF]
                                        : (inputList[IDX_WORKSPACE] + t->wsLocalProjOff);
}

// ---------------------------------------------------------------------------
// 1. AIC: Fine-Grained 20-AIC-Parallel L1-Resident-Weight BF16 Linear Projections
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteProjBatch(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                        const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t kind = taskDesc.extra_value_0;
    const uint32_t splitNum = taskDesc.task_split_num;
    const uint32_t splitVal = taskDesc.task_split_value;
    // Unpack 2D (mt, nt) & (mTile, nTile) when upper 16 bits of splitVal != 0; else 1D M-only
    const bool hasNSlice = ((splitVal >> 16) != 0U);
    const uint32_t mt = hasNSlice ? (splitNum >> 16) : splitNum;
    const uint32_t nt = hasNSlice ? (splitNum & 0xFFFFU) : 0U;
    const uint32_t mTile = hasNSlice ? (splitVal >> 16) : splitVal;
    const uint32_t nTile = hasNSlice ? (splitVal & 0xFFFFU) : 0U;

    const uint32_t mStart = mt * mTile;
    const uint32_t curM = (mStart + mTile <= t->seqLocal) ? mTile : (t->seqLocal - mStart);
    const uint32_t D = t->hiddenSize;
    const uint32_t d = t->headDim;
    const uint32_t hG = t->headsPerGroup;
    const uint32_t groupDim = hG * d;
    const uint32_t totalHeadDim = t->numHeads * d;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    __gm__ bfloat16_t *xPtr =
        reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_HIDDEN]) + static_cast<uint64_t>(mStart) * D;

    if (kind == PROJ_KIND_FA_GA_B) {
      RunRawLinearBf16(
          reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsFaOutOff) + static_cast<uint64_t>(mStart) * d,
          xPtr, reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WFA]), curM, d, D);
      if (!t->useFullRankGate) {
        RunRawLinearBf16(
            reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsGaOutOff) + static_cast<uint64_t>(mStart) * d,
            xPtr, reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WGA]), curM, d, D);
      }
      RunRawLinearBf16(
          reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsBetaAllOff) + static_cast<uint64_t>(mStart) * t->numHeads,
          xPtr, reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WB]), curM, t->numHeads, D);
    } else if (kind >= PROJ_KIND_Q && kind <= PROJ_KIND_FB) {
      const uint32_t g = taskDesc.task_index;
      const uint32_t dstRank = taskDesc.extra_value_1;
      const bool isUlyssesRemote = (t->cpMode == CP_MODE_ULYSSES) && (t->cpSize > 1U) && (dstRank != t->cpRank);
      const uint64_t headOff =
          (t->cpMode == CP_MODE_ULYSSES)
              ? (static_cast<uint64_t>(dstRank) * t->numLocalHeads + static_cast<uint64_t>(g) * hG)
              : (static_cast<uint64_t>(g) * hG);
      const uint32_t nStart = hasNSlice ? (nt * nTile) : 0U;
      const uint32_t curN = hasNSlice ? ((nStart + nTile <= groupDim) ? nTile : (groupDim - nStart)) : groupDim;
      const uint64_t dimOff = headOff * d + nStart;

      __gm__ bfloat16_t *dstPtr = nullptr;
      if (isUlyssesRemote) {
        const uint64_t slot = (static_cast<uint64_t>(dstRank) * t->headGroups + g) * t->sendSlotBytes;
        const uint64_t off = (kind == PROJ_KIND_Q) ? t->sendQOff
                           : (kind == PROJ_KIND_K) ? t->sendKOff
                           : (kind == PROJ_KIND_V) ? t->sendVOff : t->sendGOff;
        dstPtr = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_SEND_BUF] + slot + off) +
                 static_cast<uint64_t>(mStart) * groupDim + nStart;
      } else {
        GM_ADDR rb = GetProjBase(inputList, t) + static_cast<uint64_t>(g) * t->recvGroupBytes;
        const uint64_t rowBase =
            (t->cpMode == CP_MODE_ULYSSES) ? (static_cast<uint64_t>(t->cpRank) * t->seqLocal + mStart) : mStart;
        const uint64_t off = (kind == PROJ_KIND_Q) ? t->recvQOff
                           : (kind == PROJ_KIND_K) ? t->recvKOff
                           : (kind == PROJ_KIND_V) ? t->recvVOff : t->recvGOff;
        dstPtr = reinterpret_cast<__gm__ bfloat16_t *>(rb + off) + rowBase * groupDim + nStart;
      }

      if (kind == PROJ_KIND_FB) {
        __gm__ bfloat16_t *faPtr =
            reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsFaOutOff) + static_cast<uint64_t>(mStart) * d;
        __gm__ bfloat16_t *wFbPtr =
            reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WFB]) + dimOff * d;
        RunRawLinearBf16(dstPtr, faPtr, wFbPtr, curM, curN, d, groupDim);
      } else {
        const uint32_t wIdx = (kind == PROJ_KIND_Q) ? IDX_WQ : (kind == PROJ_KIND_K) ? IDX_WK : IDX_WV;
        __gm__ bfloat16_t *wPtr = reinterpret_cast<__gm__ bfloat16_t *>(inputList[wIdx]) + dimOff * D;
        RunRawLinearBf16(dstPtr, xPtr, wPtr, curM, curN, D, groupDim);
      }
    } else if (kind == PROJ_KIND_GOUT) {
      const uint32_t g = taskDesc.task_index;
      const uint32_t srcRank = taskDesc.extra_value_1;
      const uint64_t headOff =
          (t->cpMode == CP_MODE_ULYSSES)
              ? (static_cast<uint64_t>(srcRank) * t->numLocalHeads + static_cast<uint64_t>(g) * hG)
              : (static_cast<uint64_t>(g) * hG);
      const uint32_t nStart = hasNSlice ? (nt * nTile) : 0U;
      const uint32_t curN = hasNSlice ? ((nStart + nTile <= groupDim) ? nTile : (groupDim - nStart)) : groupDim;
      const uint64_t dimOff = headOff * d + nStart;
      const uint64_t slotIdx =
          (t->cpMode == CP_MODE_ULYSSES) ? (static_cast<uint64_t>(srcRank) * t->headGroups + g) : g;
      __gm__ bfloat16_t *dstPtr = reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsGOutOff) +
                                  slotIdx * t->seqLocal * groupDim + static_cast<uint64_t>(mStart) * groupDim + nStart;
      if (t->useFullRankGate) {
        __gm__ bfloat16_t *wPtr = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WGA]) + dimOff * D;
        RunRawLinearBf16(dstPtr, xPtr, wPtr, curM, curN, D, groupDim);
      } else {
        __gm__ bfloat16_t *gaPtr =
            reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsGaOutOff) + static_cast<uint64_t>(mStart) * d;
        __gm__ bfloat16_t *wPtr = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WGB]) + dimOff * d;
        RunRawLinearBf16(dstPtr, gaPtr, wPtr, curM, curN, d, groupDim);
      }
    } else if (kind == PROJ_KIND_O) {
      const uint32_t nStart = hasNSlice ? (nt * nTile) : 0U;
      const uint32_t curN = hasNSlice ? ((nStart + nTile <= D) ? nTile : (D - nStart)) : D;
      __gm__ bfloat16_t *inPtr =
          reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsONormedOff) + static_cast<uint64_t>(mStart) * totalHeadDim;
      __gm__ bfloat16_t *wOPtr =
          reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WO]) + static_cast<uint64_t>(nStart) * totalHeadDim;
      __gm__ bfloat16_t *outPtr =
          reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_OUT]) + static_cast<uint64_t>(mStart) * D + nStart;
      RunRawLinearBf16(outPtr, inPtr, wOPtr, curM, curN, totalHeadDim, D);
    }
    PipeBarrier<PIPE_ALL>();
  }
}

// ---------------------------------------------------------------------------
// 2. AIV: Vectorized BT=64 ShortConv1d (W=4) + SiLU + Fused Q/K L2Norm
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteShortConv(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                        const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t h = taskDesc.task_index;
    const uint32_t which = taskDesc.extra_value_0;  // 0=Q, 1=K, 2=V
    const uint32_t d = t->headDim;
    const uint32_t S = t->seqEff;
    const uint32_t g = h / t->headsPerGroup;
    const uint32_t hInG = h % t->headsPerGroup;
    const uint32_t groupDim = t->headsPerGroup * d;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    const uint64_t projOff = (which == 0U) ? t->recvQOff : ((which == 1U) ? t->recvKOff : t->recvVOff);
    const uint64_t dstWsOff = (which == 0U) ? t->wsQHatOff : ((which == 1U) ? t->wsKHatOff : t->wsVConvOff);
    const uint32_t wConvIdx = (which == 0U) ? IDX_WCONV_Q : ((which == 1U) ? IDX_WCONV_K : IDX_WCONV_V);

    const bool directPreProj = (t->projMTile == 0U) && ((t->cpMode != CP_MODE_ULYSSES) || (t->cpSize == 1U));
    const uint32_t srcRowDim = directPreProj ? (t->numHeads * d) : groupDim;
    const uint32_t directIdx = (which == 0U) ? IDX_WQ : ((which == 1U) ? IDX_WK : IDX_WV);
    __gm__ bfloat16_t *srcBase =
        directPreProj
            ? (reinterpret_cast<__gm__ bfloat16_t *>(inputList[directIdx]) + static_cast<uint64_t>(h) * d)
            : (reinterpret_cast<__gm__ bfloat16_t *>(GetProjBase(inputList, t) + g * t->recvGroupBytes + projOff) +
               static_cast<uint64_t>(hInG) * d);
    __gm__ bfloat16_t *dstBase =
        reinterpret_cast<__gm__ bfloat16_t *>(ws + dstWsOff) + static_cast<uint64_t>(h) * S * d;

    constexpr uint32_t BT = 64U;
    constexpr uint32_t IN_ROWS = BT + 3U;  // 67 rows

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uW = 0;                                 // 4*d*4 = 2KB
    const uint32_t uInBf16 = uW + 4 * d * 4;               // 67*d*2 = 17.15KB
    const uint32_t uInFp = uInBf16 + ((IN_ROWS * d * 2 + 31) & ~31U);  // 67*d*4 = 34.3KB
    const uint32_t uAcc = uInFp + IN_ROWS * d * 4;         // 64*d*4 = 32KB
    const uint32_t uTmp = uAcc + BT * d * 4;               // 64*d*4 = 32KB
    const uint32_t uOutBf16 = uTmp + BT * d * 4;           // 64*d*2 = 16KB
    const uint32_t uNorm = uOutBf16 + BT * d * 2;          // 64*4 = 256B
    const uint32_t uNormBrcb = uNorm + BT * 4;             // 64*8*4 = 2KB
    pipe.InitBuffer(buf, uNormBrcb + BT * 8 * 4);

    LocalTensor<float> wFp = buf.GetWithOffset<float>(4 * d, uW);
    LocalTensor<bfloat16_t> inBf16 = buf.GetWithOffset<bfloat16_t>(IN_ROWS * d, uInBf16);
    LocalTensor<float> inFp = buf.GetWithOffset<float>(IN_ROWS * d, uInFp);
    LocalTensor<float> acc = buf.GetWithOffset<float>(BT * d, uAcc);
    LocalTensor<float> tmp = buf.GetWithOffset<float>(BT * d, uTmp);
    LocalTensor<bfloat16_t> outBf16 = buf.GetWithOffset<bfloat16_t>(BT * d, uOutBf16);
    LocalTensor<float> normVec = buf.GetWithOffset<float>(BT, uNorm);
    LocalTensor<float> normBrcb = buf.GetWithOffset<float>(BT * 8, uNormBrcb);

    // Load conv weights [d, 4] and transpose to [4, d] in FP32 UB
    {
      GlobalTensor<bfloat16_t> wGm;
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[wConvIdx]) + static_cast<uint64_t>(h) * d * 4,
                          d * 4);
      DataCopy(inBf16, wGm, d * 4);
      PipeBarrier<PIPE_ALL>();
      Cast(tmp, inBf16, RoundMode::CAST_NONE, d * 4);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t k = 0; k < d; ++k) {
        wFp.SetValue(0 * d + k, tmp.GetValue(k * 4 + 0));
        wFp.SetValue(1 * d + k, tmp.GetValue(k * 4 + 1));
        wFp.SetValue(2 * d + k, tmp.GetValue(k * 4 + 2));
        wFp.SetValue(3 * d + k, tmp.GetValue(k * 4 + 3));
      }
      PipeBarrier<PIPE_ALL>();
    }

    const uint16_t blockLen = static_cast<uint16_t>((d * 2U) / 32U);
    const uint16_t haloStride = static_cast<uint16_t>(((groupDim - d) * 2U) / 32U);
    const uint16_t srcStride = static_cast<uint16_t>(((srcRowDim - d) * 2U) / 32U);
    const bool hasHalo = (t->cpMode != CP_MODE_ULYSSES) && (t->cpRank > 0U);
    __gm__ bfloat16_t *haloBase =
        hasHalo ? (reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_RECV_BUF] + t->recvHaloOff +
                                                         g * t->sendHaloGroupBytes) +
                   which * 3ULL * groupDim + static_cast<uint64_t>(hInG) * d)
                : nullptr;

    for (uint32_t bStart = 0; bStart < S; bStart += BT) {
      const uint32_t curT = (bStart + BT <= S) ? BT : (S - bStart);
      if (bStart == 0U) {
        if (hasHalo) {
          GlobalTensor<bfloat16_t> hGm;
          hGm.SetGlobalBuffer(haloBase, 3ULL * groupDim);
          DataCopyParams hp{3, blockLen, haloStride, 0};
          DataCopy(inBf16[0], hGm, hp);
          GlobalTensor<bfloat16_t> sGm;
          sGm.SetGlobalBuffer(srcBase, static_cast<uint64_t>(curT) * srcRowDim);
          DataCopyParams cp{static_cast<uint16_t>(curT), blockLen, srcStride, 0};
          DataCopy(inBf16[3 * d], sGm, cp);
          PipeBarrier<PIPE_ALL>();
          Cast(inFp, inBf16, RoundMode::CAST_NONE, (curT + 3U) * d);
          PipeBarrier<PIPE_V>();
        } else {
          Duplicate(inFp, 0.0f, 3 * d);
          GlobalTensor<bfloat16_t> sGm;
          sGm.SetGlobalBuffer(srcBase, static_cast<uint64_t>(curT) * srcRowDim);
          DataCopyParams cp{static_cast<uint16_t>(curT), blockLen, srcStride, 0};
          DataCopy(inBf16[3 * d], sGm, cp);
          PipeBarrier<PIPE_ALL>();
          Cast(inFp[3 * d], inBf16[3 * d], RoundMode::CAST_NONE, curT * d);
          PipeBarrier<PIPE_V>();
        }
      } else {
        GlobalTensor<bfloat16_t> sGm;
        sGm.SetGlobalBuffer(srcBase + static_cast<uint64_t>(bStart - 3U) * srcRowDim,
                            static_cast<uint64_t>(curT + 3U) * srcRowDim);
        DataCopyParams cp{static_cast<uint16_t>(curT + 3U), blockLen, srcStride, 0};
        DataCopy(inBf16, sGm, cp);
        PipeBarrier<PIPE_ALL>();
        Cast(inFp, inBf16, RoundMode::CAST_NONE, (curT + 3U) * d);
        PipeBarrier<PIPE_V>();
      }

      // Full-tile vectorized 4-tap convolution (d == 128)
      BinaryRepeatParams bMul{1, 1, 1, 16, 16, 0};
      Mul(acc, inFp[0 * d], wFp[0 * d], 64, static_cast<uint8_t>(curT), bMul);
      Mul(acc[64], inFp[0 * d + 64], wFp[0 * d + 64], 64, static_cast<uint8_t>(curT), bMul);
      PipeBarrier<PIPE_V>();
      for (uint32_t tap = 1; tap < 4; ++tap) {
        Mul(tmp, inFp[tap * d], wFp[tap * d], 64, static_cast<uint8_t>(curT), bMul);
        Mul(tmp[64], inFp[tap * d + 64], wFp[tap * d + 64], 64, static_cast<uint8_t>(curT), bMul);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, tmp, curT * d);
        PipeBarrier<PIPE_V>();
      }

      // SiLU: y / (1 + exp(-y))
      Muls(tmp, acc, -1.0f, curT * d);
      PipeBarrier<PIPE_V>();
      Mins(tmp, tmp, 80.0f, curT * d);
      PipeBarrier<PIPE_V>();
      Maxs(tmp, tmp, -80.0f, curT * d);
      PipeBarrier<PIPE_V>();
      Exp(tmp, tmp, curT * d);
      PipeBarrier<PIPE_V>();
      Adds(tmp, tmp, 1.0f, curT * d);
      PipeBarrier<PIPE_V>();
      Div(acc, acc, tmp, curT * d);
      PipeBarrier<PIPE_V>();

      // Fused L2Norm for Q (0) and K (1)
      if (which < 2U) {
        Mul(tmp, acc, acc, curT * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
        Add(tmp, tmp, tmp[64], 64, static_cast<uint8_t>(curT), foldP);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(normVec, tmp, 64, static_cast<uint8_t>(curT), 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Adds(normVec, normVec, 1e-6f, curT);
        PipeBarrier<PIPE_V>();
        Sqrt(normVec, normVec, curT);
        PipeBarrier<PIPE_V>();
        Duplicate(tmp, 1.0f, curT);
        PipeBarrier<PIPE_V>();
        Div(normVec, tmp, normVec, curT);
        PipeBarrier<PIPE_V>();
        Brcb(normBrcb, normVec, static_cast<uint8_t>((curT + 7U) / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams scaleP{1, 1, 0, 16, 16, 1};
        Mul(acc, acc, normBrcb, 64, static_cast<uint8_t>(curT), scaleP);
        Mul(acc[64], acc[64], normBrcb, 64, static_cast<uint8_t>(curT), scaleP);
        PipeBarrier<PIPE_V>();
      }

      Cast(outBf16, acc, RoundMode::CAST_RINT, curT * d);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<bfloat16_t> dGm;
      dGm.SetGlobalBuffer(dstBase + static_cast<uint64_t>(bStart) * d, curT * d);
      DataCopy(dGm, outBf16, curT * d);
      PipeBarrier<PIPE_ALL>();
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 3. AIV: FP32 PreGate + 6-step Hillis-Steele Cumsum -> BF16 Cube Inputs
// ---------------------------------------------------------------------------
__aicore__ inline void ExecutePreGate(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                      const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    const uint32_t g = h / t->headsPerGroup;
    const uint32_t hInG = h % t->headsPerGroup;
    const uint32_t groupDim = t->headsPerGroup * d;
    const float scale = BitsToFloat(t->scaleBits);
    const float lowerBound = BitsToFloat(t->lowerBoundBits);
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    GM_ADDR projBase = GetProjBase(inputList, t) + g * t->recvGroupBytes;
    const bool directPreProj = (t->projMTile == 0U) && ((t->cpMode != CP_MODE_ULYSSES) || (t->cpSize == 1U));
    const uint32_t gRowDim = directPreProj ? (t->numHeads * d) : groupDim;
    __gm__ bfloat16_t *gRawBase =
        directPreProj
            ? (reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WFB]) + static_cast<uint64_t>(h) * d)
            : (reinterpret_cast<__gm__ bfloat16_t *>(projBase + t->recvGOff) + static_cast<uint64_t>(hInG) * d);
    const bool useRecvBeta = (t->cpMode == CP_MODE_ULYSSES) && (t->cpSize > 1U);
    const uint32_t betaStride = useRecvBeta ? t->headsPerGroup : t->numHeads;
    const uint32_t betaCol = useRecvBeta ? hInG : h;
    __gm__ bfloat16_t *betaRawBase =
        useRecvBeta ? reinterpret_cast<__gm__ bfloat16_t *>(projBase + t->recvBetaOff)
                    : (directPreProj ? reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WB])
                                     : reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsBetaAllOff));

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    // UB budget (~148 KB < 192 KB):
    // gCum(32KB) + qFp(32KB) + kFp(32KB) + strip1(16KB, 32*d) + strip2(16KB, 32*d)
    // + bfTmp(16KB, 64*d BF16) + dtBias(512B) + gRef(512B) + beta(4KB)
    const uint32_t uGCum = 0;
    const uint32_t uQFp = uGCum + C * d * 4;
    const uint32_t uKFp = uQFp + C * d * 4;
    const uint32_t uS1 = uKFp + C * d * 4;
    const uint32_t uS2 = uS1 + 32 * d * 4;
    const uint32_t uBf = uS2 + 32 * d * 4;
    const uint32_t uDtBias = uBf + C * d * 2;
    const uint32_t uGRef = uDtBias + d * 4;
    const uint32_t uBeta = uGRef + d * 4;
    const uint32_t uBetaTmp = uBeta + C * 4;
    const uint32_t uBetaBrcb = uBetaTmp + C * 4;
    const uint32_t uBetaRaw = uBetaBrcb + C * 8 * 4;
    const uint32_t uBetaMask = uBetaRaw + ((C * betaStride * 4 + 31U) & ~31U);
    const uint32_t uScalar = uBetaMask + (((betaStride + 7U) & ~7U) * 4U);
    pipe.InitBuffer(buf, uScalar + 64);

    LocalTensor<float> gCum = buf.GetWithOffset<float>(C * d, uGCum);
    LocalTensor<float> qFp = buf.GetWithOffset<float>(C * d, uQFp);
    LocalTensor<float> kFp = buf.GetWithOffset<float>(C * d, uKFp);
    LocalTensor<float> s1 = buf.GetWithOffset<float>(32 * d, uS1);
    LocalTensor<float> s2 = buf.GetWithOffset<float>(32 * d, uS2);
    LocalTensor<bfloat16_t> bfTmp = buf.GetWithOffset<bfloat16_t>(C * d, uBf);
    LocalTensor<float> dtBias = buf.GetWithOffset<float>(d, uDtBias);
    LocalTensor<float> gRef = buf.GetWithOffset<float>(d, uGRef);
    LocalTensor<float> betaVec = buf.GetWithOffset<float>(C, uBeta);
    LocalTensor<float> betaTmp = buf.GetWithOffset<float>(C, uBetaTmp);
    LocalTensor<float> betaBrcb = buf.GetWithOffset<float>(C * 8, uBetaBrcb);
    LocalTensor<float> betaRawFp = buf.GetWithOffset<float>(C * betaStride, uBetaRaw);
    LocalTensor<float> betaMask = buf.GetWithOffset<float>((betaStride + 7U) & ~7U, uBetaMask);
    LocalTensor<float> scalarBuf = buf.GetWithOffset<float>(16, uScalar);
    Duplicate(betaMask, 0.0f, (betaStride + 7U) & ~7U);
    PipeBarrier<PIPE_ALL>();
    betaMask.SetValue(betaCol, 1.0f);

    // Load A = exp(A_log[h]) and dt_bias[h]
    GlobalTensor<float> aLogGm, dtBiasGm;
    aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_ALOG]), t->numLocalHeads);
    Duplicate(scalarBuf, 0.0f, 8);
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(aLogGm[h]);
    scalarBuf.SetValue(0, aLogGm.GetValue(h));
    PipeBarrier<PIPE_ALL>();
    Exp(scalarBuf, scalarBuf, 8);
    PipeBarrier<PIPE_ALL>();
    const float A = scalarBuf.GetValue(0);

    dtBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_DTBIAS]) + static_cast<uint64_t>(h) * d, d);
    DataCopy(dtBias, dtBiasGm, d);
    PipeBarrier<PIPE_ALL>();

    const uint16_t blockLen = static_cast<uint16_t>((d * 2U) / 32U);
    const uint16_t srcStride = static_cast<uint16_t>(((gRowDim - d) * 2U) / 32U);

    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t tStart = static_cast<uint64_t>(c) * C;
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;

      // 1. beta = sigmoid(beta_raw)
      const uint32_t betaElems = C * betaStride;
      GlobalTensor<bfloat16_t> bGm;
      bGm.SetGlobalBuffer(betaRawBase + tStart * betaStride, betaElems);
      DataCopy(bfTmp, bGm, betaElems);
      PipeBarrier<PIPE_ALL>();
      Cast(betaRawFp, bfTmp, RoundMode::CAST_NONE, betaElems);
      if ((betaStride & 7U) == 0U && betaStride <= 64U) {
        PipeBarrier<PIPE_V>();
        const uint8_t repStride = static_cast<uint8_t>(betaStride / 8U);
        Mul(betaRawFp, betaRawFp, betaMask, static_cast<uint64_t>(betaStride), static_cast<uint8_t>(C),
            BinaryRepeatParams{1, 1, 1, repStride, repStride, 0});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(betaVec, betaRawFp, static_cast<int32_t>(betaStride), static_cast<int32_t>(C), 1, 1,
                       static_cast<uint16_t>(repStride));
        PipeBarrier<PIPE_V>();
      } else {
        PipeBarrier<PIPE_ALL>();
        for (uint32_t r = 0; r < C; ++r) {
          betaVec.SetValue(r, betaRawFp.GetValue(r * betaStride + betaCol));
        }
        PipeBarrier<PIPE_ALL>();
      }
      Muls(betaTmp, betaVec, -1.0f, C);
      PipeBarrier<PIPE_V>();
      Mins(betaTmp, betaTmp, 80.0f, C);
      PipeBarrier<PIPE_V>();
      Maxs(betaTmp, betaTmp, -80.0f, C);
      PipeBarrier<PIPE_V>();
      Exp(betaTmp, betaTmp, C);
      PipeBarrier<PIPE_V>();
      Adds(betaTmp, betaTmp, 1.0f, C);
      PipeBarrier<PIPE_V>();
      Duplicate(betaVec, 1.0f, C);
      PipeBarrier<PIPE_V>();
      Div(betaVec, betaVec, betaTmp, C);
      PipeBarrier<PIPE_V>();
      Brcb(betaBrcb, betaVec, static_cast<uint8_t>(C / 8U), {1, 8});
      PipeBarrier<PIPE_ALL>();

      // 2. Load g_raw[c] (2 strips of 32 rows) and compute kda_gate in FP32
      for (uint32_t half = 0; half < 2; ++half) {
        GlobalTensor<bfloat16_t> gGm;
        gGm.SetGlobalBuffer(gRawBase + (tStart + half * 32ULL) * gRowDim, 32ULL * gRowDim);
        DataCopyParams cp{32, blockLen, srcStride, 0};
        DataCopy(bfTmp, gGm, cp);
        PipeBarrier<PIPE_ALL>();
        Cast(s1, bfTmp, RoundMode::CAST_NONE, 32 * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bAdd{1, 1, 1, 16, 16, 0};
        Add(s1, s1, dtBias, 64, 32, bAdd);
        Add(s1[64], s1[64], dtBias[64], 64, 32, bAdd);
        PipeBarrier<PIPE_V>();
        if (t->hasLowerBound) {
          Muls(s1, s1, -A, 32 * d);
          PipeBarrier<PIPE_V>();
          Mins(s1, s1, 80.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Maxs(s1, s1, -80.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Exp(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Adds(s1, s1, 1.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Duplicate(s2, lowerBound, 32 * d);
          PipeBarrier<PIPE_V>();
          Div(gCum[half * 32 * d], s2, s1, 32 * d);
        } else {
          Maxs(s2, s1, 0.0f, 32 * d);
          Abs(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Muls(s1, s1, -1.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Exp(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Adds(s1, s1, 1.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Ln(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Add(s1, s2, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Muls(gCum[half * 32 * d], s1, -A, 32 * d);
        }
        PipeBarrier<PIPE_V>();
      }

      // 3. 6-step Hillis-Steele parallel prefix sum (C == 64, using idle qFp)
      Adds(qFp[0], gCum[0], 0.0f, 1 * d);
      Add(qFp[1 * d], gCum[1 * d], gCum[0], 63 * d);
      PipeBarrier<PIPE_V>();
      Adds(gCum[0], qFp[0], 0.0f, 2 * d);
      Add(gCum[2 * d], qFp[2 * d], qFp[0], 62 * d);
      PipeBarrier<PIPE_V>();
      Adds(qFp[0], gCum[0], 0.0f, 4 * d);
      Add(qFp[4 * d], gCum[4 * d], gCum[0], 60 * d);
      PipeBarrier<PIPE_V>();
      Adds(gCum[0], qFp[0], 0.0f, 8 * d);
      Add(gCum[8 * d], qFp[8 * d], qFp[0], 56 * d);
      PipeBarrier<PIPE_V>();
      Adds(qFp[0], gCum[0], 0.0f, 16 * d);
      Add(qFp[16 * d], gCum[16 * d], gCum[0], 48 * d);
      PipeBarrier<PIPE_V>();
      Adds(gCum[0], qFp[0], 0.0f, 32 * d);
      Add(gCum[32 * d], qFp[32 * d], qFp[0], 32 * d);
      PipeBarrier<PIPE_V>();

      // Store exp(g_last) in FP32
      Mins(gRef, gCum[(C - 1) * d], 80.0f, d);
      PipeBarrier<PIPE_V>();
      Maxs(gRef, gRef, -80.0f, d);
      PipeBarrier<PIPE_V>();
      Exp(gRef, gRef, d);
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<float> egGm;
        egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsExpGLastOff) + hcIdx * d, d);
        DataCopy(egGm, gRef, d);
        PipeBarrier<PIPE_ALL>();
      }

      // 4. v_beta = v * beta -> BF16 wsVbetaOff
      {
        GlobalTensor<bfloat16_t> vGm, vbGm;
        vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVConvOff) +
                                (static_cast<uint64_t>(h) * t->seqEff + tStart) * d,
                            C * d);
        vbGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVbetaOff) + hcIdx * C * d, C * d);
        DataCopy(bfTmp, vGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(qFp, bfTmp, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(qFp, qFp, betaBrcb, 64, C, bScale);
        Mul(qFp[64], qFp[64], betaBrcb, 64, C, bScale);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp, qFp, RoundMode::CAST_RINT, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vbGm, bfTmp, C * d);
        PipeBarrier<PIPE_ALL>();
      }

      // 5. Load q_hat & k_hat into qFp, kFp
      {
        GlobalTensor<bfloat16_t> qGm, kGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsQHatOff) +
                                (static_cast<uint64_t>(h) * t->seqEff + tStart) * d,
                            C * d);
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKHatOff) +
                                (static_cast<uint64_t>(h) * t->seqEff + tStart) * d,
                            C * d);
        DataCopy(bfTmp, qGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(qFp, bfTmp, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(bfTmp, kGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(kFp, bfTmp, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_ALL>();
      }

      // 6. Compute qg = q * scale * exp(g) AND kbg = k * beta * exp(g) in ONE exp(g) pass!
      for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t rOff = half * 32 * d;
        Maxs(s1, gCum[rOff], -80.0f, 32 * d);
        PipeBarrier<PIPE_V>();
        Exp(s1, s1, 32 * d);
        PipeBarrier<PIPE_V>();

        Mul(s2, qFp[rOff], s1, 32 * d);
        PipeBarrier<PIPE_V>();
        Muls(s2, s2, scale, 32 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[0], s2, RoundMode::CAST_RINT, 32 * d);

        Mul(s2, kFp[rOff], s1, 32 * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(s2, s2, betaBrcb[half * 32 * 8], 64, 32, bScale);
        Mul(s2[64], s2[64], betaBrcb[half * 32 * 8], 64, 32, bScale);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[32 * d], s2, RoundMode::CAST_RINT, 32 * d);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<bfloat16_t> qgGm, kbgGm;
        qgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsQgOff) + hcIdx * C * d + rOff, 32 * d);
        kbgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKbgOff) + hcIdx * C * d + rOff, 32 * d);
        DataCopy(qgGm, bfTmp[0], 32 * d);
        DataCopy(kbgGm, bfTmp[32 * d], 32 * d);
        PipeBarrier<PIPE_ALL>();
      }

      // kg = k * exp(g_last - g) -> BF16
      Adds(gRef, gCum[(C - 1) * d], 0.0f, d);
      PipeBarrier<PIPE_V>();
      for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t rOff = half * 32 * d;
        BinaryRepeatParams bSub{1, 1, 1, 16, 0, 16};
        Sub(s1, gRef, gCum[rOff], 64, 32, bSub);
        Sub(s1[64], gRef[64], gCum[rOff + 64], 64, 32, bSub);
        PipeBarrier<PIPE_V>();
        Mins(s1, s1, 80.0f, 32 * d);
        PipeBarrier<PIPE_V>();
        Maxs(s1, s1, -80.0f, 32 * d);
        PipeBarrier<PIPE_V>();
        Exp(s1, s1, 32 * d);
        PipeBarrier<PIPE_V>();
        Mul(s2, kFp[rOff], s1, 32 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[rOff], s2, RoundMode::CAST_RINT, 32 * d);
        PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<bfloat16_t> kgGm;
        kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKgOff) + hcIdx * C * d, C * d);
        DataCopy(kgGm, bfTmp, C * d);
        PipeBarrier<PIPE_ALL>();
      }

      // 7. Prepare deduplicated BF16 wsSubIn[288, d]:
      //    Rows [0..191]:   4 blocks x 48 rows = (A_top16, A_bot16, B_diag16) with gRef = gCum[a*16 + 7]
      //    Rows [192..287]: 3 off-diag B_off blocks (16 + 32 + 48 = 96 rows) reusing A[a] = rows [a*48..a*48+31]!
      const uint64_t subBase = hcIdx * 288ULL * d;
      __gm__ bfloat16_t *subInPtr = reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsSubInOff) + subBase;

      for (uint32_t a = 0; a < 4; ++a) {
        const uint32_t base = a * 16;
        const uint32_t dRowOff = a * 48U;
        Adds(gRef, gCum[(base + 7) * d], 0.0f, d);
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams bSub{1, 1, 1, 16, 16, 0};
        Sub(s2, gCum[base * d], gRef, 64, 16, bSub);
        Sub(s2[64], gCum[base * d + 64], gRef[64], 64, 16, bSub);
        PipeBarrier<PIPE_V>();
        Mins(s2, s2, 40.0f, 16 * d);
        PipeBarrier<PIPE_V>();
        Maxs(s2, s2, -40.0f, 16 * d);
        PipeBarrier<PIPE_V>();
        Exp(s2, s2, 16 * d);
        PipeBarrier<PIPE_V>();

        // A top 16: -beta * k * exp(g - g_ref)
        Mul(s1, kFp[base * d], s2, 16 * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(s1, s1, betaBrcb[base * 8], 64, 16, bScale);
        Mul(s1[64], s1[64], betaBrcb[base * 8], 64, 16, bScale);
        PipeBarrier<PIPE_V>();
        Muls(s1, s1, -1.0f, 16 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[0], s1, RoundMode::CAST_RINT, 16 * d);

        // A bottom 16: q * scale * exp(g - g_ref)
        Mul(s1, qFp[base * d], s2, 16 * d);
        PipeBarrier<PIPE_V>();
        Muls(s1, s1, scale, 16 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[16 * d], s1, RoundMode::CAST_RINT, 16 * d);

        // B_diag 16: k / exp(g - g_ref)
        Div(s1, kFp[base * d], s2, 16 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[32 * d], s1, RoundMode::CAST_RINT, 16 * d);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<bfloat16_t> dGm;
        dGm.SetGlobalBuffer(subInPtr + dRowOff * d, 48 * d);
        DataCopy(dGm, bfTmp, 48 * d);
        PipeBarrier<PIPE_ALL>();

        // Off-diag B_off[a] (for a=1..3, cols=16,32,48) using the SAME gRef = gCum[a*16+7]!
        if (a >= 1U) {
          const uint32_t offBBase[3] = {192U, 208U, 240U};
          const uint32_t cols = a * 16U;
          for (uint32_t j0 = 0; j0 < cols; j0 += 32U) {
            const uint32_t cur = (j0 + 32U <= cols) ? 32U : (cols - j0);
            BinaryRepeatParams bSubB{1, 1, 1, 16, 0, 16};
            Sub(s2, gRef, gCum[j0 * d], 64, static_cast<uint8_t>(cur), bSubB);
            Sub(s2[64], gRef[64], gCum[j0 * d + 64], 64, static_cast<uint8_t>(cur), bSubB);
            PipeBarrier<PIPE_V>();
            // j < a*16 < a*16+7 => gRef - gCum[j] <= 0 always! Only lower clamp needed!
            Maxs(s2, s2, -40.0f, cur * d);
            PipeBarrier<PIPE_V>();
            Exp(s2, s2, cur * d);
            PipeBarrier<PIPE_V>();
            Mul(s1, kFp[j0 * d], s2, cur * d);
            PipeBarrier<PIPE_V>();
            Cast(bfTmp[j0 * d], s1, RoundMode::CAST_RINT, cur * d);
            PipeBarrier<PIPE_V>();
          }
          PipeBarrier<PIPE_ALL>();
          GlobalTensor<bfloat16_t> bGm;
          bGm.SetGlobalBuffer(subInPtr + offBBase[a - 1U] * d, cols * d);
          DataCopy(bGm, bfTmp, cols * d);
          PipeBarrier<PIPE_ALL>();
        }
      }
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 4. AIC: Raw BF16 Cube Intra-Chunk Matmuls (1 TPipe for entire seg!)
// ---------------------------------------------------------------------------
__aicore__ inline void RunIntraRawTile(KdaRawCubePipe &cube, LocalTensor<bfloat16_t> &l1A,
                                       LocalTensor<bfloat16_t> &l1B, LocalTensor<bfloat16_t> &A,
                                       LocalTensor<bfloat16_t> &B, LocalTensor<float> &C,
                                       __gm__ bfloat16_t *aPtr, __gm__ bfloat16_t *bPtr,
                                       __gm__ float *cPtr, uint32_t d, uint32_t N) {
  GlobalTensor<bfloat16_t> aGm, bGm;
  GlobalTensor<float> cGm;
  aGm.SetGlobalBuffer(aPtr, 32 * d);
  bGm.SetGlobalBuffer(bPtr, N * d);
  cGm.SetGlobalBuffer(cPtr, 32 * N);
  cube.CopyGmToL1Nz(l1A, aGm, 32, d, d, 32);
  cube.CopyGmToL1Nz(l1B, bGm, N, d, d, N);
  cube.WaitMte2ToMte1();
  cube.LoadL0A_NoT(A, l1A, 32, d, 32);
  cube.LoadL0B_Trans(B, l1B, d, N);
  cube.WaitMte1ToMte2();
  cube.WaitMte1ToM();
  cube.RunMmad(C, A, B, 32, N, d, true);
  cube.WaitMToMte1();
  cube.WaitMToFix();
  cube.StoreFp32Gm(cGm, C, 32, N, N);
  cube.WaitFixToM();
}

__aicore__ inline void ExecuteIntraCube(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                        const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t d = t->headDim;
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    KdaRawCubePipe cube;
    cube.Init(64 * 1024, 16 * 1024, 16 * 1024, 32 * 1024);
    auto l1A = cube.L1At(0);
    auto l1B = cube.L1At(16 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C = cube.L0C();



    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;
      __gm__ bfloat16_t *inPtr = reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsSubInOff) + hcIdx * 288ULL * d;
      __gm__ float *outPtr = reinterpret_cast<__gm__ float *>(ws + t->wsSubOutOff) + hcIdx * 5120ULL;

      // 3 off-diagonal tiles (reusing A_block[a] at a*48*d, B_off at 192/208/240*d!)
      RunIntraRawTile(cube, l1A, l1B, A, B, C, inPtr + 48ULL * d, inPtr + 192ULL * d, outPtr + 0ULL, d, 16);
      RunIntraRawTile(cube, l1A, l1B, A, B, C, inPtr + 96ULL * d, inPtr + 208ULL * d, outPtr + 512ULL, d, 32);
      RunIntraRawTile(cube, l1A, l1B, A, B, C, inPtr + 144ULL * d, inPtr + 240ULL * d, outPtr + 1536ULL, d, 48);
      for (uint32_t a = 0; a < 4; ++a) {
        const uint64_t dRowOff = a * 48ULL;
        RunIntraRawTile(cube, l1A, l1B, A, B, C, inPtr + dRowOff * d, inPtr + (dRowOff + 32ULL) * d,
                        outPtr + 3072ULL + a * 512ULL, d, 16);
      }
    }
    cube.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 5. AIV: Ultra-Thin Mask & Scatter for Aqk + Akk (16x16 TriInv offloaded to Cube!)
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteTriDiag16(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                        const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t C = t->chunkSize;
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uA = 0;
    const uint32_t uAqk = uA + C * C * 4;
    const uint32_t uSub = uAqk + C * C * 4;
    const uint32_t uBf = uSub + 5120 * 4;
    const uint32_t uMaskStrict = uBf + C * C * 2;
    const uint32_t uMaskIncl = uMaskStrict + 256 * 4;
    const uint32_t uEye = uMaskIncl + 256 * 4;
    pipe.InitBuffer(buf, uEye + 256 * 4);

    LocalTensor<float> matA = buf.GetWithOffset<float>(C * C, uA);
    LocalTensor<float> matAqk = buf.GetWithOffset<float>(C * C, uAqk);
    LocalTensor<float> subOut = buf.GetWithOffset<float>(5120, uSub);
    LocalTensor<bfloat16_t> bfOut = buf.GetWithOffset<bfloat16_t>(C * C, uBf);
    LocalTensor<float> maskStrict = buf.GetWithOffset<float>(256, uMaskStrict);
    LocalTensor<float> maskIncl = buf.GetWithOffset<float>(256, uMaskIncl);
    LocalTensor<float> eyeFp = buf.GetWithOffset<float>(256, uEye);

    Duplicate(maskStrict, 0.0f, 256);
    Duplicate(maskIncl, 0.0f, 256);
    Duplicate(eyeFp, 0.0f, 256);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t r = 0; r < 16; ++r) {
      for (uint32_t j = 0; j < r; ++j) {
        maskStrict.SetValue(r * 16 + j, 1.0f);
        maskIncl.SetValue(r * 16 + j, 1.0f);
      }
      maskIncl.SetValue(r * 16 + r, 1.0f);
      eyeFp.SetValue(r * 16 + r, 1.0f);
    }
    PipeBarrier<PIPE_ALL>();
    // Publish 16x16 BF16 identity once to wsEye16Off for Cube Neumann 16x16 inversion
    Cast(bfOut, eyeFp, RoundMode::CAST_RINT, 256);
    PipeBarrier<PIPE_ALL>();
    {
      GlobalTensor<bfloat16_t> eyeGm;
      eyeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsEye16Off), 256);
      DataCopy(eyeGm, bfOut, 256);
      PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;
      GlobalTensor<float> subGm;
      GlobalTensor<bfloat16_t> akkGm, aqkGm;
      subGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsSubOutOff) + hcIdx * 5120ULL, 5120);
      akkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsAkkOff) + hcIdx * C * C, C * C);
      aqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsAqkOff) + hcIdx * C * C, C * C);

      DataCopy(subOut, subGm, 5120);
      Duplicate(matA, 0.0f, C * C);
      Duplicate(matAqk, 0.0f, C * C);
      PipeBarrier<PIPE_ALL>();

      // Mask 4 diagonal 16x16 blocks (strict lower Akk -> Cube will invert in L1; incl lower Aqk)
      for (uint32_t a = 0; a < 4; ++a) {
        const uint32_t dOff = 3072U + a * 512U;
        Mul(subOut[dOff], subOut[dOff], maskStrict, 256);
        Mul(subOut[dOff + 256], subOut[dOff + 256], maskIncl, 256);
      }
      PipeBarrier<PIPE_V>();

      for (uint32_t a = 0; a < 4; ++a) {
        const uint32_t dOff = 3072U + a * 512U;
        const uint32_t base = a * 16;
        UnaryRepeatParams uPack{1, 1, 8, 2};
        Adds(matA[base * C + base], subOut[dOff], 0.0f, 16, 16, uPack);
        Adds(matAqk[base * C + base], subOut[dOff + 256], 0.0f, 16, 16, uPack);
      }

      const uint32_t subOffs[3] = {0, 512, 1536};
      for (uint32_t a = 1; a <= 3; ++a) {
        const uint32_t cols = a * 16;
        const uint32_t off = subOffs[a - 1];
        UnaryRepeatParams uOff{1, 1, 8, static_cast<uint8_t>(cols / 8U)};
        Adds(matA[a * 16 * C], subOut[off], 0.0f, static_cast<uint64_t>(cols), 16, uOff);
        Adds(matAqk[a * 16 * C], subOut[off + 16 * cols], 0.0f, static_cast<uint64_t>(cols), 16, uOff);
      }
      PipeBarrier<PIPE_V>();

      Cast(bfOut, matAqk, RoundMode::CAST_RINT, C * C);
      PipeBarrier<PIPE_ALL>();
      DataCopy(aqkGm, bfOut, C * C);
      PipeBarrier<PIPE_ALL>();

      Cast(bfOut, matA, RoundMode::CAST_RINT, C * C);
      PipeBarrier<PIPE_ALL>();
      DataCopy(akkGm, bfOut, C * C);
      PipeBarrier<PIPE_ALL>();
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 6. AIC: L1-Resident 64x64 Cube Block-Schur Inverse + WY Factors u & w
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteWyCube(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                     const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    KdaRawCubePipe cube;
    cube.Init(64 * 1024, 16 * 1024, 32 * 1024, 64 * 1024);
    auto l1Akk = cube.L1At(0);           // 8 KB (64x64 NZ, pitch=64)
    auto l1Tmp = cube.L1At(8 * 1024);    // 8 KB (64x64 scratch NZ, pitch=64)
    auto l1Vb = cube.L1At(16 * 1024);    // 16 KB (64x128 NZ, pitch=64)
    auto l1Kbg = cube.L1At(32 * 1024);   // 16 KB (64x128 NZ, pitch=64)
    auto l1Eye = cube.L1At(48 * 1024);   // 512 B (16x16 BF16 identity, pitch=16)
    auto l1Work = cube.L1At(50 * 1024);  // 2 KB (4 x 16x16 scratch for Cube Neumann inverse)
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    // Load 16x16 BF16 identity into L1 ONCE for the entire segment
    {
      GlobalTensor<bfloat16_t> eyeGm;
      eyeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsEye16Off), 256);
      cube.CopyGmToL1Nz(l1Eye, eyeGm, 16, 16, 16, 16);
      cube.WaitMte2ToMte1();
    }

    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;
      GlobalTensor<bfloat16_t> akkGm, vbGm, kbgGm, wGm;
      GlobalTensor<float> uGm;
      akkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsAkkOff) + hcIdx * C * C, C * C);
      vbGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVbetaOff) + hcIdx * C * d, C * d);
      kbgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKbgOff) + hcIdx * C * d, C * d);
      uGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsUOff) + hcIdx * C * d, C * d);
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsWOff) + hcIdx * C * d, C * d);

      // Copy Akk (diag=D0..D3, off-diag=A_sub) + v_beta + k_beta_g to L1 in parallel
      cube.CopyGmToL1Nz(l1Akk, akkGm, 64, 64, 64, 64);
      cube.CopyGmToL1Nz(l1Vb, vbGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1Kbg, kbgGm, 64, d, d, 64);
      cube.WaitMte2ToMte1();

      // 1. Invert all 4 diagonal 16x16 blocks purely on Cube in L1: (I - A_a)^{-1} = (I+A)(I+A^2)(I+A^4)(I+A^8)
      for (uint32_t a = 0; a < 4; ++a) {
        cube.InvertDiag16InL1(l1Akk, a * 16U, l1Eye, l1Work);
      }

      // 2. 6 tiny Cube Mmads in L1 -> full 64x64 A_inv in l1Akk!
      // B10 = D1 @ A_sub10 @ D0
      cube.SubMatmul64ToL1(l1Akk, 16, 16, l1Akk, 16, 0, l1Tmp, 0, 0, 16, 16, 16, 64);
      cube.SubMatmul64ToL1(l1Tmp, 0, 0, l1Akk, 0, 0, l1Akk, 16, 0, 16, 16, 16, 64);
      // B32 = D3 @ A_sub32 @ D2
      cube.SubMatmul64ToL1(l1Akk, 48, 48, l1Akk, 48, 32, l1Tmp, 0, 0, 16, 16, 16, 64);
      cube.SubMatmul64ToL1(l1Tmp, 0, 0, l1Akk, 32, 32, l1Akk, 48, 32, 16, 16, 16, 64);
      // B_BL = T_BR @ A_subBL @ T_TL
      cube.SubMatmul64ToL1(l1Akk, 32, 32, l1Akk, 32, 0, l1Tmp, 0, 0, 32, 32, 32, 64);
      cube.SubMatmul64ToL1(l1Tmp, 0, 0, l1Akk, 0, 0, l1Akk, 32, 0, 32, 32, 32, 64);

      // Load A_inv (64x64) ONCE into L0A, compute u (FP32) and w (BF16)!
      cube.LoadL0A_NoT(A, l1Akk, 64, 64, 64);
      cube.LoadL0B_NoT(B, l1Vb, 64, d, 64);
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, 64, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(uGm, C0, 64, d, d);
      cube.WaitFixToM();

      // Reuse A_inv in L0A? Reload is < 50 cycles and avoids L0A hazard
      cube.LoadL0A_NoT(A, l1Akk, 64, 64, 64);
      cube.LoadL0B_NoT(B, l1Kbg, 64, d, 64);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, 64, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreBf16Gm(wGm, C0, 64, d, d);
      cube.WaitFixToM();
    }
    cube.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 7. Paired 1:2 AIC[k] <-> AIV[2k, 2k+1] Hardware CrossCore FwdH Recurrence
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteFwdHAic(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                      const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t coreTile = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    GM_ADDR scBase = ws + t->wsFwdHScratchOff + coreTile * t->wsFwdHScratchStride;
    __gm__ float *wsPtr = reinterpret_cast<__gm__ float *>(scBase);
    __gm__ float *dsPtr = reinterpret_cast<__gm__ float *>(scBase + C * d * 4ULL);

    KdaRawCubePipe cube;
    cube.Init(64 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1A = cube.L1At(0);
    auto l1B = cube.L1At(16 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    GlobalTensor<float> wsGm, dsGm;
    wsGm.SetGlobalBuffer(wsPtr, C * d);
    dsGm.SetGlobalBuffer(dsPtr, d * d);

    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;
      GlobalTensor<bfloat16_t> wGm, hGm, kgGm, vnGm;
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsWOff) + hcIdx * C * d, C * d);
      hGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsHOff) + hcIdx * d * d, d * d);
      kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKgOff) + hcIdx * C * d, C * d);
      vnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVnewOff) + hcIdx * C * d, C * d);

      // Wait for wsH[c] published by paired AIV[2k], AIV[2k+1]
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

      // Step 1: wS_fp32 = w[64, 128] @ h[128, 128]
      cube.CopyGmToL1Nz(l1A, wGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1B, hGm, d, d, d, d);
      cube.WaitMte2ToMte1();
      cube.LoadL0A_NoT(A, l1A, 64, d, 64);
      cube.LoadL0B_NoT(B, l1B, d, d, d);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, d, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(wsGm, C0, 64, d, d);
      cube.WaitFixToM();
      CrossCoreSetFlag<0x2, PIPE_FIX>(1);

      // Wait for wsVnew[c] published by paired AIVs
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(2);

      // Step 3: deltaS_fp32 = kg^T[128, 64] @ v_new[64, 128]
      cube.CopyGmToL1Nz(l1A, kgGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1B, vnGm, 64, d, d, 64);
      cube.WaitMte2ToMte1();
      cube.LoadL0A_Trans(A, l1A, d, 64);
      cube.LoadL0B_NoT(B, l1B, 64, d, 64);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, d, d, 64, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(dsGm, C0, d, d, d);
      cube.WaitFixToM();
      CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    }
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    cube.Destroy();
  }
}

__aicore__ inline void ExecuteFwdHAiv(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                      const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t wid = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t coreTile = wid >> 1;
    const uint32_t sub = wid & 1U;  // 0: top half, 1: bottom half
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t halfC = C >> 1;  // 32 rows of v_new per AIV
    const uint32_t halfD = d >> 1;  // 64 rows of state S per AIV
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    GM_ADDR scBase = ws + t->wsFwdHScratchOff + coreTile * t->wsFwdHScratchStride;
    __gm__ float *wsPtr = reinterpret_cast<__gm__ float *>(scBase);
    __gm__ float *dsPtr = reinterpret_cast<__gm__ float *>(scBase + C * d * 4ULL);
    __gm__ float *stateCarryPtr =
        reinterpret_cast<__gm__ float *>(ws + t->wsStateFp32Off) + static_cast<uint64_t>(h) * d * d;

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    // UB per AIV (~84 KB << 192 KB):
    // sUb(32KB: 64*128 FP32) + b1(32KB) + b2(16KB) + bfOut(16KB) + expG(256B) + expGBrcb(2KB)
    const uint32_t uState = 0;
    const uint32_t uB1 = uState + halfD * d * 4;
    const uint32_t uB2 = uB1 + halfD * d * 4;
    const uint32_t uBf = uB2 + halfC * d * 4;
    const uint32_t uExpG = uBf + halfD * d * 2;
    const uint32_t uExpGBrcb = uExpG + halfD * 4;
    pipe.InitBuffer(buf, uExpGBrcb + halfD * 8 * 4);

    LocalTensor<float> sUb = buf.GetWithOffset<float>(halfD * d, uState);
    LocalTensor<float> b1 = buf.GetWithOffset<float>(halfD * d, uB1);
    LocalTensor<float> b2 = buf.GetWithOffset<float>(halfC * d, uB2);
    LocalTensor<bfloat16_t> bfOut = buf.GetWithOffset<bfloat16_t>(halfD * d, uBf);
    LocalTensor<float> expG = buf.GetWithOffset<float>(halfD, uExpG);
    LocalTensor<float> expGBrcb = buf.GetWithOffset<float>(halfD * 8, uExpGBrcb);

    // Initialize sUb for this segment
    if (seg == 0U) {
      if (t->cpMode != CP_MODE_ULYSSES && t->cpRank > 0U) {
        const uint32_t g = h / t->headsPerGroup;
        const uint32_t hInG = h % t->headsPerGroup;
        __gm__ float *initPtr = reinterpret_cast<__gm__ float *>(
            ws + t->wsScanExclOff + g * 2ULL * t->sendScanGroupHalfBytes) +
            static_cast<uint64_t>(hInG) * d * d;
        GlobalTensor<float> initGm;
        initGm.SetGlobalBuffer(initPtr + sub * halfD * d, halfD * d);
        DataCopy(sUb, initGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
      } else {
        Duplicate(sUb, 0.0f, halfD * d);
        PipeBarrier<PIPE_V>();
      }
    } else {
      GlobalTensor<float> stGm;
      stGm.SetGlobalBuffer(stateCarryPtr + sub * halfD * d, halfD * d);
      DataCopy(sUb, stGm, halfD * d);
      PipeBarrier<PIPE_ALL>();
    }

    // Publish initial wsH[cStart] (BF16) and signal AIC (flag 0)
    {
      const uint64_t hc0 = static_cast<uint64_t>(h) * t->numChunks + cStart;
      GlobalTensor<bfloat16_t> hGm;
      hGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsHOff) + hc0 * d * d + sub * halfD * d,
                          halfD * d);
      Cast(bfOut, sUb, RoundMode::CAST_RINT, halfD * d);
      PipeBarrier<PIPE_ALL>();
      DataCopy(hGm, bfOut, halfD * d);
      PipeBarrier<PIPE_ALL>();
      CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    }

    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;

      // Step 2: Wait for wS_fp32 (flag 1), compute v_new = u - wS -> BF16 wsVnew[c]
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
      {
        GlobalTensor<float> uGm, wsGm;
        GlobalTensor<bfloat16_t> vnGm;
        uGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsUOff) + hcIdx * C * d + sub * halfC * d,
                            halfC * d);
        wsGm.SetGlobalBuffer(wsPtr + sub * halfC * d, halfC * d);
        vnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVnewOff) + hcIdx * C * d + sub * halfC * d,
                             halfC * d);
        DataCopy(b1, uGm, halfC * d);
        DataCopy(b2, wsGm, halfC * d);
        PipeBarrier<PIPE_ALL>();
        Sub(b1, b1, b2, halfC * d);
        PipeBarrier<PIPE_V>();
        Cast(bfOut, b1, RoundMode::CAST_RINT, halfC * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vnGm, bfOut, halfC * d);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
      }

      // Step 4: Wait for deltaS_fp32 (flag 3), update sUb = sUb * expG + deltaS
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);
      const bool needNextChunk = (c + 1 < cEnd);
      const bool needSegCarry = (!needNextChunk && (seg + 1 < t->numSegs));
      const bool needRank0Summary =
          (!needNextChunk && (seg + 1 == t->numSegs) && (t->cpMode != CP_MODE_ULYSSES) && (t->cpRank == 0U));

      if (needNextChunk || needSegCarry || needRank0Summary) {
        GlobalTensor<float> egGm, dsGm;
        egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsExpGLastOff) + hcIdx * d + sub * halfD, halfD);
        dsGm.SetGlobalBuffer(dsPtr + sub * halfD * d, halfD * d);
        DataCopy(expG, egGm, halfD);
        DataCopy(b1, dsGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
        Brcb(expGBrcb, expG, 8, {1, 8});  // 64 floats -> 8 repeats
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams mulS{1, 1, 0, 16, 16, 1};
        Mul(sUb, sUb, expGBrcb, 64, 64, mulS);
        Mul(sUb[64], sUb[64], expGBrcb, 64, 64, mulS);
        PipeBarrier<PIPE_V>();
        Add(sUb, sUb, b1, halfD * d);
        PipeBarrier<PIPE_V>();

        if (needNextChunk) {
          GlobalTensor<bfloat16_t> hNextGm;
          hNextGm.SetGlobalBuffer(
              reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsHOff) + (hcIdx + 1ULL) * d * d + sub * halfD * d,
              halfD * d);
          Cast(bfOut, sUb, RoundMode::CAST_RINT, halfD * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(hNextGm, bfOut, halfD * d);
          PipeBarrier<PIPE_ALL>();
          CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
        } else {
          if (needSegCarry) {
            GlobalTensor<float> stGm;
            stGm.SetGlobalBuffer(stateCarryPtr + sub * halfD * d, halfD * d);
            PipeBarrier<PIPE_ALL>();
            DataCopy(stGm, sUb, halfD * d);
            PipeBarrier<PIPE_ALL>();
          }
          if (needRank0Summary) {
            // Rank 0 publishes final B_0 directly to wsScanSelfOff AND comm_send_buf (slot 0)!
            const uint32_t g = h / t->headsPerGroup;
            const uint32_t hInG = h % t->headsPerGroup;
            const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
            const uint64_t headElemOff = static_cast<uint64_t>(hInG) * d * d + sub * halfD * d;
            GlobalTensor<float> selfGm, sendGm;
            selfGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsScanSelfOff + groupOff) + headElemOff,
                                   halfD * d);
            sendGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(inputList[IDX_SEND_BUF] + t->sendScanOff + groupOff) + headElemOff,
                halfD * d);
            PipeBarrier<PIPE_ALL>();
            DataCopy(selfGm, sUb, halfD * d);
            DataCopy(sendGm, sUb, halfD * d);
            PipeBarrier<PIPE_ALL>();
          }
        }
      }
    }
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 8. AIC: Fused FP32-L0C FinalizeO (O = qg @ h + Aqk @ v_new -> BF16, ZERO AIV!)
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteFinalizeO(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                        const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t h = taskDesc.task_index;
    const uint32_t seg = taskDesc.extra_value_0;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t cStart = seg * t->chunksPerSeg;
    const uint32_t cEnd = cStart + t->chunksPerSeg;
    const uint32_t g = h / t->headsPerGroup;
    const uint32_t hInG = h % t->headsPerGroup;
    const uint32_t groupDim = t->headsPerGroup * d;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    // Destination layout matches GatedRMSNorm / A2A #2 directly!
    __gm__ bfloat16_t *oDstBase = nullptr;
    uint32_t oRowStride = d;
    if (t->cpMode == CP_MODE_ULYSSES && t->cpSize > 1U && seg == t->cpRank) {
      // Local seg in Ulysses writes directly into comm_recv_buf + recvOOff (strided by groupDim!)
      const uint64_t slot = static_cast<uint64_t>(t->cpRank) * t->headGroups + g;
      oDstBase = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_RECV_BUF] + t->recvOOff +
                                                       slot * t->recvOSlotBytes) +
                 static_cast<uint64_t>(hInG) * d;
      oRowStride = groupDim;
    } else if (t->cpMode == CP_MODE_ULYSSES && t->cpSize > 1U && seg != t->cpRank) {
      // Remote seg in Ulysses writes into comm_send_buf (or wsOLocalSegOff) — write directly to wsOLocalSegOff
      // packed as [seg, g, seqLocal, groupDim] so A2A #2 PutMemSignal is 1 contiguous transfer!
      const uint64_t slot = static_cast<uint64_t>(seg) * t->headGroups + g;
      oDstBase = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_SEND_BUF] +
                                                       t->cpSize * t->headGroups * t->sendSlotBytes +
                                                       slot * t->recvOSlotBytes) +
                 static_cast<uint64_t>(hInG) * d;
      oRowStride = groupDim;
    } else {
      // CP=1 or ALLGATHER/PREFIX_SUM: write directly to wsOLocalSegOff as [G, seqLocal, groupDim]
      oDstBase = reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsOLocalSegOff +
                                                       static_cast<uint64_t>(g) * t->recvOSlotBytes) +
                 static_cast<uint64_t>(cStart * C) * groupDim + static_cast<uint64_t>(hInG) * d;
      oRowStride = groupDim;
    }

    KdaRawCubePipe cube;
    cube.Init(80 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1Qg = cube.L1At(0);           // 16 KB
    auto l1H = cube.L1At(16 * 1024);    // 32 KB
    auto l1Aqk = cube.L1At(48 * 1024);  // 8 KB
    auto l1Vn = cube.L1At(56 * 1024);   // 16 KB
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    for (uint32_t c = cStart; c < cEnd; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * t->numChunks + c;
      GlobalTensor<bfloat16_t> qgGm, hGm, aqkGm, vnGm, oGm;
      qgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsQgOff) + hcIdx * C * d, C * d);
      hGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsHOff) + hcIdx * d * d, d * d);
      aqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsAqkOff) + hcIdx * C * C, C * C);
      vnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVnewOff) + hcIdx * C * d, C * d);
      oGm.SetGlobalBuffer(oDstBase + static_cast<uint64_t>((c - cStart) * C) * oRowStride,
                          static_cast<uint64_t>(C) * oRowStride);

      cube.CopyGmToL1Nz(l1Qg, qgGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1H, hGm, d, d, d, d);
      cube.CopyGmToL1Nz(l1Aqk, aqkGm, 64, 64, 64, 64);
      cube.CopyGmToL1Nz(l1Vn, vnGm, 64, d, d, 64);
      cube.WaitMte2ToMte1();

      // Part 1: C0 = qg @ h (initZero = true)
      cube.LoadL0A_NoT(A, l1Qg, 64, d, 64);
      cube.LoadL0B_NoT(B, l1H, d, d, d);
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, d, true);
      cube.WaitMToMte1();

      // Part 2: C0 += Aqk @ v_new (initZero = false — FUSED IN FP32 L0C!)
      cube.LoadL0A_NoT(A, l1Aqk, 64, 64, 64);
      cube.LoadL0B_NoT(B, l1Vn, 64, d, 64);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, 64, false);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreBf16Gm(oGm, C0, 64, d, oRowStride);
      cube.WaitFixToM();
    }
    cube.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 9. Paired 1:2 Pass-1 InterSummary (B_r, M_r) for ALLGATHER / PREFIX_SUM CP
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteInterSummaryAic(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                              const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t h = taskDesc.task_index;
    const uint32_t coreTile = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t NT = t->numChunks;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    GM_ADDR scBase = ws + t->wsFwdHScratchOff + coreTile * t->wsFwdHScratchStride;
    __gm__ float *wBPtr = reinterpret_cast<__gm__ float *>(scBase);
    __gm__ float *dBPtr = reinterpret_cast<__gm__ float *>(scBase + C * d * 4ULL);
    __gm__ float *wMPtr = reinterpret_cast<__gm__ float *>(scBase + (C * d + d * d) * 4ULL);
    __gm__ float *dMPtr = reinterpret_cast<__gm__ float *>(scBase + (2ULL * C * d + d * d) * 4ULL);
    __gm__ bfloat16_t *vBPtr = reinterpret_cast<__gm__ bfloat16_t *>(scBase + (2ULL * C * d + 2ULL * d * d) * 4ULL);
    __gm__ bfloat16_t *vMPtr = vBPtr + C * d;
    __gm__ bfloat16_t *bBfPtr = vMPtr + C * d;
    __gm__ bfloat16_t *mBfPtr = bBfPtr + d * d;

    KdaRawCubePipe cube;
    cube.Init(80 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1W = cube.L1At(0);           // 16 KB
    auto l1B = cube.L1At(16 * 1024);   // 32 KB
    auto l1M = cube.L1At(48 * 1024);   // 32 KB
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    GlobalTensor<float> wBGm, wMGm, dBGm, dMGm;
    GlobalTensor<bfloat16_t> vBGm, vMGm, bBfGm, mBfGm;
    wBGm.SetGlobalBuffer(wBPtr, C * d);
    wMGm.SetGlobalBuffer(wMPtr, C * d);
    dBGm.SetGlobalBuffer(dBPtr, d * d);
    dMGm.SetGlobalBuffer(dMPtr, d * d);
    vBGm.SetGlobalBuffer(vBPtr, C * d);
    vMGm.SetGlobalBuffer(vMPtr, C * d);
    bBfGm.SetGlobalBuffer(bBfPtr, d * d);
    mBfGm.SetGlobalBuffer(mBfPtr, d * d);

    for (uint32_t c = 0; c < NT; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * NT + c;
      GlobalTensor<bfloat16_t> wGm, kgGm;
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsWOff) + hcIdx * C * d, C * d);
      kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKgOff) + hcIdx * C * d, C * d);

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

      // wB = w @ B, wM = w @ M
      cube.CopyGmToL1Nz(l1W, wGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1B, bBfGm, d, d, d, d);
      cube.CopyGmToL1Nz(l1M, mBfGm, d, d, d, d);
      cube.WaitMte2ToMte1();

      cube.LoadL0A_NoT(A, l1W, 64, d, 64);
      cube.LoadL0B_NoT(B, l1B, d, d, d);
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, d, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(wBGm, C0, 64, d, d);
      cube.WaitFixToM();

      cube.LoadL0A_NoT(A, l1W, 64, d, 64);
      cube.LoadL0B_NoT(B, l1M, d, d, d);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, 64, d, d, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(wMGm, C0, 64, d, d);
      cube.WaitFixToM();
      CrossCoreSetFlag<0x2, PIPE_FIX>(1);

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(2);

      // dB = kg^T @ vB, dM = kg^T @ vM
      cube.CopyGmToL1Nz(l1W, kgGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1B, vBGm, 64, d, d, 64);
      cube.CopyGmToL1Nz(l1M, vMGm, 64, d, d, 64);
      cube.WaitMte2ToMte1();

      cube.LoadL0A_Trans(A, l1W, d, 64);
      cube.LoadL0B_NoT(B, l1B, 64, d, 64);
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, d, d, 64, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(dBGm, C0, d, d, d);
      cube.WaitFixToM();

      cube.LoadL0A_Trans(A, l1W, d, 64);
      cube.LoadL0B_NoT(B, l1M, 64, d, 64);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, d, d, 64, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(dMGm, C0, d, d, d);
      cube.WaitFixToM();
      CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    }
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    cube.Destroy();
  }
}

__aicore__ inline void ExecuteInterSummaryAiv(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                              const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t h = taskDesc.task_index;
    const uint32_t wid = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t coreTile = wid >> 1;
    const uint32_t sub = wid & 1U;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t halfC = C >> 1;
    const uint32_t halfD = d >> 1;
    const uint32_t NT = t->numChunks;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    GM_ADDR scBase = ws + t->wsFwdHScratchOff + coreTile * t->wsFwdHScratchStride;
    __gm__ float *wBPtr = reinterpret_cast<__gm__ float *>(scBase);
    __gm__ float *dBPtr = reinterpret_cast<__gm__ float *>(scBase + C * d * 4ULL);
    __gm__ float *wMPtr = reinterpret_cast<__gm__ float *>(scBase + (C * d + d * d) * 4ULL);
    __gm__ float *dMPtr = reinterpret_cast<__gm__ float *>(scBase + (2ULL * C * d + d * d) * 4ULL);
    __gm__ bfloat16_t *vBPtr = reinterpret_cast<__gm__ bfloat16_t *>(scBase + (2ULL * C * d + 2ULL * d * d) * 4ULL);
    __gm__ bfloat16_t *vMPtr = vBPtr + C * d;
    __gm__ bfloat16_t *bBfPtr = vMPtr + C * d;
    __gm__ bfloat16_t *mBfPtr = bBfPtr + d * d;

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    // UB per AIV (~130 KB < 192 KB):
    // bUb(32KB) + mUb(32KB) + t1(32KB) + t2(16KB) + bfOut(16KB) + expG(256B) + expGBrcb(2KB)
    const uint32_t uB = 0;
    const uint32_t uM = uB + halfD * d * 4;
    const uint32_t uT1 = uM + halfD * d * 4;
    const uint32_t uT2 = uT1 + halfD * d * 4;
    const uint32_t uBf = uT2 + halfC * d * 4;
    const uint32_t uExpG = uBf + halfD * d * 2;
    const uint32_t uExpGBrcb = uExpG + halfD * 4;
    pipe.InitBuffer(buf, uExpGBrcb + halfD * 8 * 4);

    LocalTensor<float> bUb = buf.GetWithOffset<float>(halfD * d, uB);
    LocalTensor<float> mUb = buf.GetWithOffset<float>(halfD * d, uM);
    LocalTensor<float> t1 = buf.GetWithOffset<float>(halfD * d, uT1);
    LocalTensor<float> t2 = buf.GetWithOffset<float>(halfC * d, uT2);
    LocalTensor<bfloat16_t> bfOut = buf.GetWithOffset<bfloat16_t>(halfD * d, uBf);
    LocalTensor<float> expG = buf.GetWithOffset<float>(halfD, uExpG);
    LocalTensor<float> expGBrcb = buf.GetWithOffset<float>(halfD * 8, uExpGBrcb);

    // Init B = 0, M = I_d
    Duplicate(bUb, 0.0f, halfD * d);
    Duplicate(mUb, 0.0f, halfD * d);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t r = 0; r < halfD; ++r) {
      mUb.SetValue(r * d + (sub * halfD + r), 1.0f);
    }
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<bfloat16_t> bBfGm, mBfGm;
    bBfGm.SetGlobalBuffer(bBfPtr + sub * halfD * d, halfD * d);
    mBfGm.SetGlobalBuffer(mBfPtr + sub * halfD * d, halfD * d);

    // Publish initial B_0, M_0 -> signal flag 0
    Cast(bfOut, bUb, RoundMode::CAST_RINT, halfD * d);
    PipeBarrier<PIPE_ALL>();
    DataCopy(bBfGm, bfOut, halfD * d);
    PipeBarrier<PIPE_ALL>();
    Cast(bfOut, mUb, RoundMode::CAST_RINT, halfD * d);
    PipeBarrier<PIPE_ALL>();
    DataCopy(mBfGm, bfOut, halfD * d);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

    for (uint32_t c = 0; c < NT; ++c) {
      const uint64_t hcIdx = static_cast<uint64_t>(h) * NT + c;
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
      // vB = u - wB, vM = -wM -> BF16
      {
        GlobalTensor<float> uGm, wBGm, wMGm;
        GlobalTensor<bfloat16_t> vBGm, vMGm;
        uGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsUOff) + hcIdx * C * d + sub * halfC * d,
                            halfC * d);
        wBGm.SetGlobalBuffer(wBPtr + sub * halfC * d, halfC * d);
        wMGm.SetGlobalBuffer(wMPtr + sub * halfC * d, halfC * d);
        vBGm.SetGlobalBuffer(vBPtr + sub * halfC * d, halfC * d);
        vMGm.SetGlobalBuffer(vMPtr + sub * halfC * d, halfC * d);
        DataCopy(t1, uGm, halfC * d);
        DataCopy(t2, wBGm, halfC * d);
        PipeBarrier<PIPE_ALL>();
        Sub(t1, t1, t2, halfC * d);
        PipeBarrier<PIPE_V>();
        Cast(bfOut, t1, RoundMode::CAST_RINT, halfC * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vBGm, bfOut, halfC * d);

        DataCopy(t2, wMGm, halfC * d);
        PipeBarrier<PIPE_ALL>();
        Muls(t2, t2, -1.0f, halfC * d);
        PipeBarrier<PIPE_V>();
        Cast(bfOut, t2, RoundMode::CAST_RINT, halfC * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vMGm, bfOut, halfC * d);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
      }

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);
      // Update bUb = expG * bUb + dB, mUb = expG * mUb + dM
      GlobalTensor<float> egGm, dBGm, dMGm;
      egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsExpGLastOff) + hcIdx * d + sub * halfD, halfD);
      dBGm.SetGlobalBuffer(dBPtr + sub * halfD * d, halfD * d);
      dMGm.SetGlobalBuffer(dMPtr + sub * halfD * d, halfD * d);
      DataCopy(expG, egGm, halfD);
      DataCopy(t1, dBGm, halfD * d);
      PipeBarrier<PIPE_ALL>();
      Brcb(expGBrcb, expG, 8, {1, 8});
      PipeBarrier<PIPE_V>();
      BinaryRepeatParams mulS{1, 1, 0, 16, 16, 1};
      Mul(bUb, bUb, expGBrcb, 64, 64, mulS);
      Mul(bUb[64], bUb[64], expGBrcb, 64, 64, mulS);
      PipeBarrier<PIPE_V>();
      Add(bUb, bUb, t1, halfD * d);

      DataCopy(t1, dMGm, halfD * d);
      PipeBarrier<PIPE_ALL>();
      Mul(mUb, mUb, expGBrcb, 64, 64, mulS);
      Mul(mUb[64], mUb[64], expGBrcb, 64, 64, mulS);
      PipeBarrier<PIPE_V>();
      Add(mUb, mUb, t1, halfD * d);
      PipeBarrier<PIPE_V>();

      if (c + 1 < NT) {
        Cast(bfOut, bUb, RoundMode::CAST_RINT, halfD * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(bBfGm, bfOut, halfD * d);
        PipeBarrier<PIPE_ALL>();
        Cast(bfOut, mUb, RoundMode::CAST_RINT, halfD * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(mBfGm, bfOut, halfD * d);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
      }
    }

    // Store final FP32 (B_r, M_r) to wsScanSelfOff AND comm_send_buf (slot 0)
    const uint32_t g = h / t->headsPerGroup;
    const uint32_t hInG = h % t->headsPerGroup;
    const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
    const uint64_t headElemOff = static_cast<uint64_t>(hInG) * d * d + sub * halfD * d;
    GlobalTensor<float> selfBGm, selfMGm, sendBGm, sendMGm;
    selfBGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsScanSelfOff + groupOff) + headElemOff,
                            halfD * d);
    selfMGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(ws + t->wsScanSelfOff + groupOff + t->sendScanGroupHalfBytes) + headElemOff,
        halfD * d);
    sendBGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(inputList[IDX_SEND_BUF] + t->sendScanOff + groupOff) + headElemOff,
        halfD * d);
    sendMGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_SEND_BUF] + t->sendScanOff + groupOff +
                                                             t->sendScanGroupHalfBytes) +
                                headElemOff,
                            halfD * d);
    PipeBarrier<PIPE_ALL>();
    DataCopy(selfBGm, bUb, halfD * d);
    DataCopy(selfMGm, mUb, halfD * d);
    DataCopy(sendBGm, bUb, halfD * d);
    DataCopy(sendMGm, mUb, halfD * d);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 10. Paired 1:2 ScanMerge for ALLGATHER (LASP-2) & PREFIX_SUM (Tree-Scan)
// ---------------------------------------------------------------------------
// Helper: 128x128x128 matmul via raw BF16 Cube (AIV casts FP32 -> BF16, AIC Mmad -> FP32)
// extra0: 0 = ALLGATHER_REDUCE (src = 0..rank-1), 1 = PREFIX_ROUND (extra1 = round_k)
__aicore__ inline void ExecuteScanMergeAic(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                           const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t h = taskDesc.task_index;
    const uint32_t opKind = taskDesc.extra_value_0;
    const uint32_t roundK = taskDesc.extra_value_1;
    const uint32_t d = t->headDim;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    // Per-head scan scratch: [tmpOut_fp32 (64KB), lhs_bf16 (32KB), rhs_bf16 (32KB)]
    GM_ADDR tmpBase = ws + t->wsScanTmpOff + static_cast<uint64_t>(h) * 3ULL * d * d * 4ULL;
    __gm__ float *tmpOutPtr = reinterpret_cast<__gm__ float *>(tmpBase);
    __gm__ bfloat16_t *lhsBfPtr = reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL);
    __gm__ bfloat16_t *rhsBfPtr = lhsBfPtr + d * d;

    // Count exact number of 128x128x128 matmuls this task performs
    uint32_t numMatmuls = 0;
    if (opKind == 0U) {
      // ALLGATHER_REDUCE on rank >= 1: for src = 1 .. cpRank - 1 -> 1 matmul each (M_src @ S_acc)
      numMatmuls = (t->cpRank > 1U) ? (t->cpRank - 1U) : 0U;
    } else {
      // PREFIX_ROUND on rank >= (1 << roundK)
      const uint32_t stride = 1U << roundK;
      const uint32_t src = t->cpRank - stride;
      const bool msgHasM = (src >= stride);
      if (roundK > 0U) {
        numMatmuls += 1U;             // M_excl @ B_msg
        if (msgHasM) numMatmuls += 1U; // M_excl @ M_msg
      }
      const uint32_t nextStride = stride << 1;
      const bool needIncl = (t->cpRank < t->cpSize - 1U) && (t->cpRank + nextStride < t->cpSize);
      if (needIncl) {
        numMatmuls += 1U;                        // M_self @ B_excl
        if (t->cpRank >= nextStride) numMatmuls += 1U;  // M_self @ M_excl
      }
    }
    if (numMatmuls == 0U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(6);
      return;
    }

    KdaRawCubePipe cube;
    cube.Init(64 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1A = cube.L1At(0);
    auto l1B = cube.L1At(32 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    GlobalTensor<bfloat16_t> lhsGm, rhsGm;
    GlobalTensor<float> outGm;
    lhsGm.SetGlobalBuffer(lhsBfPtr, d * d);
    rhsGm.SetGlobalBuffer(rhsBfPtr, d * d);
    outGm.SetGlobalBuffer(tmpOutPtr, d * d);

    for (uint32_t m = 0; m < numMatmuls; ++m) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(4);
      cube.CopyGmToL1Nz(l1A, lhsGm, d, d, d, d);
      cube.CopyGmToL1Nz(l1B, rhsGm, d, d, d, d);
      cube.WaitMte2ToMte1();
      cube.LoadL0A_NoT(A, l1A, d, d, d);
      cube.LoadL0B_NoT(B, l1B, d, d, d);
      cube.WaitMte1ToMte2();
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, d, d, d, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(outGm, C0, d, d, d);
      cube.WaitFixToM();
      CrossCoreSetFlag<0x2, PIPE_FIX>(5);
    }
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(6);
    cube.Destroy();
  }
}

__aicore__ inline void RunScanCubeMulAiv(LocalTensor<float> &fA, LocalTensor<bfloat16_t> &bf,
                                         GlobalTensor<bfloat16_t> &lhsGm, GlobalTensor<bfloat16_t> &rhsGm,
                                         GlobalTensor<float> &tmpOutGm, GlobalTensor<float> &srcLhsGm,
                                         GlobalTensor<float> &srcRhsGm, uint32_t elems) {
  DataCopy(fA, srcLhsGm, elems);
  PipeBarrier<PIPE_ALL>();
  Cast(bf, fA, RoundMode::CAST_RINT, elems);
  PipeBarrier<PIPE_ALL>();
  DataCopy(lhsGm, bf, elems);

  DataCopy(fA, srcRhsGm, elems);
  PipeBarrier<PIPE_ALL>();
  Cast(bf, fA, RoundMode::CAST_RINT, elems);
  PipeBarrier<PIPE_ALL>();
  DataCopy(rhsGm, bf, elems);
  PipeBarrier<PIPE_ALL>();
  CrossCoreSetFlag<0x2, PIPE_MTE3>(4);
  CrossCoreWaitFlag<0x2, PIPE_MTE2>(5);
  DataCopy(fA, tmpOutGm, elems);
  PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void ExecuteScanMergeAiv(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                           const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t h = taskDesc.task_index;
    const uint32_t opKind = taskDesc.extra_value_0;
    const uint32_t roundK = taskDesc.extra_value_1;
    const uint32_t wid = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t sub = wid & 1U;
    const uint32_t d = t->headDim;
    const uint32_t halfD = d >> 1;
    const uint32_t g = h / t->headsPerGroup;
    const uint32_t hInG = h % t->headsPerGroup;
    const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
    const uint64_t mOffBytes = t->sendScanGroupHalfBytes;
    const uint64_t hOff = static_cast<uint64_t>(hInG) * d * d + sub * halfD * d;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    GM_ADDR tmpBase = ws + t->wsScanTmpOff + static_cast<uint64_t>(h) * 3ULL * d * d * 4ULL;
    __gm__ float *tmpOutPtr = reinterpret_cast<__gm__ float *>(tmpBase) + sub * halfD * d;
    __gm__ bfloat16_t *lhsBfPtr = reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL) + sub * halfD * d;
    __gm__ bfloat16_t *rhsBfPtr =
        reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL) + d * d + sub * halfD * d;

    __gm__ float *exclBPtr = reinterpret_cast<__gm__ float *>(ws + t->wsScanExclOff + groupOff) + hOff;
    __gm__ float *exclMPtr = reinterpret_cast<__gm__ float *>(ws + t->wsScanExclOff + groupOff + mOffBytes) + hOff;
    __gm__ float *selfBPtr = reinterpret_cast<__gm__ float *>(ws + t->wsScanSelfOff + groupOff) + hOff;
    __gm__ float *selfMPtr = reinterpret_cast<__gm__ float *>(ws + t->wsScanSelfOff + groupOff + mOffBytes) + hOff;

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uA = 0;
    const uint32_t uB = uA + halfD * d * 4;
    const uint32_t uBf = uB + halfD * d * 4;
    pipe.InitBuffer(buf, uBf + halfD * d * 2);
    LocalTensor<float> fA = buf.GetWithOffset<float>(halfD * d, uA);
    LocalTensor<float> fB = buf.GetWithOffset<float>(halfD * d, uB);
    LocalTensor<bfloat16_t> bf = buf.GetWithOffset<bfloat16_t>(halfD * d, uBf);

    GlobalTensor<float> tmpOutGm, exclBGm, exclMGm, selfBGm, selfMGm;
    GlobalTensor<bfloat16_t> lhsGm, rhsGm;
    tmpOutGm.SetGlobalBuffer(tmpOutPtr, halfD * d);
    lhsGm.SetGlobalBuffer(lhsBfPtr, halfD * d);
    rhsGm.SetGlobalBuffer(rhsBfPtr, halfD * d);
    exclBGm.SetGlobalBuffer(exclBPtr, halfD * d);
    exclMGm.SetGlobalBuffer(exclMPtr, halfD * d);
    selfBGm.SetGlobalBuffer(selfBPtr, halfD * d);
    selfMGm.SetGlobalBuffer(selfMPtr, halfD * d);



    if (opKind == 0U) {
      // ALLGATHER_REDUCE: S_acc = recv[0].B; for src=1..rank-1: S_acc = recv[src].M @ S_acc + recv[src].B
      GM_ADDR recv0 = inputList[IDX_RECV_BUF] + t->recvScanOff + groupOff;
      GlobalTensor<float> r0Gm;
      r0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(recv0) + hOff, halfD * d);
      DataCopy(fA, r0Gm, halfD * d);
      PipeBarrier<PIPE_ALL>();
      DataCopy(exclBGm, fA, halfD * d);
      PipeBarrier<PIPE_ALL>();

      for (uint32_t src = 1; src < t->cpRank; ++src) {
        GM_ADDR rSlot = inputList[IDX_RECV_BUF] + t->recvScanOff + src * t->sendScanSlotBytes + groupOff;
        GlobalTensor<float> rBGm, rMGm;
        rBGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rSlot) + hOff, halfD * d);
        rMGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rSlot + mOffBytes) + hOff, halfD * d);
        RunScanCubeMulAiv(fA, bf, lhsGm, rhsGm, tmpOutGm, rMGm, exclBGm, halfD * d);  // fA = M_src @ S_acc
        DataCopy(fB, rBGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
        Add(fA, fA, fB, halfD * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(exclBGm, fA, halfD * d);
        PipeBarrier<PIPE_ALL>();
      }
    } else {
      // PREFIX_ROUND
      const uint32_t stride = 1U << roundK;
      const uint32_t src = t->cpRank - stride;
      const bool msgHasM = (src >= stride);
      GM_ADDR msgSlot = inputList[IDX_RECV_BUF] + t->recvScanOff + roundK * t->sendScanSlotBytes + groupOff;
      GlobalTensor<float> msgBGm, msgMGm;
      msgBGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(msgSlot) + hOff, halfD * d);
      msgMGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(msgSlot + mOffBytes) + hOff, halfD * d);

      if (roundK == 0U) {
        DataCopy(fA, msgBGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(exclBGm, fA, halfD * d);
        if (msgHasM) {
          DataCopy(fB, msgMGm, halfD * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(exclMGm, fB, halfD * d);
        }
        PipeBarrier<PIPE_ALL>();
      } else {
        // B_excl = M_excl @ B_msg + B_excl
        RunScanCubeMulAiv(fA, bf, lhsGm, rhsGm, tmpOutGm, exclMGm, msgBGm, halfD * d);
        DataCopy(fB, exclBGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
        Add(fA, fA, fB, halfD * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(exclBGm, fA, halfD * d);
        PipeBarrier<PIPE_ALL>();
        if (msgHasM) {
          // M_excl = M_excl @ M_msg
          RunScanCubeMulAiv(fA, bf, lhsGm, rhsGm, tmpOutGm, exclMGm, msgMGm, halfD * d);
          DataCopy(exclMGm, fA, halfD * d);
          PipeBarrier<PIPE_ALL>();
        }
      }

      const uint32_t nextStride = stride << 1;
      const bool needIncl = (t->cpRank < t->cpSize - 1U) && (t->cpRank + nextStride < t->cpSize);
      if (needIncl) {
        const uint32_t nextPingPong = (roundK + 1U) & 1U;
        GM_ADDR sendSlot =
            inputList[IDX_SEND_BUF] + t->sendScanOff + nextPingPong * t->sendScanSlotBytes + groupOff;
        GlobalTensor<float> sendBGm, sendMGm;
        sendBGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sendSlot) + hOff, halfD * d);
        sendMGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sendSlot + mOffBytes) + hOff, halfD * d);

        // B_incl = M_self @ B_excl + B_self
        RunScanCubeMulAiv(fA, bf, lhsGm, rhsGm, tmpOutGm, selfMGm, exclBGm, halfD * d);
        DataCopy(fB, selfBGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
        Add(fA, fA, fB, halfD * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(sendBGm, fA, halfD * d);
        PipeBarrier<PIPE_ALL>();
        if (t->cpRank >= nextStride) {
          RunScanCubeMulAiv(fA, bf, lhsGm, rhsGm, tmpOutGm, selfMGm, exclMGm, halfD * d);
          DataCopy(sendMGm, fA, halfD * d);
          PipeBarrier<PIPE_ALL>();
        }
      }
    }
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(6);
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 11. AIV: Gated RMSNorm (Vectorized Add + WholeReduceSum + Brcb + RowMultiply)
// ---------------------------------------------------------------------------
__aicore__ inline void ExecuteGatedRmsNorm(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                           const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t g = taskDesc.task_index;
    const uint32_t srcRank = taskDesc.extra_value_0;
    const uint32_t ot = taskDesc.task_split_num;
    const uint32_t oTile = taskDesc.task_split_value;
    const uint32_t rowStart = ot * oTile;
    const uint32_t curRows = (rowStart + oTile <= t->seqLocal) ? oTile : (t->seqLocal - rowStart);
    const uint32_t d = t->headDim;
    const uint32_t hG = t->headsPerGroup;
    const uint32_t groupDim = hG * d;
    const uint32_t totalHeadDim = t->numHeads * d;
    const float eps = BitsToFloat(t->normEpsBits);
    const float invD = 1.0f / static_cast<float>(static_cast<int32_t>(d));
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    const uint64_t slotIdx =
        (t->cpMode == CP_MODE_ULYSSES) ? (static_cast<uint64_t>(srcRank) * t->headGroups + g) : g;
    __gm__ bfloat16_t *oSrcBase =
        (t->cpMode == CP_MODE_ULYSSES && t->cpSize > 1U)
            ? reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_RECV_BUF] + t->recvOOff +
                                                    slotIdx * t->recvOSlotBytes)
            : reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsOLocalSegOff +
                                                    static_cast<uint64_t>(g) * t->recvOSlotBytes);
    const bool preProj = (t->projMTile == 0U);
    const uint64_t headOff =
        (t->cpMode == CP_MODE_ULYSSES)
            ? (static_cast<uint64_t>(srcRank) * t->numLocalHeads + static_cast<uint64_t>(g) * hG)
            : (static_cast<uint64_t>(g) * hG);
    const uint32_t gRowDim = preProj ? totalHeadDim : groupDim;
    __gm__ bfloat16_t *gOutBase =
        preProj
            ? (reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WGB]) + headOff * d)
            : (reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsGOutOff) + slotIdx * t->seqLocal * groupDim);
    __gm__ bfloat16_t *dstBase =
        reinterpret_cast<__gm__ bfloat16_t *>(preProj ? inputList[IDX_OUT] : (ws + t->wsONormedOff)) + headOff * d;

    constexpr uint32_t BT = 32U;  // 32 rows per vector batch
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uW = 0;                       // d*4 = 512B
    const uint32_t uOBf = uW + d * 4;            // 32*d*2 = 8KB
    const uint32_t uGBf = uOBf + BT * d * 2;     // 8KB
    const uint32_t uOFp = uGBf + BT * d * 2;     // 16KB
    const uint32_t uGFp = uOFp + BT * d * 4;     // 16KB
    const uint32_t uTmp = uGFp + BT * d * 4;     // 16KB
    const uint32_t uRms = uTmp + BT * d * 4;     // 128B
    const uint32_t uRmsBrcb = uRms + BT * 4;     // 1KB
    pipe.InitBuffer(buf, uRmsBrcb + BT * 8 * 4);

    LocalTensor<float> wFp = buf.GetWithOffset<float>(d, uW);
    LocalTensor<bfloat16_t> oBf = buf.GetWithOffset<bfloat16_t>(BT * d, uOBf);
    LocalTensor<bfloat16_t> gBf = buf.GetWithOffset<bfloat16_t>(BT * d, uGBf);
    LocalTensor<float> oFp = buf.GetWithOffset<float>(BT * d, uOFp);
    LocalTensor<float> gFp = buf.GetWithOffset<float>(BT * d, uGFp);
    LocalTensor<float> tmp = buf.GetWithOffset<float>(BT * d, uTmp);
    LocalTensor<float> rmsVec = buf.GetWithOffset<float>(BT, uRms);
    LocalTensor<float> rmsBrcb = buf.GetWithOffset<float>(BT * 8, uRmsBrcb);

    {
      GlobalTensor<bfloat16_t> wGm;
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_ONORMW]), d);
      DataCopy(oBf, wGm, d);
      PipeBarrier<PIPE_ALL>();
      Cast(wFp, oBf, RoundMode::CAST_NONE, d);
      PipeBarrier<PIPE_ALL>();
    }

    const uint16_t blockLen = static_cast<uint16_t>((d * 2U) / 32U);
    const uint16_t srcStride = static_cast<uint16_t>(((groupDim - d) * 2U) / 32U);
    const uint16_t dstStride = static_cast<uint16_t>(((totalHeadDim - d) * 2U) / 32U);

    for (uint32_t hi = 0; hi < hG; ++hi) {
      for (uint32_t r0 = 0; r0 < curRows; r0 += BT) {
        const uint32_t cur = (r0 + BT <= curRows) ? BT : (curRows - r0);
        const uint64_t rAbs = rowStart + r0;

        GlobalTensor<bfloat16_t> oGm, gGm, dGm;
        oGm.SetGlobalBuffer(oSrcBase + rAbs * groupDim + hi * d, static_cast<uint64_t>(cur) * groupDim);
        gGm.SetGlobalBuffer(gOutBase + rAbs * gRowDim + hi * d, static_cast<uint64_t>(cur) * gRowDim);
        dGm.SetGlobalBuffer(dstBase + rAbs * totalHeadDim + hi * d, static_cast<uint64_t>(cur) * totalHeadDim);

        const uint16_t gSrcStride = static_cast<uint16_t>(((gRowDim - d) * 2U) / 32U);
        DataCopyParams inPO{static_cast<uint16_t>(cur), blockLen, srcStride, 0};
        DataCopyParams inPG{static_cast<uint16_t>(cur), blockLen, gSrcStride, 0};
        DataCopy(oBf, oGm, inPO);
        DataCopy(gBf, gGm, inPG);
        PipeBarrier<PIPE_ALL>();

        Cast(oFp, oBf, RoundMode::CAST_NONE, cur * d);
        Cast(gFp, gBf, RoundMode::CAST_NONE, cur * d);
        PipeBarrier<PIPE_V>();

        // RMSNorm
        Mul(tmp, oFp, oFp, cur * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
        Add(tmp, tmp, tmp[64], 64, static_cast<uint8_t>(cur), foldP);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(rmsVec, tmp, 64, static_cast<uint8_t>(cur), 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Muls(rmsVec, rmsVec, invD, cur);
        PipeBarrier<PIPE_V>();
        Adds(rmsVec, rmsVec, eps, cur);
        PipeBarrier<PIPE_V>();
        Sqrt(rmsVec, rmsVec, cur);
        PipeBarrier<PIPE_V>();
        Duplicate(tmp, 1.0f, cur);
        PipeBarrier<PIPE_V>();
        Div(rmsVec, tmp, rmsVec, cur);
        PipeBarrier<PIPE_V>();
        Brcb(rmsBrcb, rmsVec, static_cast<uint8_t>((cur + 7U) / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams scaleP{1, 1, 0, 16, 16, 1};
        Mul(oFp, oFp, rmsBrcb, 64, static_cast<uint8_t>(cur), scaleP);
        Mul(oFp[64], oFp[64], rmsBrcb, 64, static_cast<uint8_t>(cur), scaleP);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams wMul{1, 1, 1, 16, 16, 0};
        Mul(oFp, oFp, wFp, 64, static_cast<uint8_t>(cur), wMul);
        Mul(oFp[64], oFp[64], wFp[64], 64, static_cast<uint8_t>(cur), wMul);
        PipeBarrier<PIPE_V>();

        // * sigmoid(g)
        Muls(gFp, gFp, -1.0f, cur * d);
        PipeBarrier<PIPE_V>();
        Mins(gFp, gFp, 80.0f, cur * d);
        PipeBarrier<PIPE_V>();
        Maxs(gFp, gFp, -80.0f, cur * d);
        PipeBarrier<PIPE_V>();
        Exp(gFp, gFp, cur * d);
        PipeBarrier<PIPE_V>();
        Adds(gFp, gFp, 1.0f, cur * d);
        PipeBarrier<PIPE_V>();
        Div(oFp, oFp, gFp, cur * d);
        PipeBarrier<PIPE_V>();

        Cast(oBf, oFp, RoundMode::CAST_RINT, cur * d);
        PipeBarrier<PIPE_ALL>();
        DataCopyParams outP{static_cast<uint16_t>(cur), blockLen, 0, dstStride};
        DataCopy(dGm, oBf, outP);
        PipeBarrier<PIPE_ALL>();
      }
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// 12. Comm AIV: Unified SHMEM PutMemSignal (Ulysses A2A #1/#2, Halo, State)
// ---------------------------------------------------------------------------
__aicore__ inline void RunShmemPutBytes(GM_ADDR rb, GM_ADDR sb, GM_ADDR ev, int64_t signalOff,
                                        int64_t dstByteOff, int64_t srcByteOff, int64_t nBytes,
                                        int64_t dstPe, bool publish) {
  hyper_multicore::PutMemSignalKernel<uint16_t> op;
  op.Init(rb, dstByteOff / 2, sb, srcByteOff / 2, nBytes / 2, ev, signalOff, 1, 1, dstPe, false, nullptr);
  op.Process(publish);
}

__aicore__ inline void PackGroupSlice(__gm__ bfloat16_t *dstPtr, __gm__ bfloat16_t *srcAllPtr,
                                      uint32_t seqLocal, uint32_t totalDim, uint32_t dimOff, uint32_t groupDim) {
  // Choose strip so strip * groupDim * 2 <= 128 KB (fits comfortably in 192 KB UB)
  const uint32_t strip = (groupDim <= 1024U) ? 64U : 32U;
  const uint16_t blockLen = static_cast<uint16_t>((groupDim * 2U) / 32U);
  const uint16_t srcStride = static_cast<uint16_t>(((totalDim - groupDim) * 2U) / 32U);
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  pipe.InitBuffer(buf, strip * groupDim * 2U + 64U);
  LocalTensor<bfloat16_t> tmp = buf.GetWithOffset<bfloat16_t>(strip * groupDim, 0);
  for (uint32_t r0 = 0; r0 < seqLocal; r0 += strip) {
    const uint32_t cur = (r0 + strip <= seqLocal) ? strip : (seqLocal - r0);
    GlobalTensor<bfloat16_t> sGm, dGm;
    sGm.SetGlobalBuffer(srcAllPtr + static_cast<uint64_t>(r0) * totalDim + dimOff,
                        static_cast<uint64_t>(cur) * totalDim);
    dGm.SetGlobalBuffer(dstPtr + static_cast<uint64_t>(r0) * groupDim, static_cast<uint64_t>(cur) * groupDim);
    DataCopyParams cp{static_cast<uint16_t>(cur), blockLen, srcStride, 0};
    DataCopy(tmp, sGm, cp);
    PipeBarrier<PIPE_ALL>();
    DataCopy(dGm, tmp, cur * groupDim);
    PipeBarrier<PIPE_ALL>();
  }
  pipe.Destroy();
}

__aicore__ inline void PackBetaSlice(__gm__ bfloat16_t *dstPtr, __gm__ bfloat16_t *srcAllPtr,
                                     uint32_t seqLocal, uint32_t numHeads, uint32_t headOff, uint32_t hG) {
  const uint32_t strip = 256U;
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  const uint32_t inBytes = strip * numHeads * 2U;
  const uint32_t outBytes = strip * hG * 2U;
  pipe.InitBuffer(buf, inBytes + outBytes + 64U);
  LocalTensor<uint16_t> inU16 = buf.GetWithOffset<uint16_t>(strip * numHeads, 0);
  LocalTensor<uint16_t> outU16 = buf.GetWithOffset<uint16_t>(strip * hG, inBytes);
  for (uint32_t r0 = 0; r0 < seqLocal; r0 += strip) {
    const uint32_t cur = (r0 + strip <= seqLocal) ? strip : (seqLocal - r0);
    GlobalTensor<uint16_t> sGm, dGm;
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(srcAllPtr) + static_cast<uint64_t>(r0) * numHeads,
                        cur * numHeads);
    dGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(dstPtr) + static_cast<uint64_t>(r0) * hG, cur * hG);
    DataCopy(inU16, sGm, cur * numHeads);
    PipeBarrier<PIPE_ALL>();
    if ((hG & 3U) == 0U && (headOff & 3U) == 0U && (numHeads & 3U) == 0U) {
      LocalTensor<uint64_t> s64 = inU16.ReinterpretCast<uint64_t>();
      LocalTensor<uint64_t> d64 = outU16.ReinterpretCast<uint64_t>();
      const uint32_t sStride = numHeads >> 2;
      const uint32_t dStride = hG >> 2;
      const uint32_t hOff64 = headOff >> 2;
      for (uint32_t r = 0; r < cur; ++r) {
        for (uint32_t c = 0; c < dStride; ++c) {
          d64.SetValue(r * dStride + c, s64.GetValue(r * sStride + hOff64 + c));
        }
      }
    } else {
      for (uint32_t r = 0; r < cur; ++r) {
        for (uint32_t c = 0; c < hG; ++c) {
          outU16.SetValue(r * hG + c, inU16.GetValue(r * numHeads + headOff + c));
        }
      }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(dGm, outU16, cur * hG);
    PipeBarrier<PIPE_ALL>();
  }
  pipe.Destroy();
}

__aicore__ inline void ExecuteShmemPut(const TaskDesc &taskDesc, GM_ADDR *inputList,
                                       const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t g = taskDesc.task_index;
    const uint32_t mode = taskDesc.extra_value_0;  // 0=A2A1, 1=A2A2, 2=HALO, 3=STATE, 4=LOCAL_BETA_PACK
    const uint32_t param = taskDesc.extra_value_1;
    const uint32_t groupDim = t->headsPerGroup * t->headDim;
    const int64_t signalOff = static_cast<int64_t>(taskDesc.trigger_event);
    GM_ADDR sb = inputList[IDX_SEND_BUF];
    GM_ADDR rb = inputList[IDX_RECV_BUF];
    GM_ADDR ev = inputList[IDX_EVENT];

    if (mode == 0U || mode == 4U || mode == 5U || mode == 6U) {
      const uint32_t dstPe = param;
      const int64_t slotOff =
          static_cast<int64_t>((static_cast<uint64_t>(dstPe) * t->headGroups + g) * t->sendSlotBytes);
      const int64_t rGrp = static_cast<int64_t>(g * t->recvGroupBytes);
      const int64_t qkvBytes = static_cast<int64_t>(static_cast<uint64_t>(t->seqLocal) * groupDim * 2ULL);
      const int64_t betaBytes = static_cast<int64_t>(static_cast<uint64_t>(t->seqLocal) * t->headsPerGroup * 2ULL);
      const int64_t rankQkvOff = static_cast<int64_t>(t->cpRank) * qkvBytes;
      const int64_t rankBetaOff = static_cast<int64_t>(t->cpRank) * betaBytes;

      const uint32_t headOff = dstPe * t->numLocalHeads + g * t->headsPerGroup;
      const uint32_t totalHeadDim = t->numHeads * t->headDim;
      const uint32_t dimOff = headOff * t->headDim;
      __gm__ bfloat16_t *betaSrc =
          (mode >= 5U) ? reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WB])
                       : reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WORKSPACE] + t->wsBetaAllOff);
      __gm__ bfloat16_t *sendBetaPtr = reinterpret_cast<__gm__ bfloat16_t *>(sb + slotOff + t->sendBetaOff);
      PackBetaSlice(sendBetaPtr, betaSrc, t->seqLocal, t->numHeads, headOff, t->headsPerGroup);

      if (mode == 5U) {
        // Pre-projected Ulysses remote A2A #1: pack Q,K,V,G from IDX_WQ/WK/WV/WFB -> comm_send_buf
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(sb + slotOff + t->sendQOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WQ]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(sb + slotOff + t->sendKOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WK]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(sb + slotOff + t->sendVOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WV]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(sb + slotOff + t->sendGOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WFB]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
      } else if (mode == 6U) {
        // Pre-projected Ulysses local slice: pack Q,K,V,G directly into local comm_recv_buf!
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(rb + rGrp + static_cast<int64_t>(t->recvQOff) + rankQkvOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WQ]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(rb + rGrp + static_cast<int64_t>(t->recvKOff) + rankQkvOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WK]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(rb + rGrp + static_cast<int64_t>(t->recvVOff) + rankQkvOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WV]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
        PackGroupSlice(reinterpret_cast<__gm__ bfloat16_t *>(rb + rGrp + static_cast<int64_t>(t->recvGOff) + rankQkvOff),
                       reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WFB]),
                       t->seqLocal, totalHeadDim, dimOff, groupDim);
      }

      if (mode == 0U || mode == 5U) {
        RunShmemPutBytes(rb, sb, ev, signalOff, rGrp + static_cast<int64_t>(t->recvQOff) + rankQkvOff,
                         slotOff + static_cast<int64_t>(t->sendQOff), qkvBytes, dstPe, false);
        RunShmemPutBytes(rb, sb, ev, signalOff, rGrp + static_cast<int64_t>(t->recvKOff) + rankQkvOff,
                         slotOff + static_cast<int64_t>(t->sendKOff), qkvBytes, dstPe, false);
        RunShmemPutBytes(rb, sb, ev, signalOff, rGrp + static_cast<int64_t>(t->recvVOff) + rankQkvOff,
                         slotOff + static_cast<int64_t>(t->sendVOff), qkvBytes, dstPe, false);
        RunShmemPutBytes(rb, sb, ev, signalOff, rGrp + static_cast<int64_t>(t->recvGOff) + rankQkvOff,
                         slotOff + static_cast<int64_t>(t->sendGOff), qkvBytes, dstPe, false);
      }
      RunShmemPutBytes(rb, sb, ev, signalOff, rGrp + static_cast<int64_t>(t->recvBetaOff) + rankBetaOff,
                       slotOff + static_cast<int64_t>(t->sendBetaOff), betaBytes, dstPe, true);
    } else if (mode == 1U) {
      // Ulysses A2A #2: send O_attn for (seg=dstPe=param, g) -> remote recvOOff[myRank, g]
      const uint32_t dstPe = param;
      const int64_t srcOff = static_cast<int64_t>(
          t->cpSize * t->headGroups * t->sendSlotBytes +
          (static_cast<uint64_t>(dstPe) * t->headGroups + g) * t->recvOSlotBytes);
      const int64_t dstOff = static_cast<int64_t>(
          t->recvOOff + (static_cast<uint64_t>(t->cpRank) * t->headGroups + g) * t->recvOSlotBytes);
      const int64_t oBytes = static_cast<int64_t>(static_cast<uint64_t>(t->seqLocal) * groupDim * 2ULL);
      RunShmemPutBytes(rb, sb, ev, signalOff, dstOff, srcOff, oBytes, dstPe, true);
    } else if (mode == 2U) {
      // ALLGATHER / PREFIX_SUM: ShortConv 3-token Halo (rank -> rank + 1)
      // Copy last 3 rows of Q, K, V from wsLocalProjOff to comm_send_buf + sendHaloOff first!
      const uint32_t dstPe = param;
      const uint64_t haloOff = t->sendHaloOff + g * t->sendHaloGroupBytes;
      const uint32_t elems3 = 3U * groupDim;
      const bool preProj = (t->projMTile == 0U);
      const uint32_t totalHeadDim = t->numHeads * t->headDim;
      GM_ADDR projGrp = GetProjBase(inputList, t) + g * t->recvGroupBytes;
      {
        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, elems3 * 2U + 64U);
        LocalTensor<bfloat16_t> tmp = buf.GetWithOffset<bfloat16_t>(elems3, 0);
        const uint64_t qkvOffs[3] = {t->recvQOff, t->recvKOff, t->recvVOff};
        const uint32_t directIdxs[3] = {IDX_WQ, IDX_WK, IDX_WV};
        const uint16_t blockLen = static_cast<uint16_t>((groupDim * 2U) / 32U);
        const uint16_t srcStride = static_cast<uint16_t>(((totalHeadDim - groupDim) * 2U) / 32U);
        for (uint32_t w = 0; w < 3; ++w) {
          GlobalTensor<bfloat16_t> sGm, dGm;
          dGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(sb + haloOff) + w * elems3, elems3);
          if (preProj) {
            sGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t *>(inputList[directIdxs[w]]) +
                    static_cast<uint64_t>(t->seqLocal - 3U) * totalHeadDim + static_cast<uint64_t>(g) * groupDim,
                3ULL * totalHeadDim);
            DataCopyParams cp{3, blockLen, srcStride, 0};
            DataCopy(tmp, sGm, cp);
          } else {
            sGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(projGrp + qkvOffs[w]) +
                                    static_cast<uint64_t>(t->seqLocal - 3U) * groupDim,
                                elems3);
            DataCopy(tmp, sGm, elems3);
          }
          PipeBarrier<PIPE_ALL>();
          DataCopy(dGm, tmp, elems3);
          PipeBarrier<PIPE_ALL>();
        }
        pipe.Destroy();
      }
      RunShmemPutBytes(rb, sb, ev, signalOff, static_cast<int64_t>(t->recvHaloOff + g * t->sendHaloGroupBytes),
            static_cast<int64_t>(haloOff),
            static_cast<int64_t>(3ULL * elems3 * 2ULL), dstPe, true);
    } else if (mode == 3U) {
      // ALLGATHER / PREFIX_SUM State Put:
      // param = dstPe, split_num = dstSlotIdx (srcRank for ALLGATHER, roundK for PREFIX_SUM),
      // split_val = (srcPingPong << 1) | hasM
      const uint32_t dstPe = param;
      const uint32_t dstSlot = taskDesc.task_split_num;
      const uint32_t srcPingPong = taskDesc.task_split_value >> 1;
      const uint32_t hasM = taskDesc.task_split_value & 1U;
      const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
      const int64_t srcOff = static_cast<int64_t>(t->sendScanOff + srcPingPong * t->sendScanSlotBytes + groupOff);
      const int64_t dstOff = static_cast<int64_t>(t->recvScanOff + dstSlot * t->sendScanSlotBytes + groupOff);
      const int64_t nBytes = static_cast<int64_t>(hasM ? (2ULL * t->sendScanGroupHalfBytes) : t->sendScanGroupHalfBytes);
      RunShmemPutBytes(rb, sb, ev, signalOff, dstOff, srcOff, nBytes, dstPe, true);
    }
  }
}

}  // namespace MulticoreKda

#endif  // HYPER_MEGA_KDA_COMPUTE_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 * Ultra-fast raw BF16 Cube primitives (CopyNz + LoadData + Mmad + Fixpipe)
 * for Ascend 910B/910C (arch22 / dav-2201). Zero MatmulImpl overhead!
 */
#ifndef HYPER_MEGA_KDA_CUBE_RAW_H
#define HYPER_MEGA_KDA_CUBE_RAW_H

#include "kernel_operator.h"

namespace MulticoreKda {

using namespace AscendC;

constexpr FixpipeConfig KDA_NZ_OUT = {CO2Layout::NZ, false};

struct KdaRawCubePipe {
  TPipe pipe;
  TBuf<TPosition::A1> l1;
  TBuf<TPosition::A2> l0a;
  TBuf<TPosition::B2> l0b;
  TBuf<TPosition::CO1> l0c;
  TEventID evMte2Mte1;
  TEventID evMte1M;
  TEventID evMMte1;
  TEventID evMFix;
  TEventID evFixM;
  TEventID evFixMte1;
  TEventID evMte1Mte2;

  __aicore__ inline void Init(uint32_t l1Bytes = 256 * 1024, uint32_t l0aBytes = 64 * 1024,
                              uint32_t l0bBytes = 64 * 1024, uint32_t l0cBytes = 128 * 1024) {
    pipe.InitBuffer(l1, l1Bytes);
    pipe.InitBuffer(l0a, l0aBytes);
    pipe.InitBuffer(l0b, l0bBytes);
    pipe.InitBuffer(l0c, l0cBytes);
    evMte2Mte1 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    evMte1M = pipe.AllocEventID<HardEvent::MTE1_M>();
    evMMte1 = pipe.AllocEventID<HardEvent::M_MTE1>();
    evMFix = pipe.AllocEventID<HardEvent::M_FIX>();
    evFixM = pipe.AllocEventID<HardEvent::FIX_M>();
    evFixMte1 = pipe.AllocEventID<HardEvent::FIX_MTE1>();
    evMte1Mte2 = pipe.AllocEventID<HardEvent::MTE1_MTE2>();
  }

  __aicore__ inline void Destroy() {
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
  }

  __aicore__ inline LocalTensor<bfloat16_t> L1At(uint32_t byteOff = 0) {
    return l1.Get<uint8_t>()[byteOff].ReinterpretCast<bfloat16_t>();
  }
  __aicore__ inline LocalTensor<bfloat16_t> L0A() { return l0a.Get<bfloat16_t>(); }
  __aicore__ inline LocalTensor<bfloat16_t> L0B() { return l0b.Get<bfloat16_t>(); }
  __aicore__ inline LocalTensor<float> L0C() { return l0c.Get<float>(); }

  __aicore__ inline void WaitMte2ToMte1() {
    SetFlag<HardEvent::MTE2_MTE1>(evMte2Mte1);
    WaitFlag<HardEvent::MTE2_MTE1>(evMte2Mte1);
  }
  __aicore__ inline void WaitMte1ToM() {
    SetFlag<HardEvent::MTE1_M>(evMte1M);
    WaitFlag<HardEvent::MTE1_M>(evMte1M);
  }
  __aicore__ inline void WaitMToMte1() {
    SetFlag<HardEvent::M_MTE1>(evMMte1);
    WaitFlag<HardEvent::M_MTE1>(evMMte1);
  }
  __aicore__ inline void WaitMToFix() {
    SetFlag<HardEvent::M_FIX>(evMFix);
    WaitFlag<HardEvent::M_FIX>(evMFix);
  }
  __aicore__ inline void WaitFixToM() {
    SetFlag<HardEvent::FIX_M>(evFixM);
    WaitFlag<HardEvent::FIX_M>(evFixM);
  }
  __aicore__ inline void WaitFixToMte1() {
    SetFlag<HardEvent::FIX_MTE1>(evFixMte1);
    WaitFlag<HardEvent::FIX_MTE1>(evFixMte1);
  }
  __aicore__ inline void WaitMte1ToMte2() {
    SetFlag<HardEvent::MTE1_MTE2>(evMte1Mte2);
    WaitFlag<HardEvent::MTE1_MTE2>(evMte1Mte2);
  }

  // GM (row-major M x N, stride srcD) -> L1 NZ (pitch = dstC0Stride, default M)
  __aicore__ inline void CopyGmToL1Nz(LocalTensor<bfloat16_t> dstL1, GlobalTensor<bfloat16_t> srcGm,
                                      uint32_t M, uint32_t N, uint32_t srcD, uint32_t dstC0Stride = 0) {
    Nd2NzParams p{};
    p.ndNum = 1;
    p.nValue = static_cast<uint16_t>(M);
    p.dValue = static_cast<uint16_t>(N);
    p.srcDValue = static_cast<uint16_t>(srcD);
    p.dstNzNStride = 1;
    p.dstNzC0Stride = static_cast<uint16_t>(dstC0Stride == 0 ? M : dstC0Stride);
    DataCopy(dstL1, srcGm, p);
  }

  // L1 NZ (M x K, pitch l1Pitch, non-transposed) -> L0A (ZZ)
  __aicore__ inline void LoadL0A_NoT(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                     uint32_t M, uint32_t K, uint32_t l1Pitch,
                                     uint32_t rowOff = 0, uint32_t colOff = 0) {
    LoadData2DParams p{};
    p.repeatTimes = static_cast<uint8_t>(K / 16U);
    p.srcStride = static_cast<uint16_t>(l1Pitch / 16U);
    p.ifTranspose = false;
    for (uint32_t r = 0; r < M / 16U; ++r) {
      const uint32_t srcIdx = (colOff / 16U) * (l1Pitch * 16U) + (rowOff + r * 16U) * 16U;
      LoadData(dst[r * (K / 16U) * 256U], src[srcIdx], p);
    }
  }

  // L1 NZ (K x M, pitch = K) transposed -> L0A (M x K in ZZ)
  __aicore__ inline void LoadL0A_Trans(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                       uint32_t M, uint32_t K) {
    LoadData2DParams p{};
    p.repeatTimes = static_cast<uint8_t>((M / 16U) * (K / 16U));
    p.srcStride = 1;
    p.ifTranspose = true;
    LoadData(dst, src, p);
  }

  // L1 NZ (K x N, pitch l1Pitch, non-transposed) -> L0B (ZN)
  __aicore__ inline void LoadL0B_NoT(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                     uint32_t K, uint32_t N, uint32_t l1Pitch,
                                     uint32_t rowOff = 0, uint32_t colOff = 0) {
    LoadData2DParams p{};
    p.repeatTimes = static_cast<uint8_t>(N / 16U);
    p.srcStride = static_cast<uint16_t>(l1Pitch / 16U);
    p.ifTranspose = true;
    for (uint32_t k = 0; k < K / 16U; ++k) {
      const uint32_t srcIdx = (colOff / 16U) * (l1Pitch * 16U) + (rowOff + k * 16U) * 16U;
      LoadData(dst[k * (N / 16U) * 256U], src[srcIdx], p);
    }
  }

  // L1 NZ (N x K, pitch = N) transposed -> L0B (K x N in ZN)
  __aicore__ inline void LoadL0B_Trans(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                       uint32_t K, uint32_t N) {
    LoadData2DParams p{};
    p.repeatTimes = static_cast<uint8_t>((K / 16U) * (N / 16U));
    p.srcStride = 1;
    p.ifTranspose = false;
    LoadData(dst, src, p);
  }

  // Single Mmad (BF16 -> FP32 L0C)
  __aicore__ inline void RunMmad(LocalTensor<float> c, LocalTensor<bfloat16_t> a,
                                 LocalTensor<bfloat16_t> b, uint32_t M, uint32_t N, uint32_t K,
                                 bool initZero = true) {
    MmadParams mm{};
    mm.m = static_cast<uint16_t>(M);
    mm.n = static_cast<uint16_t>(N);
    mm.k = static_cast<uint16_t>(K);
    mm.cmatrixInitVal = initZero;
    Mmad(c, a, b, mm);
  }

  // Fixpipe L0C (FP32) -> GM (FP32 row-major)
  __aicore__ inline void StoreFp32Gm(GlobalTensor<float> dstGm, LocalTensor<float> c,
                                     uint32_t M, uint32_t N, uint32_t dstStride) {
    auto p = FixpipeParamsV220(static_cast<uint16_t>(N), static_cast<uint16_t>(M),
                               static_cast<uint16_t>(M), static_cast<uint32_t>(dstStride), false);
    p.quantPre = QuantMode_t::NoQuant;
    Fixpipe<float, float, CFG_ROW_MAJOR>(dstGm, c, p);
  }

  // Fixpipe L0C (FP32) -> GM (BF16 row-major)
  __aicore__ inline void StoreBf16Gm(GlobalTensor<bfloat16_t> dstGm, LocalTensor<float> c,
                                     uint32_t M, uint32_t N, uint32_t dstStride) {
    auto p = FixpipeParamsV220(static_cast<uint16_t>(N), static_cast<uint16_t>(M),
                               static_cast<uint16_t>(M), static_cast<uint32_t>(dstStride), false);
    p.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(dstGm, c, p);
  }

  // Fixpipe L0C (FP32) -> L1 NZ (BF16, sub-block at (dstRow, dstCol) in L1 with pitch l1Pitch)
  __aicore__ inline void StoreBf16L1Nz(LocalTensor<bfloat16_t> dstL1Base, LocalTensor<float> c,
                                       uint32_t M, uint32_t N, uint32_t l1Pitch,
                                       uint32_t dstRow = 0, uint32_t dstCol = 0) {
    auto p = FixpipeParamsV220(static_cast<uint16_t>(N), static_cast<uint16_t>(M),
                               static_cast<uint16_t>(M), static_cast<uint32_t>(l1Pitch), false);
    p.quantPre = QuantMode_t::F322BF16;
    const uint32_t off = (dstCol / 16U) * (l1Pitch * 16U) + dstRow * 16U;
    Fixpipe<bfloat16_t, float, KDA_NZ_OUT>(dstL1Base[off], c, p);
  }

  // Convenience: L1 sub-block matmul C_sub = A_sub @ B_sub -> L1 NZ (both in 64x64 pitch-64 buffers)
  __aicore__ inline void SubMatmul64ToL1(LocalTensor<bfloat16_t> l1A, uint32_t aRow, uint32_t aCol,
                                         LocalTensor<bfloat16_t> l1B, uint32_t bRow, uint32_t bCol,
                                         LocalTensor<bfloat16_t> l1Dst, uint32_t dstRow, uint32_t dstCol,
                                         uint32_t M, uint32_t N, uint32_t K, uint32_t dstPitch = 64) {
    auto A = L0A();
    auto B = L0B();
    auto C = L0C();
    LoadL0A_NoT(A, l1A, M, K, 64, aRow, aCol);
    LoadL0B_NoT(B, l1B, K, N, 64, bRow, bCol);
    WaitMte1ToM();
    RunMmad(C, A, B, M, N, K, true);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(l1Dst, C, M, N, dstPitch, dstRow, dstCol);
    WaitFixToM();
    WaitFixToMte1();
  }

  // Compute exact 16x16 inverse (I - A)^{-1} = (I + A)(I + A^2)(I + A^4)(I + A^8) purely in L1/L0!
  // l1A holds 64x64 (pitch 64) with strict lower A_a at (base, base).
  // l1Eye holds 16x16 (pitch 16) BF16 identity. l1Work holds 4 x 512B scratch (2048B, pitch 16).
  __aicore__ inline void InvertDiag16InL1(LocalTensor<bfloat16_t> l1A, uint32_t base,
                                          LocalTensor<bfloat16_t> l1Eye, LocalTensor<bfloat16_t> l1Work) {
    auto A = L0A();
    auto B = L0B();
    auto C = L0C();
    auto wB1 = l1Work[0];
    auto wP2 = l1Work[256];
    auto wB2 = l1Work[512];
    auto wP4 = l1Work[768];

    // 1. B1 = A + I (using C = A @ I + I @ I)
    LoadL0A_NoT(A, l1A, 16, 16, 64, base, base);
    LoadL0B_NoT(B, l1Eye, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    LoadL0A_NoT(A, l1Eye, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, false);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(wB1, C, 16, 16, 16, 0, 0);
    WaitFixToM();
    WaitFixToMte1();

    // 2. P2 = A @ A
    LoadL0A_NoT(A, l1A, 16, 16, 64, base, base);
    LoadL0B_NoT(B, l1A, 16, 16, 64, base, base);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(wP2, C, 16, 16, 16, 0, 0);
    WaitFixToM();
    WaitFixToMte1();

    // 3. B2 = B1 @ P2 + B1 @ I = (I + A)(I + A^2)
    LoadL0A_NoT(A, wB1, 16, 16, 16, 0, 0);
    LoadL0B_NoT(B, wP2, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    LoadL0B_NoT(B, l1Eye, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, false);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(wB2, C, 16, 16, 16, 0, 0);
    WaitFixToM();
    WaitFixToMte1();

    // 4. P4 = P2 @ P2
    LoadL0A_NoT(A, wP2, 16, 16, 16, 0, 0);
    LoadL0B_NoT(B, wP2, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(wP4, C, 16, 16, 16, 0, 0);
    WaitFixToM();
    WaitFixToMte1();

    // 5. B4 = B2 @ P4 + B2 @ I -> wB1
    LoadL0A_NoT(A, wB2, 16, 16, 16, 0, 0);
    LoadL0B_NoT(B, wP4, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    LoadL0B_NoT(B, l1Eye, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, false);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(wB1, C, 16, 16, 16, 0, 0);
    WaitFixToM();
    WaitFixToMte1();

    // 6. P8 = P4 @ P4 -> wP2
    LoadL0A_NoT(A, wP4, 16, 16, 16, 0, 0);
    LoadL0B_NoT(B, wP4, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(wP2, C, 16, 16, 16, 0, 0);
    WaitFixToM();
    WaitFixToMte1();

    // 7. B8 = B4 @ P8 + B4 @ I -> overwrite l1A(base, base)!
    LoadL0A_NoT(A, wB1, 16, 16, 16, 0, 0);
    LoadL0B_NoT(B, wP2, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, true);
    WaitMToMte1();
    LoadL0B_NoT(B, l1Eye, 16, 16, 16, 0, 0);
    WaitMte1ToM();
    RunMmad(C, A, B, 16, 16, 16, false);
    WaitMToMte1();
    WaitMToFix();
    StoreBf16L1Nz(l1A, C, 16, 16, 64, base, base);
    WaitFixToM();
    WaitFixToMte1();
  }
};


// Ultra-fast Raw BF16 Linear GEMM: C[M, N] (row stride dstStride) = A[M, K] @ B[N, K]^T
// Path 1 (L1-Resident Weights): When M >= 256 and K in [512..4096], loads B[nBlk, K] (<= 256 KB)
//   ONCE into L1 SRAM and streams A[mBlk, 64] across all M/mBlk tiles with ZERO weight HBM re-reads!
//   Cuts Q/K/V/O HBM traffic by 16x-20x!
// Path 2 (Triple-Overlap 128x128x128): Used for small M, small N (b_proj), or K=128 (f_b/g_b).
__aicore__ inline void RunRawLinearBf16(__gm__ bfloat16_t *cPtr, __gm__ bfloat16_t *aPtr,
                                        __gm__ bfloat16_t *bPtr, uint32_t M, uint32_t N, uint32_t K,
                                        uint32_t dstStride = 0) {
  if (dstStride == 0U) dstStride = N;

  // Path 1: L1-Resident B (nBlk=64 for K<=2048, nBlk=32 for K<=4096; mBlk=256 if M%256==0 else 128)
  const uint32_t l1NBlk = (K <= 2048U && (N & 63U) == 0U) ? 64U
                        : (K <= 4096U && (N & 31U) == 0U) ? 32U : 0U;
  if (l1NBlk > 0U && M >= 256U && (M & 127U) == 0U && K >= 512U && (K & 63U) == 0U) {
    const uint32_t mBlk = ((M & 255U) == 0U) ? 256U : 128U;
    const uint32_t kBlk = 64U;
    const uint32_t kSteps = K / kBlk;
    const uint32_t aTileBytes = mBlk * kBlk * 2U;        // <= 32 KB
    const uint32_t bChunkBytes = l1NBlk * kBlk * 2U;     // <= 8 KB
    const uint32_t bAllBytes = l1NBlk * K * 2U;          // <= 256 KB
    const uint32_t cTileBytes = mBlk * l1NBlk * 4U;      // <= 64 KB

    TPipe pipe;
    TBuf<TPosition::A1> l1Buf;
    TBuf<TPosition::A2> l0aBuf;
    TBuf<TPosition::B2> l0bBuf;
    TBuf<TPosition::CO1> l0cBuf;
    pipe.InitBuffer(l1Buf, bAllBytes + 2U * aTileBytes);  // <= 320 KB (< 512 KB L1!)
    pipe.InitBuffer(l0aBuf, 2U * aTileBytes);             // <= 64 KB
    pipe.InitBuffer(l0bBuf, 2U * bChunkBytes);            // <= 16 KB
    pipe.InitBuffer(l0cBuf, 2U * cTileBytes);             // <= 128 KB

    LocalTensor<bfloat16_t> l1BAll = l1Buf.GetWithOffset<bfloat16_t>(bAllBytes / 2U, 0);
    LocalTensor<bfloat16_t> l1A[2] = {
        l1Buf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, bAllBytes),
        l1Buf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, bAllBytes + aTileBytes)};
    LocalTensor<bfloat16_t> l0a[2] = {
        l0aBuf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, 0),
        l0aBuf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, aTileBytes)};
    LocalTensor<bfloat16_t> l0b[2] = {
        l0bBuf.GetWithOffset<bfloat16_t>(bChunkBytes / 2U, 0),
        l0bBuf.GetWithOffset<bfloat16_t>(bChunkBytes / 2U, bChunkBytes)};
    LocalTensor<float> l0c[2] = {
        l0cBuf.GetWithOffset<float>(cTileBytes / 4U, 0),
        l0cBuf.GetWithOffset<float>(cTileBytes / 4U, cTileBytes)};

    TEventID e21[2] = {pipe.AllocEventID<HardEvent::MTE2_MTE1>(), pipe.AllocEventID<HardEvent::MTE2_MTE1>()};
    TEventID e12[2] = {pipe.AllocEventID<HardEvent::MTE1_MTE2>(), pipe.AllocEventID<HardEvent::MTE1_MTE2>()};
    TEventID e1m[2] = {pipe.AllocEventID<HardEvent::MTE1_M>(), pipe.AllocEventID<HardEvent::MTE1_M>()};
    TEventID em1[2] = {pipe.AllocEventID<HardEvent::M_MTE1>(), pipe.AllocEventID<HardEvent::M_MTE1>()};
    TEventID emf[2] = {pipe.AllocEventID<HardEvent::M_FIX>(), pipe.AllocEventID<HardEvent::M_FIX>()};
    TEventID efm[2] = {pipe.AllocEventID<HardEvent::FIX_M>(), pipe.AllocEventID<HardEvent::FIX_M>()};
    TEventID eBLoad = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID eBFree = pipe.AllocEventID<HardEvent::MTE1_MTE2>();

    SetFlag<HardEvent::MTE1_MTE2>(e12[0]);
    SetFlag<HardEvent::MTE1_MTE2>(e12[1]);
    SetFlag<HardEvent::M_MTE1>(em1[0]);
    SetFlag<HardEvent::M_MTE1>(em1[1]);
    SetFlag<HardEvent::FIX_M>(efm[0]);
    SetFlag<HardEvent::FIX_M>(efm[1]);
    SetFlag<HardEvent::MTE1_MTE2>(eBFree);

    uint32_t tileIdx = 0;
    uint32_t stepIdx = 0;
    for (uint32_t n0 = 0; n0 < N; n0 += l1NBlk) {
      // Load ENTIRE weight strip B[n0..n0+l1NBlk-1, 0..K-1] into L1 ONCE!
      WaitFlag<HardEvent::MTE1_MTE2>(eBFree);
      for (uint32_t ks = 0; ks < kSteps; ++ks) {
        GlobalTensor<bfloat16_t> bGm;
        bGm.SetGlobalBuffer(bPtr + static_cast<uint64_t>(n0) * K + ks * kBlk, static_cast<uint64_t>(l1NBlk) * K);
        Nd2NzParams pb{};
        pb.ndNum = 1;
        pb.nValue = static_cast<uint16_t>(l1NBlk);
        pb.dValue = static_cast<uint16_t>(kBlk);
        pb.srcDValue = static_cast<uint16_t>(K);
        pb.dstNzNStride = 1;
        pb.dstNzC0Stride = static_cast<uint16_t>(l1NBlk);
        DataCopy(l1BAll[ks * l1NBlk * kBlk], bGm, pb);
      }
      SetFlag<HardEvent::MTE2_MTE1>(eBLoad);
      WaitFlag<HardEvent::MTE2_MTE1>(eBLoad);

      for (uint32_t m0 = 0; m0 < M; m0 += mBlk) {
        const uint32_t cBuf = tileIdx & 1U;
        ++tileIdx;

        // Prologue: prefetch A[m0, 0..63]
        const uint32_t firstBuf = stepIdx & 1U;
        {
          GlobalTensor<bfloat16_t> aGm;
          aGm.SetGlobalBuffer(aPtr + static_cast<uint64_t>(m0) * K, static_cast<uint64_t>(mBlk) * K);
          WaitFlag<HardEvent::MTE1_MTE2>(e12[firstBuf]);
          Nd2NzParams pa{};
          pa.ndNum = 1;
          pa.nValue = static_cast<uint16_t>(mBlk);
          pa.dValue = static_cast<uint16_t>(kBlk);
          pa.srcDValue = static_cast<uint16_t>(K);
          pa.dstNzNStride = 1;
          pa.dstNzC0Stride = static_cast<uint16_t>(mBlk);
          DataCopy(l1A[firstBuf], aGm, pa);
          SetFlag<HardEvent::MTE2_MTE1>(e21[firstBuf]);
        }

        for (uint32_t ks = 0; ks < kSteps; ++ks) {
          const uint32_t cur = stepIdx & 1U;
          const uint32_t nxt = cur ^ 1U;
          ++stepIdx;

          // 1. Prefetch NEXT A chunk (ZERO MTE2 for B!)
          if (ks + 1U < kSteps) {
            GlobalTensor<bfloat16_t> aGm;
            aGm.SetGlobalBuffer(aPtr + static_cast<uint64_t>(m0) * K + (ks + 1U) * kBlk,
                                static_cast<uint64_t>(mBlk) * K);
            WaitFlag<HardEvent::MTE1_MTE2>(e12[nxt]);
            Nd2NzParams pa{};
            pa.ndNum = 1;
            pa.nValue = static_cast<uint16_t>(mBlk);
            pa.dValue = static_cast<uint16_t>(kBlk);
            pa.srcDValue = static_cast<uint16_t>(K);
            pa.dstNzNStride = 1;
            pa.dstNzC0Stride = static_cast<uint16_t>(mBlk);
            DataCopy(l1A[nxt], aGm, pa);
            SetFlag<HardEvent::MTE2_MTE1>(e21[nxt]);
          }

          // 2. MTE1: l1A[cur] -> l0a[cur], l1BAll[ks] -> l0b[cur]
          WaitFlag<HardEvent::MTE2_MTE1>(e21[cur]);
          WaitFlag<HardEvent::M_MTE1>(em1[cur]);
          {
            LoadData2DParams la{};
            la.repeatTimes = static_cast<uint8_t>(kBlk / 16U);
            la.srcStride = static_cast<uint16_t>(mBlk / 16U);
            la.ifTranspose = false;
            for (uint32_t r = 0; r < mBlk / 16U; ++r) {
              LoadData(l0a[cur][r * (kBlk / 16U) * 256U], l1A[cur][r * 256U], la);
            }
            LoadData2DParams lb{};
            lb.repeatTimes = static_cast<uint8_t>((kBlk / 16U) * (l1NBlk / 16U));
            lb.srcStride = 1;
            lb.ifTranspose = false;
            LoadData(l0b[cur], l1BAll[ks * l1NBlk * kBlk], lb);
          }
          SetFlag<HardEvent::MTE1_MTE2>(e12[cur]);
          SetFlag<HardEvent::MTE1_M>(e1m[cur]);

          // 3. Cube Mmad
          if (ks == 0U) {
            WaitFlag<HardEvent::FIX_M>(efm[cBuf]);
          }
          WaitFlag<HardEvent::MTE1_M>(e1m[cur]);
          {
            MmadParams mm{};
            mm.m = static_cast<uint16_t>(mBlk);
            mm.n = static_cast<uint16_t>(l1NBlk);
            mm.k = static_cast<uint16_t>(kBlk);
            mm.cmatrixInitVal = (ks == 0U);
            Mmad(l0c[cBuf], l0a[cur], l0b[cur], mm);
          }
          SetFlag<HardEvent::M_MTE1>(em1[cur]);
        }

        // 4. Async Fixpipe L0C[cBuf] -> GM (row stride = dstStride!)
        SetFlag<HardEvent::M_FIX>(emf[cBuf]);
        WaitFlag<HardEvent::M_FIX>(emf[cBuf]);
        {
          GlobalTensor<bfloat16_t> cGm;
          cGm.SetGlobalBuffer(cPtr + static_cast<uint64_t>(m0) * dstStride + n0,
                              static_cast<uint64_t>(mBlk) * dstStride);
          auto fp = FixpipeParamsV220(static_cast<uint16_t>(l1NBlk), static_cast<uint16_t>(mBlk),
                                      static_cast<uint16_t>(mBlk), static_cast<uint32_t>(dstStride), false);
          fp.quantPre = QuantMode_t::F322BF16;
          Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(cGm, l0c[cBuf], fp);
        }
        SetFlag<HardEvent::FIX_M>(efm[cBuf]);
      }
      SetFlag<HardEvent::MTE1_MTE2>(eBFree);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(eBFree);
    WaitFlag<HardEvent::MTE1_MTE2>(e12[0]);
    WaitFlag<HardEvent::MTE1_MTE2>(e12[1]);
    WaitFlag<HardEvent::M_MTE1>(em1[0]);
    WaitFlag<HardEvent::M_MTE1>(em1[1]);
    WaitFlag<HardEvent::FIX_M>(efm[0]);
    WaitFlag<HardEvent::FIX_M>(efm[1]);

    pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(e21[0]);
    pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(e21[1]);
    pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(e12[0]);
    pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(e12[1]);
    pipe.ReleaseEventID<HardEvent::MTE1_M>(e1m[0]);
    pipe.ReleaseEventID<HardEvent::MTE1_M>(e1m[1]);
    pipe.ReleaseEventID<HardEvent::M_MTE1>(em1[0]);
    pipe.ReleaseEventID<HardEvent::M_MTE1>(em1[1]);
    pipe.ReleaseEventID<HardEvent::M_FIX>(emf[0]);
    pipe.ReleaseEventID<HardEvent::M_FIX>(emf[1]);
    pipe.ReleaseEventID<HardEvent::FIX_M>(efm[0]);
    pipe.ReleaseEventID<HardEvent::FIX_M>(efm[1]);
    pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(eBLoad);
    pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(eBFree);
    pipe.Destroy();
    return;
  }

  // Path 2: Triple-Overlap 128x128x128 (small M, small N, or K=128)
  const uint32_t kBlk = ((K & 127U) == 0U) ? 128U : 64U;
  const uint32_t mBlk = (M >= 128U) ? 128U : M;
  uint32_t nBlk = 128U;
  while (nBlk > 16U && (N % nBlk != 0U || nBlk > N)) {
    nBlk >>= 1;
  }

  const uint32_t aTileBytes = mBlk * kBlk * 2U;
  const uint32_t bTileBytes = nBlk * kBlk * 2U;
  const uint32_t l1SlotBytes = aTileBytes + bTileBytes;
  const uint32_t cTileBytes = mBlk * nBlk * 4U;

  TPipe pipe;
  TBuf<TPosition::A1> l1Buf;
  TBuf<TPosition::A2> l0aBuf;
  TBuf<TPosition::B2> l0bBuf;
  TBuf<TPosition::CO1> l0cBuf;
  pipe.InitBuffer(l1Buf, 2U * l1SlotBytes);
  pipe.InitBuffer(l0aBuf, 2U * aTileBytes);
  pipe.InitBuffer(l0bBuf, 2U * bTileBytes);
  pipe.InitBuffer(l0cBuf, 2U * cTileBytes);

  LocalTensor<bfloat16_t> l1A[2] = {
      l1Buf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, 0),
      l1Buf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, l1SlotBytes)};
  LocalTensor<bfloat16_t> l1B[2] = {
      l1Buf.GetWithOffset<bfloat16_t>(bTileBytes / 2U, aTileBytes),
      l1Buf.GetWithOffset<bfloat16_t>(bTileBytes / 2U, l1SlotBytes + aTileBytes)};
  LocalTensor<bfloat16_t> l0a[2] = {
      l0aBuf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, 0),
      l0aBuf.GetWithOffset<bfloat16_t>(aTileBytes / 2U, aTileBytes)};
  LocalTensor<bfloat16_t> l0b[2] = {
      l0bBuf.GetWithOffset<bfloat16_t>(bTileBytes / 2U, 0),
      l0bBuf.GetWithOffset<bfloat16_t>(bTileBytes / 2U, bTileBytes)};
  LocalTensor<float> l0c[2] = {
      l0cBuf.GetWithOffset<float>(cTileBytes / 4U, 0),
      l0cBuf.GetWithOffset<float>(cTileBytes / 4U, cTileBytes)};

  TEventID e21[2] = {pipe.AllocEventID<HardEvent::MTE2_MTE1>(), pipe.AllocEventID<HardEvent::MTE2_MTE1>()};
  TEventID e12[2] = {pipe.AllocEventID<HardEvent::MTE1_MTE2>(), pipe.AllocEventID<HardEvent::MTE1_MTE2>()};
  TEventID e1m[2] = {pipe.AllocEventID<HardEvent::MTE1_M>(), pipe.AllocEventID<HardEvent::MTE1_M>()};
  TEventID em1[2] = {pipe.AllocEventID<HardEvent::M_MTE1>(), pipe.AllocEventID<HardEvent::M_MTE1>()};
  TEventID emf[2] = {pipe.AllocEventID<HardEvent::M_FIX>(), pipe.AllocEventID<HardEvent::M_FIX>()};
  TEventID efm[2] = {pipe.AllocEventID<HardEvent::FIX_M>(), pipe.AllocEventID<HardEvent::FIX_M>()};

  SetFlag<HardEvent::MTE1_MTE2>(e12[0]);
  SetFlag<HardEvent::MTE1_MTE2>(e12[1]);
  SetFlag<HardEvent::M_MTE1>(em1[0]);
  SetFlag<HardEvent::M_MTE1>(em1[1]);
  SetFlag<HardEvent::FIX_M>(efm[0]);
  SetFlag<HardEvent::FIX_M>(efm[1]);

  const uint32_t kSteps = K / kBlk;
  uint32_t tileIdx = 0;
  uint32_t stepIdx = 0;
  for (uint32_t m0 = 0; m0 < M; m0 += mBlk) {
    const uint32_t curM = (m0 + mBlk <= M) ? mBlk : (M - m0);
    for (uint32_t n0 = 0; n0 < N; n0 += nBlk) {
      const uint32_t curN = (n0 + nBlk <= N) ? nBlk : (N - n0);
      const uint32_t cBuf = tileIdx & 1U;
      ++tileIdx;

      const uint32_t firstBuf = stepIdx & 1U;
      {
        GlobalTensor<bfloat16_t> aGm, bGm;
        aGm.SetGlobalBuffer(aPtr + static_cast<uint64_t>(m0) * K, static_cast<uint64_t>(curM) * K);
        bGm.SetGlobalBuffer(bPtr + static_cast<uint64_t>(n0) * K, static_cast<uint64_t>(curN) * K);
        WaitFlag<HardEvent::MTE1_MTE2>(e12[firstBuf]);
        Nd2NzParams pa{};
        pa.ndNum = 1;
        pa.nValue = static_cast<uint16_t>(curM);
        pa.dValue = static_cast<uint16_t>(kBlk);
        pa.srcDValue = static_cast<uint16_t>(K);
        pa.dstNzNStride = 1;
        pa.dstNzC0Stride = static_cast<uint16_t>(curM);
        DataCopy(l1A[firstBuf], aGm, pa);

        Nd2NzParams pb{};
        pb.ndNum = 1;
        pb.nValue = static_cast<uint16_t>(curN);
        pb.dValue = static_cast<uint16_t>(kBlk);
        pb.srcDValue = static_cast<uint16_t>(K);
        pb.dstNzNStride = 1;
        pb.dstNzC0Stride = static_cast<uint16_t>(curN);
        DataCopy(l1B[firstBuf], bGm, pb);
        SetFlag<HardEvent::MTE2_MTE1>(e21[firstBuf]);
      }

      for (uint32_t ks = 0; ks < kSteps; ++ks) {
        const uint32_t cur = stepIdx & 1U;
        const uint32_t nxt = cur ^ 1U;
        ++stepIdx;

        if (ks + 1U < kSteps) {
          const uint32_t kNext = (ks + 1U) * kBlk;
          GlobalTensor<bfloat16_t> aGm, bGm;
          aGm.SetGlobalBuffer(aPtr + static_cast<uint64_t>(m0) * K + kNext, static_cast<uint64_t>(curM) * K);
          bGm.SetGlobalBuffer(bPtr + static_cast<uint64_t>(n0) * K + kNext, static_cast<uint64_t>(curN) * K);
          WaitFlag<HardEvent::MTE1_MTE2>(e12[nxt]);
          Nd2NzParams pa{};
          pa.ndNum = 1;
          pa.nValue = static_cast<uint16_t>(curM);
          pa.dValue = static_cast<uint16_t>(kBlk);
          pa.srcDValue = static_cast<uint16_t>(K);
          pa.dstNzNStride = 1;
          pa.dstNzC0Stride = static_cast<uint16_t>(curM);
          DataCopy(l1A[nxt], aGm, pa);

          Nd2NzParams pb{};
          pb.ndNum = 1;
          pb.nValue = static_cast<uint16_t>(curN);
          pb.dValue = static_cast<uint16_t>(kBlk);
          pb.srcDValue = static_cast<uint16_t>(K);
          pb.dstNzNStride = 1;
          pb.dstNzC0Stride = static_cast<uint16_t>(curN);
          DataCopy(l1B[nxt], bGm, pb);
          SetFlag<HardEvent::MTE2_MTE1>(e21[nxt]);
        }

        WaitFlag<HardEvent::MTE2_MTE1>(e21[cur]);
        WaitFlag<HardEvent::M_MTE1>(em1[cur]);
        {
          LoadData2DParams la{};
          la.repeatTimes = static_cast<uint8_t>(kBlk / 16U);
          la.srcStride = static_cast<uint16_t>(curM / 16U);
          la.ifTranspose = false;
          for (uint32_t r = 0; r < curM / 16U; ++r) {
            LoadData(l0a[cur][r * (kBlk / 16U) * 256U], l1A[cur][r * 256U], la);
          }

          LoadData2DParams lb{};
          lb.repeatTimes = static_cast<uint8_t>((kBlk / 16U) * (curN / 16U));
          lb.srcStride = 1;
          lb.ifTranspose = false;
          LoadData(l0b[cur], l1B[cur], lb);
        }
        SetFlag<HardEvent::MTE1_MTE2>(e12[cur]);
        SetFlag<HardEvent::MTE1_M>(e1m[cur]);

        if (ks == 0U) {
          WaitFlag<HardEvent::FIX_M>(efm[cBuf]);
        }
        WaitFlag<HardEvent::MTE1_M>(e1m[cur]);
        {
          MmadParams mm{};
          mm.m = static_cast<uint16_t>(curM);
          mm.n = static_cast<uint16_t>(curN);
          mm.k = static_cast<uint16_t>(kBlk);
          mm.cmatrixInitVal = (ks == 0U);
          Mmad(l0c[cBuf], l0a[cur], l0b[cur], mm);
        }
        SetFlag<HardEvent::M_MTE1>(em1[cur]);
      }

      SetFlag<HardEvent::M_FIX>(emf[cBuf]);
      WaitFlag<HardEvent::M_FIX>(emf[cBuf]);
      {
        GlobalTensor<bfloat16_t> cGm;
        cGm.SetGlobalBuffer(cPtr + static_cast<uint64_t>(m0) * dstStride + n0,
                            static_cast<uint64_t>(curM) * dstStride);
        auto fp = FixpipeParamsV220(static_cast<uint16_t>(curN), static_cast<uint16_t>(curM),
                                    static_cast<uint16_t>(curM), static_cast<uint32_t>(dstStride), false);
        fp.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(cGm, l0c[cBuf], fp);
      }
      SetFlag<HardEvent::FIX_M>(efm[cBuf]);
    }
  }

  WaitFlag<HardEvent::MTE1_MTE2>(e12[0]);
  WaitFlag<HardEvent::MTE1_MTE2>(e12[1]);
  WaitFlag<HardEvent::M_MTE1>(em1[0]);
  WaitFlag<HardEvent::M_MTE1>(em1[1]);
  WaitFlag<HardEvent::FIX_M>(efm[0]);
  WaitFlag<HardEvent::FIX_M>(efm[1]);

  pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(e21[0]);
  pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(e21[1]);
  pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(e12[0]);
  pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(e12[1]);
  pipe.ReleaseEventID<HardEvent::MTE1_M>(e1m[0]);
  pipe.ReleaseEventID<HardEvent::MTE1_M>(e1m[1]);
  pipe.ReleaseEventID<HardEvent::M_MTE1>(em1[0]);
  pipe.ReleaseEventID<HardEvent::M_MTE1>(em1[1]);
  pipe.ReleaseEventID<HardEvent::M_FIX>(emf[0]);
  pipe.ReleaseEventID<HardEvent::M_FIX>(emf[1]);
  pipe.ReleaseEventID<HardEvent::FIX_M>(efm[0]);
  pipe.ReleaseEventID<HardEvent::FIX_M>(efm[1]);
  pipe.Destroy();
}

}  // namespace MulticoreKda

#endif  // HYPER_MEGA_KDA_CUBE_RAW_H

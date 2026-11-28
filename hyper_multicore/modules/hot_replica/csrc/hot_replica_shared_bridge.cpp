// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Bind expert transfers to the same SHMEM library used by MegaMoE.
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "shmem.h"   // NOLINT(build/include_subdir)
#include "csrc/launch.h"

extern "C" int hmc_hot_replica_grad_return(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks, int32_t borrowed,
    bool native_home, bool bf16, bool publish, bool finish);

static void check_runtime(int64_t rank, int64_t ranks) {
    TORCH_CHECK(rank == aclshmem_my_pe(), "expert-transfer rank differs from active MegaMoE SHMEM PE");
    TORCH_CHECK(ranks == aclshmem_n_pes(), "expert-transfer world differs from active MegaMoE SHMEM root");
}

static void prefetch(torch::Tensor home, torch::Tensor slots, torch::Tensor copies,
                     int64_t rank, int64_t ranks, int64_t blocks, bool publish, bool finish) {
    check_runtime(rank, ranks);
    TORCH_CHECK(home.is_contiguous() && slots.is_contiguous() && copies.is_contiguous(),
                "expert transfer requires contiguous tensors");
    TORCH_CHECK(home.scalar_type() == at::kBFloat16 && slots.scalar_type() == at::kBFloat16,
                "prefetch weights must be BF16");
    TORCH_CHECK(copies.scalar_type() == at::kInt && copies.dim() == 1,
                "prefetch copies must be INT32 slots");
    TORCH_CHECK(home.dim() == 3 && slots.dim() == 3 && home.size(0) % 2 == 0
                    && home.size(0) / 2 >= slots.size(0),
                "prefetch home/slot geometry changed");
    TORCH_CHECK(home.size(1) == slots.size(1) && home.size(2) == slots.size(2)
                    && copies.numel() == slots.size(0),
                "prefetch projection shape changed");
    // Match prefetch_weight_kernel's reserved SHMEM region and double-buffer
    // geometry. Keep this guard on the host so unsupported shapes fail before
    // the kernel's collective barriers, rather than silently skipping a copy.
    constexpr int64_t ub_limit_bytes = 188 * 1024;
    constexpr int64_t scratch_offset = 64;
    constexpr int64_t alignment = 32;
    constexpr int64_t buffers = 2;
    const int64_t slot_count = slots.size(0);
    TORCH_CHECK(slot_count > 0 && slot_count <= (ub_limit_bytes - scratch_offset) / (2 * sizeof(int32_t)),
                "HotReplica prefetch UB budget cannot hold the expert copy table");
    const int64_t table_end = scratch_offset + 2 * slot_count * sizeof(int32_t);
    const int64_t data_offset = (table_end + alignment - 1) / alignment * alignment;
    const int64_t remaining_bytes = ub_limit_bytes - data_offset;
    TORCH_CHECK(remaining_bytes > 0, "HotReplica prefetch UB budget has no staging space");
    const int64_t buffer_bytes = remaining_bytes / buffers / alignment * alignment;
    TORCH_CHECK(home.size(1) > 0 && home.size(2) > 0,
                "HotReplica prefetch projection dimensions must be positive");
    TORCH_CHECK(buffer_bytes / home.element_size() / home.size(2) > 0,
                "HotReplica prefetch UB budget cannot hold one BF16 projection row: width=", home.size(2),
                ", buffer_bytes=", buffer_bytes);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // OpCommand may execute after this binding returns; own the tensor handles.
    auto run = [=]() -> int {
        // The device kernel publishes home writes and completes all peer reads.
        return hmc_hot_replica_prefetch_weight_kernel_do(stream, static_cast<uint8_t*>(home.data_ptr()),
            static_cast<uint8_t*>(slots.data_ptr()), static_cast<uint8_t*>(copies.data_ptr()),
            ranks*(home.size(0)/2), home.size(1), home.size(2), slots.size(0), blocks, publish, finish);
    };
    at_npu::native::OpCommand::RunOpApi("HyperMulticoreHotReplicaPrefetch", run);
}

static void reduce(torch::Tensor source, torch::Tensor home, torch::Tensor slots,
                   torch::Tensor copies, int64_t rank, int64_t blocks, bool publish, bool finish) {
    check_runtime(rank, copies.size(0));
    TORCH_CHECK(source.is_contiguous() && home.is_contiguous() && slots.is_contiguous()
                    && copies.is_contiguous(),
                "gradient transfer requires contiguous tensors");
    const bool native_home = source.scalar_type() == at::kBFloat16;
    if (native_home) {
        TORCH_CHECK(home.scalar_type() == at::kBFloat16,
                    "native BF16 local weight gradients require BF16 output");
    } else {
        TORCH_CHECK(source.scalar_type() == at::kFloat
                        && (home.scalar_type() == at::kFloat || home.scalar_type() == at::kBFloat16),
                    "FP32 owner reduction requires FP32 inputs and FP32 or BF16 outputs");
    }
    TORCH_CHECK(slots.scalar_type() == at::kFloat, "replica weight gradients must remain FP32");
    TORCH_CHECK(copies.scalar_type() == at::kInt && copies.dim() == 2,
                "gradient copy plan must be INT32 [root, slots]");
    TORCH_CHECK(home.sizes() == source.sizes() && home.dim() == 3 && slots.dim() == 3
                    && slots.size(0) == copies.size(1) && slots.size(0) <= home.size(0)
                    && slots.size(1) == home.size(1) && slots.size(2) == home.size(2),
                "gradient transfer shape changed");
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        return hmc_hot_replica_grad_return(stream, source.data_ptr(), home.data_ptr(), slots.data_ptr(),
            copies.data_ptr<int32_t>(), home.size(0), home.size(1), home.size(2),
            copies.size(0), rank, blocks, copies.size(1), native_home, home.scalar_type() == at::kBFloat16,
            publish, finish);
    };
    at_npu::native::OpCommand::RunOpApi("HyperMulticoreHotReplicaGradReduce", run);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prefetch", &prefetch, pybind11::arg("home"), pybind11::arg("slots"),
               pybind11::arg("copies"), pybind11::arg("rank"), pybind11::arg("ranks"),
               pybind11::arg("blocks"),
               pybind11::arg("publish") = true, pybind11::arg("finish") = true);
    module.def("reduce", &reduce, pybind11::arg("source"), pybind11::arg("home"),
               pybind11::arg("slots"), pybind11::arg("copies"), pybind11::arg("rank"),
               pybind11::arg("blocks"),
               pybind11::arg("publish") = true, pybind11::arg("finish") = true);
    module.attr("native_home_gradient") = true;
    module.def("runtime_state", []() { return std::vector<int>{aclshmem_my_pe(), aclshmem_n_pes()}; });
}

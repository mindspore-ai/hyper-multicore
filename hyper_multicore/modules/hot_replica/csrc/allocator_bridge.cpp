// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "csrc/allocator_launch.h"
static inline int64_t aligned(int64_t n) { return (n + 7) / 8 * 8; }
constexpr int32_t kPlannerMaxPage = 1024;
std::vector<torch::Tensor> allocate(torch::Tensor counts, int64_t capacity, int64_t ub_bytes) {
    TORCH_CHECK(counts.dim() == 2 && counts.scalar_type() == at::kInt && counts.is_contiguous(),
                "counts must be contiguous int32 [R, E]");
    int64_t R = counts.size(0), E = counts.size(1);
    TORCH_CHECK(R > 0 && E > 0 && E % R == 0 && capacity > 0, "invalid EP geometry");
    int64_t B = E / R;
    int64_t elements = 4 * aligned(R*E) + 3 * aligned(E) + 2 * aligned(R) + aligned(R*R)
        + aligned(R*(E+B)) + aligned(R*B);
    const int64_t budget = ub_bytes * 3 / 4 / 4;
    const int64_t sizes[] = {R*E, E, R, R*R, R*E, E, R, R*E, R*E, R*(E+B), R*B, E};
    uint32_t resident_mask = 0;
    const int64_t page_size = std::min<int64_t>(kPlannerMaxPage, budget / 12 / 8 * 8);
    TORCH_CHECK(page_size >= 8, "UB is too small for planner pages");
    int64_t local_elements = 12 * page_size;
    for (int i = 0; i < 12; ++i) {
        if (aligned(sizes[i]) <= page_size) {
            resident_mask |= 1U << i;
            local_elements += aligned(sizes[i]) - page_size;
        }
    }
    // Keep frequently revisited allocation and transfer tables resident first.
    const int order[] = {4, 3, 1, 2, 5, 6, 11, 10, 0, 7, 8, 9};
    for (int i : order) {
        if (resident_mask & (1U << i)) continue;
        int64_t extra = aligned(sizes[i]) - page_size;
        if (local_elements + extra <= budget) {
            resident_mask |= 1U << i;
            local_elements += extra;
        }
    }
    auto workspace = torch::empty({elements}, counts.options());
    auto cumulative = torch::empty({E, R}, counts.options());
    auto offsets = torch::empty({R, E}, counts.options());
    auto boundaries = torch::empty({R, E+B}, counts.options());
    auto copies = torch::empty({R, B}, counts.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        return hmc_hot_replica_allocate(stream, counts.data_ptr(), cumulative.data_ptr(),
            offsets.data_ptr(), boundaries.data_ptr(), copies.data_ptr(), E, R, capacity,
            local_elements * 4, workspace.data_ptr(), resident_mask, page_size);
    };
    at_npu::native::OpCommand::RunOpApi("HotReplicaA3Allocate", run);
    if (!(resident_mask & (1U << 4)))
        cumulative = cumulative.view({R, E}).cumsum(0, at::kInt).t().contiguous();
    return {cumulative, offsets, boundaries, copies};
}
std::vector<torch::Tensor> render(torch::Tensor gathered, torch::Tensor cumulative,
                                torch::Tensor copies, int64_t rank, int64_t width) {
    const int64_t p = gathered.size(0), e = gathered.size(1);
    auto counts = torch::empty({p, p * width}, gathered.options());
    auto repeats = torch::empty({e * p}, gathered.options());
    auto labels = torch::empty({e * p}, gathered.options());
    int64_t group = 1;
    while ((group * p * width) % 32 != 0) ++group;
    const int64_t bytes = (5 * aligned(p * e) + aligned(e) + aligned(p * width)) * 4;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        return hmc_hot_replica_render(stream, gathered.data_ptr(), cumulative.data_ptr(),
            copies.data_ptr(), counts.data_ptr(), repeats.data_ptr(), labels.data_ptr(),
            p, e, rank, width, group, bytes);
    };
    at_npu::native::OpCommand::RunOpApi("HotReplicaVirtualCounts", run);
    return {counts, repeats, labels};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allocate", &allocate);
    m.def("render", &render);
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#pragma once
#include "csrc/cached_op_api.h"
#include <ATen/record_function.h>

namespace hyper_multicore::mxfp8_binding {
// Per-invocation binding only. Native PTA retains its stream, allocator,
// executor cache and queue behavior; no tensor address survives this call.
struct ByteTensorList {
  at::TensorList tensors;
  mutable std::vector<TensorStructPtr> descriptors;
};

inline void FlattenBytes(const at::Tensor& tensor, TensorStruct& descriptor) {
  TORCH_CHECK(tensor.is_contiguous(), "MXFP8 task buffers must be contiguous; implicit copies are forbidden");
  const auto item_size = tensor.element_size();
  TORCH_CHECK(tensor.numel() <= INT64_MAX / item_size &&
              tensor.storage_offset() <= INT64_MAX / item_size,
              "MXFP8 byte extent exceeds int64 bounds");
  const int64_t bytes = tensor.numel() * item_size;
  const int64_t offset = tensor.storage_offset() * item_size;
  TORCH_CHECK(offset >= 0 && static_cast<uint64_t>(offset) <= descriptor.nbytes &&
              static_cast<uint64_t>(bytes) <= descriptor.nbytes - offset,
              "MXFP8 byte view exceeds its underlying storage");
  descriptor.acl_type = ACL_UINT8;
  descriptor.acl_format = ACL_FORMAT_ND;
  descriptor.itemsize = 1;
  descriptor.storage_offset = offset;
  descriptor.sizes.assign(1, bytes);
  descriptor.strides.assign(1, 1);
  descriptor.storage_sizes.assign(1, descriptor.nbytes);
}

inline const std::vector<TensorStructPtr>& Descriptors(const ByteTensorList& buffers) {
  if (buffers.descriptors.empty()) {
    RECORD_FUNCTION("hmc_prepare/mxfp8/byte_descriptors", c10::ArrayRef<c10::IValue>{});
    // Native conversion keeps the real storage address and allocator handling.
    // Only its logical descriptor changes, exactly like view(uint8).reshape(-1).
    buffers.descriptors = ::CopyTypeV2(buffers.tensors);
    for (size_t i = 0; i < buffers.tensors.size(); ++i) {
      auto& descriptor = buffers.descriptors[i];
      if (descriptor->acl_format != ACL_FORMAT_ND && descriptor->acl_format != ACL_FORMAT_NCHW) {
        // Preserve native dtype-view behavior for physical formats that cannot
        // be interpreted as a flat dense byte span by this adapter.
        descriptor = ::CopyTypeV2(buffers.tensors[i].view(at::kByte).reshape({-1}));
      } else {
        FlattenBytes(buffers.tensors[i], *descriptor);
      }
    }
  }
  return buffers.descriptors;
}

// The torch-npu V2 task queue takes descriptor snapshots before submission.
// Its worker uses those descriptors for both cache lookup and ACL conversion.
inline std::vector<TensorStructPtr> CopyTypeV2(const ByteTensorList& buffers) {
  return Descriptors(buffers);
}

// On a V1 cache miss, ConvertTypes calls this overload on the submitting thread.
// ConvertTypeV2 is the native descriptor-to-ACL converter, independent of
// which task queue selected this overload.
inline aclTensorList* ConvertType(const ByteTensorList& buffers) {
  return ::ConvertTypeV2(Descriptors(buffers));
}

inline void add_param_to_buf(const ByteTensorList& buffers) {
  // The V1 executor cache looks up the original arguments before converting
  // them. Tag the TensorList key so the original dtype/shape is interpreted as
  // a flat byte descriptor, distinct from ordinary TensorList binding to the
  // same ACLNN entry. Native hashing still handles the original tensors and
  // their addresses. The tag does not become an ACLNN argument.
  constexpr int64_t kByteDescriptorCacheTag = 0x484d434259544553;  // ASCII "HMCBYTES"
  ::add_param_to_buf(kByteDescriptorCacheTag);
  ::add_param_to_buf(buffers.tensors);
}
}  // namespace hyper_multicore::mxfp8_binding

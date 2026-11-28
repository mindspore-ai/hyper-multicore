# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Validate native MX byte views without requiring an MXFP8-capable device."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from tests.common.native_sources import MXFP8_DESCRIPTOR_STUB, native_source


def test_native_mx_operands_preserve_storage_and_reject_implicit_copies(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for native alias validation")
    header = tmp_path / "csrc/cached_op_api.h"
    header.parent.mkdir()
    header.write_text(MXFP8_DESCRIPTOR_STUB + r"""
#include <tuple>
template <typename T>
auto CopyTypesV2(const T& value) { return std::make_tuple(CopyTypeV2(value)); }
template <typename T>
auto ConvertTypes(const T& value) { return std::make_tuple(ConvertType(value)); }
namespace hyper_multicore::mxfp8_binding { struct ByteTensorList; }
namespace launch_test {
using ByteBuffers = hyper_multicore::mxfp8_binding::ByteTensorList;
void Capture(const ByteBuffers&, const at::Tensor&, const at::Tensor&,
             const at::Tensor&, const at::Tensor&, const ByteBuffers&, int64_t);
}
#define HYPER_EXEC_NPU_CMD_EXT(api, ...) launch_test::Capture(__VA_ARGS__)
""")
    byte_header = native_source(Path("hyper_multicore/torch/csrc/mxfp8_byte_tensor_list.h"), tmp_path)
    shutil.copy2(byte_header, tmp_path / "csrc/mxfp8_byte_tensor_list.h")
    source = native_source(Path("hyper_multicore/torch/csrc/mega_moe_mxfp8.cpp"), tmp_path)
    program = tmp_path / "aliases.cpp"
    program.write_text('''
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <vector>
#include "csrc/mxfp8_byte_tensor_list.h"
namespace launch_test {
std::vector<at::Tensor> originals;
int calls = 0;
void Capture(const ByteBuffers& buffers, const at::Tensor&, const at::Tensor&, const at::Tensor&,
             const at::Tensor&, const ByteBuffers& outputs, int64_t count) {
  ++calls;
  const auto descriptors = CopyTypeV2(buffers);
  TORCH_CHECK(count == originals.size() && descriptors.size() == originals.size());
  for (int64_t i = 0; i < count; ++i) {
    const auto& original = originals[i];
    const auto& descriptor = *descriptors[i];
    TORCH_CHECK(buffers.tensors[i].storage().unsafeGetStorageImpl() == original.storage().unsafeGetStorageImpl());
    TORCH_CHECK(descriptor.acl_type == ACL_UINT8 && descriptor.itemsize == 1);
    TORCH_CHECK(descriptor.sizes == std::vector<int64_t>{original.numel() * original.element_size()});
    TORCH_CHECK(descriptor.data_ptr == original.storage().data());
    TORCH_CHECK(descriptor.storage_offset == original.storage_offset() * original.element_size());
  }
  TORCH_CHECK(CopyTypeV2(outputs) == descriptors);
  buffers.tensors[0].zero_();
}
}
int main() {
  using Launch = void(at::TensorList, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);
  auto runtime = at::zeros({64}, at::TensorOptions().dtype(at::kByte));
  auto launch = at::zeros({592}, runtime.options());
  using Version = int64_t();
  TORCH_CHECK(c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::mega_moe_mxfp8_abi_version", "")
              .typed<Version>().call() == 4);
  for (const auto* name : {"hyper_multicore::mega_moe_mxfp8", "hyper_multicore::mega_moe_mxfp8_grad"}) {
    auto invoke = c10::Dispatcher::singleton().findSchemaOrThrow(name, "").typed<Launch>();
    auto base = at::ones({7, 8}, at::TensorOptions().dtype(at::kBFloat16));
    launch_test::originals = {base.slice(0, 2, 5), at::ones({2, 3}, at::TensorOptions().dtype(at::kLong)),
                              at::empty({0, 8}, base.options())};
    invoke.redispatch(c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), launch_test::originals,
                     runtime, runtime, runtime, launch);
    TORCH_CHECK(base.slice(0, 2, 5).eq(0).all().item<bool>());
    TORCH_CHECK(base.slice(0, 0, 2).eq(1).all().item<bool>());
    launch_test::originals.clear();
    const auto before = launch_test::calls;
    launch_test::originals = {base.transpose(0, 1)};
    bool rejected = false;
    try {
      invoke.redispatch(c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), launch_test::originals,
                       runtime, runtime, runtime, launch);
    } catch (const c10::Error&) { rejected = true; }
    TORCH_CHECK(rejected && launch_test::calls == before);
  }
  return 0;
}
''')
    torch_root = Path(torch.__file__).resolve().parent
    binary = tmp_path / "aliases"
    standard = re.search(r"C\+\+ Version: (\d+)", torch.__config__.show())
    if standard is None:
        raise RuntimeError("Torch does not report its C++ standard")
    cpp_standard = "c++20" if int(standard[1]) >= 202002 else "c++17"
    command = [compiler, f"-std={cpp_standard}", "-O0",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
               "-I", str(tmp_path), "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"), str(program), str(source),
               "-L", str(torch_root / "lib"), f"-Wl,-rpath,{torch_root / 'lib'}",
               "-ltorch", "-ltorch_cpu", "-lc10", "-o", str(binary)]
    compiled = subprocess.run(command, check=False, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    test = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert test.returncode == 0, test.stdout + test.stderr

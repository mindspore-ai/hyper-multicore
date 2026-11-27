# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Validate native MX byte views without requiring an MXFP8-capable device."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from tests.common.native_sources import native_source


def test_native_mx_operands_preserve_storage_and_reject_implicit_copies(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for native alias validation")
    header = tmp_path / "csrc/cached_op_api.h"
    header.parent.mkdir()
    header.write_text('''#pragma once
#include <ATen/ATen.h>
namespace launch_test {
void Capture(at::TensorList, const at::Tensor&, const at::Tensor&, const at::Tensor&,
             const at::Tensor&, at::TensorList, int64_t);
}
#define HYPER_EXEC_NPU_CMD_EXT(api, ...) launch_test::Capture(__VA_ARGS__)
''')
    source = native_source(Path("hyper_multicore/torch/csrc/mega_moe_mxfp8.cpp"), tmp_path)
    program = tmp_path / "aliases.cpp"
    program.write_text('''
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <vector>
namespace launch_test {
std::vector<at::Tensor> originals, retained;
int calls = 0;
void Capture(at::TensorList aliases, const at::Tensor&, const at::Tensor&, const at::Tensor&,
             const at::Tensor&, at::TensorList outputs, int64_t count) {
  ++calls;
  TORCH_CHECK(count == originals.size() && aliases.size() == originals.size());
  for (int64_t i = 0; i < count; ++i) {
    TORCH_CHECK(aliases[i].scalar_type() == at::kByte && aliases[i].dim() == 1);
    TORCH_CHECK(aliases[i].data_ptr() == originals[i].data_ptr());
    TORCH_CHECK(aliases[i].numel() == originals[i].numel() * originals[i].element_size());
    TORCH_CHECK(aliases[i].storage().unsafeGetStorageImpl() == originals[i].storage().unsafeGetStorageImpl());
    TORCH_CHECK(outputs[i].data_ptr() == aliases[i].data_ptr());
  }
  aliases[0].zero_();
  retained.assign(aliases.begin(), aliases.end());
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
    TORCH_CHECK(launch_test::retained[0].eq(0).all().item<bool>());
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
    command = [compiler, f"-std={cpp_standard}", "-O0", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
               "-I", str(tmp_path), "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"), str(program), str(source),
               "-L", str(torch_root / "lib"), f"-Wl,-rpath,{torch_root / 'lib'}",
               "-ltorch", "-ltorch_cpu", "-lc10", "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

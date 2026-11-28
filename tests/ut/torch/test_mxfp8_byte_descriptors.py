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
"""Check real byte-descriptor lowering with CPU tensors and a native API stub."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import torch

from tests.common.native_sources import MXFP8_DESCRIPTOR_STUB, native_source


def test_byte_descriptors_preserve_storage_bounds():
    torch_root = Path(torch.__file__).resolve().parent
    version = re.search(r'C\+\+ Version: (\d+)', torch.__config__.show())
    compiler = shutil.which('g++')
    if compiler is None:
        raise unittest.SkipTest('g++ is required for native descriptor verification')
    if version is None:
        raise RuntimeError('Torch does not report its required C++ standard')
    standard = 'c++20' if int(version[1]) >= 202002 else 'c++17'
    with tempfile.TemporaryDirectory(prefix='hmc-byte-descriptors-') as directory:
        root = Path(directory)
        (root / 'csrc').mkdir()
        (root / 'csrc/cached_op_api.h').write_text(MXFP8_DESCRIPTOR_STUB)
        byte_header = native_source(Path('hyper_multicore/torch/csrc/mxfp8_byte_tensor_list.h'), root)
        shutil.copy2(byte_header, root / 'csrc/mxfp8_byte_tensor_list.h')
        source = root / 'test.cpp'
        source.write_text(TEST)
        executable = root / 'test'
        result = subprocess.run([
            compiler, '-std=' + standard, '-O0',
            '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch.compiled_with_cxx11_abi())),
            '-I' + str(root),
            '-I' + str(torch_root / 'include'), '-I' + str(torch_root / 'include/torch/csrc/api/include'),
            str(source), '-L' + str(torch_root / 'lib'), '-Wl,-rpath,' + str(torch_root / 'lib'),
            '-ltorch', '-ltorch_cpu', '-lc10', '-o', str(executable),
        ], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


TEST = r'''#include "csrc/mxfp8_byte_tensor_list.h"
#include <iostream>
using namespace hyper_multicore::mxfp8_binding;
int main() {
  for (auto dtype : {at::kBFloat16,at::kFloat,at::kLong,at::kInt,at::kByte,at::kFloat8_e4m3fn}) {
    auto base=at::empty({19,16},at::TensorOptions().dtype(dtype).requires_grad(false));
    for (auto bounds : {std::pair<int,int>{0,0},{0,19},{3,9},{19,19}}) {
      auto tensor=base.slice(0,bounds.first,bounds.second);
      std::vector<at::Tensor> inputs{tensor};
      ByteTensorList binding{inputs,{}};
      auto first=CopyTypeV2(binding);auto second=CopyTypeV2(binding);
      TORCH_CHECK(first[0]==second[0],"one invocation must reuse its native snapshot");
      auto expected=tensor.view(at::kByte).reshape({-1});auto d=first[0];
      TORCH_CHECK(d->sizes==expected.sizes().vec() && d->strides==expected.strides().vec(),"byte layout mismatch");
      TORCH_CHECK(d->storage_offset==expected.storage_offset() && d->nbytes==expected.storage().nbytes(),"byte storage mismatch");
      TORCH_CHECK(d->data_ptr==expected.storage().data() && d->itemsize==1 && d->acl_type==ACL_UINT8,"byte address mismatch");
      auto replacement=at::empty_like(tensor);std::vector<at::Tensor> fresh{replacement};
      ByteTensorList next{fresh,{}};auto n=CopyTypeV2(next)[0];
      TORCH_CHECK(n!=d && n->data_ptr==replacement.storage().data(),"new invocation retained a previous address");
    }
    auto bad=base.transpose(0,1);std::vector<at::Tensor> input{bad};ByteTensorList binding{input,{}};
    bool rejected=false;try{CopyTypeV2(binding);}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected,"noncontiguous storage must not be copied implicitly");
  }
  std::cout<<"byte descriptor bounds and invocation ownership passed\n";
}
'''

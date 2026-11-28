# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Cache executable addresses without caching per-call or per-thread memory state."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_cached_context_keeps_each_threads_lifecycle(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required for header validation')
    header = Path(__file__).resolve().parents[3] / 'hyper_multicore/torch/csrc/cached_op_api.h'
    stub = tmp_path / 'op_plugin/include/npu_cpp_extension.h'
    stub.parent.mkdir(parents=True)
    stub.write_text('''#pragma once
#include <atomic>
#include <cstring>
#include <cstdlib>
inline std::atomic<int> lookups[4]{};
inline std::atomic<int> calls[4]{};
inline thread_local int active = 0;
inline int initialize(void*, bool) { if(active++) std::abort(); ++calls[0]; return 0; }
inline void release(void*, bool) { if(active != 1) std::abort(); ++calls[1]; }
inline void uninitialize(void*, bool) { if(active-- != 1) std::abort(); ++calls[2]; }
inline void clear() { if(active) std::abort(); ++calls[3]; }
inline void* GetOpApiFuncAddr(const char* name) {
  if(!std::strcmp(name, "InitHugeMemThreadLocal")) {++lookups[0]; return reinterpret_cast<void*>(&initialize);}
  if(!std::strcmp(name, "ReleaseHugeMem")) {++lookups[1]; return reinterpret_cast<void*>(&release);}
  if(!std::strcmp(name, "UnInitHugeMemThreadLocal")) {++lookups[2]; return reinterpret_cast<void*>(&uninitialize);}
  if(!std::strcmp(name, "UnInitPTACacheThreadLocal")) {++lookups[3]; return reinterpret_cast<void*>(&clear);}
  std::abort();
}
inline void GetApiFunc(const char*, const char*, void*&, void*&) {}
''')
    source = tmp_path / 'test.cpp'
    source.write_text('#include "' + str(header) + '"\n' + '''#include <thread>
#include <vector>
int main() {
  auto worker=[]() {
    for(int i=0;i<32;++i) {
      hyper_multicore::InitExecCommonCtx();
      hyper_multicore::ReleaseExecCommonCtx();
      hyper_multicore::UnInitExecCommonCtx();
    }
  };
  std::vector<std::thread> threads;
  for(int i=0;i<4;++i) threads.emplace_back(worker);
  for(auto& thread:threads) thread.join();
  for(int i=0;i<4;++i) if(lookups[i] != 1 || calls[i] != 128) return i+1;
}
''')
    binary = tmp_path / 'context-test'
    subprocess.run([compiler, '-std=c++17', '-pthread', '-I', str(tmp_path), str(source), '-o', str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

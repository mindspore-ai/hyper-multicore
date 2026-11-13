/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details of the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "runtime/config.h"

namespace {

using hyper_multicore::shmem::runtime::DataEngine;
using hyper_multicore::shmem::runtime::ErrorCode;
using hyper_multicore::shmem::runtime::LoadConfigFromEnvironment;

void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void test_data_engine_configuration() {
  const struct {
    const char *engine;
    DataEngine expected;
  } cases[] = {{nullptr, DataEngine::Mte}, {"mte", DataEngine::Mte}, {"mte_udma", DataEngine::MteUdma}};
  for (const auto &entry : cases) {
    unsetenv("HYPER_MULTICORE_SHMEM_DATA_ENGINE");
    if (entry.engine != nullptr) setenv("HYPER_MULTICORE_SHMEM_DATA_ENGINE", entry.engine, 1);
    const auto config = LoadConfigFromEnvironment();
    check(config.ok(), "valid configuration was rejected");
    check(config.value().data_engine == entry.expected, "wrong data engine");
  }
  setenv("HYPER_MULTICORE_SHMEM_DATA_ENGINE", "udma", 1);
  const auto invalid_engine = LoadConfigFromEnvironment();
  check(!invalid_engine.ok(), "invalid data engine was accepted");
  check(invalid_engine.error().error_code == ErrorCode::InvalidConfig, "wrong engine error category");
}

}  // namespace

int main() {
  unsetenv("HYPER_MULTICORE_SHMEM_HEAP_SIZE");
  unsetenv("HYPER_MULTICORE_SHMEM_TIMEOUT_SEC");
  unsetenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT");
  try {
    test_data_engine_configuration();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}

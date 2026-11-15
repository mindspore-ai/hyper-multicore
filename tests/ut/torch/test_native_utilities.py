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
"""Exercise native launch-status propagation without a CANN or NPU backend."""

import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from tests.common.mark_utils import arg_mark

_STUB_HEADERS = {
    "launch_test_state.h": """#pragma once
namespace launch_test {
extern int callback_status;
extern int last_status;
extern int status_reads;
}
""",
    "torch_npu/csrc/core/npu/NPUStream.h": """#pragma once
namespace c10_npu {
struct NPUStream { void* stream(bool) const { return nullptr; } };
inline NPUStream getCurrentNPUStream(int = -1) { return {}; }
}
""",
    "torch_npu/csrc/framework/OpCommand.h": """
#pragma once
#include <stdexcept>
#include <string>
#include "launch_test_state.h"
namespace at_npu::native {
struct OpCommand {
    template<typename Callback>
    static void RunOpApi(const char*, Callback run) {
        launch_test::callback_status = run();
        if (launch_test::callback_status != 0) {
            throw std::runtime_error("launch status " + std::to_string(launch_test::callback_status));
        }
    }
};
}
""",
    "acl/acl.h": """
#pragma once
#include <cstdint>
#include "launch_test_state.h"
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_DEV_ATTR_VECTOR_CORE_NUM = 1;
constexpr int ACL_RT_THREAD_LEVEL = 0;
inline int aclrtGetDeviceInfo(int, int, int64_t* cores) { *cores = 40; return ACL_SUCCESS; }
inline int aclrtPeekAtLastError(int) {
    ++launch_test::status_reads;
    return launch_test::last_status;
}
""",
}


def _native_utility_source(root: Path) -> Path:
    """Read production C++ from a checkout or the source archive in the CI test bundle."""
    source_root = Path(__file__).resolve().parents[3]
    relative = Path("hyper_multicore/torch/csrc/mega_moe_utilities.cpp")
    source = source_root / relative
    if source.is_file():
        return source
    with tarfile.open(source_root / "hyper-multicore.tar") as archive:
        members = [member for member in archive.getmembers()
                   if member.isfile() and Path(member.name).parts[-len(relative.parts):] == relative.parts]
        if len(members) != 1:
            raise RuntimeError("The test bundle must contain exactly one production mega_moe_utilities.cpp")
        with archive.extractfile(members[0]) as content:
            source = root / relative.name
            source.write_bytes(content.read())
    return source


def _compile_native_utilities(root: Path) -> Path:
    """Compile the real utility with CPU tensors and framework-independent launch stubs."""
    compiler = shutil.which("g++")
    if compiler is None:
        raise unittest.SkipTest("g++ is required for the native launch-status test")
    for name, content in _STUB_HEADERS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    torch_root = Path(torch.__file__).resolve().parent
    source = _native_utility_source(root)
    standard = re.search(r"C\+\+ Version: (\d+)", torch.__config__.show())
    if standard is None:
        raise RuntimeError("Torch does not report its required C++ standard")
    cpp_standard = "c++20" if int(standard[1]) >= 202002 else "c++17"
    executable = root / "native_utilities_test"
    library_path = ":".join(str(path) for path in (torch_root / "lib", torch_root.parent / "torch.libs")
                            if path.is_dir())
    result = subprocess.run([
        compiler, f"-std={cpp_standard}", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
        "-I", str(root), "-I", str(torch_root / "include"),
        "-I", str(torch_root / "include/torch/csrc/api/include"),
        str(Path(__file__).with_name("native_utilities_test.cpp")),
        str(source),
        "-L", str(torch_root / "lib"), f"-Wl,-rpath,{library_path}", f"-Wl,-rpath-link,{library_path}",
        "-Wl,--disable-new-dtags",
        "-ltorch", "-ltorch_cpu", "-lc10", "-o", str(executable),
    ], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    return executable


@unittest.skipUnless(sys.platform == "linux", "CANN utility bindings use Linux framework libraries")
class TestNativeUtilities(unittest.TestCase):
    """Run production C++ with real CPU tensors and an injected launch status."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_route_offsets_propagates_launch_status(self) -> None:
        """Feature: Native routing launch error propagation.
        Description: Execute the real RouteOffsets callback with successful and failed launches.
        Expectation: RunOpApi receives the immediate status and reports launch failures to the caller.
        """
        with tempfile.TemporaryDirectory(prefix="hmc-native-utilities-") as directory:
            executable = _compile_native_utilities(Path(directory))
            for status in (0, 17):
                with self.subTest(status=status):
                    result = subprocess.run([str(executable), str(status)],
                                            check=False, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

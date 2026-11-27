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
"""Compile HotReplica in the normal native-build workspace; publication owns staging."""

import argparse
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sysconfig


def cann_includes(cann: Path, source: Path, sdk: Path) -> list[str]:
    """Use the same CANN and private SHMEM SDK as the enclosing native build."""
    paths = [source.parent, sdk / "include", sdk / "include/device", sdk / "src", sdk / "src/device"]
    paths += [
        cann / path
        for path in (
            "include",
            "include/runtime",
            "pkg_inc",
            "pkg_inc/runtime",
            "asc",
            "asc/include",
            "asc/include/interface",
            "asc/include/utils",
            "asc/impl",
            "compiler/tikcpp",
            "compiler/tikcpp/tikcfw",
            "compiler/tikcpp/tikcfw/impl",
            "compiler/tikcpp/tikcfw/interface",
            f"{platform.machine()}-linux/include",
        )
    ]
    paths += [p.parent.parent for p in cann.rglob("bfloat16.h") if p.parent.name == "opdev"]
    return [f"-I{path}" for path in paths if path.is_dir()]


def build_kernels(output: Path, sdk: Path, npu_arch: int) -> None:
    """Compile framework-independent AscendC code for the selected SHMEM backend."""
    source = Path(__file__).parent / "csrc"
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "bisheng",
        "-x",
        "asc",
        f"--npu-arch=dav-{npu_arch}",
        "--cce-auto-sync=off",
        "--cce-fatobj-link",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        "-Wl,-z,relro,-z,now,-z,noexecstack",
        *[
            str(source / name)
            for name in ("hot_replica_expert_prefetch.cpp", "hot_replica_expert_grad_reduce_fp32.cpp",
                         "replica_counts.cpp")
        ],
        "-o",
        str(output / "libhmc_hot_replica_kernels.so"),
        "-Wl,-soname,libhmc_hot_replica_kernels.so",
        *cann_includes(cann, source, sdk),
        f"-L{sdk / 'lib'}",
        f"-L{cann / 'lib64'}",
        "-lhyper_multicore_shmem",
        "-lhyper_multicore_shmem_utils",
        "-lascendcl",
        "-lruntime",
        "-ldl",
    ]
    subprocess.run(command, check=True)


def build_adapters(output: Path, sdk: Path, device_root: Path, standard: int) -> None:
    """Compile only host bridges with each selected framework's compiler and ABI."""
    import torch  # pylint: disable=import-outside-toplevel
    import torch_npu  # pylint: disable=import-outside-toplevel

    source = Path(__file__).parent / "csrc"
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    torch_root, npu_root = Path(torch.__file__).parent, Path(torch_npu.__file__).parent
    includes = cann_includes(cann, source, sdk)
    includes += [
        f"-I{p}"
        for p in (
            torch_root / "include",
            torch_root / "include/torch/csrc/api/include",
            npu_root / "include",
            Path(sysconfig.get_path("include")),
        )
    ]
    output.mkdir(parents=True, exist_ok=True)
    common = [
        *shlex.split(os.environ.get("CXX", "g++")),
        "-shared",
        "-fPIC",
        "-O2",
        f"-std=c++{standard}",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",  # pylint: disable=protected-access
        "-Wl,-z,relro,-z,now,-z,noexecstack",
    ]
    libraries = [f"-L{p}" for p in (device_root, torch_root / "lib", npu_root / "lib", sdk / "lib", cann / "lib64")]
    libraries += [
        "-lhmc_hot_replica_kernels",
        "-ltorch_npu",
        "-ltorch",
        "-lc10",
        "-ltorch_python",
        "-lhyper_multicore_shmem",
        "-lhyper_multicore_shmem_utils",
        "-lascendcl",
        "-lruntime",
        "-ldl",
    ]
    subprocess.run(
        [
            *common,
            "-DTORCH_EXTENSION_NAME=_hot_replica_shared_bridge",
            str(source / "hot_replica_shared_bridge.cpp"),
            "-o",
            str(output / "_hot_replica_shared_bridge.so"),
            *includes,
            *libraries,
        ],
        check=True,
    )
    build_host_planner(output)


def build_host_planner(output: Path) -> None:
    """Build the portable CPU selector with the target interpreter's headers."""
    source = Path(__file__).parent / "csrc" / "replica_planner.cpp"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "g++")),
            "-shared", "-fPIC", "-O3", "-std=c++17", "-Wl,-z,relro,-z,now,-z,noexecstack",
            f"-I{sysconfig.get_path('include')}", f"-I{source.parents[1]}",
            str(source), "-o", str(output / "_replica_host.so"),
        ],
        check=True,
    )


def main() -> None:
    """Build isolated artifacts; the enclosing build publishes and hashes them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("common", "adapter", "planner"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--device-root", type=Path)
    parser.add_argument("--cxx-standard", type=int, default=17)
    parser.add_argument("--npu-arch", type=int, choices=(2201,), default=2201)
    args = parser.parse_args()
    if args.phase == "planner":
        build_host_planner(args.output_dir)
        return
    if args.sdk_root is None:
        parser.error("--sdk-root is required for BF16 HotReplica builds")
    if args.phase == "common":
        build_kernels(args.output_dir, args.sdk_root, args.npu_arch)
    else:
        if args.device_root is None:
            parser.error("--device-root is required for adapter builds")
        build_adapters(args.output_dir, args.sdk_root, args.device_root, args.cxx_standard)


if __name__ == "__main__":
    main()

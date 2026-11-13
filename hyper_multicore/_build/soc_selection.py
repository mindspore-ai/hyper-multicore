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
"""Resolve build SoCs and their device architecture groups."""

import argparse

ARCHITECTURES = {
    "arch22": {"npu_arch": 2201, "compile_soc": "ascend910b", "sdk_soc_type": "Ascend910B",
               "soc_series": ["ascend910b", "ascend910_93"]},
    "arch35": {"npu_arch": 3510, "compile_soc": "ascend950", "sdk_soc_type": "Ascend950",
               "soc_series": ["ascend950"]},
}
SOC_ARCHITECTURES = {soc: name for name, spec in ARCHITECTURES.items() for soc in spec["soc_series"]}
SUPPORTED_SOCS = frozenset(SOC_ARCHITECTURES)
DEFAULT_SOC_LIST = ",".join(SOC_ARCHITECTURES)


def resolve_soc_selection(value: str) -> tuple[list[str], dict[str, list[str]]]:
    """Preserve requested order and group SoCs sharing one device architecture."""
    socs = [soc.strip() for soc in value.split(",")]
    if any(soc not in SUPPORTED_SOCS for soc in socs):
        raise ValueError("--soc-list must contain only ascend910b, ascend910_93, or ascend950; "
                         "empty entries are invalid")
    if len(set(socs)) != len(socs):
        raise ValueError("--soc-list must contain unique SoCs")
    groups = {}
    for soc in socs:
        groups.setdefault(SOC_ARCHITECTURES[soc], []).append(soc)
    return socs, groups


def main() -> None:
    """Print normalized selection or one backend group per line for Shell callers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("soc_list", nargs="?", default=DEFAULT_SOC_LIST)
    parser.add_argument("--groups", action="store_true")
    args = parser.parse_args()
    try:
        socs, groups = resolve_soc_selection(args.soc_list)
    except ValueError as error:
        parser.error(str(error))
    if args.groups:
        for backend, selected in groups.items():
            spec = ARCHITECTURES[backend]
            print(backend, spec["sdk_soc_type"], spec["compile_soc"],
                  ",".join(spec["soc_series"]), ",".join(selected))
    else:
        print(",".join(socs), ",".join(ARCHITECTURES[name]["sdk_soc_type"] for name in groups))


if __name__ == "__main__":
    main()

# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Remove the MXFP8 device registrations from one generated arch22 ops-info file."""

from pathlib import Path
import re
import sys


_MX_OP_SECTIONS = re.compile(
    r"^\[HyperMegaMoeMxfp8(?:Grad)?\][ \t]*\n.*?(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _filter_device_ops(path: Path) -> None:
    """Leave other generated operator sections intact."""
    content = path.read_text(encoding="utf-8")
    filtered = _MX_OP_SECTIONS.sub("", content)
    if content != filtered:
        path.write_text(filtered, encoding="utf-8")


if __name__ == "__main__":
    _filter_device_ops(Path(sys.argv[1]))

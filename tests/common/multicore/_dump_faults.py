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
"""Filesystem fault injection shared by Dumper CPU and NPU tests."""

from collections.abc import Iterator
from contextlib import contextmanager
import errno
from pathlib import Path
from unittest import mock


@contextmanager
def fail_dump_output_payload_write() -> Iterator[None]:
    """Fail the payload write while allowing Torch's smaller ZIP finalizer writes."""
    open_file = Path.open

    def _open(path, *args, **kwargs):
        # The original caller still owns the file handle.
        handle = open_file(path, *args, **kwargs)  # pylint: disable=consider-using-with
        if handle.mode == "wb" and path.name.startswith("output_") and path.suffix == ".pt":
            write = handle.write

            def _write(data):
                if len(data) >= 4096:
                    raise OSError(errno.ENOSPC, "injected disk full")
                return write(data)

            handle.write = _write
        return handle

    with mock.patch.object(Path, "open", new=_open):
        yield

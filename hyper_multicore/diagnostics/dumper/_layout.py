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
"""Dump-specific safe stage directories, independent of profiler names."""

from collections.abc import Sequence
from io import BufferedWriter
from pathlib import Path
import re

import torch

from hyper_multicore.diagnostics.common.graph import operator_stage_name
from hyper_multicore.scheduler.graph import OperatorNode


_SAFE_FILE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_KERNEL_MANIFEST_VERSION = 5
_TASK_MANIFEST_VERSION = 6


class _TensorFileWriter:
    """Keep the first stream error even if Torch's ZIP finalizer replaces it."""

    def __init__(self, stream: BufferedWriter) -> None:
        """Track errors on the stream opened by the dump writer."""
        self._stream = stream
        self.error: OSError | None = None

    def write(self, data: bytes) -> int:
        """Write a ZIP record, retaining the original file error."""
        try:
            return self._stream.write(data)
        except OSError as error:
            self.error = self.error or error
            raise

    def flush(self) -> None:
        """Flush pending ZIP writes and retain any file error."""
        try:
            self._stream.flush()
        except OSError as error:
            self.error = self.error or error
            raise


def _save_tensor(tensor: torch.Tensor, path: Path) -> None:
    """Separate file stream errors from serialization failures."""
    with path.open("wb") as stream:
        writer = _TensorFileWriter(stream)
        try:
            torch.save(tensor, writer)
        except (OSError, RuntimeError) as error:
            if writer.error is not None:
                raise writer.error from None
            if isinstance(error, OSError):
                raise RuntimeError(f"failed to serialize Tensor to {path}") from error
            raise


def _stage_names(operators: Sequence[OperatorNode]) -> dict[str, str]:
    """Reject duplicate, colliding or reserved dump stage directories."""
    names = {}
    used_directories = set()
    for operator in operators:
        name = operator_stage_name(operator)
        directory = _SAFE_FILE_NAME.sub("_", name)
        if directory in ("", ".", "..") or directory in used_directories:
            raise ValueError(f"duplicate, colliding or reserved MegaKernel dump stage directory: {name!r}")
        used_directories.add(directory)
        names[operator.name] = name
    return names

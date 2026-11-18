"""CANN byte-list storage bounds may exceed a contiguous logical metadata view."""
import subprocess
from pathlib import Path

from tests.common.native_sources import native_source


def test_pooled_metadata_view_uses_logical_bounds_within_physical_storage(tmp_path):
    """Check launch bounds when a byte view borrows pooled storage."""
    header = native_source(Path("hyper_multicore/ops/common/mega_moe/contracts/mx_byte_tensor_list.h"), tmp_path)
    source = tmp_path / 'contract.cpp'
    source.write_text('''
#include "mx_byte_tensor_list.h"
int main() {
  using MulticoreMx::IsMxByteVectorShapeValid;
  if (!IsMxByteVectorShapeValid(0x100000001ULL, 2144, 48)) return 1;
  if (!IsMxByteVectorShapeValid(1, 48, 48)) return 2;
  if (IsMxByteVectorShapeValid(1, 40, 48)) return 3;
  if (IsMxByteVectorShapeValid(2, 2144, 48)) return 4;
  if (!IsMxByteVectorShapeValid(1, 0, 0)) return 5;
  return 0;
}
''', encoding='utf-8')
    binary = tmp_path / 'contract'
    subprocess.run(['g++', '-std=c++17', '-I', str(header.parent),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

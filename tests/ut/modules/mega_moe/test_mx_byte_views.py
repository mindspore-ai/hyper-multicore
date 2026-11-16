"""CANN byte-list storage bounds may exceed a contiguous logical metadata view."""
from pathlib import Path
import subprocess


def test_pooled_metadata_view_uses_logical_bounds_within_physical_storage(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / 'contract.cpp'
    source.write_text('''
#include "hyper_mega_moe/op_kernel/contracts/mx_byte_tensor_list.h"
int main() {
  using MulticoreMx::IsMxByteVectorShapeValid;
  if (!IsMxByteVectorShapeValid(0x100000001ULL, 2144, 48)) return 1;
  if (!IsMxByteVectorShapeValid(1, 48, 48)) return 2;
  if (IsMxByteVectorShapeValid(1, 40, 48)) return 3;
  if (IsMxByteVectorShapeValid(2, 2144, 48)) return 4;
  if (!IsMxByteVectorShapeValid(1, 0, 0)) return 5;
  return 0;
}
''')
    binary = tmp_path / 'contract'
    subprocess.run(['g++', '-std=c++17', '-I', str(root / 'hyper_multicore/ops'),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

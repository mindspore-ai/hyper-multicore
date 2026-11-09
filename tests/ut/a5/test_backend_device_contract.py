"""The compiled private SDK family must agree with the runtime device gate."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('a5', [False, True])
def test_backend_device_family_is_compiled_explicitly(tmp_path: Path, a5: bool):
    include = Path(__file__).resolve().parents[3]/'hyper_multicore/shmem/ccsrc'
    expected = 'true' if a5 else 'false'
    source = tmp_path/'backend.cpp'
    source.write_text('''#include "device/device_model.h"
using namespace hyper_multicore::shmem;
static_assert(BackendSupportsDevice(DeviceModel::kAscend950) == ''' + expected + ''');
static_assert(BackendSupportsDevice(DeviceModel::kAscend910B) != ''' + expected + ''');
static_assert(BackendSupportsDevice(DeviceModel::kAscend910C) != ''' + expected + ''');
int main() { return 0; }
''')
    command = ['g++', '-std=c++17', '-I'+str(include), str(source), '-o', str(tmp_path/'backend')]
    if a5:
        command.insert(1, '-DHP_SHMEM_ASCEND950')
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run([str(tmp_path/'backend')], check=True)


def test_a5_transport_requires_a_live_udma_bank_before_launch(monkeypatch):
    from types import SimpleNamespace
    import torch
    from hyper_multicore.modules.mega_moe.udma_order import validate_device_transport
    tensor = SimpleNamespace(device=SimpleNamespace(type='npu'))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(get_device_name=lambda _: 'Ascend950DT'), raising=False)
    monkeypatch.delenv('HYPER_MULTICORE_SHMEM_DATA_ENGINE', raising=False)
    with pytest.raises(ValueError, match='requires.*mte_udma'):
        validate_device_transport(tensor)
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_DATA_ENGINE', 'mte_udma')
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '32')
    monkeypatch.delenv('HYPER_MULTICORE_MX_AIV_POLICY', raising=False)
    validate_device_transport(tensor)


@pytest.mark.parametrize('a5', [False, True])
def test_cube_event_publication_uses_a_legal_architecture_path(a5):
    """A5 has no L1-to-GM event copy; preserve the scalar atomic completion contract."""
    source = (Path(__file__).resolve().parents[3] /
              'hyper_multicore/ops/runtime/worker_kernel.h').read_text()
    body = source.split('void AtomicAddForAllEventCounters(uint32_t event_index) {', 1)[1]
    body = body.split('__aicore__ inline void ExecuteTaskFast', 1)[0]
    command = ['g++', '-E', '-P', '-x', 'c++', '-D__DAV_CUBE__',
               '-D__NPU_ARCH__=' + ('3510' if a5 else '2201'), '-']
    result = subprocess.run(command, input=body, text=True, check=True, capture_output=True).stdout
    assert ('AscendC::AtomicAdd<int32_t>' in result) is a5
    assert ('TPipe eventPipe' in result) is not a5

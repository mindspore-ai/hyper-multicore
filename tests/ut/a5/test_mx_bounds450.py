"""Regression coverage for unsigned MX descriptor shapes and wide element products."""
from pathlib import Path
import subprocess

import pytest

from hyper_multicore.scheduler.mxfp8 import make_mx_gmm_descriptor


def test_capacity20_product_is_supported() -> None:
    record = make_mx_gmm_descriptor(layout='NN', m=655360, n=3584, k=5120,
                                    x_scale_position=17, w_scale_position=23, cube_count=32)
    assert (record.m, record.n, record.k) == (655360, 3584, 5120)


def test_single_dimension_can_exceed_signed_int32() -> None:
    record = make_mx_gmm_descriptor(layout='NN', m=2**31, n=1, k=1,
                                    x_scale_position=0, w_scale_position=1, cube_count=32)
    assert record.m == 2**31


def test_uint32_serialization_overflow_is_still_rejected() -> None:
    with pytest.raises(ValueError, match='serialized range'):
        make_mx_gmm_descriptor(layout='NN', m=2**32, n=1, k=1,
                               x_scale_position=0, w_scale_position=1, cube_count=32)


def test_device_descriptor_accepts_unsigned_shape(tmp_path: Path) -> None:
    source = tmp_path / 'descriptor.cpp'
    source.write_text('''#include "mx_gmm_descriptor.h"
int main() {
  MulticoreMx::MxGmmDescriptor r{};
  r.magic = MulticoreMx::MX_GMM_MAGIC;
  r.version = 1; r.size = sizeof(r);
  r.m = 0x80000000u; r.n = 1; r.k = 1; r.cube_count = 32;
  r.base_m = 64; r.base_n = 128; r.base_k = 128;
  r.k_al1 = r.k_bl1 = r.scale_k = 256; r.db_l0c = 2;
  return MulticoreMx::IsMxGmmDescriptorValid(r) ? 0 : 1;
}
''')
    include = Path(__file__).resolve().parents[3] / 'hyper_multicore/ops/mxfp8'
    binary = tmp_path / 'descriptor'
    subprocess.run(['g++', '-std=c++17', '-I' + str(include), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize('forward', [True, False])
def test_ep64_capacity20_builds_both_direction_plans(monkeypatch: pytest.MonkeyPatch, forward: bool) -> None:
    from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
    from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
    from hyper_multicore.scheduler.mxfp8 import MxGmmDescriptor
    import ctypes

    monkeypatch.setenv('HYPER_MULTICORE_MX_GMM_TILE', 't6')
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=384, top_k=8, initial_capacity_factor=20, receive_capacity=655360,
                       ep_size=64, ep_group=None, rank_id=0, num_cube_cores=32,
                       mx_vector_policy='window2', dispatch_split=4096, combine_split=4096)
    runtime = build_mxfp8_runtime(spec, forward=forward, ready_depth=8)
    size = ctypes.sizeof(MxGmmDescriptor)
    records = [MxGmmDescriptor.from_buffer_copy(runtime.gmm_descriptors[i:i + size])
               for i in range(0, len(runtime.gmm_descriptors), size)]
    assert records
    assert any(max(r.m * r.n, r.m * r.k, r.n * r.k) > 2**31 - 1 for r in records)
    assert all(r.cube_count == 32 for r in records)

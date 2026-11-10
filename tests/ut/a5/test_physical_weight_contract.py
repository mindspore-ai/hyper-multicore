"""The standalone package borrows refreshed physical roots without downstream imports."""
import torch
import pytest
from hyper_multicore.modules.mega_moe.mxfp8_weights import MXFP8ExpertWeightBinding


def root():
    value = torch.empty((128, 128), dtype=torch.bfloat16)
    value.col_data = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    value.col_scale = torch.full((2, 128, 2), 127, dtype=torch.uint8)
    value.row_data = value.row_scale = None
    return value


def test_current_root_refresh_is_borrowed_at_backward():
    value = root()
    binding = MXFP8ExpertWeightBinding(value, (2, 64, 128))
    forward = binding.resolve('forward')
    old = forward.col_data.data_ptr()
    value.col_data = value.col_data.clone()
    value.col_scale = value.col_scale.clone()
    value.row_data = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    value.row_scale = torch.full((128, 2, 2), 127, dtype=torch.uint8)
    backward = binding.resolve('backward')
    assert backward.col_data.data_ptr() == value.col_data.data_ptr() != old
    assert backward.row_data.data_ptr() == value.row_data.data_ptr()
    assert forward.row_data is None


def test_missing_backward_axis_fails_explicitly():
    with pytest.raises(ValueError, match='row data/scale'):
        MXFP8ExpertWeightBinding(root(), (2, 64, 128)).resolve('backward')


def test_fixed_mx_constructor_preserves_automatic_bf16_default(monkeypatch):
    from hyper_multicore import MegaMoeExperts, MXFP8MegaMoeExperts
    monkeypatch.setenv('HYPER_MULTICORE_MX_AIV_POLICY', 'window2')
    monkeypatch.setenv('HYPER_MULTICORE_MX_COMM_SPLIT', '512')
    fixed = MXFP8MegaMoeExperts(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                              num_experts=12, top_k=8, ep_size=2)
    automatic = MegaMoeExperts(hidden_size=128, intermediate_size=128, num_experts=2, top_k=1,
                               create_parameters=False)
    assert fixed._resource_group.specification['local_num_tokens'] == 4096
    assert fixed._resource_group.specification['auto_tokens'] is False
    assert automatic._resource_group.specification['auto_tokens'] is True
    fixed.close()
    automatic.close()

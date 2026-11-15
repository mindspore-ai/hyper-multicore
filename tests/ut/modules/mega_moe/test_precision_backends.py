"""The common expert API preserves parameter roots and backend contracts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hyper_multicore import MegaMoeExperts
from hyper_multicore.modules.mega_moe.backends import get_backend
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend as mx_backend
from hyper_multicore.modules.mega_moe.resources import _MegaMoeExecutionResources


@pytest.mark.parametrize('precision', ['bf16', 'mxfp8'])
def test_one_frontend_dispatches_without_owning_external_weights(precision):
    layer = MegaMoeExperts(precision=precision, local_num_tokens=512, hidden_size=64,
                           intermediate_size=64, num_experts=6, top_k=2,
                           create_parameters=False)
    try:
        assert not layer.state_dict()
        backend = get_backend(precision)
        assert issubclass(backend.Resources, _MegaMoeExecutionResources)
        hidden, flat, ids, probs, counts, resources = [object() for _ in range(6)]
        roots = (object(), object())
        with (patch.object(layer, '_validate_forward_inputs', return_value=flat),
              patch.object(layer, '_get_execution_resources', return_value=resources),
              patch.object(torch.npu, 'is_current_stream_capturing', return_value=False),
              patch.object(backend, 'forward', return_value=object()) as forward):
            result = layer(hidden, ids, probs, tokens_per_expert=counts, expert_weights=roots)
            assert result is forward.return_value
            forward.assert_called_once_with(layer, hidden, flat, ids, probs, counts, roots, resources)
        assert not layer.state_dict()
    finally:
        layer.close()


def test_bf16_parameter_names_stay_flat_and_mx_requires_physical_roots():
    layer = MegaMoeExperts(hidden_size=64, intermediate_size=64, num_experts=6, top_k=2)
    try:
        assert set(layer.state_dict()) == {'gate_up_weight', 'down_weight'}
    finally:
        layer.close()
    with pytest.raises(ValueError, match='create_parameters=False'):
        MegaMoeExperts(precision='mxfp8', local_num_tokens=512, hidden_size=64,
                       intermediate_size=64, num_experts=6, top_k=2)
    with pytest.raises(ValueError, match='physical weight roots'):
        MegaMoeExperts.from_module(torch.nn.Module(), top_k=2, precision='mxfp8')
    with pytest.raises(ValueError, match='precision'):
        get_backend('unknown')


def test_mx_passes_current_roots_counts_and_no_clone_to_autograd():
    roots = (object(), object())
    layer = SimpleNamespace(local_experts=6, hidden_size=64, intermediate_size=64,
                            nograd_no_clone=True)
    hidden, flat, ids, probs, counts, resources = [object() for _ in range(6)]
    with (patch.object(mx_backend, 'MXFP8Execution') as execution,
          patch.object(mx_backend, 'execute_mxfp8_experts') as run):
        result = mx_backend.forward(layer, hidden, flat, ids, probs, counts, roots, resources)
        execution.assert_called_once_with(resources, tokens_per_expert=counts, nograd_no_clone=True)
        run.assert_called_once_with(hidden, *roots, ids, probs, ((6, 64, 128), (6, 64, 64)),
                                    execution.return_value)
        assert result is run.return_value

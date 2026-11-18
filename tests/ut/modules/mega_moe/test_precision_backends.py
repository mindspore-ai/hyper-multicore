"""The common expert API preserves parameter roots and backend contracts."""
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

import pytest
import torch

from hyper_multicore import MegaMoeExperts
from hyper_multicore.modules.mega_moe import module as experts_module
from hyper_multicore.modules.mega_moe.backends import get_backend
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend as mx_backend
from hyper_multicore.modules.mega_moe.resources import _MegaMoeExecutionResources


@pytest.mark.parametrize('precision', ['bf16', 'mxfp8'])
def test_one_frontend_dispatches_without_owning_external_weights(precision):
    """Verify backend selection with caller-owned weights."""
    layer = MegaMoeExperts(precision=precision, hidden_size=64,
                           intermediate_size=64, num_experts=6, top_k=2,
                           )
    try:
        assert not layer.state_dict()
        backend = get_backend(precision)
        assert issubclass(backend.Resources, _MegaMoeExecutionResources)
        hidden, flat, ids, probs, counts, resources = [object() for _ in range(6)]
        roots = (object(), object())
        route = object()
        with (patch.object(experts_module, 'prepare_expert_invocation', return_value=nullcontext(route)) as prepare,
              patch.object(layer, '_validate_forward_inputs', return_value=flat),
              patch.object(layer, '_get_execution_resources', return_value=resources),
              patch.object(torch.npu, 'is_current_stream_capturing', return_value=False),
              patch.object(backend, 'forward', return_value=object()) as forward):
            result = layer(hidden, ids, probs, tokens_per_expert=counts, expert_weights=roots)
            assert result is forward.return_value
            prepare.assert_called_once_with(flat, ids, probs, counts, roots, resources, weight_error=False)
            forward.assert_called_once_with(layer, hidden, flat, ids, probs, roots, resources, route)
        assert not layer.state_dict()
    finally:
        layer.close()


def test_frontend_never_owns_parameters_and_rejects_legacy_entry_points():
    """Check parameter ownership and the supported frontend interface."""
    for precision in ('bf16', 'mxfp8'):
        layer = MegaMoeExperts(precision=precision, hidden_size=64, intermediate_size=64, num_experts=6, top_k=2)
        try:
            assert not layer.state_dict()
            assert not hasattr(layer, 'from_module')
        finally:
            layer.close()
    with pytest.raises(TypeError, match='create_parameters'):
        MegaMoeExperts(hidden_size=64, intermediate_size=64, num_experts=6, top_k=2, create_parameters=True)
    with pytest.raises(ValueError, match='precision'):
        get_backend('unknown')


def test_mx_passes_current_roots_and_prepared_route_to_autograd():
    """Pass current roots and the prepared route through to autograd."""
    roots = (object(), object())
    layer = SimpleNamespace(local_experts=6, hidden_size=64, intermediate_size=64,
                            nograd_no_clone=True)
    hidden, flat, ids, probs, resources, route = [object() for _ in range(6)]
    with (patch.object(mx_backend, 'MXFP8Execution') as execution,
          patch.object(mx_backend, 'execute_mxfp8_experts') as run):
        result = mx_backend.forward(layer, hidden, flat, ids, probs, roots, resources, route)
        execution.assert_called_once_with(resources, nograd_no_clone=True)
        run.assert_called_once_with(hidden, *roots, route, probs, ((6, 64, 128), (6, 64, 64)),
                                    execution.return_value)
        assert result is run.return_value

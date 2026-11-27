"""The common expert API preserves parameter roots and backend contracts."""
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
from contextlib import nullcontext

import pytest
import torch

from hyper_multicore import MegaMoeExperts
from hyper_multicore.diagnostics import DiagnosticAction, dumper, profiler
from hyper_multicore.modules.mega_moe import module as experts_module
from hyper_multicore.modules.mega_moe.backends import get_backend
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend as mx_backend
from hyper_multicore.modules.mega_moe.backends.mxfp8 import execution as mx_execution
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
            prepare.assert_called_once_with(flat, ids, probs, counts, roots, resources, weight_error=False, planner=None)
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


@pytest.mark.parametrize(
    "direction, filters, action, global_rank, warns",
    [
        ("forward", {"kernel_names": "mega_gate"}, DiagnosticAction.RECORD, None, False),
        ("forward", {"directions": "backward"}, DiagnosticAction.RECORD, None, False),
        ("forward", {"ranks": 1}, DiagnosticAction.RECORD, None, False),
        ("forward", {}, DiagnosticAction.WARMUP, None, False),
        ("forward", {"kernel_names": "HyperMegaMoeMxfp8"}, DiagnosticAction.RECORD, None, True),
        ("backward", {"level": "task", "stages": "Dispatch", "kernel_names": "HyperMegaMoeMxfp8Grad"},
         DiagnosticAction.RECORD, None, True),
        ("forward", {"ranks": 5}, DiagnosticAction.RECORD, 5, True),
        ("forward", {"ranks": 1}, DiagnosticAction.RECORD, 5, False),
    ],
    ids=[
        "kernel-miss", "direction-miss", "rank-miss", "warmup", "kernel-match", "task-match",
        "global-rank-match", "ep-rank-miss",
    ],
)
def test_mx_dump_selection_preserves_ordinary_launch(tmp_path, caplog, direction, filters, action, global_rank, warns):
    """Apply dump filters without requesting MXFP8 diagnostic operands."""
    kernel_name = "HyperMegaMoeMxfp8" if direction == "forward" else "HyperMegaMoeMxfp8Grad"
    runtime = SimpleNamespace(host=SimpleNamespace(native_operator=kernel_name), runtime=object(), event_capacity=1)
    ep_rank = 0 if global_rank is None else 1
    plan = SimpleNamespace(spec=SimpleNamespace(rank_id=ep_rank))
    buffers, events = [object()], torch.zeros(4)
    config = dumper.DumpConfig(tmp_path, **filters)
    execution = mx_execution.MXFP8Execution(None)
    with (
        patch.object(torch.distributed, "is_initialized", return_value=global_rank is not None),
        patch.object(torch.distributed, "get_rank", return_value=global_rank),
        patch.object(mx_execution, "prepare_mxfp8_launch") as prepare,
        patch.object(mx_execution, "launch_mxfp8_experts") as launch,
        dumper.dump(config, schedule=lambda _: action) as session,
    ):
        execution._launch(buffers, runtime, events, plan, direction)
        execution._launch(buffers, runtime, events, plan, direction)
    assert prepare.call_args_list == [call(buffers, runtime.runtime, events, plan.spec)] * 2
    assert launch.call_args_list == [
        call(buffers, runtime.runtime, events, events, prepare.return_value, direction=direction),
    ] * 2
    if warns:
        assert len(caplog.messages) == 1
        rank = ep_rank if global_rank is None else global_rank
        expected = f"kernel={kernel_name} rank={rank} direction={direction} step=0 reason=unsupported_kernel"
        assert expected in caplog.messages[0]
        assert "invocation=" not in caplog.messages[0]
    else:
        assert not caplog.messages
    assert not session.last_manifest_paths
    assert not list(tmp_path.iterdir())


def test_mx_profiler_uses_prepared_launch_arguments():
    """Continue launching with the runtime and buffer supplied by the profiler."""
    buffers, events = [object()], torch.zeros(4)
    prepared = SimpleNamespace(runtime_config_tensor=object(), event_counters=object(), diagnostic_buffer=object())
    diagnostics = SimpleNamespace(launch=Mock(return_value=nullcontext(prepared)))
    runtime = SimpleNamespace(host=SimpleNamespace(native_operator="HyperMegaMoeMxfp8"), runtime=object(),
                              event_capacity=1, diagnostics=diagnostics)
    plan = SimpleNamespace(spec=SimpleNamespace(rank_id=0))
    with (
        patch.object(mx_execution, "prepare_mxfp8_launch") as prepare,
        patch.object(mx_execution, "launch_mxfp8_experts") as launch,
        profiler.profile(),
    ):
        mx_execution.MXFP8Execution(None)._launch(buffers, runtime, events, plan)
    prepare.assert_called_once_with(buffers, prepared.runtime_config_tensor, prepared.event_counters, plan.spec)
    launch.assert_called_once_with(buffers, prepared.runtime_config_tensor, prepared.event_counters,
                                   prepared.diagnostic_buffer, prepare.return_value, direction="forward")

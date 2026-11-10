"""The public diagnostic boundary must preserve MX worker and invocation identity."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_multicore.diagnostics.api import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.diagnostics.common.session import _activate_session, _deactivate_session


def test_active_worker_metadata_reaches_the_generic_runtime():
    with patch('hyper_multicore.diagnostics.api.PreparedDiagnosticRuntime') as runtime:
        adapter = prepare_mega_kernel_diagnostics(object(), object(), kernel_name='MX',
            parameter_count=68, device='cpu', vector_worker_stride=1, active_vector_workers=32)
    assert adapter._runtime is runtime.return_value
    spec = runtime.call_args.args[2]
    assert spec.active_vector_workers == 32
    assert spec.vector_worker_stride == 1


def test_invocation_tags_reach_the_selected_profiler():
    adapter = MegaKernelDiagnostics(SimpleNamespace(parameter_count=1))
    selected = Mock()
    selected.is_capture_enabled.return_value = True
    _activate_session('profiler', selected)
    try:
        tags = {'layer': '25', 'phase': 'recompute'}
        adapter.launch([object()], direction='forward', tags=tags)
        assert selected.prepare_call.call_args.args[0].tags == tags
    finally:
        _deactivate_session(selected)

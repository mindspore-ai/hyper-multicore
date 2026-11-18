"""Partial-gradient precision changes storage, not scheduling or activation math."""

# The assertions verify internal ownership and the native ABI.
# pylint: disable=protected-access
import ctypes
from dataclasses import replace

import pytest

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.schedule import explain_schedule
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.tasks.gmm_mxfp8 import MxGmmDescriptor, MX_GMM_OUTPUT_FP32


def test_partial_gradient_policy_reaches_common_resource_contract():
    """Pass the partial gradient dtype into the shared execution resources."""
    layer = MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                           numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype="float32"))
    try:
        assert layer._resource_group.specification['partial_weight_grad_dtype'] == 'float32'
    finally:
        layer.close()
    with pytest.raises(ValueError, match='partial_weight_grad_dtype'):
        MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                       numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype='float16'))


def test_mx_fp32_partials_have_disjoint_full_width_outputs_and_same_dag():
    """Check that FP32 partial gradients preserve the graph and output layout."""
    spec = MegaMoeSpec(512, 512, 128, 24, 8, 1.25, 8192, 4, None, 0, 32,
                       precision='mxfp8', transport_engine='udma', dispatch_split=512, combine_split=512)
    ordinary = build_mxfp8_runtime(spec, forward=False)
    fp32 = build_mxfp8_runtime(replace(spec, partial_weight_grad_dtype='float32'), forward=False)
    before, after = explain_schedule(ordinary.config), explain_schedule(fp32.config)
    assert before['queues'] == after['queues']
    assert before['tasks'] == after['tasks']
    assert before['event_triggers'] == after['event_triggers']
    size = ctypes.sizeof(MxGmmDescriptor)
    records = [MxGmmDescriptor.from_buffer_copy(fp32.gmm_descriptors, i)
               for i in range(0, len(fp32.gmm_descriptors), size)]
    seen = 0
    for record in records:
        if record.layout == 2:  # TN partial weight gradients.
            assert record.output_dtype == MX_GMM_OUTPUT_FP32
            expert = record.group_list_offset // 8
            assert record.y_offset == expert * record.m * record.n * 4
            seen += 1
        else:
            assert record.output_dtype == 0
    assert seen == 2 * spec.local_experts * spec.num_cube_cores
    assert build_mxfp8_runtime(spec, forward=True).gmm_descriptors == build_mxfp8_runtime(
        replace(spec, partial_weight_grad_dtype='float32'), forward=True).gmm_descriptors

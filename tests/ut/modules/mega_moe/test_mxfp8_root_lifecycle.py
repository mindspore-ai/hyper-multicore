"""Backward must resolve freshly gathered physical storage on the original root."""
import torch

from hyper_multicore.modules.mega_moe.backends.mxfp8.autograd import MXFP8ForwardState, execute_mxfp8_experts


def physical(root, shape):
    e, m, n = shape
    root.col_data = torch.empty(shape, dtype=torch.float8_e4m3fn)
    root.col_scale = torch.empty((e, m // 64, n, 2), dtype=torch.uint8)
    root.row_data = torch.empty(shape, dtype=torch.float8_e4m3fn)
    root.row_scale = torch.empty((e, m, n // 64, 2), dtype=torch.uint8)


def test_fresh_physical_root_in_backward_and_no_grad_retention():
    shapes = ((1, 64, 128), (1, 64, 64))
    roots = [torch.nn.Parameter(torch.empty(shape, dtype=torch.bfloat16)) for shape in shapes]
    for root, shape in zip(roots, shapes):
        physical(root, shape)
    x = torch.ones((4, 64), dtype=torch.bfloat16, requires_grad=True)
    probs = torch.ones((4, 1), requires_grad=True)

    class Execution:
        def forward(self, hidden, ids, probabilities, w1, w2, retained):
            self.retained = retained
            self.old_data = w1.col_data
            return MXFP8ForwardState(hidden.clone(), (), None)

        def backward(self, grad, saved, metadata, w1, w2):
            assert w1.col_data.data_ptr() == roots[0].col_data.data_ptr()
            assert w1.col_data.data_ptr() != self.old_data.data_ptr()
            assert w1.row_data is not None and w2.row_data is not None
            return grad, torch.zeros_like(probs), *(torch.zeros_like(root) for root in roots)

    execution = Execution()
    y = execute_mxfp8_experts(x, *roots, torch.zeros((4, 1), dtype=torch.int64), probs, shapes, execution)
    assert execution.retained
    for root, shape in zip(roots, shapes):
        physical(root, shape)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert all(torch.equal(root.grad, torch.zeros_like(root)) for root in roots)
    with torch.no_grad():
        execute_mxfp8_experts(x, *roots, torch.zeros((4, 1), dtype=torch.int64), probs, shapes, execution)
    assert not execution.retained

"""Saved-permutation device checks kept out of the lightweight launcher."""
import torch
import torch_npu

from hyper_multicore.torch import ops


def run_saved_permutation_staging() -> None:
    """Reuse a forward map bitwise, including empty rows and multi-tile payloads."""
    for tokens, top_k, hidden in ((0, 2, 512), (17, 2, 129), (128, 4, 1537), (129, 8, 20001)):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(tokens, hidden, dtype=dtype, device="npu")
            ids = torch.arange(tokens * top_k, dtype=torch.int32, device="npu").view(tokens, top_k) % 16
            mapping = torch.empty(tokens * top_k, dtype=torch.int32, device="npu")
            expected = torch.empty(tokens * top_k, hidden, dtype=dtype, device="npu")
            probs = torch.randn(tokens, top_k, dtype=torch.float32, device="npu")
            expected_probs = torch.empty(tokens * top_k, dtype=torch.float32, device="npu")
            if tokens:
                x[0, :3] = torch.tensor([-0., float("nan"), float("inf")], device="npu", dtype=dtype)
                probs[0, 0] = -0.
                ops.moe_token_permute_out(x, ids, expected, mapping)
                torch_npu.npu_scatter_nd_update_(expected_probs, mapping.view(-1, 1), probs.flatten())
            allocation = torch.full((tokens * top_k + 2, hidden), 17., dtype=dtype, device="npu")
            scalar_allocation = torch.full((tokens * top_k + 32,), 123., dtype=torch.float32, device="npu")
            output, output_probs = allocation[1:-1], scalar_allocation[16:-16]
            for _ in range(3):
                ops.moe_token_permute_from_mapping_out(x, mapping, probs, output, output_probs)
            assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(output_probs.view(torch.uint8), expected_probs.view(torch.uint8))
            assert (allocation[0] == 17).all() and (allocation[-1] == 17).all()
            assert (scalar_allocation[:16] == 123).all() and (scalar_allocation[-16:] == 123).all()

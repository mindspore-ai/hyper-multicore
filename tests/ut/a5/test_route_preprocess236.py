"""Contracts for exact route counts and dtype-preserving metadata scaling."""
import importlib.util
from pathlib import Path

import pytest
import torch


def load_preprocess():
    path = Path(__file__).resolve().parents[3] / "hyper_multicore/modules/mega_moe/preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess236", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_histogram_describes_final_ids_and_preserves_empty_experts():
    module = load_preprocess()
    ids = torch.tensor([0, 3, 3, 7], dtype=torch.int64)
    assert torch.equal(module.fixed_route_counts(ids, 8), torch.tensor([1, 0, 0, 2, 0, 0, 0, 1], dtype=torch.int32))
    assert torch.equal(module.fixed_route_counts(ids[:0], 8), torch.zeros(8, dtype=torch.int32))


def test_invalid_indices_are_rejected_using_existing_global_load_readback():
    module = load_preprocess()
    for ids in (torch.tensor([-1, 0, 3]), torch.tensor([0, 3, 8])):
        counts = module.fixed_route_counts(ids, 8)
        with pytest.raises(ValueError, match="count conservation"):
            module.validate_global_loads([int(counts.sum())], ids.numel())
    module.validate_global_loads([1, 3], 4)


def test_policy_validation_rejects_unsupported_count_geometry():
    module = load_preprocess()
    module.validate_count_policy("baseline", 1, None)
    module.validate_count_policy("histc", 8, None)
    for args in (("invalid", 8, None), ("histc", 1, None), ("histc", 8, 128)):
        with pytest.raises(ValueError):
            module.validate_count_policy(*args)

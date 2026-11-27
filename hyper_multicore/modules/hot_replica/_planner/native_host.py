# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native implementation of the host planner's prefix search."""

import importlib.util
from pathlib import Path


_backend = None
_PLANNER_ABI = 3


def load_replica_planner(library: Path) -> None:
    """Bind the planner built for the selected SHMEM adapter's Python ABI."""
    global _backend  # pylint: disable=global-statement
    if _backend is not None:
        if Path(_backend.__file__).resolve() != library.resolve():
            raise RuntimeError("Replica planner is already loaded from another adapter")
        return
    spec = importlib.util.spec_from_file_location(__package__ + "._replica_host", library)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Replica planner from {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _backend = module


def select_prefix(totals, counts, config, quantum, levels, nodes, multiplier, prices, capacity):
    """Parse the counts once and return the winning moves and best visited peak."""
    if getattr(_backend, "PLANNER_ABI", None) != _PLANNER_ABI:
        return None
    if len(prices.phases) > 2 or max(quantum, *levels, capacity or 0) > 2**30:
        return None
    ranks = config.ep_size
    return _backend.select_prefix(
        tuple(totals), counts, ranks, config.replica_slots_per_rank, quantum, levels,
        nodes or (0,) * ranks, prices.phases, multiplier, prices.compute, prices.setup,
        -1 if capacity is None else capacity, prices.search_single_slot,
    )

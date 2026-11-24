# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared host search and aligned source quota materialization."""

import importlib.util
from pathlib import Path


_backend = None


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


def select_prefix(totals, counts, config, quantum, levels, nodes, multiplier, prices, capacity, resident, resize):
    """Parse the counts once and return the winning moves and best visited peak."""
    native = getattr(_backend, "select_prefix", None)
    if native is None or len(prices.phases) > 2 or max(quantum, *levels, capacity or 0, resident or 0) > 2**30:
        return None
    ranks = config.ep_size
    return native(tuple(totals), counts, ranks, config.replica_slots_per_rank, quantum, levels,
                  nodes or (0,) * ranks, prices.phases, multiplier, prices.compute, prices.setup, resize,
                  -1 if capacity is None else capacity, resident or 0)

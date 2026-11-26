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


def select_prefix(totals, counts, config, quantum, levels, nodes, multiplier, prices, capacity, resident, resize,
                  minimum_rows=1, peer_edges=()):
    """Parse the counts once and return the winning moves and best visited peak."""
    single_slot = prices.search_single_slot and config.replica_slots_per_rank > 1
    extended = prices.matrix_copy or peer_edges
    if single_slot:
        method = "select_prefix_with_slot_search"
    elif extended:
        method = "select_prefix_with_matrix_cost"
    else:
        method = "select_prefix" if minimum_rows == 1 else "select_prefix_with_minimum_rows"
    native = getattr(_backend, method, None)
    if native is None or len(prices.phases) > 2 or max(quantum, *levels, capacity or 0, resident or 0) > 2**30:
        return None
    ranks = config.ep_size
    args = (tuple(totals), counts, ranks, config.replica_slots_per_rank, quantum, levels,
            nodes or (0,) * ranks, prices.phases, multiplier, prices.compute, prices.setup, resize,
            -1 if capacity is None else capacity, resident or 0)
    if single_slot:
        return native(*args, minimum_rows, prices.matrix_copy, peer_edges or None, True)
    if extended:
        return native(*args, minimum_rows, prices.matrix_copy, peer_edges or None)
    return native(*args) if minimum_rows == 1 else native(*args, minimum_rows)

def aligned_dispatch(counts, slots, spans, ranks, home, width):
    """Intersect original source intervals with the chosen aligned fragments."""
    native = getattr(_backend, "aligned_dispatch", None)
    if native is None or ranks > 128 or width > 8192 // ranks:
        return None
    return native(counts, slots, spans, ranks, home, width)

# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native search and quota materialization for the aligned MX planner."""

import importlib.util
from pathlib import Path


_backend = None


def load_mx_planner(library: Path) -> None:
    """Bind the planner built for the selected SHMEM adapter's Python ABI."""
    global _backend  # pylint: disable=global-statement
    if _backend is not None:
        if Path(_backend.__file__).resolve() != library.resolve():
            raise RuntimeError("MX replica planner is already loaded from another adapter")
        return
    spec = importlib.util.spec_from_file_location(__package__ + "._replica_host", library)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MX replica planner from {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _backend = module


def _search_aligned(totals, ranks, budget, minimum, nodes):
    if _backend is None or minimum > 2**30:
        return None
    return _backend.search_aligned(totals, ranks, budget, minimum, nodes)


def _search_whole(totals, ranks, budget, minimum, capacity_required, nodes):
    if _backend is None or minimum > 2**30:
        return None
    return _backend.search_whole(totals, ranks, budget, minimum, capacity_required, nodes)


def _aligned_dispatch(counts, slots, spans, ranks, home, width):
    if _backend is None or ranks > 128 or width > 8192 // ranks:
        return None
    return _backend.aligned_dispatch(counts, slots, spans, ranks, home, width)

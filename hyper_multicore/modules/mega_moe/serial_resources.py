"""Opt-in workspace sharing for compatible, serially scheduled model layers."""

from __future__ import annotations

from typing import Any
from weakref import WeakSet

_OWNERS: WeakSet[Any] = WeakSet()


def register_serial_executor(executor: Any, policy: str) -> bool:
    """Register an owner without retaining its model or allocating device memory."""
    if policy not in ("0", "1"):
        raise ValueError("HYPER_MULTICORE_MOE_SHARE_EXECUTION_RESOURCES must be 0 or 1")
    if policy == "0":
        return False
    _OWNERS.add(executor)
    return True


def _unbound_single_owner(executor: Any) -> bool:
    """Leave closed, explicit-shared and already allocated groups untouched."""
    group = executor._resource_group
    return (not executor._resource_closed and not group.closed and not group.closing
            and not group.shared and group.resources is None and group.binding is None
            and len(group.members) == 1)


def share_serial_resources_before_first_use(executor: Any) -> None:
    """Batch existing unbound owners through the established sharing/lifetime API.

    Enabling this policy promises serial scheduling of compatible owners. Each
    invocation still retains its own backward tensors; only execution scratch is
    shared. New owners never join a group that has already allocated resources.
    """
    if executor not in _OWNERS or not _unbound_single_owner(executor):
        return
    group = executor._resource_group
    candidates = [owner for owner in _OWNERS
                  if type(owner) is type(executor) and _unbound_single_owner(owner)
                  and owner._resource_group.scope_key == group.scope_key
                  and owner._resource_group.compatibility_key == group.compatibility_key]
    candidates.sort(key=lambda owner: owner._resource_group.identifier)
    if len(candidates) > 1:
        type(executor).share_execution_resources(candidates)

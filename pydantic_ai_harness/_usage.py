"""Shared helpers for nested agent usage accounting."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.usage import UsageLimits


def reserved_usage_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """Hold one request back for a nested model call made from a hook.

    A hook can run after the parent request has passed its pre-request limit
    check. Reserving one slot prevents the nested call from allowing that
    already-approved parent request to exceed ``request_limit``.
    """
    if limits is None or limits.request_limit is None:
        return limits
    return replace(limits, request_limit=max(0, limits.request_limit - 1))

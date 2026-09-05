"""Shared helpers for nested agent usage accounting."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.usage import UsageLimits


def reserved_usage_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """Reserve the pending parent request before a nested model call made from a hook.

    The hook may run after the parent request's limit check. Reducing a finite request limit
    prevents the nested call from spending the request that was already approved for the parent.
    """
    if limits is None or limits.request_limit is None:
        return limits
    return replace(limits, request_limit=max(0, limits.request_limit - 1))


def forwarded_usage_limits(limits: UsageLimits | None, *, reserve_tool_call: bool = False) -> UsageLimits | None:
    """The parent's `UsageLimits` as a nested run started from a function tool should inherit them.

    Every ceiling carries over, which is what makes the budget tree-wide. `count_tokens_before_request`
    does not: it selects a request pipeline rather than setting a budget, and `Model.count_tokens`
    raises `NotImplementedError` on the models that do not implement it. A nested run can be on a
    different model from the parent, so inheriting the flag aborts runs whose parent-side token
    counting works.

    Dropping it costs preflight enforcement on a nested model that does implement `count_tokens`.
    The flag is what folds the pending request's counted tokens and its priced cost into the usage
    checked before the request, so without it `per_request_input_tokens_limit`, `input_tokens_limit`,
    `total_tokens_limit` and `cost_limit` no longer see that request ahead of time and are checked
    against the response instead: one oversized nested request is sent before the ceiling trips.
    Keeping it only where it works needs a run-and-fall-back, tracked in #697; there is no way to
    ask a model whether it implements `count_tokens`.

    Set `reserve_tool_call` when the nested run can itself make tool calls. The tool wrapping it is
    counted against `tool_calls_limit` once it returns, not when it starts, so a nested run checking
    the raw limit spends a budget that does not yet include the call containing it and the tree lands
    one over. `request_limit` never needs this: a function tool runs after the parent's request was
    made and counted.
    """
    if limits is None:
        return None
    tool_calls_limit = limits.tool_calls_limit
    if reserve_tool_call and tool_calls_limit is not None:
        tool_calls_limit = max(0, tool_calls_limit - 1)
    if tool_calls_limit == limits.tool_calls_limit and not limits.count_tokens_before_request:
        return limits
    return replace(limits, tool_calls_limit=tool_calls_limit, count_tokens_before_request=False)

"""Uniform, deterministic markers for elided tool output.

Every path that removes content from a tool return -- spilling it to the store, truncating
it to a budget, capping a read-back -- leaves one of these markers in the model-visible
text. The model always learns that content was removed, how much, and how to get it back
(or that it is gone). One helper builds each marker so the wording stays uniform across
paths, and no marker carries a timestamp or other run-varying data, so identical inputs
produce identical bytes (cache-prefix stability).
"""

from __future__ import annotations

READ_TOOL_NAME = 'read_tool_result'
"""Tool that reads a line or byte range from a spilled payload."""

GREP_TOOL_NAME = 'grep_tool_result'
"""Tool that searches a spilled payload for a pattern."""


def retrieval_hint(handle: str) -> str:
    """The shared 'how to get the full payload' clause, naming both query tools.

    Kept in one place so every marker points the model at the same tools with the same
    phrasing, whatever path produced the elision.
    """
    return (
        f'search it with {GREP_TOOL_NAME}(handle={handle!r}, pattern=...), '
        f'read ranges with {READ_TOOL_NAME}(handle={handle!r}, offset=..., limit=...)'
    )


def spill_header(*, size_desc: str, handle: str) -> str:
    """Lead line of a spilled tool return: what was stored, where, and how to query it."""
    return (
        f'[Tool output too large ({size_desc}); stored as {handle!r}. '
        f'{retrieval_hint(handle)}. A head/tail preview follows.]'
    )


def elision_marker(*, omitted: str, handle: str | None) -> str:
    """One marker for any elided span. `omitted` names the amount (e.g. '12 lines / 400 bytes').

    When `handle` is set the span was spilled and is retrievable through the query tools;
    otherwise it was truncated and the omitted span is gone (re-run the tool to recover it).
    """
    if handle is not None:
        return f'[... {omitted} omitted; {retrieval_hint(handle)} ...]'
    return f'[... {omitted} omitted; not stored, re-run the tool for the full output ...]'


def summary_header(*, size_desc: str, handle: str | None) -> str:
    """Lead line of a summarized tool return: what was replaced, by whom, and how to recover it.

    Like `spill_header`/`elision_marker`, this makes the elision explicit: the model learns the
    body below is a harness-generated summary standing in for the real tool output, not the tool
    result itself. When `handle` is set the original was also spilled and stays retrievable
    through the query tools; otherwise the summary is all that remains (re-run the tool for the
    full output).
    """
    recover = f'{retrieval_hint(handle)}' if handle is not None else 're-run the tool for the full output'
    return f'[Tool output too large ({size_desc}); summarized by harness. {recover}. The summary follows.]'


def missing_handle_message(handle: str) -> str:
    """Guidance returned (not raised) when a handle does not resolve to a stored payload.

    Returned rather than raised so a wrong or expired handle cannot consume a tool retry and
    escalate to a fatal `UnexpectedModelBehavior` (see PR #293). The store's error is not
    echoed -- it can carry the resolved filesystem path, which the model has no need for.
    """
    return (
        f'[No stored tool result for handle {handle!r}. Use the exact handle string from a '
        '"[Tool output too large ... stored as ...]" marker; if the result is no longer '
        'available, re-run the original tool.]'
    )

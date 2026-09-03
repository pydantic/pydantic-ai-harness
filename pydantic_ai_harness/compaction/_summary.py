"""Summary artifact identity shared across compaction and its consumers.

`SummarizingCompaction` replaces the summarized prefix of a history with one
message carrying the summary text. The summary is written as a user-turn part so
the post-compaction request stays valid on OpenAI-compatible backends that
accept a single leading `system` message (SGLang, some vLLM deployments): the
adapter already inserts the run's instructions as one leading system message,
and a second one from the summary makes those backends reject the request.

Identity rides model-invisible metadata, not the text. The marker is carried by
`TextContent.metadata` (the same convention `_pinning` uses) and repeated on the
containing `ModelRequest.metadata` so UI adapters that flatten `TextContent`
can preserve it. Prefix matching on an unmarked user turn would let a message
that merely opens with the same sentence hide itself from rendering, search,
and goal anchoring. The rendered text still carries `SUMMARY_PREFIX` for the
model to read; only the legacy `SystemPromptPart` shape -- which users cannot
author -- is identified by that prefix. Histories written before the shape
change carry the legacy shape and every consumer of the identity below matches
all three persisted forms.

Deliberately internal (not exported from the package `__init__`): summary
identity is an implementation detail of the strategies. `conversation_search`
and `system_reminders` mirror the constants they need as local literals instead
of importing them, keeping those capabilities decoupled from compaction
internals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
    TextContent,
    UserPromptPart,
)

if TYPE_CHECKING:
    from datetime import datetime

    from pydantic_ai.messages import ModelMessage

SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'
"""The exact prefix a `SummarizingCompaction` summary part carries in its text."""

_SUMMARY_METADATA_KEY = 'pydantic-ai-harness.compaction.summary.v1'
"""Key repeated on the containing request so UI adapters can preserve summary identity."""

_SUMMARY_METADATA = {_SUMMARY_METADATA_KEY: True}
"""Model-invisible marker identifying a summary part; mirrors `_pinning`'s convention."""


def make_summary_part(body: str, *, timestamp: datetime | None = None) -> UserPromptPart:
    """Build the user-turn part a compaction writes its summary into.

    The prefix is part of the rendered text (the model reads it as context);
    the metadata marker is what consumers match on, so a user turn cannot
    forge or collide with summary identity by opening with the same sentence.
    """
    content = TextContent(content=f'{SUMMARY_PREFIX}{body}', metadata=dict(_SUMMARY_METADATA))
    return UserPromptPart(content=[content], timestamp=timestamp) if timestamp else UserPromptPart(content=[content])


def make_summary_message(body: str, system_parts: Sequence[SystemPromptPart]) -> ModelRequest:
    """Build a summary request with part- and message-level identity markers."""
    return ModelRequest(parts=[*system_parts, make_summary_part(body)], metadata=dict(_SUMMARY_METADATA))


def _marked_summary_text(part: UserPromptPart) -> str | None:
    """Return the marked summary body, or `None` when the part carries no marker."""
    if isinstance(part.content, str):
        return None
    for item in part.content:
        if isinstance(item, TextContent) and item.metadata == _SUMMARY_METADATA:
            text = item.content
            return text[len(SUMMARY_PREFIX) :] if text.startswith(SUMMARY_PREFIX) else text
    return None


def is_summary_part(part: ModelRequestPart, *, message_metadata: Mapping[str, object] | None = None) -> bool:
    """Return whether *part* is a compaction summary, in either written shape.

    Matches the structured marked user turn, its flattened request-marked form,
    and the earlier `SystemPromptPart` shape (identified by the prefix, which
    users cannot author), so persisted histories keep reading as summarized. A
    plain user turn whose text starts with the prefix is not a summary.
    """
    return summary_text(part, message_metadata=message_metadata) is not None


def summary_text(part: ModelRequestPart, *, message_metadata: Mapping[str, object] | None = None) -> str | None:
    """Return the summary body when *part* is a summary in either shape, else `None`.

    The prefix is stripped, so the caller receives the text the summarizer
    produced. Doubles as a type-narrowing guard for the content shapes.
    """
    if isinstance(part, UserPromptPart):
        marked = _marked_summary_text(part)
        if marked is not None:
            return marked
        if (
            message_metadata is not None
            and message_metadata.get(_SUMMARY_METADATA_KEY) is True
            and isinstance(part.content, str)
            and part.content.startswith(SUMMARY_PREFIX)
        ):
            return part.content[len(SUMMARY_PREFIX) :]
        return None
    if isinstance(part, SystemPromptPart) and part.dynamic_ref is None and part.content.startswith(SUMMARY_PREFIX):
        return part.content[len(SUMMARY_PREFIX) :]
    return None


def normalize_legacy_summaries(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Rewrite `SystemPromptPart` summaries into the marked user-turn shape, copy-on-write.

    Identical rendered text, different part type; the input list is left untouched and a
    new one is returned only when a rewrite happened. A history persisted by an
    older release maps to two leading `system` messages once the adapter inserts
    the run's instructions, which single-system backends reject; rewriting on the
    way through keeps those histories sendable instead of waiting for the next
    compaction to replace the summary. Returns the input list when nothing
    needed rewriting, so callers can rely on the identity as an unchanged flag.

    A part with a `dynamic_ref` is left alone even when its text carries the
    prefix: rewriting it would drop the ref core uses to re-evaluate the
    instruction every turn, freezing it in the wrong role. The prefix match on
    system parts cannot be triggered by user input.
    """
    changed = False
    out: list[ModelMessage] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest) or not any(
            isinstance(part, SystemPromptPart) and part.dynamic_ref is None and part.content.startswith(SUMMARY_PREFIX)
            for part in msg.parts
        ):
            out.append(msg)
            continue
        parts: list[ModelRequestPart] = []
        for part in msg.parts:
            if (
                isinstance(part, SystemPromptPart)
                and part.dynamic_ref is None
                and part.content.startswith(SUMMARY_PREFIX)
            ):
                parts.append(make_summary_part(part.content[len(SUMMARY_PREFIX) :], timestamp=part.timestamp))
                changed = True
            else:
                parts.append(part)
        metadata = {**(msg.metadata or {}), **_SUMMARY_METADATA}
        out.append(replace(msg, parts=parts, metadata=metadata))
    return out if changed else messages

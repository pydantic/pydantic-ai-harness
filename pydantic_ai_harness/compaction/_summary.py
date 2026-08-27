"""Summary artifact identity shared across compaction and its consumers.

`SummarizingCompaction` replaces the summarized prefix of a history with one
message carrying the summary text. The summary is written as a user-turn part so
the post-compaction request stays valid on OpenAI-compatible backends that
accept a single leading `system` message (SGLang, some vLLM deployments): the
adapter already inserts the run's instructions as one leading system message,
and a second one from the summary makes those backends reject the request.

Identity rides model-invisible `TextContent.metadata` (the same convention
`_pinning` uses), not the text: a user turn is ordinary user-controlled text,
so prefix matching on it would let a message that merely opens with the same
sentence hide itself from rendering, search, and goal anchoring. The rendered
text still carries `SUMMARY_PREFIX` for the model to read; only the legacy
`SystemPromptPart` shape -- which users cannot author -- is identified by that
prefix. Histories written before the shape change carry the legacy shape and
every consumer of the identity below matches both.

Deliberately internal (not exported from the package `__init__`): summary
identity is an implementation detail of the strategies. `conversation_search`
and `system_reminders` mirror the constants they need as local literals instead
of importing them, keeping those capabilities decoupled from compaction
internals.
"""

from __future__ import annotations

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

_SUMMARY_METADATA = {'pydantic-ai-harness.compaction.summary.v1': True}
"""Model-invisible marker identifying a summary part; mirrors `_pinning`'s convention."""


def make_summary_part(body: str, *, timestamp: datetime | None = None) -> UserPromptPart:
    """Build the user-turn part a compaction writes its summary into.

    The prefix is part of the rendered text (the model reads it as context);
    the metadata marker is what consumers match on, so a user turn cannot
    forge or collide with summary identity by opening with the same sentence.
    """
    content = TextContent(content=f'{SUMMARY_PREFIX}{body}', metadata=dict(_SUMMARY_METADATA))
    return UserPromptPart(content=[content], timestamp=timestamp) if timestamp else UserPromptPart(content=[content])


def _marked_summary_text(part: UserPromptPart) -> str | None:
    """Return the marked summary body, or `None` when the part carries no marker."""
    if isinstance(part.content, str):
        return None
    for item in part.content:
        if isinstance(item, TextContent) and item.metadata == _SUMMARY_METADATA:
            text = item.content
            return text[len(SUMMARY_PREFIX) :] if text.startswith(SUMMARY_PREFIX) else text
    return None


def is_summary_part(part: ModelRequestPart) -> bool:
    """Return whether *part* is a compaction summary, in either written shape.

    Matches the marked user-turn shape and the earlier `SystemPromptPart` shape
    (identified by the prefix, which users cannot author), so persisted
    histories from older releases keep reading as summarized. A plain user
    turn whose text starts with the prefix is not a summary.
    """
    return summary_text(part) is not None


def summary_text(part: ModelRequestPart) -> str | None:
    """Return the summary body when *part* is a summary in either shape, else `None`.

    The prefix is stripped, so the caller receives the text the summarizer
    produced. Doubles as a type-narrowing guard for the content shapes.
    """
    if isinstance(part, UserPromptPart):
        return _marked_summary_text(part)
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
        out.append(replace(msg, parts=parts))
    return out if changed else messages

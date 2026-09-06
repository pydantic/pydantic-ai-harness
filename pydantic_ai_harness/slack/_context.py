"""Typed context for a Slack-hosted agent run."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackFile:
    """A file attached to the Slack message that started this run."""

    file_id: str
    name: str | None = None
    mimetype: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackContext:
    """Slack identities and conversation coordinates for one agent run."""

    team_id: str
    channel_id: str
    thread_ts: str
    message_ts: str
    user_id: str
    enterprise_id: str | None = None
    files: tuple[SlackFile, ...] = ()


_slack_context: ContextVar[SlackContext | None] = ContextVar('pydantic_ai_harness_slack_context', default=None)


@contextmanager
def bind_slack_run(context: SlackContext) -> Generator[None]:
    token = _slack_context.set(context)
    try:
        yield
    finally:
        _slack_context.reset(token)


def current_slack_context() -> SlackContext | None:
    """Return the Slack context bound to the current agent run, if any."""
    return _slack_context.get()

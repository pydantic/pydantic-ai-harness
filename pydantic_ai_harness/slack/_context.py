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
_slack_token: ContextVar[str | None] = ContextVar('pydantic_ai_harness_slack_token', default=None)


@contextmanager
def bind_slack_run(context: SlackContext, *, token: str | None = None) -> Generator[None]:
    """Bind the Slack context, and the token the sender's Slack tools should use, to the current run.

    `SlackApp` calls this around each agent run. `Slack` reads the token through
    `current_slack_token` when it was constructed without one.
    """
    context_reset = _slack_context.set(context)
    token_reset = _slack_token.set(token)
    try:
        yield
    finally:
        _slack_token.reset(token_reset)
        _slack_context.reset(context_reset)


def current_slack_context() -> SlackContext | None:
    """Return the Slack context bound to the current agent run, if any."""
    return _slack_context.get()


def current_slack_token() -> str | None:
    """Return the Slack token bound to the current agent run, if any. Private to this package."""
    return _slack_token.get()

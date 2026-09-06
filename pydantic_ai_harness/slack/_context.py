"""Typed context for a Slack-hosted agent run."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class _SlackRun:
    context: SlackContext
    user_token: str | None = field(default=None, repr=False)


_slack_run: ContextVar[_SlackRun | None] = ContextVar('pydantic_ai_harness_slack_run', default=None)


def _current_slack_run() -> _SlackRun | None:
    return _slack_run.get()


def current_slack_user_token() -> str | None:
    run = _current_slack_run()
    return None if run is None else run.user_token


@contextmanager
def bind_slack_run(context: SlackContext, user_token: str | None = None) -> Generator[None]:
    token = _slack_run.set(_SlackRun(context=context, user_token=user_token))
    try:
        yield
    finally:
        _slack_run.reset(token)


def current_slack_context() -> SlackContext | None:
    """Return the Slack context bound to the current agent run, if any."""
    run = _current_slack_run()
    return None if run is None else run.context

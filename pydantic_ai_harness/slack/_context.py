"""Typed context for a Slack-hosted agent run."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a non-empty string')


def _require_optional_string(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f'{name} must be a string or None')


def _require_files(value: object) -> None:
    if not isinstance(value, tuple) or not all(  # pyright: ignore[reportUnknownVariableType]
        isinstance(file, SlackFile)
        for file in value  # pyright: ignore[reportUnknownVariableType]
    ):
        raise ValueError('files must be a tuple of SlackFile instances')


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackFile:
    """A file attached to the Slack message that started this run."""

    file_id: str
    name: str | None = None
    mimetype: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.file_id, 'file_id')
        _require_optional_string(self.name, 'name')
        _require_optional_string(self.mimetype, 'mimetype')


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

    def __post_init__(self) -> None:
        for name in ('team_id', 'channel_id', 'thread_ts', 'message_ts', 'user_id'):
            _require_non_empty_string(getattr(self, name), name)
        _require_optional_string(self.enterprise_id, 'enterprise_id')
        _require_files(self.files)

    @property
    def conversation_id(self) -> str:
        return f'{self.team_id}:{self.channel_id}:{self.thread_ts}'


@dataclass(frozen=True)
class _SlackRun:
    context: SlackContext
    user_token: str | None = field(default=None, repr=False)


_slack_run: ContextVar[_SlackRun | None] = ContextVar('pydantic_ai_harness_slack_run', default=None)


def _current_slack_run() -> _SlackRun | None:
    return _slack_run.get()


def _current_slack_user_token() -> str | None:  # pyright: ignore[reportUnusedFunction]
    run = _current_slack_run()
    return None if run is None else run.user_token


@contextmanager
def _bind_slack_run(context: SlackContext, user_token: str | None = None) -> Generator[None]:  # pyright: ignore[reportUnusedFunction]
    token = _slack_run.set(_SlackRun(context=context, user_token=user_token))
    try:
        yield
    finally:
        _slack_run.reset(token)


def current_slack_context() -> SlackContext | None:
    """Return the Slack context bound to the current agent run, if any."""
    run = _current_slack_run()
    return None if run is None else run.context

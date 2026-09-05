"""Typed context for the Slack message that started an agent run."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_harness.slack._client import SlackClient


@dataclass(frozen=True, slots=True)
class SlackContextEntity:
    """One item in Slack's relevance-ordered active-view context."""

    entity_type: str
    """Slack entity type, for example `slack#/types/channel_id`."""

    value: str
    """Slack ID of the channel, thread, canvas, list, or other entity."""

    team_id: str | None = None
    """Workspace containing this entity, when Slack supplies one."""


@dataclass(frozen=True, slots=True)
class SlackContext:
    """The Slack identities and conversation coordinates for one agent run."""

    channel_id: str
    """Conversation containing the message."""

    thread_ts: str
    """Timestamp of the root message for this conversation."""

    message_ts: str
    """Timestamp of the message that started this run."""

    user_id: str
    """Slack user that sent the message."""

    team_id: str | None = None
    """Workspace containing the conversation."""

    enterprise_id: str | None = None
    """Enterprise Grid organization containing the workspace, when present."""

    active_entities: tuple[SlackContextEntity, ...] = ()
    """What the user is viewing, ordered from most to least relevant."""

    user_token: str | None = None
    """OAuth user token for this invocation. Kept out of `repr`."""

    def __repr__(self) -> str:
        fields = (
            f'channel_id={self.channel_id!r}',
            f'thread_ts={self.thread_ts!r}',
            f'message_ts={self.message_ts!r}',
            f'user_id={self.user_id!r}',
            f'team_id={self.team_id!r}',
            f'enterprise_id={self.enterprise_id!r}',
            f'active_entities={self.active_entities!r}',
            'user_token=<redacted>' if self.user_token is not None else 'user_token=None',
        )
        return f'SlackContext({", ".join(fields)})'


_current_context: ContextVar[SlackContext | None] = ContextVar('pydantic_ai_harness_slack_context', default=None)
_current_delivery_client: ContextVar[SlackClient | None] = ContextVar(
    'pydantic_ai_harness_slack_delivery_client', default=None
)
_fixed_mcp_fallback_allowed: ContextVar[bool] = ContextVar(
    'pydantic_ai_harness_slack_fixed_mcp_fallback_allowed', default=True
)


def current_slack_context() -> SlackContext | None:
    """Return the Slack context bound to the current agent run, if any."""
    return _current_context.get()


def current_delivery_client() -> SlackClient | None:
    """Return the Bolt-authorized client bound to the current Slack run."""
    return _current_delivery_client.get()


def fixed_mcp_fallback_allowed() -> bool:
    """Whether this run may fall back to the process-wide MCP token."""
    return _fixed_mcp_fallback_allowed.get()


@contextmanager
def bind_slack_context(
    context: SlackContext, client: SlackClient | None = None, *, allow_fixed_mcp_fallback: bool = True
) -> Generator[None]:
    """Bind `context` while a Slack-hosted agent run is executing."""
    context_token = _current_context.set(context)
    client_token = _current_delivery_client.set(client)
    fallback_token = _fixed_mcp_fallback_allowed.set(allow_fixed_mcp_fallback)
    try:
        yield
    finally:
        _fixed_mcp_fallback_allowed.reset(fallback_token)
        _current_delivery_client.reset(client_token)
        _current_context.reset(context_token)

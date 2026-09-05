"""Typed context for the Slack message that started an agent run."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_harness.slack._client import SlackClient


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackMessageContext:
    """A message Slack says is relevant to the user's active view."""

    channel_id: str
    """Conversation containing the message."""

    message_ts: str
    """Timestamp identifying the message."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackFile:
    """A file attached to the Slack message that started this run."""

    file_id: str
    """Slack file ID accepted by `slack_read_file`."""

    name: str | None = None
    """Original file name, when Slack supplies one."""

    mimetype: str | None = None
    """Media type reported by Slack, when present."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackContextEntity:
    """One item in Slack's relevance-ordered active-view context."""

    entity_type: str
    """Slack entity type, for example `slack#/types/channel_id`."""

    value: str | SlackMessageContext
    """Slack ID, or typed coordinates when the entity is a message context."""

    team_id: str | None = None
    """Workspace containing this entity, when Slack supplies one."""

    def __post_init__(self) -> None:
        is_message = self.entity_type == 'slack#/types/message_context'
        if is_message != isinstance(self.value, SlackMessageContext):
            raise ValueError(
                "The 'slack#/types/message_context' entity requires SlackMessageContext; "
                'all other Slack context entities require a string value.'
            )


@dataclass(frozen=True, slots=True, kw_only=True)
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

    files: tuple[SlackFile, ...] = ()
    """Files attached to the message that started this run."""

    @contextmanager
    def bind(self) -> Generator[None]:
        """Bind this context around a run hosted outside `SlackApp`.

        This supplies conversation coordinates only. Configure a fixed MCP token
        on `Slack` and an appropriate tool selection for the external host.
        """
        with bind_slack_context(self):
            yield


_current_context: ContextVar[SlackContext | None] = ContextVar('pydantic_ai_harness_slack_context', default=None)
_current_user_token: ContextVar[str | None] = ContextVar('pydantic_ai_harness_slack_user_token', default=None)
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


def current_user_token() -> str | None:
    """Return the private OAuth credential bound to this Slack run."""
    return _current_user_token.get()


def fixed_mcp_fallback_allowed() -> bool:
    """Whether this run may fall back to the process-wide MCP token."""
    return _fixed_mcp_fallback_allowed.get()


@contextmanager
def bind_slack_context(
    context: SlackContext,
    client: SlackClient | None = None,
    *,
    user_token: str | None = None,
    allow_fixed_mcp_fallback: bool = True,
) -> Generator[None]:
    """Bind `context` while a Slack-hosted agent run is executing."""
    context_token = _current_context.set(context)
    user_token_token = _current_user_token.set(user_token)
    client_token = _current_delivery_client.set(client)
    fallback_token = _fixed_mcp_fallback_allowed.set(allow_fixed_mcp_fallback)
    try:
        yield
    finally:
        _fixed_mcp_fallback_allowed.reset(fallback_token)
        _current_delivery_client.reset(client_token)
        _current_user_token.reset(user_token_token)
        _current_context.reset(context_token)

"""Give an agent tools for reading and writing Slack."""

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, overload, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from pydantic_ai_harness.slack._context import current_slack_token

_INSTRUCTIONS = """\
Slack tools act with the identity of the connected token.
Before answering about discussion elsewhere in Slack, read the relevant thread or channel with these tools rather than guessing.
Distinguish channel-wide results from thread replies.
If you cannot find the needed context, say so instead of making it up.
"""

# Externally owned endpoint verified 2026-09-06 against Slack's official MCP overview
# (https://docs.slack.dev/ai/slack-mcp-server/); recheck when integration changes.
_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'

_SLACK_MCP_ACCEPTS_MARKDOWN_TEXT = (
    'markdown_text'
    in inspect.signature(
        AsyncWebClient.chat_postMessage  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    ).parameters
)


@runtime_checkable
class _SlackResponse(Protocol):
    @overload
    def __getitem__(self, key: Literal['ts']) -> str: ...

    @overload
    def __getitem__(self, key: Literal['messages']) -> list[dict[str, str]]: ...

    def __getitem__(self, key: str) -> object: ...


def _as_slack_response(response: object) -> _SlackResponse:
    if not isinstance(response, _SlackResponse):  # pragma: no cover - Slack SDK returns this response shape
        raise ModelRetry('Slack API returned an invalid response.')
    return response


class _SlackClient(Protocol):
    async def chat_postMessage(
        self, *, channel: str, text: str | None = None, thread_ts: str | None = None, markdown_text: str | None = None
    ) -> object: ...

    async def reactions_add(self, *, channel: str, timestamp: str, name: str) -> object: ...

    async def conversations_replies(self, *, channel: str, ts: str, limit: int) -> object: ...


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack tools using a user or bot token."""

    token: str | None = field(default=None, repr=False)
    id: str | None = 'slack'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Resolve a token into a fresh capability for this run."""
        del ctx
        token = self.token
        if token is None or not token.strip():
            token = current_slack_token()
        if token is None or not token.strip():
            token = os.environ.get('SLACK_USER_TOKEN')
        if token is None or not token.strip():
            token = os.environ.get('SLACK_BOT_TOKEN')
        if token is None or not token.strip():
            raise UserError(
                'Slack tools need a token. Pass Slack(token=...) or set SLACK_USER_TOKEN or SLACK_BOT_TOKEN.'
            )
        return replace(self, token=token.strip())

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge equal-token configurations and reject ambiguous credentials."""
        first = capabilities[0]
        assert isinstance(first, cls)
        for capability in capabilities[1:]:
            assert isinstance(capability, cls)
            if capability.token != first.token:
                raise UserError('Multiple Slack capabilities with different credentials cannot be combined.')
        return super().combine(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Return the Slack toolset for the configured token."""
        if self.token is None:
            return None
        if not self.token.startswith('xoxb-'):
            return MCPToolset(
                _SLACK_MCP_URL,
                id=f'{self.id or "slack"}-mcp',
                headers={'Authorization': f'Bearer {self.token}'},
                include_instructions=True,  # Core defaults this to false; Slack MCP supplies required instructions.
            )

        # Slack's hosted MCP server accepts user tokens only, verified 2026-09-07 against Slack's official MCP server
        # documentation (https://docs.slack.dev/ai/slack-mcp-server/); recheck when token support changes.
        client: _SlackClient = AsyncWebClient(token=self.token)

        async def send_message(channel: str, text: str, thread_ts: str | None = None) -> str:
            """Send a message to a Slack channel."""
            try:
                if _SLACK_MCP_ACCEPTS_MARKDOWN_TEXT:
                    response = _as_slack_response(
                        await client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
                            channel=channel, markdown_text=text, thread_ts=thread_ts
                        )
                    )
                else:
                    response = _as_slack_response(
                        await client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
                            channel=channel, text=text, thread_ts=thread_ts
                        )
                    )
            except SlackApiError as error:
                raise ModelRetry(f'Slack API error: {error}') from error
            return response['ts']

        async def add_reaction(channel: str, timestamp: str, name: str) -> str:
            """Add a reaction to a Slack message."""
            try:
                await client.reactions_add(  # pyright: ignore[reportUnknownMemberType]
                    channel=channel, timestamp=timestamp, name=name
                )
            except SlackApiError as error:
                raise ModelRetry(f'Slack API error: {error}') from error
            return 'ok'

        async def read_thread(channel: str, thread_ts: str, limit: int = 50) -> list[dict[str, str]]:
            """Read one page of replies in a Slack thread."""
            try:
                response = _as_slack_response(
                    await client.conversations_replies(  # pyright: ignore[reportUnknownMemberType]
                        channel=channel,
                        ts=thread_ts,
                        limit=min(limit, 200),
                    )
                )
            except SlackApiError as error:
                raise ModelRetry(f'Slack API error: {error}') from error
            return [
                {'user': message.get('user', ''), 'ts': message['ts'], 'text': message.get('text', '')}
                for message in response['messages']
            ]

        return FunctionToolset(
            [send_message, add_reaction, read_thread],
            id=f'{self.id or "slack"}-bot',
        )

    def get_instructions(self) -> str:
        """Return guidance for using Slack tools."""
        return _INSTRUCTIONS

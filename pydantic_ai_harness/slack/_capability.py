"""Give an agent tools for reading and writing Slack."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

_INSTRUCTIONS = """\
Slack tools act with the identity of the connected token.
Before answering about discussion elsewhere in Slack, read the relevant thread or channel with these tools rather than guessing.
Distinguish channel-wide results from thread replies.
If you cannot find the needed context, say so instead of making it up.
"""

# Slack's hosted MCP server. It accepts user tokens only, so bot tokens get the small toolset below instead.
# Verified 2026-09-07 against https://docs.slack.dev/ai/slack-mcp-server/; recheck when the integration changes.
_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack tools using a user or bot token.

    A user token (`xoxp-`) gives Slack's hosted MCP catalog, acting as that user. A bot token (`xoxb-`) gives
    three built-in tools: `send_message`, `add_reaction`, and `read_thread`.
    """

    token: str | None = field(default=None, repr=False)
    """The Slack token. Defaults to `SLACK_USER_TOKEN`, then `SLACK_BOT_TOKEN`."""
    id: str | None = 'slack'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Return a copy with the token resolved for this run."""
        token = self.token or os.environ.get('SLACK_USER_TOKEN') or os.environ.get('SLACK_BOT_TOKEN')
        if not token:
            raise UserError(
                'Slack tools need a token. Pass Slack(token=...) or set SLACK_USER_TOKEN or SLACK_BOT_TOKEN.'
            )
        return replace(self, token=token)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge equal-token configurations and reject different credentials."""
        first = capabilities[0]
        assert isinstance(first, cls)
        for capability in capabilities[1:]:
            assert isinstance(capability, cls)
            if capability.token != first.token:
                raise UserError('Multiple Slack capabilities with different credentials cannot be combined.')
        return super().combine(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Return the toolset for the resolved token, or `None` before a run resolves one."""
        if self.token is None:
            return None
        if self.token.startswith('xoxb-'):
            return _bot_toolset(self.token, f'{self.id or "slack"}-bot')
        return MCPToolset(
            _SLACK_MCP_URL,
            id=f'{self.id or "slack"}-mcp',
            headers={'Authorization': f'Bearer {self.token}'},
            include_instructions=True,  # Core defaults this to false; Slack MCP supplies required instructions.
        )

    def get_instructions(self) -> str:
        """Return guidance for using Slack tools."""
        return _INSTRUCTIONS


def _bot_toolset(token: str, toolset_id: str) -> FunctionToolset[Any]:
    # The SDK's methods take untyped **kwargs, so the client is typed as Any to keep the tools plain.
    client: Any = AsyncWebClient(token=token)

    async def send_message(channel: str, text: str, thread_ts: str | None = None) -> str:
        """Send a markdown message to a Slack channel, or reply in a thread when `thread_ts` is given."""
        response = await _call(client.chat_postMessage, channel=channel, markdown_text=text, thread_ts=thread_ts)
        return str(response['ts'])

    async def add_reaction(channel: str, timestamp: str, name: str) -> str:
        """Add an emoji reaction, by name without colons, to the message with this timestamp."""
        await _call(client.reactions_add, channel=channel, timestamp=timestamp, name=name)
        return 'ok'

    async def read_thread(channel: str, thread_ts: str, limit: int = 50) -> list[dict[str, str]]:
        """Read up to `limit` replies (at most 200) in a Slack thread."""
        response = await _call(client.conversations_replies, channel=channel, ts=thread_ts, limit=min(limit, 200))
        messages: list[dict[str, Any]] = response['messages']
        return [
            {'user': message.get('user', ''), 'ts': message.get('ts', ''), 'text': message.get('text', '')}
            for message in messages
        ]

    return FunctionToolset([send_message, add_reaction, read_thread], id=toolset_id)


async def _call(method: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    """Call one Slack Web API method, turning a Slack error into a retry the model can act on."""
    try:
        return await method(**kwargs)
    except SlackApiError as error:
        raise ModelRetry(f'Slack API error: {error}') from error

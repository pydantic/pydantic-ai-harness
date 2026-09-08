"""Slack hosted MCP capability.

Provider contract, verified 2026-09-08: `https://mcp.slack.com/mcp` is the Streamable HTTP endpoint,
it authenticates a Slack user token (`xoxp-`) or an OAuth token from the authorization server it
advertises, and only internal or directory-published apps may connect to it.

Source: https://docs.slack.dev/ai/slack-mcp-server/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from typing import Literal

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Slack capability. Install it with: uv add "pydantic-ai-harness[slack]"'
    ) from _import_error

_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'
_DEFAULT_DESCRIPTION = 'Search Slack, read conversations, and post as the token owner.'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Connect an agent to Slack's hosted MCP server.

    The agent acts as the person whose user token it holds, and the default exposes Slack's write
    tools; the token's scopes decide what the agent can read or change.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: Auth | Literal['oauth'] | str | None = field(default=None, repr=False)
    """A Slack user token (`xoxp-`), `'oauth'`, or a custom `httpx.Auth`. Defaults to `$SLACK_USER_TOKEN`.

    Slack's OAuth wants your own app's client credentials, so reach for a configured `fastmcp` OAuth
    client rather than the `'oauth'` shorthand.
    """

    read_only: bool = False
    """Expose only the tools Slack marks read-only; a tool Slack leaves unannotated counts as a write."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Slack MCP connection."""
        auth = self.auth or environ.get('SLACK_USER_TOKEN')
        if not auth:
            raise UserError('Slack needs a user token: pass auth= or set SLACK_USER_TOKEN.')
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(
            _SLACK_MCP_URL, id=self.id or 'slack', auth=auth, include_instructions=True
        )
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool_def: is_read_only(tool_def))
        return toolset

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
        read_only: bool = False,
    ) -> Slack[AgentDepsT]:
        """Construct a Slack capability from serializable options.

        `auth` is absent by design, so a spec file cannot carry a Slack token: the credential comes
        from `$SLACK_USER_TOKEN`.
        """
        return cls(id=id, description=description, defer_loading=defer_loading, read_only=read_only)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'Slack'

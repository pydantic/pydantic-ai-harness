"""Stripe hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Stripe support with: uv add "pydantic-ai-harness[stripe]"') from exc


@dataclass(kw_only=True)
class Stripe(AbstractCapability[AgentDepsT]):
    """Use Stripe's hosted tools with the permissions of the connected credential."""

    description: str | None = 'Read and change Stripe resources.'
    auth: Auth | str | None = field(default=None, repr=False)
    """Restricted API key, `'oauth'`, or HTTP authentication. Defaults to `STRIPE_API_KEY`, then OAuth."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """
    connected_account: str | None = None
    """Stripe Connect account sent through the native `Stripe-Account` header."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Stripe connection."""
        if self.client is not None:
            return MCPToolset(self.client, id=self.id or 'stripe', include_instructions=self.include_instructions)
        return MCPToolset(
            'https://mcp.stripe.com',
            id=self.id or 'stripe',
            auth=self.auth if self.auth is not None else environ.get('STRIPE_API_KEY', 'oauth'),
            headers={'Stripe-Account': self.connected_account} if self.connected_account is not None else None,
            include_instructions=self.include_instructions,
        )

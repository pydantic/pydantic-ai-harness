"""Cloudflare hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Cloudflare support with: uv add "pydantic-ai-harness[cloudflare]"') from exc

from enum import Enum


class CloudflareServer(str, Enum):
    """Official Cloudflare managed MCP server selection."""

    API = 'api'
    DOCS = 'docs'
    AGENTS_SDK_DOCS = 'agents_sdk_docs'
    WORKERS_BINDINGS = 'workers_bindings'
    WORKERS_BUILDS = 'workers_builds'
    OBSERVABILITY = 'observability'
    CONTAINERS = 'containers'
    BROWSER = 'browser'
    LOGPUSH = 'logpush'
    AI_GATEWAY = 'ai_gateway'
    AUDIT_LOGS = 'audit_logs'
    DNS_ANALYTICS = 'dns_analytics'
    DEX = 'dex'
    CASB = 'casb'
    DEVELOPER_STACK = 'developer_stack'
    BLOG = 'blog'
    DEMO_DAY = 'demo_day'


_URLS: dict[CloudflareServer, str] = {
    CloudflareServer.API: 'https://mcp.cloudflare.com/mcp',
    CloudflareServer.DOCS: 'https://docs.mcp.cloudflare.com/mcp',
    CloudflareServer.AGENTS_SDK_DOCS: 'https://agents.cloudflare.com/mcp',
    CloudflareServer.WORKERS_BINDINGS: 'https://bindings.mcp.cloudflare.com/mcp',
    CloudflareServer.WORKERS_BUILDS: 'https://builds.mcp.cloudflare.com/mcp',
    CloudflareServer.OBSERVABILITY: 'https://observability.mcp.cloudflare.com/mcp',
    CloudflareServer.CONTAINERS: 'https://containers.mcp.cloudflare.com/mcp',
    CloudflareServer.BROWSER: 'https://browser.mcp.cloudflare.com/mcp',
    CloudflareServer.LOGPUSH: 'https://logs.mcp.cloudflare.com/mcp',
    CloudflareServer.AI_GATEWAY: 'https://ai-gateway.mcp.cloudflare.com/mcp',
    CloudflareServer.AUDIT_LOGS: 'https://auditlogs.mcp.cloudflare.com/mcp',
    CloudflareServer.DNS_ANALYTICS: 'https://dns-analytics.mcp.cloudflare.com/mcp',
    CloudflareServer.DEX: 'https://dex.mcp.cloudflare.com/mcp',
    CloudflareServer.CASB: 'https://casb.mcp.cloudflare.com/mcp',
    CloudflareServer.DEVELOPER_STACK: 'https://stack.mcp.cloudflare.com/mcp',
    CloudflareServer.BLOG: 'https://blog.mcp.cloudflare.com/mcp',
    CloudflareServer.DEMO_DAY: 'https://demo-day.mcp.cloudflare.com/mcp',
}
_PUBLIC_SERVERS = frozenset(
    {
        CloudflareServer.DOCS,
        CloudflareServer.AGENTS_SDK_DOCS,
        CloudflareServer.DEVELOPER_STACK,
        CloudflareServer.BLOG,
        CloudflareServer.DEMO_DAY,
    }
)


@dataclass(kw_only=True)
class Cloudflare(AbstractCapability[AgentDepsT]):
    """Connect to a Cloudflare MCP server with provider-controlled permissions."""

    description: str | None = 'Use Cloudflare API, product, and documentation tools.'
    auth: Auth | str | None = field(default=None, repr=False)
    """API token, `'oauth'`, or HTTP authentication. Defaults to `CLOUDFLARE_API_TOKEN`."""
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """
    server: CloudflareServer = CloudflareServer.DOCS
    """Managed server to connect to. Documentation is public; other servers may require OAuth."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Cloudflare connection and optional read-only selection."""
        auth = self.auth if self.auth is not None else environ.get('CLOUDFLARE_API_TOKEN')
        if auth is None and self.server not in _PUBLIC_SERVERS:
            auth = 'oauth'
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'cloudflare', include_instructions=self.include_instructions
            )
        else:
            toolset = MCPToolset(
                _URLS[self.server],
                id=self.id or 'cloudflare',
                auth=auth,
                headers=None,
                include_instructions=self.include_instructions,
            )
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset

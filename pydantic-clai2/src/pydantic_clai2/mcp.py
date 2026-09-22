"""Built-in MCP configuration and tool discovery using core's managed toolsets."""

from typing import Annotated, Literal

from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_ai.capabilities import Toolset
from pydantic_ai.mcp import MCPToolset

from .commands import Command
from .plugins import PluginHost


class ServerSettings(BaseModel):
    """Common server options at the trusted plugin-settings boundary."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    enabled: bool = True


class StdioServer(ServerSettings):
    """A local program, launched without a shell by the MCP client."""

    transport: Literal['stdio']
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


class HTTPServer(ServerSettings):
    """A Streamable HTTP MCP endpoint."""

    transport: Literal['http']
    url: HttpUrl
    headers: dict[str, str] | None = None


class MCPSettings(BaseModel):
    """Server names also prefix tool names to avoid cross-server collisions."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: dict[
        Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9_]*$')],
        Annotated[StdioServer | HTTPServer, Field(discriminator='transport')],
    ] = Field(default_factory=dict)


def activate(host: PluginHost[None]) -> None:
    """Register configured servers without connecting until a run or explicit discovery."""
    settings = host.settings(MCPSettings)
    servers: dict[str, MCPToolset[None]] = {}
    for name, config in settings.servers.items():
        if not config.enabled:
            continue
        if isinstance(config, StdioServer):
            transport = StdioTransport(command=config.command, args=config.args, env=config.env, cwd=config.cwd)
        else:
            transport = StreamableHttpTransport(url=str(config.url), headers=config.headers)
        server: MCPToolset[None] = MCPToolset(transport, id=f'mcp_{name}')
        servers[name] = server
        host.add(Toolset(server.prefixed(name)))

    async def command(args: list[str]) -> str:
        if not args or args == ['list']:
            if not settings.servers:
                return (
                    'No MCP servers configured. Configure the mcp plugin with '
                    '/plugins add mcp pydantic_clai2.mcp JSON. See PLUGINS.md for examples.'
                )
            return '\n'.join(
                f'{name}: {config.transport}, {"enabled" if config.enabled else "disabled"}'
                for name, config in settings.servers.items()
            )
        if len(args) == 2 and args[0] == 'tools':
            name = args[1]
            if name not in servers:
                return f'Unknown or disabled MCP server: {name}'
            server = servers[name]
            async with server:
                tools = await server.list_tools()
            return '\n'.join(f'{name}_{tool.name}' for tool in tools) or f'No tools provided by {name}.'
        return 'Usage: /mcp [list | tools NAME]'

    host.commands.register(
        Command(
            name='mcp',
            description='List configured MCP servers or discover their tools.',
            handler=command,
            complete=lambda args: servers if len(args) > 1 and args[0] == 'tools' else ('list', 'tools'),
        )
    )

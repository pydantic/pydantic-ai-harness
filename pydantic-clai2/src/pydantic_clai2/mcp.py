"""The built-in MCP plugin: explicit config approval, with core owning connections per turn."""

from pathlib import Path
from tempfile import mkdtemp

from fastmcp.client.transports import StdioTransport
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.capabilities import Capability
from pydantic_ai.mcp import MCPToolset, load_mcp_toolsets
from pydantic_ai.toolsets import PrefixedToolset

from .commands import Command
from .plugins import DepsT, PluginHost, SessionStart


class MCPSettings(BaseModel):
    """An absolute `config_path` opts into loading trusted configuration on every plugin activation."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    config_path: str | None = Field(
        default=None, description='Absolute path to a trusted MCP config, including its future edits.'
    )


def activate(host: PluginHost[DepsT]) -> None:
    """Offer `/mcp` without reading or connecting to project servers until explicitly approved."""
    try:
        settings = host.settings(MCPSettings)
    except ValidationError:
        raise ValueError('Invalid MCP settings. Use only config_path with an absolute path or null.') from None
    config_path = Path(settings.config_path) if settings.config_path is not None else Path.cwd() / '.mcp.json'
    if not config_path.is_absolute():
        raise ValueError('MCP config_path must be absolute; relative paths cannot be persistently trusted.')

    capability = Capability[DepsT]()
    host.add(capability)
    loaded = False
    log_directory: Path | None = None

    def load() -> str:
        nonlocal loaded, log_directory
        try:
            toolsets = load_mcp_toolsets(config_path)
            logs: Path | None = None
            for index, toolset in enumerate(toolsets, start=1):
                assert isinstance(toolset, PrefixedToolset)
                assert isinstance(toolset.wrapped, MCPToolset)
                transport = toolset.wrapped.client.transport
                if isinstance(transport, StdioTransport):
                    transport.keep_alive = False  # Otherwise FastMCP keeps subprocesses alive after a turn.
                    if logs is None:
                        logs = Path(mkdtemp(prefix='clai-mcp-'))
                    transport.log_file = logs / f'server-{index}.log'
                    transport.log_file.touch(mode=0o600)
        except (OSError, ValueError):
            return (
                'Cannot load MCP config. Check file access, JSON, mcpServers entries, and environment references. '
                'Config values are hidden; previously loaded servers are unchanged.'
            )
        capability.toolsets = tuple(toolsets)
        loaded = True
        log_directory = logs
        notice = f'\nStdio logs: {logs}' if logs is not None else ''
        return f'Loaded {len(toolsets)} MCP server(s) from {config_path}. Connections open only during agent turns.{notice}'

    @host.on('session_start')
    async def start(event: SessionStart) -> None:
        if settings.config_path is not None:
            host.console.print(load(), markup=False)

    def command(args: list[str]) -> str:
        if not args or args == ['status']:
            state = f'{len(capability.toolsets)} server(s) loaded' if loaded else 'not loaded'
            logs = f'\nStdio logs: {log_directory}' if log_directory is not None else ''
            return (
                f'MCP: {state}. Config: {config_path}\n'
                f'Connections are scoped to agent turns. Use /mcp load to review approval instructions.{logs}'
            )
        if args == ['load']:
            return (
                f'Review {config_path} before loading. MCP configs can execute commands and read your environment.\n'
                'To trust and load this file for this plugin activation, run /mcp load --approve. '
                'This does not save approval.'
            )
        if args == ['load', '--approve']:
            return load()
        raise ValueError('Usage: /mcp [status|load [--approve]]')

    host.commands.register(
        Command(
            name='mcp',
            description='Show MCP status or explicitly approve loading .mcp.json',
            handler=command,
            complete=lambda args: ('status', 'load') if len(args) <= 1 else ('--approve',),
        )
    )

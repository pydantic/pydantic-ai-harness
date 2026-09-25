"""The built-in `notion` plugin: declared disabled, connected by token or browser sign-in, and failing loudly."""

import io
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.notion import Notion
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.notion import NOTION_MCP_URL, TOKENS
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture(autouse=True)
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NOTION_ACCESS_TOKEN', raising=False)


def loader(tmp_path: Path, commands: Commands | None = None) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'settings.db')
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands or Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=DEFAULT_PLUGINS,
    )


def names(commands: Commands) -> set[str]:
    return {command.name for command in commands}


def notion(capabilities: list[AgentCapability[None]]) -> Notion[None]:
    found: list[Notion[None]] = [capability for capability in capabilities if isinstance(capability, Notion)]
    [capability] = found
    return capability


def test_declared_as_a_disabled_builtin_not_a_raw_catalog_entry() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'notion']
    assert declaration.factory == 'pydantic_clai2.notion'
    assert not declaration.enabled
    assert all(plugin.factory != 'pydantic_ai_harness.notion:Notion' for plugin in DEFAULT_PLUGINS)


async def test_token_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NOTION_ACCESS_TOKEN', 'ntn-token')
    commands = Commands()
    plugins = loader(tmp_path, commands)
    await plugins.enable('notion')
    capability = notion(plugins.capabilities())
    assert capability.auth == 'ntn-token' and capability.client is None and not capability.read_only
    assert isinstance(capability.get_toolset(), MCPToolset)
    assert 'notion' in names(commands)
    await plugins.close('exit')
    assert 'notion' not in names(commands)


async def test_browser_sign_in_without_a_token(tmp_path: Path) -> None:
    plugins = loader(tmp_path)
    await plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"read_only": true}'])
    capability = notion(plugins.capabilities())
    assert capability.auth is None and capability.read_only
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == NOTION_MCP_URL and isinstance(transport.auth, OAuth)
    await plugins.close('exit')


async def test_oauth_setting_ignores_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NOTION_ACCESS_TOKEN', 'ntn-token')
    plugins = loader(tmp_path)
    await plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"auth": "oauth"}'])
    capability = notion(plugins.capabilities())
    assert capability.auth is None and isinstance(capability.client, Client)
    await plugins.close('exit')


async def test_token_setting_without_a_token_fails_to_load(tmp_path: Path) -> None:
    commands = Commands()
    plugins = loader(tmp_path, commands)
    with pytest.raises(PluginError, match='Set NOTION_ACCESS_TOKEN'):
        await plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"auth": "token"}'])
    assert plugins.capabilities() == [] and 'notion' not in names(commands)
    [entry] = [entry for entry in plugins.entries() if entry.name == 'notion']
    assert entry.state.startswith('enabled, failed: UserError: Set NOTION_ACCESS_TOKEN')


async def test_unknown_settings_are_rejected(tmp_path: Path) -> None:
    plugins = loader(tmp_path)
    with pytest.raises(PluginError, match='extra_forbidden|Extra inputs'):
        await plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"token": "ntn-token"}'])
    assert plugins.capabilities() == []


async def test_logout_forgets_the_browser_sign_in(tmp_path: Path, vault: dict[tuple[str, str], str]) -> None:
    commands = Commands()
    plugins = loader(tmp_path, commands)
    await plugins.enable('notion')
    await TOKENS.put('token', {'access_token': 'a'}, collection='mcp-oauth-token')
    assert TOKENS.signed_in()
    [command] = [command for command in commands if command.name == 'notion']
    assert list(command.complete([''])) == ['logout'] and list(command.complete(['logout', ''])) == []
    with pytest.raises(ValueError, match='Usage: /notion logout'):
        commands.execute('/notion')
    assert commands.execute('/notion logout') == (
        'Signed out of Notion. The next browser sign-in asks for your account again.'
    )
    assert vault == {}
    await plugins.close('exit')

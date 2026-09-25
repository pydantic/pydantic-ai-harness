"""The built-in `ordinal` plugin: a token from the environment, or a browser sign-in kept in the keyring."""

import io
import sys
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringLocked, PasswordDeleteError
from mcp.shared.auth import OAuthToken
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.ordinal import Ordinal
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.capability_catalog import HARNESS_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.ordinal import TOKEN_ENV, TOKENS, URL, USAGE, activate
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore

Vault = dict[tuple[str, str], str]


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Vault:
    entries: Vault = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        if (service, account) not in entries:
            raise PasswordDeleteError('Not found')
        del entries[service, account]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    return entries


def terminal(monkeypatch: pytest.MonkeyPatch, *, attached: bool) -> None:
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: attached)


async def sign_in() -> None:
    token = OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh', expires_in=3600)
    await TokenStorageAdapter(TokenStore(TOKENS), server_url=URL).set_tokens(token)


def host() -> tuple[PluginHost[None], io.StringIO]:
    output = io.StringIO()
    return PluginHost[None](name='ordinal', console=Console(file=output), settings={}), output


def ordinal_client(plugin: PluginHost[None]) -> MCPToolsetClient | None:
    """The one capability is an `Ordinal`; return the connection it was given, if any."""
    [capability] = plugin.capabilities
    assert isinstance(capability, Ordinal)
    return capability.client


def run(plugin: PluginHost[None], *args: str) -> object:
    [command] = list(plugin.commands)
    return command.handler(list(args))


def test_declared_as_a_disabled_built_in_not_a_catalog_entry() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ordinal']
    assert declaration.factory == 'pydantic_clai2.ordinal'
    assert not declaration.enabled
    assert not any(plugin.factory.startswith('pydantic_ai_harness.ordinal') for plugin in HARNESS_PLUGINS)


def test_url_matches_the_harness_endpoint() -> None:
    toolset = Ordinal[None](auth='token').get_toolset()
    assert isinstance(toolset, MCPToolset)
    assert isinstance(toolset.client, Client)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == URL


def test_environment_token_is_used_as_is(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV, 'token')
    terminal(monkeypatch, attached=False)
    plugin, output = host()
    activate(plugin)
    assert ordinal_client(plugin) is None
    assert output.getvalue() == ''
    assert run(plugin) == f'Ordinal uses `{TOKEN_ENV}`.'


def test_terminal_without_a_token_signs_in_through_the_browser(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=True)
    plugin, output = host()
    activate(plugin)
    transport = ordinal_client(plugin)
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == URL
    assert isinstance(transport.auth, OAuth)
    assert 'browser opens to sign in on first use' in output.getvalue()
    assert run(plugin) == 'Ordinal: not signed in; the browser opens on first use.'


async def test_saved_sign_in_loads_without_a_terminal(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    await sign_in()
    assert ('pydantic-clai2', f'mcp-{TOKENS}') in vault
    terminal(monkeypatch, attached=False)
    plugin, output = host()
    activate(plugin)
    assert isinstance(ordinal_client(plugin), StreamableHttpTransport)
    assert output.getvalue() == ''
    assert run(plugin) == 'Ordinal: signed in through the browser.'

    assert 'Signed out of Ordinal' in str(run(plugin, 'logout'))
    assert vault == {}
    assert run(plugin) == 'Ordinal: not signed in; the browser opens on first use.'


def test_unreadable_keyring_is_reported(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)

    def locked(service: str, account: str) -> str | None:
        raise KeyringLocked('locked')

    monkeypatch.setattr(keyring, 'get_password', locked)
    assert run(plugin) == 'Ordinal: sign-in unknown; the keyring could not be read.'


def test_command_usage_and_completion(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    [command] = list(plugin.commands)
    assert command.name == 'ordinal'
    assert run(plugin, 'nope') == USAGE
    assert list(command.complete([''])) == ['logout']
    assert list(command.complete(['logout', ''])) == []


async def test_no_token_and_no_terminal_fails_to_enable(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    terminal(monkeypatch, attached=False)
    plugin, _ = host()
    with pytest.raises(UserError, match=TOKEN_ENV):
        activate(plugin)
    assert plugin.capabilities == []

    store = SettingsStore(tmp_path / 'settings.db')
    loader = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ordinal'],
    )
    with pytest.raises(PluginError, match=TOKEN_ENV):
        await loader.enable('ordinal')
    assert loader.capabilities() == []

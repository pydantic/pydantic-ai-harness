"""The built-in `grain` plugin: harness `Grain` with a token from the environment or a keyring-backed sign-in."""

import io
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import PasswordDeleteError
from mcp.shared.auth import OAuthToken
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.grain import Grain
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2._app import create_shell
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.grain import GRAIN_MCP_URL, TOKEN_ACCOUNT, GrainSignIn, activate
from pydantic_clai2.headless import no_screen
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugins import FullScreen, PluginHost, SessionStart, bare_screen
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore

RETIRED = 'pydantic_ai_harness.grain:Grain'
"""The raw factory the retired `/plugins` catalog saved under `grain`."""

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def keyring_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    """No token in the environment, and a keyring that can also forget, for `/grain logout`."""
    monkeypatch.delenv('GRAIN_ACCESS_TOKEN', raising=False)
    entries: dict[tuple[str, str], str] = {}

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        if entries.pop((service, account), None) is None:
            raise PasswordDeleteError('Not found')  # pragma: no cover -- `forget` deletes only what it read.

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)


def host(
    settings: dict[str, bool] | None = None, *, full_screen: FullScreen = bare_screen, output: io.StringIO | None = None
) -> PluginHost[None]:
    return PluginHost(
        name='grain',
        console=Console(file=output or io.StringIO(), width=200),
        settings={**(settings or {})},
        full_screen=full_screen,
    )


def grain(plugin: PluginHost[None]) -> Grain[None]:
    [capability] = plugin.capabilities
    assert isinstance(capability, Grain)
    return capability  # pyright: ignore[reportUnknownVariableType] -- `isinstance` cannot narrow the type argument.


def sign_in(capability: Grain[None]) -> GrainSignIn:
    assert isinstance(capability.client, Client)
    transport = capability.client.transport
    assert isinstance(transport, StreamableHttpTransport) and transport.url == GRAIN_MCP_URL
    assert isinstance(transport.auth, GrainSignIn)
    return transport.auth


def test_declared_as_a_disabled_builtin() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'grain']
    assert declaration.factory == 'pydantic_clai2.grain'
    assert not declaration.enabled


@pytest.mark.parametrize('enabled', [True, False])
def test_a_saved_catalog_declaration_moves_to_the_builtin(tmp_path: Path, enabled: bool) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(PluginSettings(id='grain', factory=RETIRED, enabled=enabled))
    store.save_plugin(PluginSettings(id='other', factory=RETIRED))
    shell_for(store)
    saved = {plugin.id: plugin for plugin in store.plugins()}
    assert saved['grain'] == PluginSettings(id='grain', factory='pydantic_clai2.grain', enabled=enabled)
    assert saved['other'].factory == RETIRED, 'only the id the built-in replaced moves'


def test_a_saved_declaration_with_settings_stays(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    chosen = PluginSettings(id='grain', factory=RETIRED, settings={'read_only': True})
    store.save_plugin(chosen)
    shell_for(store)
    assert store.plugins() == [chosen]


def shell_for(store: SettingsStore) -> None:
    create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=store,
        builtin_plugins=DEFAULT_PLUGINS,
        project=ProjectSettings(),
        headless=True,
    )


async def test_enabling_the_builtin_adds_grain_and_the_command(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'grain'],
    )
    await loader.load_all()
    assert loader.capabilities() == []

    await loader.enable('grain')
    [capability] = loader.capabilities()
    assert isinstance(capability, Grain) and capability.read_only
    assert 'Not signed in' in await commands.execute_async('/grain')


async def test_token_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GRAIN_ACCESS_TOKEN', 'grain-token')
    plugin = host({'read_only': False})
    activate(plugin)
    capability = grain(plugin)
    assert capability.client is None and capability.auth is None and not capability.read_only
    assert await plugin.commands.execute_async('/grain') == 'Grain uses GRAIN_ACCESS_TOKEN.'
    assert 'cannot revoke' in await plugin.commands.execute_async('/grain logout')


async def test_without_a_token_it_signs_in_through_the_browser_and_keeps_tokens_in_the_keyring() -> None:
    plugin = host()
    activate(plugin)
    capability = grain(plugin)
    assert capability.auth is None and capability.read_only
    oauth = sign_in(capability)
    assert oauth.tokens.name == TOKEN_ACCOUNT
    assert 'Not signed in' in await plugin.commands.execute_async('/grain')

    storage = TokenStorageAdapter(oauth.tokens, server_url=GRAIN_MCP_URL)
    token = OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh')
    await storage.set_tokens(token)
    oauth.context.current_tokens = token
    assert 'Signed in to Grain' in await plugin.commands.execute_async('/grain')

    assert 'Signed out of Grain' in await plugin.commands.execute_async('/grain logout')
    assert await storage.get_tokens() is None
    assert oauth.context.current_tokens is None, 'the loaded client cannot keep using the old token'
    with pytest.raises(ValueError, match='Usage: /grain'):
        await plugin.commands.execute_async('/grain login')


async def test_sign_in_prints_the_url_before_opening_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    async def open_browser(self: OAuth, authorization_url: str) -> None:
        opened.append(authorization_url)

    monkeypatch.setattr(OAuth, 'redirect_handler', open_browser)
    output = io.StringIO()
    plugin = host(output=output)
    activate(plugin)
    url = 'https://api.grain.com/oauth/authorize?state=abc'
    await sign_in(grain(plugin)).redirect_handler(url)
    assert opened == [url]
    assert url in output.getvalue()


async def test_headless_sign_in_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    async def open_browser(self: OAuth, authorization_url: str) -> None:
        raise AssertionError('no browser without someone to sign in')  # pragma: no cover

    monkeypatch.setattr(OAuth, 'redirect_handler', open_browser)
    plugin = host(full_screen=no_screen)
    activate(plugin)
    with pytest.raises(UserError, match='Run clai2 interactively once to sign in, or set GRAIN_ACCESS_TOKEN'):
        await sign_in(grain(plugin)).redirect_handler('https://api.grain.com/oauth/authorize')


def test_settings_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        activate(host({'readonly': True}))


def test_completes_logout() -> None:
    plugin = host()
    activate(plugin)
    [command] = plugin.commands
    assert list(command.complete([])) == ['logout']
    assert list(command.complete(['logout', ''])) == []

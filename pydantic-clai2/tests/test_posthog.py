"""The built-in `posthog` plugin: which credential it connects with, read-only mode, and `/posthog`."""

import io
from pathlib import Path

import keyring
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringLocked
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.posthog import PostHog
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.api_keys import save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.posthog import KEY_NAME, TOKENS, activate
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_NAME, raising=False)


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    entries: dict[tuple[str, str], str] = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        del entries[service, account]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    return entries


def load(settings: dict[str, JsonValue] | None = None) -> tuple[PluginHost[None], StreamableHttpTransport]:
    host = PluginHost[None](name='posthog', console=Console(file=io.StringIO()), settings=settings or {})
    activate(host)
    [capability] = host.capabilities
    assert isinstance(capability, PostHog)
    assert not capability.read_only, 'the read-only header does the filtering, which keeps the `posthog` tool'
    assert isinstance(capability.client, StreamableHttpTransport)
    assert capability.client.url == 'https://mcp.posthog.com/mcp'
    return host, capability.client


async def test_environment_key_wins_and_read_only_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_NAME, 'phx_env')
    save_key(name=KEY_NAME, value='phx_saved')
    host, transport = load()
    assert transport.headers == {'x-posthog-read-only': 'true'}
    assert not isinstance(transport.auth, OAuth)
    assert transport.auth is not None
    assert await host.commands.execute_async('/posthog') == f'PostHog (read-only) uses {KEY_NAME} from the environment.'
    assert 'unset or delete it' in await host.commands.execute_async('/posthog logout')


async def test_saved_key_and_read_write() -> None:
    save_key(name=KEY_NAME, value='phx_saved')
    host, transport = load({'read_only': False, 'auth': 'api_key'})
    assert transport.headers == {}
    assert not isinstance(transport.auth, OAuth)
    assert await host.commands.execute_async('/posthog') == f'PostHog (read-write) uses {KEY_NAME} from /keys.'


async def test_oauth_when_no_key_or_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    _, transport = load()
    assert isinstance(transport.auth, OAuth)
    monkeypatch.setenv(KEY_NAME, 'phx_env')
    _, transport = load({'auth': 'oauth'})
    assert isinstance(transport.auth, OAuth), 'an explicit `oauth` ignores the key'


async def test_api_key_mode_without_a_key_fails_on_enable(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'posthog')
    assert builtin.factory == 'pydantic_clai2.posthog'
    assert not builtin.enabled
    commands = Commands()
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(builtin.model_copy(update={'settings': {'auth': 'api_key'}}),),
    )
    await loader.load_all()
    assert loader.capabilities() == []
    with pytest.raises(PluginError, match=f'Set {KEY_NAME} or save it in /keys'):
        await loader.enable('posthog')
    with pytest.raises(UserError):
        load({'auth': 'api_key'})
    assert loader.capabilities() == []
    assert not list(commands)
    await loader.close('exit')


async def test_status_and_logout_replace_the_sign_in(vault: dict[tuple[str, str], str]) -> None:
    host, _ = load()
    [capability] = host.capabilities
    assert isinstance(capability, PostHog)
    before = capability.client
    assert 'not signed in' in await host.commands.execute_async('/posthog')
    await TokenStore(TOKENS).put('token', {'access_token': 'x'}, collection='mcp-oauth-token')
    assert vault
    assert await host.commands.execute_async('/posthog') == 'PostHog (read-only) is signed in through the browser.'
    assert 'Signed out' in await host.commands.execute_async('/posthog logout')
    assert not vault
    assert capability.client is not before
    assert isinstance(capability.client, StreamableHttpTransport)
    assert isinstance(capability.client.auth, OAuth)
    assert capability.client.headers == {'x-posthog-read-only': 'true'}
    with pytest.raises(ValueError, match='Usage'):
        await host.commands.execute_async('/posthog login')
    [command] = host.commands
    assert list(command.complete([''])) == ['logout']
    assert list(command.complete(['logout', ''])) == []


async def test_status_when_the_keyring_is_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    host, _ = load()

    def locked(service: str, account: str) -> str | None:
        raise KeyringLocked('locked')

    monkeypatch.setattr(keyring, 'get_password', locked)
    assert 'unknown sign-in state' in await host.commands.execute_async('/posthog')

"""The built-in `linear` plugin: declared disabled, and it adds `Linear` only with a credential."""

import io
import json
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import PasswordDeleteError
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.linear import Linear
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.api_keys import save_key
from pydantic_clai2.capability_catalog import HARNESS_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.linear import TOKEN_ACCOUNT
from pydantic_clai2.mcp import TokenStore, http_client
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio
Vault = dict[tuple[str, str], str]


@pytest.fixture(autouse=True)
def no_linear_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('LINEAR_ACCESS_TOKEN', raising=False)


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Vault:
    """A keyring that can also delete, so signing out is observable."""
    entries: Vault = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        if entries.pop((service, account), None) is None:
            raise PasswordDeleteError(account)

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    return entries


def make(tmp_path: Path) -> tuple[PluginLoader[None], Commands]:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    loader = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear'],
    )
    return loader, commands


async def declare(loader: PluginLoader[None], settings: dict[str, JsonValue]) -> None:
    await loader.remove('linear')
    await loader.command(['add', 'linear', 'pydantic_clai2.linear', json.dumps(settings)])


def test_declared_as_a_disabled_built_in_not_a_catalog_entry() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear']
    assert declaration.factory == 'pydantic_clai2.linear'
    assert not declaration.enabled
    assert all('linear' not in plugin.factory for plugin in HARNESS_PLUGINS)


async def test_environment_token_and_read_only_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader, _ = make(tmp_path)
    await loader.load_all()
    assert loader.capabilities() == [], 'disabled until the user enables it'
    monkeypatch.setenv('LINEAR_ACCESS_TOKEN', 'lin_env')
    assert await loader.command(['enable', 'linear']) == 'Enabled linear.'
    assert loader.capabilities() == [Linear(auth='lin_env', read_only=True)]

    await declare(loader, {'read_only': False})
    assert loader.capabilities() == [Linear(auth='lin_env', read_only=False)]
    await loader.close('exit')


async def test_missing_credential_fails_the_load(tmp_path: Path) -> None:
    loader, _ = make(tmp_path)
    with pytest.raises(PluginError, match='Linear needs a credential: set LINEAR_ACCESS_TOKEN'):
        await loader.enable('linear')
    assert loader.capabilities() == []
    assert 'enabled, failed' in await loader.command(['list'])


async def test_saved_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LINEAR_ACCESS_TOKEN', 'lin_env')
    loader, _ = make(tmp_path)
    save_key(name='LINEAR', value='lin_saved')
    await declare(loader, {'api_key': 'LINEAR'})
    assert loader.capabilities() == [Linear(auth='lin_saved', read_only=True)], 'a named key beats the environment'

    with pytest.raises(PluginError, match='Saved API key MISSING is missing'):
        await declare(loader, {'api_key': 'MISSING'})
    assert loader.capabilities() == []


async def test_oauth_and_api_key_are_exclusive(tmp_path: Path) -> None:
    loader, _ = make(tmp_path)
    with pytest.raises(PluginError, match='Choose `oauth` or `api_key`, not both'):
        await declare(loader, {'oauth': True, 'api_key': 'LINEAR'})
    assert loader.capabilities() == []


@pytest.mark.parametrize(
    ('read_only', 'url'), [(True, 'https://mcp.linear.app/mcp/readonly'), (False, 'https://mcp.linear.app/mcp')]
)
async def test_oauth_signs_in_with_keyring_tokens(tmp_path: Path, vault: Vault, read_only: bool, url: str) -> None:
    loader, commands = make(tmp_path)
    await declare(loader, {'oauth': True, 'read_only': read_only})
    [capability] = loader.capabilities()
    assert isinstance(capability, Linear)
    assert not capability.read_only, 'the URL carries read_only, not tool annotations'
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == url
    assert transport.httpx_client_factory is http_client, 'redirects stay off, as for /mcp servers'
    assert isinstance(transport.auth, OAuth)

    await TokenStore(TOKEN_ACCOUNT).put('x', {'access_token': 'a'}, collection='mcp-oauth-token')
    assert TokenStore(TOKEN_ACCOUNT).signed_in()
    with pytest.raises(ValueError, match='Usage: /linear logout'):
        await run(commands, '/linear')
    assert (await run(commands, '/linear logout')).startswith('Signed out of Linear.')
    assert vault == {}

    await loader.disable('linear')
    assert 'linear' not in {command.name for command in commands}


async def run(commands: Commands, text: str) -> str:
    result = commands.execute(text)
    return result if isinstance(result, str) else await result

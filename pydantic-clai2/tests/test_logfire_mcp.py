"""The `logfire_mcp` built-in: named keys, fail-closed loading, OAuth, and its declaration."""

import asyncio
import io
import webbrowser
from collections.abc import Mapping
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from keyring.errors import PasswordDeleteError
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP
from rich.console import Console
from typing_extensions import TypeIs

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, logfire_mcp
from pydantic_clai2.commands import Commands
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.logfire_mcp import ACCOUNT, LogfireMCPSettings, activate, choose_key, chosen_key, command
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore

CTX = RunContext(deps=None, model=TestModel(), usage=RunUsage())


class Prompt:
    def __init__(self, *values: str | BaseException) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def loaded(settings: Mapping[str, JsonValue] | None = None) -> PluginHost[None]:
    host: PluginHost[None] = PluginHost(
        name='logfire_mcp', console=Console(file=io.StringIO()), settings=dict(settings or {})
    )
    activate(host)
    return host


def added(settings: Mapping[str, JsonValue] | None = None) -> LogfireMCP[None]:
    [capability] = loaded(settings).capabilities
    assert is_logfire_mcp(capability)
    return capability


def is_logfire_mcp(capability: object) -> TypeIs[LogfireMCP[None]]:
    return isinstance(capability, LogfireMCP)


def press(monkeypatch: pytest.MonkeyPatch, *keys: str) -> None:
    pressed = iter(keys)
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))


@pytest.fixture
def deletable_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    entries: dict[tuple[str, str], str] = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        if entries.pop((service, account), None) is None:
            raise PasswordDeleteError('Not found')

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)


def no_browser() -> webbrowser.BaseBrowser:
    raise webbrowser.Error('could not locate runnable browser')


def a_browser() -> webbrowser.BaseBrowser:
    return webbrowser.GenericBrowser('true')


def test_settings_hold_no_credentials() -> None:
    assert set(LogfireMCPSettings.model_fields) == {'oauth', 'region', 'read_only'}
    with pytest.raises(ValidationError):
        loaded({'key': 'LOGFIRE_API_KEY'})


def test_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    capability = added()
    assert (capability.auth, capability.client, capability.url, capability.read_only) == (
        None,
        None,
        LOGFIRE_US_MCP_URL,
        True,
    )
    capability = added({'region': 'eu', 'read_only': False})
    assert (capability.url, capability.read_only) == (LOGFIRE_EU_MCP_URL, False)


def test_conventional_saved_key_resolves_each_run() -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='first')
    capability = added()
    assert callable(capability.auth)
    assert capability.auth(CTX) == 'first'
    api_keys.save_key(name='LOGFIRE_API_KEY', value='second')
    assert capability.auth(CTX) == 'second'
    api_keys.delete_key(name='LOGFIRE_API_KEY')
    with pytest.raises(UserError, match='Saved API key LOGFIRE_API_KEY is missing'):
        capability.auth(CTX)


async def test_choosing_a_shared_key_stores_only_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    api_keys.save_key(name='SHARED', value='shared-secret')
    press(monkeypatch, 'enter')
    assert (
        await choose_key(prompt=Prompt())
        == 'Logfire MCP uses SHARED. Run /plugins reload logfire_mcp to connect with it.'
    )
    assert 'shared-secret' not in (load_codex_credentials(account=ACCOUNT) or '')
    capability = added()
    assert callable(capability.auth)
    assert capability.auth(CTX) == 'shared-secret'
    with pytest.raises(ValueError, match='used by logfire_mcp'):
        api_keys.rename_key(name='SHARED', new_name='OTHER')
    api_keys.delete_key(name='SHARED')
    with pytest.raises(UserError, match='Saved API key SHARED is missing'):
        loaded()


async def test_entering_a_key_saves_it_under_the_conventional_name() -> None:
    prompt = Prompt(' typed-secret ')
    assert await choose_key(prompt=prompt) == (
        'Logfire MCP uses LOGFIRE_API_KEY. Run /plugins reload logfire_mcp to connect with it.'
    )
    assert prompt.labels == [('Logfire API key (saved to /keys as LOGFIRE_API_KEY): ', True)]
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'typed-secret'
    assert chosen_key() == api_keys.KeyReference(name='LOGFIRE_API_KEY')


@pytest.mark.parametrize(('answer', 'value'), [('n', 'old'), (EOFError(), 'old'), ('y', 'new')])
async def test_entering_a_key_asks_before_replacing_a_shared_one(
    monkeypatch: pytest.MonkeyPatch, answer: str | BaseException, value: str
) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='old')
    press(monkeypatch, 'down', 'enter')
    await choose_key(prompt=Prompt('new', answer))
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == value
    assert (chosen_key() is None) == (value == 'old')


async def test_cancelled_or_empty_entry() -> None:
    assert await choose_key(prompt=Prompt(EOFError())) == 'Logfire key unchanged.'
    with pytest.raises(ValueError, match='A Logfire API key is required'):
        await choose_key(prompt=Prompt('  '))
    assert chosen_key() is None


@pytest.mark.usefixtures('deletable_keyring')
async def test_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logfire_mcp, 'PromptSession', lambda: Prompt('secret'))
    monkeypatch.setattr(webbrowser, 'get', a_browser)
    host = loaded()
    assert [c.name for c in host.commands] == ['logfire_mcp']
    assert 'logout' in await command(['help'])
    with pytest.raises(ValueError, match='Usage'):
        await command([])
    assert 'LOGFIRE_API_KEY' in await command(['key'])
    await TokenStore(ACCOUNT).put('token', {'access_token': 'x'}, collection='mcp-oauth-token')
    assert 'Forgot' in await command(['logout'])
    assert chosen_key() is None
    assert not TokenStore(ACCOUNT).signed_in()
    assert 'LOGFIRE_API_KEY' in api_keys.load_keys()


def test_invalid_choice_fails_closed() -> None:
    save_codex_credentials(account=ACCOUNT, value='{}')
    with pytest.raises(UserError, match='Run /logfire_mcp logout'):
        loaded()


def test_oauth_uses_a_keyring_client_with_a_sign_in_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webbrowser, 'get', a_browser)
    capability = added({'region': 'eu'})
    assert capability.auth is None
    assert isinstance(capability.client, Client)
    assert capability.client._init_timeout == 330  # pyright: ignore[reportPrivateUsage]
    assert str(capability.client.transport.url) == LOGFIRE_EU_MCP_URL  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]


def test_oauth_without_a_browser_fails_unless_already_signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webbrowser, 'get', no_browser)
    with pytest.raises(UserError, match='Logfire sign-in needs a browser'):
        added()
    asyncio.run(TokenStore(ACCOUNT).put('token', {'access_token': 'x'}, collection='mcp-oauth-token'))
    assert isinstance(added().client, Client)


def test_no_credential_with_oauth_off_fails() -> None:
    with pytest.raises(UserError, match='Save a key named LOGFIRE_API_KEY in /keys'):
        added({'oauth': False})


def test_declared_as_disabled_builtin_that_fails_clearly(tmp_path: Path) -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'logfire_mcp']
    assert (declaration.factory, declaration.enabled) == ('pydantic_clai2.logfire_mcp', False)
    store = SettingsStore(tmp_path / 'settings.db')
    plugins: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[declaration.model_copy(update={'settings': {'oauth': False}})],
    )
    asyncio.run(plugins.load_all())
    assert plugins.capabilities() == []
    with pytest.raises(PluginError, match="Plugin 'logfire_mcp': UserError: Save a key named"):
        asyncio.run(plugins.enable('logfire_mcp'))
    assert plugins.capabilities() == []

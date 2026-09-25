"""The `logfire_mcp` built-in: credential selection, fail-closed loading, and its declaration."""

import asyncio
import io
import webbrowser
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastmcp import Client
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP
from rich.console import Console
from typing_extensions import TypeIs

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.commands import Commands
from pydantic_clai2.logfire_mcp import TOKEN_ACCOUNT, activate
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore


def added(settings: Mapping[str, JsonValue] | None = None) -> LogfireMCP[None]:
    host: PluginHost[None] = PluginHost(
        name='logfire_mcp', console=Console(file=io.StringIO()), settings=dict(settings or {})
    )
    activate(host)
    [capability] = host.capabilities
    assert is_logfire_mcp(capability)
    return capability


def is_logfire_mcp(capability: object) -> TypeIs[LogfireMCP[None]]:
    return isinstance(capability, LogfireMCP)


def no_browser() -> webbrowser.BaseBrowser:
    raise webbrowser.Error('could not locate runnable browser')


def test_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    capability = added()
    assert (capability.auth, capability.client, capability.url, capability.read_only) == (
        None,
        None,
        LOGFIRE_US_MCP_URL,
        True,
    )
    capability = added({'region': 'eu', 'read_only': False})
    assert (capability.url, capability.read_only) == (LOGFIRE_EU_MCP_URL, False)


def test_saved_key_resolves_each_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    with pytest.raises(UserError, match='No saved API key is named LOGFIRE'):
        added({'key': 'logfire'})
    api_keys.save_key(name='LOGFIRE', value='first')
    capability = added({'key': 'logfire'})
    assert callable(capability.auth)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert capability.auth(ctx) == 'first'
    api_keys.save_key(name='LOGFIRE', value='second')
    assert capability.auth(ctx) == 'second'
    api_keys.delete_key(name='LOGFIRE')
    with pytest.raises(UserError, match='Saved API key LOGFIRE is missing'):
        capability.auth(ctx)
    with pytest.raises(ValidationError):
        added({'key': 'not a name'})


def a_browser() -> webbrowser.BaseBrowser:
    return webbrowser.GenericBrowser('true')


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
    asyncio.run(TokenStore(TOKEN_ACCOUNT).put('token', {'access_token': 'x'}, collection='mcp-oauth-token'))
    assert isinstance(added().client, Client)


def test_no_credential_with_oauth_off_fails() -> None:
    with pytest.raises(UserError, match='Set `LOGFIRE_API_KEY`'):
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
    with pytest.raises(PluginError, match="Plugin 'logfire_mcp': UserError: Set `LOGFIRE_API_KEY`"):
        asyncio.run(plugins.enable('logfire_mcp'))
    assert plugins.capabilities() == []

"""The built-in `day_ai` plugin: declared disabled, and loaded only with a token or a finished browser sign-in."""

import io
from pathlib import Path
from types import TracebackType

import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.day_ai import DayAI
from rich.console import Console

import pydantic_clai2.day_ai as day_ai
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore


class SignIn:
    """Stands in for FastMCP's `Client`, whose connection would open the browser."""

    connected: list[StreamableHttpTransport] = []
    error: Exception | None = None

    def __init__(self, transport: StreamableHttpTransport) -> None:
        self.transport = transport

    async def __aenter__(self) -> None:
        if SignIn.error is not None:
            raise SignIn.error
        SignIn.connected.append(self.transport)

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None


@pytest.fixture(autouse=True)
def sign_in(monkeypatch: pytest.MonkeyPatch) -> type[SignIn]:
    monkeypatch.delenv(day_ai.TOKEN_ENV, raising=False)
    monkeypatch.setattr(day_ai, 'Client', SignIn)
    monkeypatch.setattr(SignIn, 'connected', [])
    monkeypatch.setattr(SignIn, 'error', None)
    return SignIn


def make(
    tmp_path: Path,
    *,
    terminal: bool = True,
    settings: dict[str, JsonValue] | None = None,
    output: io.StringIO | None = None,
) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'settings.db')
    if settings is not None:
        store.save_plugin(PluginSettings(id='day_ai', factory='pydantic_clai2.day_ai', settings=settings))
    return PluginLoader(
        store=store,
        console=Console(file=output or io.StringIO(), force_terminal=terminal),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=DEFAULT_PLUGINS,
    )


def day_ai_client(loader: PluginLoader[None]) -> object:
    [client] = [capability.client for capability in loader.capabilities() if isinstance(capability, DayAI)]
    return client


def test_declared_as_a_disabled_built_in_not_the_raw_capability() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'day_ai']
    assert declaration == PluginSettings(id='day_ai', factory='pydantic_clai2.day_ai', enabled=False)


async def test_environment_token_skips_browser_sign_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(day_ai.TOKEN_ENV, 'token')
    loader = make(tmp_path)
    await loader.enable('day_ai')
    assert loader.capabilities() == [DayAI[None]()], 'harness `DayAI` reads the variable itself'
    assert SignIn.connected == []
    await loader.close('exit')


async def test_browser_sign_in_runs_on_enable(tmp_path: Path) -> None:
    output = io.StringIO()
    loader = make(tmp_path, output=output)
    await loader.enable('day_ai')
    transport = day_ai_client(loader)
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == day_ai.DAY_AI_MCP_URL
    assert isinstance(transport.auth, OAuth)
    [signed_in_with] = SignIn.connected
    assert signed_in_with.url == day_ai.DAY_AI_MCP_URL and signed_in_with is not transport
    assert 'Opening your browser to sign in to Day AI.' in output.getvalue()
    await loader.close('exit')


async def test_stored_tokens_skip_the_browser(tmp_path: Path) -> None:
    tokens = TokenStorageAdapter(TokenStore(day_ai.TOKEN_ACCOUNT), server_url=day_ai.DAY_AI_MCP_URL)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))
    loader = make(tmp_path, terminal=False)
    await loader.enable('day_ai')
    assert isinstance(day_ai_client(loader), StreamableHttpTransport)
    assert SignIn.connected == []
    await loader.close('exit')


async def test_failed_sign_in_leaves_nothing_loaded(tmp_path: Path) -> None:
    SignIn.error = RuntimeError('authorization denied')
    loader = make(tmp_path)
    with pytest.raises(PluginError, match="Plugin 'day_ai': RuntimeError: authorization denied"):
        await loader.enable('day_ai')
    assert loader.capabilities() == []


async def test_headless_without_credentials_fails_clearly(tmp_path: Path) -> None:
    loader = make(tmp_path, terminal=False)
    with pytest.raises(PluginError, match='sign in to Day AI from an interactive CLAI session first'):
        await loader.enable('day_ai')
    assert loader.capabilities() == [] and SignIn.connected == []


async def test_oauth_off_needs_the_environment_token(tmp_path: Path) -> None:
    loader = make(tmp_path, settings={'oauth': False})
    with pytest.raises(PluginError, match='Set DAY_AI_ACCESS_TOKEN to connect to Day AI'):
        await loader.enable('day_ai')
    assert loader.capabilities() == []

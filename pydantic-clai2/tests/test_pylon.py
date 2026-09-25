"""The built-in `pylon` plugin: how it is declared and which credential it connects with."""

import io
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.pylon import Pylon
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.capability_catalog import HARNESS_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.mcp import OAUTH_TIMEOUT
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.pylon import PYLON_MCP_URL, activate
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio


def activated(**settings: JsonValue) -> AgentCapability[None]:
    host: PluginHost[None] = PluginHost(name='pylon', console=Console(file=io.StringIO()), settings=settings)
    activate(host)
    [capability] = host.capabilities
    return capability


@pytest.fixture
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PYLON_ACCESS_TOKEN', raising=False)


class TestPylonPlugin:
    def test_declared_as_disabled_clai_built_in(self) -> None:
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pylon']
        assert declaration.factory == 'pydantic_clai2.pylon'
        assert not declaration.enabled
        assert 'pydantic_ai_harness.pylon:Pylon' not in {plugin.factory for plugin in HARNESS_PLUGINS}

    def test_environment_token_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PYLON_ACCESS_TOKEN', 'pylon-token')
        assert activated(read_only=True) == Pylon[None](read_only=True), 'harness reads the variable itself'

    @pytest.mark.usefixtures('no_token')
    def test_browser_sign_in_without_token(self) -> None:
        capability = activated()
        assert isinstance(capability, Pylon)
        client = capability.client
        assert isinstance(client, Client)
        transport = client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == PYLON_MCP_URL == 'https://mcp.usepylon.com'
        assert isinstance(transport.auth, OAuth)
        assert client._init_timeout == OAUTH_TIMEOUT, 'the browser gets as long as `/mcp` gives it'  # pyright: ignore[reportPrivateUsage]
        assert not capability.read_only

    def test_settings_reject_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            activated(token='pylon-token')

    @pytest.mark.usefixtures('no_token')
    async def test_enable_without_any_credential_fails_clearly(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'settings.db')
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pylon']
        loader: PluginLoader[None] = PluginLoader(
            store=store,
            console=Console(file=io.StringIO()),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=(declaration.model_copy(update={'settings': {'browser_sign_in': False}}),),
        )
        with pytest.raises(PluginError, match='Set `PYLON_ACCESS_TOKEN`'):
            await loader.enable('pylon')
        assert loader.capabilities() == []

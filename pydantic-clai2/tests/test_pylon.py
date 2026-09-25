"""The built-in `pylon` plugin: its declaration, its `/keys` reference, and which credential it connects with."""

import io
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.pylon import Pylon
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, pylon
from pydantic_clai2.commands import Commands
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.mcp import OAUTH_TIMEOUT
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio


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


def make_host(settings: dict[str, JsonValue] | None = None) -> PluginHost[None]:
    return PluginHost(name='pylon', console=Console(file=io.StringIO()), settings=settings or {})


def activated(**settings: JsonValue) -> AbstractCapability[None]:
    host = make_host(settings)
    pylon.activate(host)
    [capability] = host.capabilities
    assert isinstance(capability, AbstractCapability)
    # Narrowing `AgentCapability`'s callable arm leaves pyright a `Capability[Unknown]` alternative.
    return capability  # pyright: ignore[reportUnknownVariableType]


def use_prompt(monkeypatch: pytest.MonkeyPatch, prompt: Prompt) -> None:
    monkeypatch.setattr(pylon, 'PromptSession', lambda: prompt)


async def command(host: PluginHost[None], *args: str) -> str:
    [registered] = host.commands
    result = registered.handler(list(args))
    assert not isinstance(result, str)
    return await result


def pylon_tools(capability: AbstractCapability[None]) -> list[str]:
    """Run once and report the Pylon tools the run could see; `TestModel` calls none of them."""
    model = TestModel(call_tools=[])
    Agent(model, deps_type=type(None), capabilities=[capability]).run_sync('hi')
    assert model.last_model_request_parameters is not None
    return [tool.name for tool in model.last_model_request_parameters.function_tools]


class TestPylonPlugin:
    def test_declared_as_disabled_clai_built_in(self) -> None:
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pylon']
        assert declaration.factory == 'pydantic_clai2.pylon'
        assert not declaration.enabled
        assert declaration.settings == {}, 'nothing secret, or otherwise, is declared'

    def test_no_key_chosen_means_no_pylon_tools(self) -> None:
        assert pylon_tools(activated()) == []

    async def test_new_token_is_saved_in_keys_and_only_its_name_is_referenced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt = Prompt(' pylon-secret ')
        use_prompt(monkeypatch, prompt)
        assert await pylon.choose_key() == 'Pylon connects with PYLON_ACCESS_TOKEN from /keys.'
        assert prompt.labels[0][1], 'the value is entered masked'
        assert api_keys.load_keys()['PYLON_ACCESS_TOKEN'].get_secret_value() == 'pylon-secret'
        raw = load_codex_credentials(account='pylon')
        assert raw is not None and 'pylon-secret' not in raw
        assert pylon.saved_key() == api_keys.KeyReference(name='PYLON_ACCESS_TOKEN')

    async def test_shared_key_is_referenced_and_resolved_each_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='SHARED_PYLON', value='first')

        async def pick(**_: object) -> api_keys.KeyReference:
            return api_keys.KeyReference(name='SHARED_PYLON')

        monkeypatch.setattr(pylon, 'prompt_api_key', pick)
        assert await pylon.choose_key() == 'Pylon connects with SHARED_PYLON from /keys.'
        capability = activated()
        assert isinstance(capability, Pylon)
        auth = capability.auth
        assert callable(auth)
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        assert auth(ctx) == 'first'
        api_keys.save_key(name='SHARED_PYLON', value='replaced')
        assert auth(ctx) == 'replaced', 'replacing the key in /keys reaches the next run without a reload'
        with pytest.raises(ValueError, match='used by pylon'):
            api_keys.rename_key(name='SHARED_PYLON', new_name='OTHER')

    def test_deleted_key_fails_the_run_closed(self) -> None:
        api_keys.save_key(name='PYLON_ACCESS_TOKEN', value='secret')
        save_codex_credentials(account='pylon', value='{"token": {"name": "PYLON_ACCESS_TOKEN"}}')
        api_keys.delete_key(name='PYLON_ACCESS_TOKEN')
        with pytest.raises(UserError, match='PYLON_ACCESS_TOKEN is missing'):
            pylon_tools(activated())

    def test_invalid_saved_reference_fails_closed(self) -> None:
        save_codex_credentials(account='pylon', value='{"token": "inline-secret"}')
        with pytest.raises(UserError, match='/pylon key'):
            pylon.saved_key()

    @pytest.mark.parametrize('answer', ['n', EOFError()])
    async def test_existing_label_is_not_replaced_without_consent(
        self, monkeypatch: pytest.MonkeyPatch, answer: str | BaseException
    ) -> None:
        api_keys.save_key(name='PYLON_ACCESS_TOKEN', value='keep')

        async def enter(**_: object) -> str:
            return 'new'

        monkeypatch.setattr(pylon, 'prompt_api_key', enter)
        use_prompt(monkeypatch, Prompt(answer))
        assert await pylon.choose_key() == 'Pylon key unchanged.'
        assert api_keys.load_keys()['PYLON_ACCESS_TOKEN'].get_secret_value() == 'keep'
        assert pylon.saved_key() is None

    async def test_replacing_the_label_with_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='PYLON_ACCESS_TOKEN', value='old')

        async def enter(**_: object) -> str:
            return 'new'

        monkeypatch.setattr(pylon, 'prompt_api_key', enter)
        use_prompt(monkeypatch, Prompt('y'))
        assert await pylon.choose_key() == 'Pylon connects with PYLON_ACCESS_TOKEN from /keys.'
        assert api_keys.load_keys()['PYLON_ACCESS_TOKEN'].get_secret_value() == 'new'

    async def test_cancel_and_blank_entries_save_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_prompt(monkeypatch, Prompt(KeyboardInterrupt(), '  '))
        assert await pylon.choose_key() == 'Pylon key unchanged.'
        with pytest.raises(ValueError, match='access token is required'):
            await pylon.choose_key()
        assert pylon.saved_key() is None and api_keys.load_keys() == {}

    async def test_command_reports_and_chooses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        host = make_host()
        pylon.activate(host)
        assert 'no key yet' in await command(host)
        use_prompt(monkeypatch, Prompt('secret'))
        assert await command(host, 'key') == 'Pylon connects with PYLON_ACCESS_TOKEN from /keys.'
        assert 'connects with PYLON_ACCESS_TOKEN' in await command(host)
        with pytest.raises(ValueError, match='Usage: /pylon'):
            await command(host, 'nope')
        [registered] = host.commands
        assert list(registered.complete([])) == ['key'] and list(registered.complete(['key', ''])) == []

    @pytest.mark.parametrize('terminal', [True, False])
    async def test_enabling_asks_for_a_key_only_in_a_terminal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal: bool
    ) -> None:
        prompt = Prompt('secret')
        use_prompt(monkeypatch, prompt)
        store = SettingsStore(tmp_path / 'settings.db')
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pylon']
        loader: PluginLoader[None] = PluginLoader(
            store=store,
            console=Console(file=io.StringIO(), force_terminal=terminal),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=(declaration,),
        )
        await loader.enable('pylon')
        assert len(loader.capabilities()) == 1
        assert bool(prompt.labels) == terminal
        assert (pylon.saved_key() is not None) == terminal
        await loader.enable('pylon')
        assert len(prompt.labels) == int(terminal), 'a chosen key is not asked for again'

    def test_browser_sign_in(self) -> None:
        capability = activated(auth='browser')
        assert isinstance(capability, Pylon)
        client = capability.client
        assert isinstance(client, Client)
        transport = client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == pylon.PYLON_MCP_URL == 'https://mcp.usepylon.com'
        assert isinstance(transport.auth, OAuth)
        assert client._init_timeout == OAUTH_TIMEOUT, 'the browser gets as long as `/mcp` gives it'  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize('settings', [{'token': 'pylon-token'}, {'auth': 'env'}])
    def test_settings_reject_secrets_and_unknown_modes(self, settings: dict[str, JsonValue]) -> None:
        with pytest.raises(ValidationError):
            activated(**settings)

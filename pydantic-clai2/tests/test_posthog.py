"""The built-in `posthog` plugin: its declaration, its `/keys` reference, read-only mode, and browser sign-in."""

import io
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringLocked
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.posthog import PostHog
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, posthog
from pydantic_clai2.commands import Commands
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.mcp import OAUTH_TIMEOUT, TokenStore
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
    return PluginHost(name='posthog', console=Console(file=io.StringIO()), settings=settings or {})


def loaded(**settings: JsonValue) -> tuple[PluginHost[None], PostHog[None]]:
    host = make_host(settings)
    posthog.activate(host)
    [capability] = host.capabilities
    assert isinstance(capability, PostHog)
    return host, capability  # pyright: ignore[reportUnknownVariableType]


def activated(**settings: JsonValue) -> PostHog[None]:
    return loaded(**settings)[1]


def use_prompt(monkeypatch: pytest.MonkeyPatch, prompt: Prompt) -> None:
    monkeypatch.setattr(posthog, 'PromptSession', lambda: prompt)


async def command(host: PluginHost[None], *args: str) -> str:
    return await host.commands.execute_async(' '.join(['/posthog', *args]))


def browser_transport(capability: PostHog[None]) -> StreamableHttpTransport:
    client = capability.client
    assert isinstance(client, Client)
    assert client._init_timeout == OAUTH_TIMEOUT, 'the browser gets as long as `/mcp` gives it'  # pyright: ignore[reportPrivateUsage]
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == posthog.POSTHOG_MCP_URL == 'https://mcp.posthog.com/mcp'
    assert isinstance(transport.auth, OAuth)
    return transport


def posthog_tools(capability: PostHog[None]) -> list[str]:
    """Run once and report the PostHog tools the run could see; `TestModel` calls none of them."""
    model = TestModel(call_tools=[])
    Agent(model, deps_type=type(None), capabilities=[capability]).run_sync('hi')
    assert model.last_model_request_parameters is not None
    return [tool.name for tool in model.last_model_request_parameters.function_tools]


class TestPostHogPlugin:
    def test_declared_as_disabled_clai_built_in(self) -> None:
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'posthog']
        assert declaration.factory == 'pydantic_clai2.posthog'
        assert not declaration.enabled
        assert declaration.settings == {}

    def test_key_mode_is_read_only_by_default_and_ignores_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('POSTHOG_PERSONAL_API_KEY', 'phx_env')
        capability = activated()
        assert capability.read_only and callable(capability.auth) and capability.client is None
        assert not activated(read_only=False).read_only
        assert posthog_tools(capability) == [], 'no key chosen means no PostHog tools, not the environment variable'

    async def test_new_key_is_saved_in_keys_and_only_its_name_is_referenced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt = Prompt(' phx_secret ')
        use_prompt(monkeypatch, prompt)
        assert await posthog.choose_key() == 'PostHog connects with POSTHOG_PERSONAL_API_KEY from /keys.'
        assert prompt.labels[0][1], 'the value is entered masked'
        assert api_keys.load_keys()['POSTHOG_PERSONAL_API_KEY'].get_secret_value() == 'phx_secret'
        raw = load_codex_credentials(account='posthog')
        assert raw is not None and 'phx_secret' not in raw
        assert posthog.saved_key() == api_keys.KeyReference(name='POSTHOG_PERSONAL_API_KEY')

    async def test_shared_key_is_referenced_and_resolved_each_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='SHARED_POSTHOG', value='first')

        async def pick(**_: object) -> api_keys.KeyReference:
            return api_keys.KeyReference(name='SHARED_POSTHOG')

        monkeypatch.setattr(posthog, 'prompt_api_key', pick)
        assert await posthog.choose_key() == 'PostHog connects with SHARED_POSTHOG from /keys.'
        auth = activated().auth
        assert callable(auth)
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        assert auth(ctx) == 'first'
        api_keys.save_key(name='SHARED_POSTHOG', value='replaced')
        assert auth(ctx) == 'replaced', 'replacing the key in /keys reaches the next run without a reload'
        with pytest.raises(ValueError, match='used by posthog'):
            api_keys.rename_key(name='SHARED_POSTHOG', new_name='OTHER')

    def test_deleted_key_fails_the_run_closed(self) -> None:
        api_keys.save_key(name='POSTHOG_PERSONAL_API_KEY', value='secret')
        save_codex_credentials(account='posthog', value='{"token": {"name": "POSTHOG_PERSONAL_API_KEY"}}')
        api_keys.delete_key(name='POSTHOG_PERSONAL_API_KEY')
        with pytest.raises(UserError, match='POSTHOG_PERSONAL_API_KEY is missing'):
            posthog_tools(activated())

    def test_invalid_saved_reference_fails_closed(self) -> None:
        save_codex_credentials(account='posthog', value='{"token": "inline-secret"}')
        with pytest.raises(UserError, match='/posthog key'):
            posthog.saved_key()

    @pytest.mark.parametrize('answer', ['n', EOFError()])
    async def test_existing_label_is_not_replaced_without_consent(
        self, monkeypatch: pytest.MonkeyPatch, answer: str | BaseException
    ) -> None:
        api_keys.save_key(name='POSTHOG_PERSONAL_API_KEY', value='keep')

        async def enter(**_: object) -> str:
            return 'new'

        monkeypatch.setattr(posthog, 'prompt_api_key', enter)
        use_prompt(monkeypatch, Prompt(answer))
        assert await posthog.choose_key() == 'PostHog key unchanged.'
        assert api_keys.load_keys()['POSTHOG_PERSONAL_API_KEY'].get_secret_value() == 'keep'
        assert posthog.saved_key() is None

    async def test_replacing_the_label_with_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='POSTHOG_PERSONAL_API_KEY', value='old')

        async def enter(**_: object) -> str:
            return 'new'

        monkeypatch.setattr(posthog, 'prompt_api_key', enter)
        use_prompt(monkeypatch, Prompt('y'))
        assert await posthog.choose_key() == 'PostHog connects with POSTHOG_PERSONAL_API_KEY from /keys.'
        assert api_keys.load_keys()['POSTHOG_PERSONAL_API_KEY'].get_secret_value() == 'new'

    async def test_cancel_and_blank_entries_save_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_prompt(monkeypatch, Prompt(KeyboardInterrupt(), '  '))
        assert await posthog.choose_key() == 'PostHog key unchanged.'
        with pytest.raises(ValueError, match='personal API key is required'):
            await posthog.choose_key()
        assert posthog.saved_key() is None and api_keys.load_keys() == {}

    async def test_command_reports_and_chooses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        host = make_host()
        posthog.activate(host)
        assert 'no key yet' in await command(host)
        use_prompt(monkeypatch, Prompt('secret'))
        assert await command(host, 'key') == 'PostHog connects with POSTHOG_PERSONAL_API_KEY from /keys.'
        assert await command(host) == (
            'PostHog (read-only) connects with POSTHOG_PERSONAL_API_KEY from /keys. /posthog key chooses another.'
        )
        with pytest.raises(ValueError, match='Usage: /posthog'):
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
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'posthog']
        output = io.StringIO()
        loader: PluginLoader[None] = PluginLoader(
            store=store,
            console=Console(file=output, force_terminal=terminal, width=200),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=(declaration,),
        )
        await loader.enable('posthog')
        assert len(loader.capabilities()) == 1
        assert bool(prompt.labels) == terminal
        assert (posthog.saved_key() is not None) == terminal
        assert ('PostHog has no key' in output.getvalue()) != terminal, 'without a key, enabling says so'
        await loader.reload('posthog')
        assert len(prompt.labels) == int(terminal), 'a chosen key is not asked for again'
        await loader.close('exit')

    async def test_browser_sign_in_status_and_logout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        host, capability = loaded(auth='browser')
        assert not capability.read_only, 'the read-only header does the filtering, which keeps the `posthog` tool'
        before = capability.client
        assert browser_transport(capability).headers == {'x-posthog-read-only': 'true'}
        assert 'not signed in' in await command(host)
        await TokenStore(posthog.TOKENS).put('token', {'access_token': 'x'}, collection='mcp-oauth-token')
        assert await command(host) == 'PostHog (read-only) is signed in through the browser.'

        def delete(service: str, account: str) -> None:
            pass

        monkeypatch.setattr(keyring, 'delete_password', delete)
        assert 'Signed out' in await command(host, 'logout')
        assert capability.client is not before, 'the live sign-in is replaced, not reused'
        assert browser_transport(capability).headers == {'x-posthog-read-only': 'true'}
        with pytest.raises(ValueError, match='Usage'):
            await command(host, 'login')
        [registered] = host.commands
        assert list(registered.complete([''])) == ['logout'] and list(registered.complete(['logout', ''])) == []

        def locked(service: str, account: str) -> str | None:
            raise KeyringLocked('locked')

        monkeypatch.setattr(keyring, 'get_password', locked)
        assert 'unknown sign-in state' in await command(host)

    def test_browser_read_write_sends_no_read_only_header(self) -> None:
        assert browser_transport(activated(auth='browser', read_only=False)).headers == {}

    @pytest.mark.parametrize('settings', [{'api_key': 'phx_secret'}, {'auth': 'api_key'}, {'auth': 'oauth'}])
    def test_settings_reject_secrets_and_unknown_modes(self, settings: dict[str, JsonValue]) -> None:
        with pytest.raises(ValidationError):
            activated(**settings)

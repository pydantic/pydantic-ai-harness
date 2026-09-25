"""The built-in `day_ai` plugin: a token from `/keys` named in settings, or a browser sign-in, never a stored secret."""

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
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.day_ai import DayAI
from rich.console import Console

import pydantic_clai2.day_ai as day_ai
from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.api_keys import KeyReference, SavedKey
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'day_ai')


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


class Prompts:
    """Stands in for `prompt_api_key` and the confirmation prompt, answering from a script."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None, confirm: str | None = ''
    ) -> None:
        self.labels: list[str] = []
        self.confirm = confirm
        prompts = self

        async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
            assert optional, 'an empty answer must be allowed, since it means browser sign-in'
            prompts.labels.append(label)
            return choice

        class Session:
            async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
                prompts.labels.append(label)
                if prompts.confirm is None:
                    raise KeyboardInterrupt
                return prompts.confirm

        monkeypatch.setattr(day_ai, 'prompt_api_key', prompt_api_key)
        monkeypatch.setattr(day_ai, 'PromptSession', Session)


class Shell:
    """A loader plus what a test inspects: typed commands, printed output, and the plaintext settings file."""

    def __init__(self, tmp_path: Path, *, terminal: bool = True, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'settings.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        self.commands = Commands()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, force_terminal=terminal, width=200),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    async def run(self, text: str) -> str:
        return await self.commands.execute_async(text)


@pytest.fixture(autouse=True)
def sign_in(monkeypatch: pytest.MonkeyPatch) -> type[SignIn]:
    monkeypatch.setattr(day_ai, 'Client', SignIn)
    monkeypatch.setattr(SignIn, 'connected', [])
    monkeypatch.setattr(SignIn, 'error', None)
    return SignIn


def with_key(name: str = day_ai.KEY_NAME) -> list[DayAI[None]]:
    return [DayAI[None](auth=SavedKey(name=name, setup=day_ai.SETUP))]


def browser_client(shell: Shell) -> object:
    [client] = [capability.client for capability in shell.loader.capabilities() if isinstance(capability, DayAI)]
    return client


def test_declared_disabled_with_no_settings() -> None:
    assert BUILTIN == PluginSettings(id='day_ai', factory='pydantic_clai2.day_ai', enabled=False)


async def test_conventional_key_is_used_and_resolved_on_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('DAY_AI_ACCESS_TOKEN', 'a-label-not-a-source')
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='first')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == with_key()
    assert SignIn.connected == [], 'a token needs no browser sign-in'
    assert await shell.run('/day_ai') == 'Day AI token: DAY_AI_ACCESS_TOKEN in /keys (saved).'
    token = SavedKey(name='DAY_AI_ACCESS_TOKEN', setup=day_ai.SETUP)
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='replaced')
    assert token(None) == 'replaced'
    api_keys.delete_key(name='DAY_AI_ACCESS_TOKEN')
    with pytest.raises(UserError, match='DAY_AI_ACCESS_TOKEN is missing. Run /day_ai connect'):
        token(None)
    await shell.loader.close('exit')


async def test_named_key_missing_still_loads_and_fails_closed(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False, settings={'token': {'name': 'WORK_DAY_AI'}})
    await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == with_key('WORK_DAY_AI')
    assert 'Day AI has no token: WORK_DAY_AI is not in /keys.' in shell.output.getvalue()
    assert await shell.run('/day_ai') == 'Day AI token: WORK_DAY_AI in /keys (missing).'
    assert SignIn.connected == [], 'a named key never falls back to the browser'
    await shell.loader.close('exit')


async def test_settings_cannot_hold_a_token(tmp_path: Path) -> None:
    shell = Shell(tmp_path, settings={'token': 'secret'})
    with pytest.raises(PluginError, match='token'):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == []


async def test_environment_is_not_a_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DAY_AI_ACCESS_TOKEN', 'ignored')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert isinstance(browser_client(shell), StreamableHttpTransport)
    assert len(SignIn.connected) == 1
    await shell.loader.close('exit')


async def test_browser_sign_in_runs_on_enable(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    transport = browser_client(shell)
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == day_ai.DAY_AI_MCP_URL
    assert isinstance(transport.auth, OAuth)
    [signed_in_with] = SignIn.connected
    assert signed_in_with.url == day_ai.DAY_AI_MCP_URL and signed_in_with is not transport
    assert 'Opening your browser to sign in to Day AI.' in shell.output.getvalue()
    assert await shell.run('/day_ai') == (
        'Day AI signs in through the browser. /day_ai connect chooses a token from /keys instead.'
    )
    await shell.loader.close('exit')


async def test_stored_sign_in_skips_the_browser(tmp_path: Path) -> None:
    tokens = TokenStorageAdapter(TokenStore(day_ai.TOKEN_ACCOUNT), server_url=day_ai.DAY_AI_MCP_URL)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('day_ai')
    assert isinstance(browser_client(shell), StreamableHttpTransport)
    assert SignIn.connected == []
    await shell.loader.close('exit')


async def test_failed_sign_in_leaves_nothing_loaded(tmp_path: Path) -> None:
    SignIn.error = RuntimeError('authorization denied')
    shell = Shell(tmp_path)
    with pytest.raises(PluginError, match="Plugin 'day_ai': RuntimeError: authorization denied"):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == []


async def test_headless_without_credentials_fails_clearly(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    with pytest.raises(PluginError, match='Save DAY_AI_ACCESS_TOKEN in /keys, or sign in to Day AI'):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == [] and SignIn.connected == []


async def test_connect_saves_a_new_token_in_keys_and_only_its_name_in_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    prompts = Prompts(monkeypatch, ' day-secret ')
    assert await shell.run('/day_ai connect') == (
        'Day AI will use the saved key DAY_AI_ACCESS_TOKEN; manage it in /keys. Run /plugins reload day_ai to connect with it.'
    )
    assert prompts.labels == [
        'Day AI access token, saved in /keys as DAY_AI_ACCESS_TOKEN (leave empty to sign in through the browser): '
    ]
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'day-secret'
    [saved] = shell.store.plugins()
    assert saved.enabled and saved.settings == {'token': {'name': 'DAY_AI_ACCESS_TOKEN'}}
    assert b'day-secret' not in shell.path.read_bytes()
    await shell.loader.reload('day_ai')
    assert shell.loader.capabilities() == with_key()
    await shell.loader.close('exit')


async def test_connect_to_an_existing_key_then_back_to_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='WORK_DAY_AI', value='work-secret')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    Prompts(monkeypatch, KeyReference(name='WORK_DAY_AI'))
    await shell.run('/day_ai connect')
    assert shell.store.plugins()[0].settings == {'token': {'name': 'WORK_DAY_AI'}}
    assert b'work-secret' not in shell.path.read_bytes()
    await shell.loader.reload('day_ai')
    assert shell.loader.capabilities() == with_key('WORK_DAY_AI')
    Prompts(monkeypatch, '  ')
    assert await shell.run('/day_ai connect') == (
        'Day AI will sign in through the browser. Run /plugins reload day_ai to connect with it.'
    )
    assert shell.store.plugins()[0].settings == {'token': None}
    await shell.loader.close('exit')


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    prompts = Prompts(monkeypatch, 'other', confirm='n')
    assert await shell.run('/day_ai connect') == 'Day AI connection unchanged.'
    assert prompts.labels[-1] == 'Replace DAY_AI_ACCESS_TOKEN for every plugin that uses it? [y/N]: '
    prompts.confirm = None
    assert await shell.run('/day_ai connect') == 'Day AI connection unchanged.'
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'shared'
    prompts.confirm = 'y'
    await shell.run('/day_ai connect')
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'other'
    await shell.loader.close('exit')


async def test_cancelled_connect_and_bad_arguments_change_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='kept')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    Prompts(monkeypatch, None)
    assert await shell.run('/day_ai connect') == 'Day AI connection unchanged.'
    with pytest.raises(ValueError, match=r'Usage: /day_ai \[connect\]'):
        await shell.run('/day_ai status')
    assert [plugin.settings for plugin in shell.store.plugins()] == [{}]
    await shell.loader.close('exit')

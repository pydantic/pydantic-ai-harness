"""The built-in `linear` plugin: declared disabled, its key named in `/keys`, and resolved on every run."""

import io
import json
from pathlib import Path
from typing import TypeGuard

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import PasswordDeleteError
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.linear import Linear
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, linear
from pydantic_clai2.api_keys import delete_key, key_users, load_keys, rename_key, save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.linear import ACCOUNT, KEY_NAME, TOKEN_ACCOUNT, choose_key, reference
from pydantic_clai2.mcp import TokenStore, http_client
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio
Vault = dict[tuple[str, str], str]


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Vault:
    """A keyring that can also delete, so signing out and deleting keys are observable."""
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


class Prompt:
    def __init__(self, *values: str | BaseException) -> None:
        self.values = iter(values)
        self.labels: list[str] = []

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        self.labels.append(label)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def make(tmp_path: Path) -> tuple[PluginLoader[None], Commands, io.StringIO]:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    output = io.StringIO()
    loader = PluginLoader[None](
        store=store,
        console=Console(file=output, width=400),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear'],
    )
    return loader, commands, output


async def declare(loader: PluginLoader[None], settings: dict[str, JsonValue]) -> None:
    await loader.remove('linear')
    await loader.command(['add', 'linear', 'pydantic_clai2.linear', json.dumps(settings)])


def is_linear(capability: object) -> TypeGuard[Linear[None]]:
    return isinstance(capability, Linear)


def token(loader: PluginLoader[None]) -> str | None:
    """What the next run connects with."""
    [capability] = loader.capabilities()
    assert is_linear(capability) and callable(capability.auth)
    return capability.auth(RunContext(deps=None, model=TestModel(), usage=RunUsage()))


async def run(commands: Commands, text: str) -> str:
    result = commands.execute(text)
    return result if isinstance(result, str) else await result


def test_declared_as_a_disabled_built_in_not_the_raw_capability() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear']
    assert declaration.factory == 'pydantic_clai2.linear'
    assert not declaration.enabled
    assert all(not plugin.factory.startswith('pydantic_ai_harness.linear') for plugin in DEFAULT_PLUGINS)


async def test_each_run_resolves_the_named_key(tmp_path: Path, vault: Vault) -> None:
    loader, _, output = make(tmp_path)
    await loader.load_all()
    assert loader.capabilities() == [], 'disabled until the user enables it'
    save_key(name=KEY_NAME, value='lin_1')
    assert await loader.command(['enable', 'linear']) == 'Enabled linear.'
    assert output.getvalue() == ''
    [capability] = loader.capabilities()
    assert isinstance(capability, Linear) and capability.read_only
    assert token(loader) == 'lin_1'

    save_key(name=KEY_NAME, value='lin_2')
    assert token(loader) == 'lin_2', 'replacing the key in /keys reaches the next run without a reload'
    delete_key(name=KEY_NAME)
    with pytest.raises(
        UserError, match=f'{KEY_NAME} is missing. Restore it in /keys or reconfigure through /linear key'
    ):
        token(loader)
    assert all(
        'lin_' not in value
        for value in (str(plugin.settings) for plugin in SettingsStore(tmp_path / 'settings.db').plugins())
    )

    await declare(loader, {'read_only': False})
    [capability] = loader.capabilities()
    assert isinstance(capability, Linear) and not capability.read_only
    await loader.close('exit')


async def test_missing_key_is_reported_at_load_and_fixed_with_linear_key(
    tmp_path: Path, vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader, commands, output = make(tmp_path)
    await loader.enable('linear')
    assert f'Linear: Saved API key {KEY_NAME} is missing' in output.getvalue()
    with pytest.raises(UserError, match='is missing'):
        token(loader)

    prompt = Prompt('lin_typed')
    monkeypatch.setattr(linear, 'PromptSession', lambda: prompt)
    assert (await run(commands, '/linear key')).endswith(f'Linear uses {KEY_NAME} from /keys from the next run.')
    assert prompt.labels == [f'Linear API key (saved in /keys as {KEY_NAME}): ']
    assert token(loader) == 'lin_typed', 'no reload needed'


async def test_settings_carry_no_credential(tmp_path: Path) -> None:
    loader, _, _ = make(tmp_path)
    with pytest.raises(PluginError, match='api_key'):
        await declare(loader, {'api_key': 'lin_secret'})
    assert loader.capabilities() == []


async def test_choose_a_saved_key_stores_only_its_name(
    tmp_path: Path, vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_key(name='GITHUB_TOKEN', value='gh')
    save_key(name='SHARED_LINEAR', value='lin_shared')
    monkeypatch.setattr(api_keys, 'menu_key', iter(['down', 'enter']).__next__)
    assert await choose_key(Prompt()) == 'Linear uses SHARED_LINEAR from /keys from the next run.'
    assert reference().name == 'SHARED_LINEAR'
    raw = load_codex_credentials(account=ACCOUNT)
    assert raw is not None and json.loads(raw) == {'token': {'name': 'SHARED_LINEAR'}}
    assert key_users(name='SHARED_LINEAR') == ['linear']
    with pytest.raises(ValueError, match='used by linear'):
        rename_key(name='SHARED_LINEAR', new_name='OTHER')

    loader, commands, _ = make(tmp_path)
    await loader.enable('linear')
    assert token(loader) == 'lin_shared'
    with pytest.raises(ValueError, match='Usage: /linear key'):
        await run(commands, '/linear')


@pytest.mark.parametrize(
    ('existing', 'answer', 'stored', 'result'),
    [
        (False, None, 'lin_new', f'Saved {KEY_NAME} in the OS keyring. Linear uses {KEY_NAME}'),
        (True, 'y', 'lin_new', f'Saved {KEY_NAME} in the OS keyring. Linear uses {KEY_NAME}'),
        (True, 'n', 'lin_old', 'Linear key unchanged.'),
        (True, EOFError(), 'lin_old', 'Linear key unchanged.'),
    ],
)
async def test_enter_a_new_key(
    vault: Vault,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    answer: str | BaseException | None,
    stored: str,
    result: str,
) -> None:
    if existing:
        save_key(name=KEY_NAME, value='lin_old')
        monkeypatch.setattr(api_keys, 'menu_key', iter(['down', 'enter']).__next__)
    prompt = Prompt(' lin_new ', *([] if answer is None else [answer]))
    assert (await choose_key(prompt)).startswith(result)
    assert load_keys()[KEY_NAME].get_secret_value() == stored
    raw = load_codex_credentials(account=ACCOUNT) or ''
    assert 'lin_' not in raw, 'the credential-store entry holds a name, never the key'
    if existing:
        assert prompt.labels[-1] == f'Replace {KEY_NAME} in /keys for every plugin using it? [y/N]: '


async def test_key_deleted_while_choosing(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    async def gone(*, prompt: object, label: str) -> api_keys.KeyReference:
        return api_keys.KeyReference(name='GONE')

    monkeypatch.setattr(linear, 'prompt_api_key', gone)
    with pytest.raises(UserError, match='Select a saved key again through /linear key'):
        await choose_key(Prompt())
    assert load_codex_credentials(account=ACCOUNT) is None


async def test_cancel_empty_and_invalid_choice(vault: Vault) -> None:
    assert await choose_key(Prompt(EOFError())) == 'Linear key unchanged.'
    with pytest.raises(ValueError, match='A Linear API key is required'):
        await choose_key(Prompt('  '))
    assert reference().name == KEY_NAME
    save_codex_credentials(account=ACCOUNT, value='{"token": "inline-secret"}')
    with pytest.raises(UserError, match='saved Linear key choice is invalid'):
        reference()


@pytest.mark.parametrize(
    ('read_only', 'url'), [(True, 'https://mcp.linear.app/mcp/readonly'), (False, 'https://mcp.linear.app/mcp')]
)
async def test_oauth_signs_in_with_keyring_tokens(tmp_path: Path, vault: Vault, read_only: bool, url: str) -> None:
    loader, commands, _ = make(tmp_path)
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

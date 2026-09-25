"""The built-in `ordinal` plugin: a named `/keys` entry, the environment token, or a browser sign-in in the keyring."""

import io
import sys
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringLocked, PasswordDeleteError
from mcp.shared.auth import OAuthToken
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ordinal import Ordinal
from rich.console import Console

import pydantic_clai2.ordinal as ordinal_plugin
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2._app import RETIRED_PLUGINS, create_shell
from pydantic_clai2.api_keys import KeyReference, delete_key, load_keys, rename_key, save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.credential_store import save_codex_credentials
from pydantic_clai2.mcp import OAUTH_TIMEOUT, TokenStore
from pydantic_clai2.ordinal import KEY_ACCOUNT, KEY_NAME, TOKENS, URL, USAGE, OrdinalAuth, activate
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore

Vault = dict[tuple[str, str], str]
Picked = str | KeyReference | None


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Vault:
    entries: Vault = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        if (service, account) not in entries:
            raise PasswordDeleteError('Not found')
        del entries[service, account]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    monkeypatch.delenv(KEY_NAME, raising=False)
    return entries


class Answers:
    """A `PromptSession` stand-in for the replace-key confirmation; `None` means Ctrl-D."""

    def __init__(self, *answers: str | None) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def __call__(self) -> 'Answers':
        return self

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        self.asked.append(label)
        answer = self.answers.pop(0)
        if answer is None:
            raise EOFError
        return answer


def pick(monkeypatch: pytest.MonkeyPatch, picked: Picked, *answers: str | None) -> Answers:
    """Answer the shared `/keys` picker with `picked`, and any confirmation with `answers`."""
    labels: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> Picked:
        assert optional
        labels.append(label)
        return picked

    session = Answers(*answers)
    monkeypatch.setattr(ordinal_plugin, 'prompt_api_key', prompt_api_key)
    monkeypatch.setattr(ordinal_plugin, 'PromptSession', session)
    return session


def terminal(monkeypatch: pytest.MonkeyPatch, *, attached: bool) -> None:
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: attached)


async def sign_in() -> None:
    token = OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh', expires_in=3600)
    await TokenStorageAdapter(TokenStore(TOKENS), server_url=URL).set_tokens(token)


def host(*, is_terminal: bool = False) -> tuple[PluginHost[None], io.StringIO]:
    output = io.StringIO()
    console = Console(file=output, force_terminal=is_terminal)
    return PluginHost[None](name='ordinal', console=console, settings={}), output


def auth_of(plugin: PluginHost[None]) -> OrdinalAuth[None]:
    [capability] = plugin.capabilities
    assert isinstance(capability, OrdinalAuth)
    return capability  # pyright: ignore[reportUnknownVariableType]


async def next_run(plugin: PluginHost[None]) -> Ordinal[None]:
    """The `Ordinal` the plugin hands the next run."""
    return await auth_of(plugin)(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))


async def run(plugin: PluginHost[None], *args: str) -> str:
    [command] = list(plugin.commands)
    result = command.handler(list(args))
    return result if isinstance(result, str) else await result


def test_declared_as_a_disabled_built_in_backed_by_the_plugin_module() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ordinal']
    assert declaration.factory == 'pydantic_clai2.ordinal'
    assert not declaration.enabled
    assert declaration.settings == {}
    assert [plugin.factory for plugin in RETIRED_PLUGINS if plugin.id == 'ordinal'] == [
        'pydantic_ai_harness.ordinal:Ordinal'
    ]


def test_url_matches_the_harness_endpoint() -> None:
    toolset = Ordinal[None](auth='token').get_toolset()
    assert isinstance(toolset, MCPToolset)
    assert isinstance(toolset.client, Client)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == URL


async def test_named_key_is_resolved_every_run_and_fails_closed(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_NAME, 'from-the-environment')
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    pick(monkeypatch, 'first-token')
    assert await run(plugin, 'key') == f'Ordinal uses {KEY_NAME} from /keys, starting with the next run.'
    assert load_keys()[KEY_NAME].get_secret_value() == 'first-token'
    assert all('first-token' not in value for (_, account), value in vault.items() if account == KEY_ACCOUNT)
    # A chosen key beats the environment, as an explicit `auth` does in harness `Ordinal`.
    assert (await next_run(plugin)).auth == 'first-token'

    save_key(name=KEY_NAME, value='replaced-token')
    assert (await next_run(plugin)).auth == 'replaced-token'
    with pytest.raises(ValueError, match='used by ordinal'):
        rename_key(name=KEY_NAME, new_name='ELSEWHERE')
    assert await run(plugin) == f'Ordinal uses {KEY_NAME} from /keys. /ordinal key chooses another.'

    delete_key(name=KEY_NAME)
    with pytest.raises(UserError, match=f'{KEY_NAME} is missing'):
        await next_run(plugin)


async def test_existing_key_is_shared_by_name(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SHARED_ORDINAL', value='shared-token')
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    pick(monkeypatch, KeyReference(name='SHARED_ORDINAL'))
    assert 'SHARED_ORDINAL from /keys' in str(await run(plugin, 'key'))
    assert (await next_run(plugin)).auth == 'shared-token'
    assert set(load_keys()) == {'SHARED_ORDINAL'}


async def test_typing_over_a_shared_key_asks_first(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name=KEY_NAME, value='kept')
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    for answer in ('n', None):
        session = pick(monkeypatch, 'unwanted', answer)
        assert await run(plugin, 'key') == 'Ordinal key unchanged.'
        assert 'every connection using it' in session.asked[0]
    assert load_keys()[KEY_NAME].get_secret_value() == 'kept'
    pick(monkeypatch, 'wanted', 'y')
    assert 'from /keys' in str(await run(plugin, 'key'))
    assert load_keys()[KEY_NAME].get_secret_value() == 'wanted'


async def test_no_key_and_cancel_choices(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name=KEY_NAME, value='token')
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    pick(monkeypatch, KeyReference(name=KEY_NAME))
    await run(plugin, 'key')
    pick(monkeypatch, None)
    assert await run(plugin, 'key') == 'Ordinal key unchanged.'
    assert (await next_run(plugin)).auth == 'token'
    pick(monkeypatch, '')
    assert 'uses no /keys entry' in str(await run(plugin, 'key'))
    assert (await next_run(plugin)).client is not None


async def test_environment_token_when_no_key_is_chosen(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_NAME, 'token')
    terminal(monkeypatch, attached=False)
    plugin, output = host()
    activate(plugin)
    ordinal = await next_run(plugin)
    assert ordinal.client is None and ordinal.auth is None
    assert output.getvalue() == ''
    assert await run(plugin) == f'Ordinal uses `{KEY_NAME}` from the environment.'
    assert await run(plugin, 'logout') == f'Signed out of the browser session; runs still use `{KEY_NAME}`.'


async def test_browser_sign_in_waits_for_the_browser(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    client = (await next_run(plugin)).client
    assert isinstance(client, Client)
    assert isinstance(client.transport, StreamableHttpTransport)
    assert client.transport.url == URL
    assert isinstance(client.transport.auth, OAuth)
    # `MCPToolset` gives a bare transport a 5 second handshake, which would end the sign-in early.
    assert client._init_timeout == OAUTH_TIMEOUT  # pyright: ignore[reportPrivateUsage]
    assert await run(plugin) == 'Ordinal: not signed in; the browser opens on first use.'


async def test_logout_ends_the_browser_session(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    await sign_in()
    assert ('pydantic-clai2', f'mcp-{TOKENS}') in vault
    terminal(monkeypatch, attached=False)
    plugin, output = host()
    activate(plugin)
    assert output.getvalue() == ''
    assert await run(plugin) == 'Ordinal: signed in through the browser.'
    signed_in = (await next_run(plugin)).client
    assert 'Signed out of Ordinal' in str(await run(plugin, 'logout'))
    assert vault == {}
    signed_out = (await next_run(plugin)).client
    assert isinstance(signed_in, Client) and isinstance(signed_out, Client)
    # FastMCP keeps tokens inside the `OAuth` once connected; the next run must not reuse it.
    assert signed_out is not signed_in and signed_out.transport.auth is not signed_in.transport.auth
    assert await run(plugin) == 'Ordinal: not signed in; the browser opens on first use.'


async def test_logout_leaves_a_chosen_key_alone(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name=KEY_NAME, value='token')
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    pick(monkeypatch, KeyReference(name=KEY_NAME))
    await run(plugin, 'key')
    assert f'runs still use {KEY_NAME} from /keys' in str(await run(plugin, 'logout'))
    assert load_keys()[KEY_NAME].get_secret_value() == 'token'


async def test_unreadable_keyring_is_reported(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)

    def locked(service: str, account: str) -> str | None:
        if account == KEY_ACCOUNT:
            return None
        raise KeyringLocked('locked')

    monkeypatch.setattr(keyring, 'get_password', locked)
    assert await run(plugin) == 'Ordinal: sign-in unknown; the keyring could not be read.'


async def test_invalid_saved_choice_fails_closed(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    save_codex_credentials(account=KEY_ACCOUNT, value='{"token": "a raw secret"}')
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    with pytest.raises(UserError, match='/ordinal key'):
        await next_run(plugin)


async def test_command_usage_and_completion(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=True)
    plugin, _ = host()
    activate(plugin)
    [command] = list(plugin.commands)
    assert command.name == 'ordinal'
    assert await run(plugin, 'nope') == USAGE
    assert list(command.complete([''])) == ['key', 'logout']
    assert list(command.complete(['key', ''])) == []


def ordinal_loader(store: SettingsStore, console: Console | None = None) -> PluginLoader[None]:
    return PluginLoader[None](
        store=store,
        console=console or Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ordinal'],
    )


async def test_enabling_offers_the_key_picker_once_nothing_is_configured(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    terminal(monkeypatch, attached=True)
    save_key(name=KEY_NAME, value='token')
    pick(monkeypatch, KeyReference(name=KEY_NAME))
    output = io.StringIO()
    loader = ordinal_loader(SettingsStore(tmp_path / 'settings.db'), Console(file=output, force_terminal=True))
    await loader.enable('ordinal')
    assert f'Ordinal uses {KEY_NAME} from /keys' in output.getvalue()

    pick(monkeypatch, 'never asked')
    await loader.reload('ordinal')
    assert 'never asked' not in output.getvalue()
    assert output.getvalue().count('from /keys') == 1


async def test_saved_catalog_toggle_becomes_the_built_in(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(KEY_NAME, 'token')
    store = SettingsStore(tmp_path / 'settings.db')

    def shell_loader() -> PluginLoader[None]:
        return create_shell(
            Agent(TestModel()),
            deps=None,
            plugins=(),
            usage_limits=None,
            settings=None,
            project=ProjectSettings(),
            console=Console(file=io.StringIO(), force_terminal=True),
            store=store,
            builtin_plugins=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ordinal'],
        ).loader

    store.save_plugin(PluginSettings(id='ordinal', factory='pydantic_ai_harness.ordinal:Ordinal', enabled=True))
    loader = shell_loader()
    [entry] = loader.entries()
    assert entry.builtin and entry.declaration.enabled
    assert entry.declaration.factory == 'pydantic_clai2.ordinal'
    await loader.load_all()
    [capability] = loader.capabilities()
    # `/plugins reload` re-imports the module, so compare where the class lives rather than its identity.
    assert type(capability).__module__ == 'pydantic_clai2.ordinal'

    customized = PluginSettings(id='ordinal', factory='pydantic_ai_harness.ordinal:Ordinal', settings={'id': 'mine'})
    store.save_plugin(customized)
    [entry] = shell_loader().entries()
    assert entry.declaration == customized and not entry.builtin


async def test_nothing_to_authenticate_with_and_no_terminal_fails_to_enable(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    terminal(monkeypatch, attached=False)
    plugin, _ = host()
    with pytest.raises(UserError, match=KEY_NAME):
        activate(plugin)
    assert plugin.capabilities == []

    loader = ordinal_loader(SettingsStore(tmp_path / 'settings.db'))
    with pytest.raises(PluginError, match=KEY_NAME):
        await loader.enable('ordinal')
    assert loader.capabilities() == []

    save_key(name=KEY_NAME, value='token')
    save_codex_credentials(account=KEY_ACCOUNT, value=f'{{"token": {{"name": "{KEY_NAME}"}}}}')
    await loader.enable('ordinal')
    assert len(loader.capabilities()) == 1

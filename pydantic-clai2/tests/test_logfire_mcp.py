"""The `logfire_mcp` built-in: its settings menu, keys kept in `/keys`, credential order, and OAuth."""

import io
import webbrowser
from pathlib import Path

import pytest
from fastmcp import Client
from menu_script import Script, pick, typed
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LogfireMCP
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInputResult  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import TypeIs

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.api_keys import KeyReference, SavedKey
from pydantic_clai2.commands import Commands
from pydantic_clai2.field_menu import CUSTOM
from pydantic_clai2.logfire_mcp import SETUP, TOKEN_ACCOUNT, LogfireMCPSource, command
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'logfire_mcp')
CLOSE = MenuResult(cancelled=True)
SELF_HOSTED = 'https://logfire.example.com/mcp'


class Shell:
    """A loader plus what a test inspects: printed output and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        return declaration.settings

    def capability(self) -> LogfireMCP[None]:
        [capability] = self.loader.capabilities()
        assert is_logfire_mcp(capability)
        return capability


def is_logfire_mcp(capability: object) -> TypeIs[LogfireMCP[None]]:
    return isinstance(capability, LogfireMCP)


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr('pydantic_clai2.logfire_mcp.RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> list[tuple[str, bool]]:
    """Answer `prompt_api_key`, whose saved-key list needs a real terminal, and record its labels."""
    calls: list[tuple[str, bool]] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        calls.append((label, optional))
        return choice

    monkeypatch.setattr('pydantic_clai2.logfire_mcp.prompt_api_key', prompt_api_key)
    return calls


def browser(monkeypatch: pytest.MonkeyPatch, *, found: bool) -> None:
    def get() -> webbrowser.BaseBrowser:
        if not found:
            raise webbrowser.Error('could not locate runnable browser')
        return webbrowser.GenericBrowser('true')

    monkeypatch.setattr(webbrowser, 'get', get)


def keyed(name: str) -> LogfireMCP[None]:
    return LogfireMCP[None](auth=SavedKey(name=name, setup=SETUP), read_only=True)


def test_declared_disabled_with_no_settings() -> None:
    assert (BUILTIN.factory, BUILTIN.enabled, BUILTIN.settings) == ('pydantic_clai2.logfire_mcp', False, {})


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser(monkeypatch, found=True)
    calls = key_choice(monkeypatch, ' typed-secret ')
    shown = script(
        monkeypatch,
        lists=[pick('key'), pick('url'), pick('read_only'), pick('include_instructions'), pick('oauth')],
        choices=[pick(CUSTOM), pick('false'), pick('false'), pick('false')],
        texts=[typed(SELF_HOSTED)],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'logfire_mcp']) == '\n'.join(
        [
            'Enabled logfire_mcp.',
            'Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.',
            'Saved Destination.',
            'Saved Tools.',
            'Saved Server instructions.',
            'Saved Browser sign-in.',
        ]
    )
    assert calls == [('Logfire API key (saved in /keys as LOGFIRE_API_KEY)', True)]
    assert shown.opened.count('list') == 6
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'typed-secret'
    assert shell.saved() == {
        'key': {'name': 'LOGFIRE_API_KEY'},
        'url': SELF_HOSTED,
        'oauth': False,
        'read_only': False,
        'include_instructions': False,
    }
    assert b'typed-secret' not in shell.path.read_bytes()
    capability = shell.capability()
    assert capability == LogfireMCP[None](
        auth=SavedKey(name='LOGFIRE_API_KEY', setup=SETUP), url=SELF_HOSTED, read_only=False, include_instructions=False
    )


async def test_reopening_repicks_a_saved_key_and_region_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='SHARED', value='shared-secret')
    shell = Shell(tmp_path, {'key': {'name': 'LOGFIRE_API_KEY'}})
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'logfire_mcp'])
    assert 'LOGFIRE_API_KEY is not in /keys. Run /plugins configure logfire_mcp' in shell.output.getvalue()
    key_choice(monkeypatch, KeyReference(name='SHARED'))
    script(monkeypatch, lists=[pick('key'), pick('url')], choices=[pick(LOGFIRE_EU_MCP_URL)])
    assert await shell.loader.command(['configure', 'logfire_mcp']) == (
        'Logfire uses the saved key SHARED. Manage it in /keys.\nSaved Destination.'
    )
    assert shell.saved()['key'] == {'name': 'SHARED'}
    assert b'shared-secret' not in shell.path.read_bytes()
    capability = shell.capability()
    assert (capability.auth, capability.url) == (SavedKey(name='SHARED', setup=SETUP), LOGFIRE_EU_MCP_URL)


async def test_no_api_key_clears_the_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    shell = Shell(tmp_path, {'key': {'name': 'OLD'}})
    script(monkeypatch, lists=[])
    await shell.loader.enable('logfire_mcp')
    key_choice(monkeypatch, '')
    script(monkeypatch, lists=[pick('key')])
    assert await shell.loader.configure('logfire_mcp') == (
        'Logfire uses LOGFIRE_API_KEY from the environment or /keys, then browser sign-in.'
    )
    assert shell.saved()['key'] is None
    assert shell.capability() == LogfireMCP[None](read_only=True)


async def test_new_key_is_typed_masked_when_no_keys_are_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    script(monkeypatch, lists=[pick('key')], texts=[typed('masked-secret')])
    await shell.loader.command(['enable', 'logfire_mcp'])
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'masked-secret'


@pytest.mark.parametrize('answer', [TextInputResult(cancelled=True), typed('   ')])
async def test_cancelled_or_blank_key_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: TextInputResult
) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    script(monkeypatch, lists=[pick('key')], texts=[answer])
    assert await shell.loader.configure('logfire_mcp') == 'Logfire MCP settings unchanged.'
    assert api_keys.load_keys() == {}


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    key_choice(monkeypatch, 'other')
    script(monkeypatch, lists=[pick('key'), pick('key')], choices=[pick(False), pick(True)])
    assert await shell.loader.configure('logfire_mcp') == (
        'Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.'
    )
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'other'


def test_menu_validates_resets_and_notes_where_the_key_comes_from(monkeypatch: pytest.MonkeyPatch) -> None:
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={'read_only': False})
    source = LogfireMCPSource(host)

    def note() -> str:
        return {row.key: row for row in source.rows()}['key'].note

    rows = {row.key: row for row in source.rows()}
    assert (source.current(rows['key']), note()) == ('(none)', 'browser sign-in, if on')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    assert note() == 'LOGFIRE_API_KEY from /keys'
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env')
    assert note() == 'LOGFIRE_API_KEY from the environment'
    source.save(source.settings.model_copy(update={'key': KeyReference(name='GONE')}))
    assert (source.current(rows['key']), note()) == ('GONE', 'missing from /keys')
    source.save(source.settings.model_copy(update={'key': KeyReference(name='LOGFIRE_API_KEY')}))
    assert note() == ''
    bad_url = 'Value error, Use an https:// URL without credentials or a query.'
    assert source.problem(rows['url'], 'http://logfire.example.com/mcp') == bad_url
    assert source.problem(rows['url'], 'https://user:pw@logfire.example.com/mcp') == bad_url
    assert source.problem(rows['read_only'], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(rows['url'], SELF_HOSTED) is None
    assert source.current(rows['read_only']) == 'false'
    assert source.reset(rows['read_only']) == 'Reset Tools.'
    assert source.current(rows['read_only']) == 'true'


@pytest.mark.parametrize(
    'settings',
    [
        {'key': 'inline-secret'},
        {'key': {'name': 'X', 'value': 'secret'}},
        {'auth': 'secret'},
        {'region': 'eu'},
        {'url': 'http://logfire-us.pydantic.dev/mcp'},
    ],
)
async def test_settings_cannot_hold_a_secret_or_invalid_options(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('logfire_mcp')
    assert shell.loader.capabilities() == []


async def test_environment_key_wins_over_the_conventional_saved_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    assert shell.capability() == LogfireMCP[None](read_only=True)


async def test_conventional_saved_key_resolves_each_run(tmp_path: Path) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='first')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    capability = shell.capability()
    assert capability == keyed('LOGFIRE_API_KEY')
    assert isinstance(capability.auth, SavedKey)
    api_keys.save_key(name='LOGFIRE_API_KEY', value='second')
    assert capability.auth(None) == 'second'
    api_keys.delete_key(name='LOGFIRE_API_KEY')
    with pytest.raises(UserError, match='LOGFIRE_API_KEY is missing. Run /plugins configure logfire_mcp'):
        capability.auth(None)


async def test_oauth_uses_a_keyring_client_with_a_sign_in_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser(monkeypatch, found=True)
    shell = Shell(tmp_path, {'url': LOGFIRE_EU_MCP_URL})
    await shell.loader.enable('logfire_mcp')
    client = shell.capability().client
    assert isinstance(client, Client)
    assert client._init_timeout == 330  # pyright: ignore[reportPrivateUsage]
    assert str(client.transport.url) == LOGFIRE_EU_MCP_URL  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]


@pytest.mark.parametrize(
    ('settings', 'problem'),
    [({}, 'browser sign-in needs a browser'), ({'oauth': False}, 'there is no API key and browser sign-in is off')],
)
async def test_no_credential_still_loads_the_menu_and_fails_runs_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, JsonValue], problem: str
) -> None:
    browser(monkeypatch, found=False)
    shell = Shell(tmp_path, settings)
    await shell.loader.enable('logfire_mcp')
    assert f'Logfire MCP: {problem}. {SETUP}' in shell.output.getvalue()
    capability = shell.capability()
    auth = capability.auth
    assert auth == SavedKey(name='LOGFIRE_API_KEY', setup=SETUP)
    assert isinstance(auth, SavedKey)
    with pytest.raises(UserError, match='LOGFIRE_API_KEY is missing'):
        auth(None)


async def test_stored_sign_in_needs_no_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    browser(monkeypatch, found=False)
    await TokenStore(TOKEN_ACCOUNT).put('token', {'access_token': 'x'}, collection='mcp-oauth-token')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    assert isinstance(shell.capability().client, Client)
    assert shell.output.getvalue() == ''


async def test_logout_forgets_only_the_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    forgotten: list[str] = []

    def forget(store: TokenStore) -> None:
        forgotten.append(store.name)

    monkeypatch.setattr(TokenStore, 'forget', forget)
    api_keys.save_key(name='LOGFIRE_API_KEY', value='kept')
    assert await command(['logout']) == 'Forgot the Logfire browser sign-in. Keys in /keys are kept.'
    assert forgotten == [TOKEN_ACCOUNT]
    assert 'LOGFIRE_API_KEY' in api_keys.load_keys()
    with pytest.raises(ValueError, match='Usage: /logfire_mcp logout'):
        await command([])


async def test_registers_the_logout_command(tmp_path: Path) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='kept')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    [host] = [entry.host for entry in shell.loader.entries() if entry.host]
    assert [c.name for c in host.commands] == ['logfire_mcp']


async def test_add_replacing_the_builtin_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    added = await shell.loader.command(['add', 'logfire_mcp', 'pydantic_clai2.logfire_mcp', '{"read_only": false}'])
    assert added == 'Replaced built-in logfire_mcp.\nSaved Tools.'
    assert shell.capability() == keyed('LOGFIRE_API_KEY')


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'logfire_mcp'])
    shell.store.save_plugin(BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.repo_context'}))
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')


async def test_plugins_menu_configure_key_opens_the_settings_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])

    def run(menu: PluginMenu[None]) -> MenuResult | None:
        [item] = menu.items()
        menu.toggle(Redraw(), item)
        assert menu.notice == 'Press C to configure logfire_mcp.'
        return menu.configure(Redraw(), item)

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Tools.'
    assert shell.capability().read_only is False


async def test_plugins_menu_reports_a_configure_error(tmp_path: Path) -> None:
    shell = Shell(tmp_path)

    def run(menu: PluginMenu[None]) -> MenuResult | None:
        assert menu.configure(Redraw(), MenuItem('none', value=None)) is None
        return menu.configure(Redraw(), menu.items()[0])

    assert 'enable it before configuring' in await open_plugins_menu(shell.loader, run=run)


class Redraw:
    def replace_items(self, items: object) -> None:
        pass

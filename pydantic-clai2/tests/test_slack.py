"""The built-in `slack` plugin: a settings menu for `Slack`'s options, with the user token kept in `/keys`."""

import io
import threading
from dataclasses import dataclass
from typing import TypeGuard

import anyio
import pytest
from menu_script import Script, pick, typed
from pydantic import JsonValue, SecretStr
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.slack import Slack
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2 import slack as slack_plugin
from pydantic_clai2.api_keys import KeyReference, delete_key, load_keys, rename_key, save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import SessionStart, TurnStart
from pydantic_clai2.promoted_plugins import adopt_promoted
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'slack')
ENABLED = BUILTIN.model_copy(update={'enabled': True})
CONTEXT = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
CLOSE = MenuResult(cancelled=True)
ESC = TextInputResult(cancelled=True)


@pytest.fixture(autouse=True)
def no_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)


def is_slack(capability: object) -> TypeGuard[Slack[None]]:
    return isinstance(capability, Slack)


@dataclass
class Shell:
    plugins: PluginLoader[None]
    store: SettingsStore
    output: io.StringIO

    def slack(self) -> Slack[None]:
        [capability] = self.plugins.capabilities()
        assert is_slack(capability)
        return capability

    async def turn_token(self) -> str | None:
        """The token the Slack connection would use for a turn started now."""
        await self.plugins.fire(TurnStart(text='hi'))
        auth = self.slack().auth
        assert callable(auth)
        return auth(CONTEXT)

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        return declaration.settings

    def source(self) -> 'slack_plugin.SlackSource[None]':
        [entry] = self.plugins.entries()
        assert entry.host is not None
        return slack_plugin.SlackSource(entry.host)


async def shell(declaration: PluginSettings = ENABLED) -> Shell:
    store = SettingsStore()
    output = io.StringIO()
    plugins: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=output, width=200),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration,),
    )
    await plugins.load_all()
    return Shell(plugins, store, output)


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(slack_plugin, 'RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None, *, think: float = 0) -> None:
    """Answer the saved-key list, which needs a real terminal; with `/keys` empty the real prompt runs instead.

    `think` keeps the picker open past the menu thread's polling interval, as a person choosing would.
    """

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        assert label.startswith('Slack user token (xoxp-)')
        await anyio.sleep(think)
        return choice

    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)


def test_declared_as_disabled_builtin() -> None:
    assert BUILTIN == PluginSettings(id='slack', factory='pydantic_clai2.slack', enabled=False)


async def test_enable_opens_the_menu_and_every_option_saves_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-the-environment-is-not-a-source')
    shown = script(
        monkeypatch,
        lists=[pick('token'), pick('read_only'), pick('include_instructions')],
        choices=[pick('false'), pick('false')],
        texts=[typed(' xoxp-new ')],
    )
    app = await shell(BUILTIN)
    assert await app.plugins.command(['enable', 'slack']) == '\n'.join(
        [
            'Enabled slack.',
            'Slack uses the saved key SLACK_USER_TOKEN. Manage it in /keys.',
            'Saved Tools.',
            'Saved Server instructions.',
        ]
    )
    assert shown.opened == ['list', 'text', 'list', 'choice', 'list', 'choice', 'list']
    assert 'CLAI does not read SLACK_USER_TOKEN from the environment.' in app.output.getvalue()
    assert load_keys()['SLACK_USER_TOKEN'].get_secret_value() == 'xoxp-new'
    assert load_codex_credentials(account='slack') == '{"token":{"name":"SLACK_USER_TOKEN"}}'
    assert app.saved() == {'read_only': False, 'include_instructions': False}
    assert not app.slack().read_only and not app.slack().include_instructions
    assert await app.turn_token() == 'xoxp-new'
    await app.plugins.close('exit')


async def test_reopening_repicks_a_shared_key_that_stays_live(monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SLACK_USER_TOKEN', value='xoxp-first')
    save_key(name='ACME_SLACK', value='xoxp-acme')
    slack_plugin.save_connection(KeyReference(name='SLACK_USER_TOKEN'))
    key_choice(monkeypatch, KeyReference(name='ACME_SLACK'), think=0.2)
    script(monkeypatch, lists=[pick('token')])
    app = await shell()
    assert await app.turn_token() == 'xoxp-first'
    assert (
        await app.plugins.command(['configure', 'slack']) == 'Slack uses the saved key ACME_SLACK. Manage it in /keys.'
    )
    assert app.store.plugins() == []  # The token choice never touches plugin settings.
    assert await app.turn_token() == 'xoxp-acme'

    save_key(name='ACME_SLACK', value='xoxp-replaced')
    assert await app.turn_token() == 'xoxp-replaced'
    with pytest.raises(ValueError, match='used by slack'):
        rename_key(name='ACME_SLACK', new_name='OTHER')
    delete_key(name='ACME_SLACK')
    assert await app.turn_token() is None
    assert 'ACME_SLACK is missing. Restore it in /keys or reconfigure through /plugins configure slack.' in (
        app.output.getvalue()
    )
    await app.plugins.close('exit')


@pytest.mark.parametrize('picked', [True, False])
async def test_bot_tokens_are_refused(monkeypatch: pytest.MonkeyPatch, picked: bool) -> None:
    if picked:
        save_key(name='BOT', value='xoxb-bot')
        key_choice(monkeypatch, KeyReference(name='BOT'))
    script(monkeypatch, lists=[pick('token')], texts=[typed('xoxb-bot')])
    app = await shell()
    message = await app.plugins.command(['configure', 'slack'])
    assert message == "That is a bot token (xoxb-). Slack's MCP server accepts only user tokens (xoxp-)."
    assert load_codex_credentials(account='slack') is None
    assert ('SLACK_USER_TOKEN' in load_keys()) is False
    await app.plugins.close('exit')


@pytest.mark.parametrize(('replace', 'expected'), [(False, 'xoxp-shared'), (True, 'xoxp-mine')])
async def test_a_shared_name_is_replaced_only_when_confirmed(
    monkeypatch: pytest.MonkeyPatch, replace: bool, expected: str
) -> None:
    save_key(name='SLACK_USER_TOKEN', value='xoxp-shared')
    key_choice(monkeypatch, 'xoxp-mine')
    script(monkeypatch, lists=[pick('token')], choices=[pick(replace)])
    app = await shell()
    await app.plugins.command(['configure', 'slack'])
    assert load_keys()['SLACK_USER_TOKEN'].get_secret_value() == expected
    assert (load_codex_credentials(account='slack') is not None) is replace
    await app.plugins.close('exit')


@pytest.mark.parametrize(('replace', 'expected'), [(False, 'xoxp-other-session'), (True, 'xoxp-mine')])
async def test_a_key_saved_meanwhile_by_another_session_still_needs_confirmation(
    monkeypatch: pytest.MonkeyPatch, replace: bool, expected: str
) -> None:
    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str:
        save_key(name='SLACK_USER_TOKEN', value='xoxp-other-session')  # After the name was found free.
        return 'xoxp-mine'

    def keys_before_the_other_session() -> dict[str, SecretStr]:
        return {}

    monkeypatch.setattr('pydantic_clai2.plugin_keys.load_keys', keys_before_the_other_session)
    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)
    shown = script(monkeypatch, lists=[pick('token')], choices=[pick(replace)])
    app = await shell()
    await app.plugins.command(['configure', 'slack'])
    assert shown.opened == ['list', 'choice', 'list']
    assert load_keys()['SLACK_USER_TOKEN'].get_secret_value() == expected
    assert (load_codex_credentials(account='slack') is not None) is replace
    await app.plugins.close('exit')


@pytest.mark.parametrize('answer', [ESC, typed('   ')])
async def test_cancelling_or_an_empty_token_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, answer: TextInputResult
) -> None:
    script(monkeypatch, lists=[pick('token')], texts=[answer])
    app = await shell()
    assert await app.plugins.command(['configure', 'slack']) == 'Slack settings unchanged.'
    assert load_keys() == {}
    assert load_codex_credentials(account='slack') is None
    await app.plugins.close('exit')


async def test_a_key_deleted_before_saving_points_back_to_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    key_choice(monkeypatch, KeyReference(name='GONE'))
    script(monkeypatch, lists=[pick('token')])
    app = await shell()
    assert await app.plugins.command(['configure', 'slack']) == (
        'The selected API key no longer exists. Select a saved key again through /plugins configure slack.'
    )
    assert load_codex_credentials(account='slack') is None
    await app.plugins.close('exit')


async def test_reset_forgets_the_key_or_restores_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SLACK_USER_TOKEN', value='xoxp-first')
    slack_plugin.save_connection(KeyReference(name='SLACK_USER_TOKEN'))
    app = await shell(ENABLED.model_copy(update={'settings': {'read_only': False}}))
    source = app.source()
    [token, read_only, _] = source.rows()
    assert source.reset(read_only) == 'Reset Tools.'
    assert app.saved() == {'read_only': True, 'include_instructions': True}
    assert source.reset(token).startswith('Slack no longer uses a key from /keys')
    assert load_codex_credentials(account='slack') is None
    assert load_keys()['SLACK_USER_TOKEN'].get_secret_value() == 'xoxp-first'
    await app.plugins.close('exit')


async def test_rows_show_the_token_state_and_validate_options() -> None:
    app = await shell()
    source = app.source()
    menu = FieldMenu(source)
    assert [row.key for row in source.rows()] == ['token', 'read_only', 'include_instructions']
    assert source.rows()[0].note == 'choose a key'
    assert source.current(source.rows()[0]) == '(not chosen)'
    assert 'read-only' in menu.items()[1].label
    assert source.problem(source.rows()[1], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(source.rows()[1], 'false') is None

    save_key(name='GONE', value='xoxp-gone')
    slack_plugin.save_connection(KeyReference(name='GONE'))
    delete_key(name='GONE')
    assert source.rows()[0].note == 'missing from /keys'
    assert source.current(source.rows()[0]) == 'GONE'

    save_codex_credentials(account='slack', value='{"token": "xoxp-inline"}')
    assert source.rows()[0].note == 'invalid; choose again'
    assert source.current(source.rows()[0]) == '(invalid)'
    assert await app.turn_token() is None
    assert 'The saved Slack connection is invalid.' in app.output.getvalue()
    await app.plugins.close('exit')


async def test_closing_the_menu_mid_pick_cancels_the_key_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    picking = anyio.Event()
    cancelled = anyio.Event()

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> None:
        picking.set()
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.set()

    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)
    script(monkeypatch, lists=[pick('token')])
    app = await shell()
    async with anyio.create_task_group() as group:
        group.start_soon(app.plugins.command, ['configure', 'slack'])
        await picking.wait()
        group.cancel_scope.cancel()
    with anyio.fail_after(5):
        await cancelled.wait()
    assert load_codex_credentials(account='slack') is None
    await app.plugins.close('exit')


class Redraw:
    def replace_items(self, items: object) -> None:
        pass


async def test_add_replacing_the_builtin_opens_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    app = await shell(BUILTIN)
    assert await app.plugins.command(['add', 'slack', 'pydantic_clai2.slack', '{"read_only": false}']) == (
        'Replaced built-in slack.\nSaved Tools.'
    )
    assert app.slack().read_only
    await app.plugins.close('exit')


async def test_configure_needs_a_loaded_plugin_with_a_menu() -> None:
    app = await shell(BUILTIN)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await app.plugins.command(['configure', 'slack'])
    app.store.save_plugin(PluginSettings(id='plain', factory='pydantic_clai2.repo_context'))
    assert await app.plugins.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await app.plugins.configure('plain')
    await app.plugins.close('exit')


async def test_plugins_menu_configure_key_opens_the_settings_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])
    app = await shell(BUILTIN)

    def run(menu: PluginMenu[None]) -> MenuResult | None:
        [item] = menu.items()
        menu.toggle(Redraw(), item)
        assert menu.notice == 'Press C to configure slack.'
        return menu.configure(Redraw(), item)

    assert await open_plugins_menu(app.plugins, run=run) == 'Saved Tools.'
    assert not app.slack().read_only
    await app.plugins.close('exit')


async def test_plugins_menu_stays_open_when_there_is_nothing_to_configure() -> None:
    app = await shell(BUILTIN)
    app.store.save_plugin(PluginSettings(id='plain', factory='pydantic_clai2.repo_context', enabled=True))
    await app.plugins.load_all()

    def run(menu: PluginMenu[None]) -> MenuResult | None:
        plain_row, slack_row = sorted(menu.items(), key=lambda item: str(item.value))
        assert menu.configure(Redraw(), MenuItem('none', value=None)) is None
        assert menu.configure(Redraw(), slack_row) is None
        assert menu.notice == 'Enable slack before configuring it.'
        assert menu.configure(Redraw(), plain_row) is None
        assert menu.notice == 'plain has no settings menu.'
        return None

    assert await open_plugins_menu(app.plugins, run=run) == ''
    await app.plugins.close('exit')


async def test_the_token_is_resolved_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    readers: list[int] = []

    def recording_load(*, account: str) -> str | None:
        readers.append(threading.get_ident())
        return load_codex_credentials(account=account)

    monkeypatch.setattr(slack_plugin, 'load_codex_credentials', recording_load)
    app = await shell()
    await app.turn_token()
    assert len(readers) == 2
    assert loop_thread not in readers
    await app.plugins.close('exit')


async def test_settings_reject_a_token() -> None:
    app = await shell(BUILTIN)
    with pytest.raises(PluginError, match='token'):
        await app.plugins.command(['add', 'slack', 'pydantic_clai2.slack', '{"token": "xoxp-inline"}'])
    assert app.plugins.capabilities() == []
    await app.plugins.close('exit')


RAW = PluginSettings(id='slack', factory='pydantic_ai_harness.slack:Slack')


@pytest.mark.parametrize('enabled', [True, False])
def test_saved_catalog_row_adopts_the_builtin(enabled: bool) -> None:
    store = SettingsStore()
    store.save_plugin(RAW.model_copy(update={'enabled': enabled}))
    store.save_plugin(PluginSettings(id='notion', factory='pydantic_ai_harness.notion:Notion'))
    adopt_promoted(store, DEFAULT_PLUGINS)
    assert store.plugins() == [
        PluginSettings(id='notion', factory='pydantic_ai_harness.notion:Notion'),
        BUILTIN.model_copy(update={'enabled': enabled}),
    ]


def test_own_declarations_are_kept() -> None:
    store = SettingsStore()
    custom = RAW.model_copy(update={'settings': {'read_only': False}})
    store.save_plugin(custom)
    adopt_promoted(store, DEFAULT_PLUGINS)
    assert store.plugins() == [custom]
    store.save_plugin(RAW)
    adopt_promoted(store, [plugin for plugin in DEFAULT_PLUGINS if plugin.id != 'slack'])
    assert store.plugins() == [RAW]

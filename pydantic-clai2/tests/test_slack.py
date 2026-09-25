"""The built-in `slack` plugin: the token is a `/keys` reference chosen by `/slack` and resolved every turn."""

import io
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeGuard

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.slack import Slack
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2 import slack as slack_plugin
from pydantic_clai2.api_keys import KeyReference, SecretPrompt, delete_key, load_keys, rename_key, save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart, TurnStart
from pydantic_clai2.promoted_plugins import adopt_promoted
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'slack')
ENABLED = BUILTIN.model_copy(update={'enabled': True})
CONTEXT = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


def is_slack(capability: object) -> TypeGuard[Slack[None]]:
    return isinstance(capability, Slack)


@dataclass
class Shell:
    plugins: PluginLoader[None]
    commands: Commands
    output: io.StringIO

    async def turn_token(self) -> str | None:
        """The token the Slack connection would use for a turn started now."""
        await self.plugins.fire(TurnStart(text='hi'))
        [slack] = self.plugins.capabilities()
        assert is_slack(slack)
        auth = slack.auth
        assert callable(auth)
        return auth(CONTEXT)


async def shell(declaration: PluginSettings = ENABLED) -> Shell:
    store = SettingsStore()
    commands = Commands()
    output = io.StringIO()
    plugins: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=output, width=200),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration,),
    )
    await plugins.load_all()
    return Shell(plugins, commands, output)


def answer(value: str | KeyReference | None) -> Callable[..., object]:
    async def prompt_api_key(*, prompt: SecretPrompt, label: str, optional: bool = False) -> str | KeyReference | None:
        assert label.startswith('Slack user token')
        return value

    return prompt_api_key


def test_declared_as_disabled_builtin() -> None:
    assert BUILTIN == PluginSettings(id='slack', factory='pydantic_clai2.slack', enabled=False)


async def test_enabling_without_a_token_warns_and_gives_no_tools() -> None:
    app = await shell(BUILTIN)
    assert app.plugins.capabilities() == []
    await app.plugins.enable('slack')
    assert 'Slack tools are off. Run /slack to choose its user token from /keys' in app.output.getvalue()
    assert 'slack' in {command.name for command in app.commands}
    assert await app.turn_token() is None
    await app.plugins.close('exit')


async def test_a_chosen_key_is_saved_by_name_and_resolved_every_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SHARED_SLACK', value='xoxp-first')
    monkeypatch.setattr(slack_plugin, 'prompt_api_key', answer(KeyReference(name='SHARED_SLACK')))
    app = await shell()
    assert await app.commands.execute_async('/slack') == 'Slack now uses the SHARED_SLACK key from /keys.'
    assert load_codex_credentials(account='slack') == '{"token":{"name":"SHARED_SLACK"}}'
    assert await app.turn_token() == 'xoxp-first'

    save_key(name='SHARED_SLACK', value='xoxp-replaced')
    assert await app.turn_token() == 'xoxp-replaced'
    with pytest.raises(ValueError, match='used by slack'):
        rename_key(name='SHARED_SLACK', new_name='OTHER')

    delete_key(name='SHARED_SLACK')
    assert await app.turn_token() is None
    assert 'Saved API key SHARED_SLACK is missing. Restore it in /keys or reconfigure through /slack.' in (
        app.output.getvalue()
    )
    await app.plugins.close('exit')


async def test_a_new_token_goes_to_keys_not_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slack_plugin, 'prompt_api_key', answer(' xoxp-new '))
    app = await shell()
    assert await app.commands.execute_async('/slack') == 'Slack now uses the SLACK_USER_TOKEN key from /keys.'
    assert load_keys()['SLACK_USER_TOKEN'].get_secret_value() == 'xoxp-new'
    assert load_codex_credentials(account='slack') == '{"token":{"name":"SLACK_USER_TOKEN"}}'
    assert await app.turn_token() == 'xoxp-new'
    await app.plugins.close('exit')


async def test_a_new_token_never_replaces_a_shared_key(monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SLACK_USER_TOKEN', value='xoxp-shared')
    monkeypatch.setattr(slack_plugin, 'prompt_api_key', answer('xoxp-other'))
    app = await shell()
    with pytest.raises(ValueError, match='SLACK_USER_TOKEN already exists in /keys'):
        await app.commands.execute_async('/slack')
    assert load_keys()['SLACK_USER_TOKEN'].get_secret_value() == 'xoxp-shared'
    assert load_codex_credentials(account='slack') is None
    await app.plugins.close('exit')


@pytest.mark.parametrize(
    ('command', 'value', 'expected'),
    [
        ('/slack', None, 'Slack connection cancelled.'),
        ('/slack', '  ', ValueError('An API key is required')),
        ('/slack token', None, ValueError('Usage: /slack')),
    ],
)
async def test_cancel_empty_and_usage(
    monkeypatch: pytest.MonkeyPatch, command: str, value: str | None, expected: str | ValueError
) -> None:
    monkeypatch.setattr(slack_plugin, 'prompt_api_key', answer(value))
    app = await shell()
    if isinstance(expected, str):
        assert await app.commands.execute_async(command) == expected
    else:
        with pytest.raises(ValueError, match=str(expected)):
            await app.commands.execute_async(command)
    assert load_codex_credentials(account='slack') is None
    await app.plugins.close('exit')


async def test_an_invalid_saved_connection_fails_closed() -> None:
    save_codex_credentials(account='slack', value='{"token": "xoxp-inline"}')
    app = await shell()
    assert 'The saved Slack connection is invalid. Run /slack' in app.output.getvalue()
    assert await app.turn_token() is None
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


async def test_read_only_setting() -> None:
    app = await shell()
    [capability] = app.plugins.capabilities()
    assert is_slack(capability) and capability.read_only
    await app.plugins.close('exit')
    app = await shell(ENABLED.model_copy(update={'settings': {'read_only': False}}))
    [capability] = app.plugins.capabilities()
    assert is_slack(capability) and not capability.read_only
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

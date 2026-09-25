"""The built-in `slack` plugin: declared off, token from the environment or `/keys`, and a clear failure without one."""

import io
import threading

import pytest
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.slack import Slack
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2 import slack as slack_plugin
from pydantic_clai2.api_keys import load_keys, save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.promoted_plugins import adopt_promoted
from pydantic_clai2.settings_store import SettingsStore

pytestmark = pytest.mark.anyio

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'slack')


@pytest.fixture(autouse=True)
def no_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)


def loader(store: SettingsStore, builtin: PluginSettings = BUILTIN) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(builtin,),
    )


def connection(token: str, *, read_only: bool = True) -> list[Slack[None]]:
    """What `capabilities()` holds once the plugin has loaded with this token."""
    return [Slack[None](auth=token, read_only=read_only)]


def test_declared_as_disabled_builtin() -> None:
    assert BUILTIN == PluginSettings(id='slack', factory='pydantic_clai2.slack', enabled=False)


async def test_disabled_until_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-env')
    plugins = loader(SettingsStore())
    await plugins.load_all()
    assert plugins.capabilities() == []
    await plugins.enable('slack')
    assert plugins.capabilities() == connection('xoxp-env')
    await plugins.close('exit')


async def test_missing_token_fails_enable_and_registers_nothing() -> None:
    plugins = loader(SettingsStore())
    with pytest.raises(PluginError, match='Set SLACK_USER_TOKEN, or save it under that name with /keys'):
        await plugins.enable('slack')
    assert plugins.capabilities() == []
    await plugins.close('exit')


async def test_saved_key_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SLACK_USER_TOKEN', value='xoxp-saved')
    plugins = loader(SettingsStore(), BUILTIN.model_copy(update={'enabled': True}))
    await plugins.load_all()
    assert plugins.capabilities() == connection('xoxp-saved')
    await plugins.close('exit')

    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-env')
    plugins = loader(SettingsStore(), BUILTIN.model_copy(update={'enabled': True}))
    await plugins.load_all()
    assert plugins.capabilities() == connection('xoxp-env')
    await plugins.close('exit')


async def test_read_only_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-env')
    declaration = BUILTIN.model_copy(update={'enabled': True, 'settings': {'read_only': False}})
    plugins = loader(SettingsStore(), declaration)
    await plugins.load_all()
    assert plugins.capabilities() == connection('xoxp-env', read_only=False)
    await plugins.close('exit')


async def test_rejects_unknown_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-env')
    plugins = loader(SettingsStore())
    await plugins.load_all()
    with pytest.raises(PluginError, match='token'):
        await plugins.command(['add', 'slack', 'pydantic_clai2.slack', '{"token": "xoxp-inline"}'])
    assert plugins.capabilities() == []
    await plugins.close('exit')


async def test_saved_key_is_read_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name='SLACK_USER_TOKEN', value='xoxp-saved')
    loop_thread = threading.get_ident()
    readers: list[int] = []

    def recording_load_keys() -> dict[str, SecretStr]:
        readers.append(threading.get_ident())
        return load_keys()

    monkeypatch.setattr(slack_plugin, 'load_keys', recording_load_keys)
    plugins = loader(SettingsStore(), BUILTIN.model_copy(update={'enabled': True}))
    await plugins.load_all()
    assert plugins.capabilities() == connection('xoxp-saved')
    assert readers and loop_thread not in readers
    await plugins.close('exit')


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

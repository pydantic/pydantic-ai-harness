"""The built-in `slack` plugin: declared off, token from the environment or `/keys`, and a clear failure without one."""

import io

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.slack import Slack
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.api_keys import save_key
from pydantic_clai2.capability_catalog import HARNESS_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
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
    assert all(plugin.id != 'slack' for plugin in HARNESS_PLUGINS)


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

"""The `google_workspace` built-in: token lookup, settings, and its declaration."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.google_workspace import GoogleWorkspace
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import TypeIs

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.api_keys import delete_key, save_key
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.google_workspace import MISSING_TOKEN, activate
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture(autouse=True)
def no_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('GOOGLE_ACCESS_TOKEN', raising=False)


class Redraw:
    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.items = items


def host(settings: dict[str, JsonValue] | None = None) -> PluginHost[None]:
    return PluginHost(name='google_workspace', console=Console(file=io.StringIO()), settings=settings or {})


def loaded(plugin: PluginHost[None]) -> GoogleWorkspace[None]:
    [capability] = plugin.capabilities
    assert is_workspace(capability)
    return capability


def is_workspace(capability: AgentCapability[None]) -> TypeIs[GoogleWorkspace[None]]:
    return isinstance(capability, GoogleWorkspace)


def run_token(capability: GoogleWorkspace[None]) -> str | None:
    assert callable(capability.auth)
    return capability.auth(RunContext(deps=None, model=TestModel(), usage=RunUsage()))


def test_activate_uses_the_environment_token_with_read_only_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'env-token')
    plugin = host()
    activate(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail', 'calendar', 'drive')
    assert capability.read_only is True
    assert capability.id == 'google-workspace-calendar-drive-gmail'
    assert run_token(capability) == 'env-token'


def test_saved_key_wins_and_is_read_on_every_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'env-token')
    save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    plugin = host()
    activate(plugin)
    capability = loaded(plugin)
    assert run_token(capability) == 'saved-token'
    save_key(name='GOOGLE_ACCESS_TOKEN', value='refreshed-token')
    assert run_token(capability) == 'refreshed-token'
    delete_key(name='GOOGLE_ACCESS_TOKEN')
    assert run_token(capability) == 'env-token'
    monkeypatch.delenv('GOOGLE_ACCESS_TOKEN')
    with pytest.raises(UserError, match='GOOGLE_ACCESS_TOKEN'):
        run_token(capability)


def test_activate_without_a_token_fails_and_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', '')
    plugin = host()
    with pytest.raises(UserError) as error:
        activate(plugin)
    assert str(error.value) == MISSING_TOKEN
    assert plugin.capabilities == []


def test_settings_choose_services_and_writable_tools() -> None:
    save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    plugin = host({'services': ['gmail'], 'read_only': False})
    activate(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail',)
    assert capability.read_only is False


@pytest.mark.parametrize(
    'settings',
    [{'services': []}, {'services': ['maps']}, {'auth': 'token'}, {'read_only': 'yes'}],
)
def test_settings_reject_bad_values_and_credentials(settings: dict[str, JsonValue]) -> None:
    save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    with pytest.raises(ValidationError):
        activate(host(settings))


def loader(store: SettingsStore, builtin: Sequence[PluginSettings] = DEFAULT_PLUGINS) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=builtin,
    )


def test_declared_as_a_disabled_builtin_that_enables_from_the_menu(tmp_path: Path) -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'google_workspace']
    assert declaration.factory == 'pydantic_clai2.google_workspace'
    assert not declaration.enabled
    assert not any(plugin.factory.startswith('pydantic_ai_harness.google_workspace') for plugin in DEFAULT_PLUGINS)

    plugins = loader(SettingsStore(tmp_path / 'settings.db'), builtin=(declaration,))

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run(action)

    asyncio.run(plugins.load_all())
    assert plugins.capabilities() == []
    menu = PluginMenu(plugins, apply=apply)
    [item] = menu.items()
    menu.toggle(Redraw(), item)
    assert menu.notice is not None
    assert 'GOOGLE_ACCESS_TOKEN' in menu.details(item)
    assert plugins.capabilities() == []

    save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    asyncio.run(plugins.load_all())
    [capability] = plugins.capabilities()
    assert isinstance(capability, GoogleWorkspace)
    asyncio.run(plugins.close('exit'))


def test_the_former_catalog_entry_saved_by_the_menu_loads_the_builtin(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    former = PluginSettings(id='google_workspace', factory='pydantic_ai_harness.google_workspace:GoogleWorkspace')
    store.save_plugin(former)
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration.factory == 'pydantic_clai2.google_workspace'
    assert entry.declaration.enabled
    assert entry.builtin

    store.save_plugin(former.model_copy(update={'enabled': False}))
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration.factory == 'pydantic_clai2.google_workspace'
    assert not entry.declaration.enabled


def test_a_former_declaration_with_its_own_settings_is_kept(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    custom = PluginSettings(
        id='google_workspace',
        factory='pydantic_ai_harness.google_workspace:GoogleWorkspace',
        settings={'services': ['gmail']},
    )
    store.save_plugin(custom)
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration == custom
    assert not entry.builtin

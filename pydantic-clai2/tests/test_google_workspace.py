"""The `google_workspace` built-in: named-key auth, `/google_workspace`, settings, and its declaration."""

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

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, google_workspace
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore


class Prompt:
    def __init__(self, *, values: list[str | BaseException]) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


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


def test_the_conventional_key_is_resolved_on_every_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'env-token')
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    plugin = host()
    activate_quietly(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail', 'calendar', 'drive')
    assert capability.read_only is True
    assert capability.id == 'google-workspace-calendar-drive-gmail'
    assert run_token(capability) == 'saved-token'
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='refreshed-token')
    assert run_token(capability) == 'refreshed-token'
    api_keys.delete_key(name='GOOGLE_ACCESS_TOKEN')
    with pytest.raises(UserError, match='GOOGLE_ACCESS_TOKEN') as error:
        run_token(capability)
    assert 'env-token' not in str(error.value)


def activate_quietly(plugin: PluginHost[None]) -> str:
    output = io.StringIO()
    plugin.console = Console(file=output)
    google_workspace.activate(plugin)
    return output.getvalue()


def test_activate_without_a_key_warns_and_fails_closed_on_use() -> None:
    plugin = host()
    warning = activate_quietly(plugin)
    assert 'Run /google_workspace' in warning
    assert [command.name for command in plugin.commands] == ['google_workspace']
    with pytest.raises(UserError) as error:
        run_token(loaded(plugin))
    assert str(error.value) == google_workspace.missing('GOOGLE_ACCESS_TOKEN')


def test_settings_choose_services_and_writable_tools() -> None:
    plugin = host({'services': ['gmail'], 'read_only': False})
    activate_quietly(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail',)
    assert capability.read_only is False


@pytest.mark.parametrize(
    'settings',
    [
        {'services': []},
        {'services': ['maps']},
        {'auth': 'token'},
        {'token': 'secret'},
        {'token': {'name': 'GOOGLE_ACCESS_TOKEN'}},
        {'read_only': 'yes'},
    ],
)
def test_settings_reject_bad_values_and_credentials(settings: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        google_workspace.activate(host(settings))


async def test_command_saves_an_entered_token_under_the_conventional_label(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = host()
    activate_quietly(plugin)
    prompt = Prompt(values=['entered-token'])
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: prompt)
    result = await plugin.commands.execute_async('/google_workspace')
    assert result == 'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN from the next turn.'
    assert prompt.labels == [('Google OAuth access token (saved in /keys as GOOGLE_ACCESS_TOKEN): ', True)]
    assert api_keys.load_keys()['GOOGLE_ACCESS_TOKEN'].get_secret_value() == 'entered-token'
    stored = load_codex_credentials(account='google-workspace')
    assert stored is not None and 'entered-token' not in stored
    assert run_token(loaded(plugin)) == 'entered-token'


async def test_command_references_a_shared_saved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='WORK_GOOGLE', value='shared-token')
    plugin = host()
    activate_quietly(plugin)
    pressed = iter(['enter'])
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: Prompt(values=[]))
    assert await plugin.commands.execute_async('/google_workspace') == (
        'Google Workspace uses the saved key WORK_GOOGLE from the next turn.'
    )
    assert run_token(loaded(plugin)) == 'shared-token'
    assert 'GOOGLE_ACCESS_TOKEN' not in api_keys.load_keys()
    with pytest.raises(ValueError, match='used by google-workspace'):
        api_keys.rename_key(name='WORK_GOOGLE', new_name='OTHER')


@pytest.mark.parametrize('value', [EOFError(), ' '])
async def test_command_cancel_and_empty_entry_leave_no_key(
    monkeypatch: pytest.MonkeyPatch, value: str | BaseException
) -> None:
    plugin = host()
    activate_quietly(plugin)
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: Prompt(values=[value]))
    if isinstance(value, str):
        with pytest.raises(ValueError, match='required'):
            await plugin.commands.execute_async('/google_workspace')
    else:
        assert await plugin.commands.execute_async('/google_workspace') == 'Google Workspace key unchanged.'
    assert api_keys.load_keys() == {}
    assert load_codex_credentials(account='google-workspace') is None
    with pytest.raises(ValueError, match='Usage'):
        await plugin.commands.execute_async('/google_workspace secret')


def test_an_invalid_saved_choice_fails_closed() -> None:
    save_codex_credentials(account='google-workspace', value='{"token": ["inline-secret"]}')
    plugin = host()
    assert 'Run /google_workspace again' in activate_quietly(plugin)
    with pytest.raises(UserError) as error:
        run_token(loaded(plugin))
    assert 'inline-secret' not in str(error.value)
    api_keys.save_key(name='ANY', value='unrelated')
    with pytest.raises(UserError, match='through /google_workspace'):
        api_keys.rename_key(name='ANY', new_name='OTHER')


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
    assert declaration.settings == {}
    plugins = loader(SettingsStore(tmp_path / 'settings.db'), builtin=(declaration,))

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run(action)

    asyncio.run(plugins.load_all())
    assert plugins.capabilities() == []
    menu = PluginMenu(plugins, apply=apply)
    [item] = menu.items()
    menu.toggle(Redraw(), item)
    assert menu.notice is None
    assert 'enabled, loaded' in menu.details(item)
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

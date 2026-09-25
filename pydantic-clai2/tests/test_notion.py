"""The built-in `notion` plugin: keys come from `/keys` by name, never from plugin settings."""

import inspect
import io
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.notion import Notion
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, notion
from pydantic_clai2.commands import Commands
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

Vault = dict[tuple[str, str], str]


class Prompt:
    def __init__(self, *values: str | BaseException) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


class Shell:
    def __init__(self, tmp_path: Path) -> None:
        self.store = SettingsStore(tmp_path / 'settings.db')
        self.output = io.StringIO()
        self.commands = Commands()
        self.plugins = PluginLoader[None](
            store=self.store,
            console=Console(file=self.output),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=DEFAULT_PLUGINS,
        )

    async def run(self, text: str) -> str:
        result = self.commands.execute(text)
        assert inspect.isawaitable(result), '`/notion` reaches the keyring off the event loop'
        return await result

    async def built(self) -> Notion[None]:
        """The run's `Notion`, from the per-run factory the plugin registers."""
        [capability] = self.plugins.capabilities()
        assert callable(capability)
        result = capability(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))
        assert inspect.isawaitable(result)
        connected = await result
        assert isinstance(connected, Notion)
        return connected


def answer(monkeypatch: pytest.MonkeyPatch, *values: str | BaseException, keys: tuple[str, ...] = ()) -> Prompt:
    """Type `values` into the masked prompt and press `keys` in the saved-key picker."""
    prompt = Prompt(*values)
    pressed = iter(keys)
    monkeypatch.setattr(notion, 'PromptSession', lambda: prompt)
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    return prompt


def test_declared_as_a_disabled_builtin_not_a_raw_catalog_entry() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'notion']
    assert declaration.factory == 'pydantic_clai2.notion'
    assert not declaration.enabled
    assert all(plugin.factory != 'pydantic_ai_harness.notion:Notion' for plugin in DEFAULT_PLUGINS)


async def test_browser_sign_in_without_a_selected_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NOTION_ACCESS_TOKEN', 'ignored')
    shell = Shell(tmp_path)
    await shell.plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"read_only": true}'])
    capability = await shell.built()
    assert capability.auth is None and capability.read_only
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == notion.NOTION_MCP_URL and isinstance(transport.auth, OAuth)
    assert (await shell.built()).client is not client, 'each run reloads the sign-in, so logout applies to the next'
    await shell.plugins.close('exit')


async def test_a_saved_key_is_chosen_by_name_and_resolved_each_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-first')
    shell = Shell(tmp_path)
    await shell.plugins.enable('notion')
    prompt = answer(monkeypatch, keys=('enter',))
    assert await shell.run('/notion key') == 'Notion uses NOTION_API_KEY; manage it in /keys.'
    assert prompt.labels == []
    assert load_codex_credentials(account='notion') == '{"token":{"name":"NOTION_API_KEY"}}'
    assert 'ntn-first' not in repr(shell.store.plugins())
    assert (await shell.built()).auth == 'ntn-first'
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-second')
    assert (await shell.built()).auth == 'ntn-second', 'replacing the key in /keys reaches every consumer'
    assert api_keys.key_users(name='NOTION_API_KEY') == ['notion']
    with pytest.raises(ValueError, match='used by notion'):
        api_keys.rename_key(name='NOTION_API_KEY', new_name='OTHER')
    api_keys.delete_key(name='NOTION_API_KEY')
    with pytest.raises(UserError, match='NOTION_API_KEY is missing'):
        await shell.built()
    await shell.plugins.close('exit')


async def test_a_new_token_is_saved_to_keys_not_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    await shell.plugins.enable('notion')
    prompt = answer(monkeypatch, ' ntn-typed ')
    assert await shell.run('/notion key') == (
        'Saved NOTION_API_KEY in the OS keyring. Notion uses NOTION_API_KEY; manage it in /keys.'
    )
    assert prompt.labels == [('Notion access token (saved in /keys as NOTION_API_KEY): ', True)]
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == 'ntn-typed'
    assert (await shell.built()).auth == 'ntn-typed'
    await shell.plugins.close('exit')


@pytest.mark.parametrize(('confirm', 'value'), [('y', 'ntn-new'), ('n', 'ntn-old'), (EOFError(), 'ntn-old')])
async def test_entering_a_token_asks_before_replacing_a_shared_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirm: str | BaseException, value: str
) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-old')
    shell = Shell(tmp_path)
    await shell.plugins.enable('notion')
    answer(monkeypatch, 'ntn-new', confirm, keys=('down', 'enter'))
    message = await shell.run('/notion key')
    assert message.endswith('Notion uses NOTION_API_KEY; manage it in /keys.') == (value == 'ntn-new')
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == value
    await shell.plugins.close('exit')


async def test_cancelled_or_empty_entry_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    await shell.plugins.enable('notion')
    answer(monkeypatch, EOFError(), '  ')
    assert await shell.run('/notion key') == 'Notion key unchanged.'
    with pytest.raises(ValueError, match='A Notion access token is required'):
        await shell.run('/notion key')
    assert load_codex_credentials(account='notion') is None and api_keys.load_keys() == {}
    await shell.plugins.close('exit')


async def test_key_mode_never_opens_a_browser(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    await shell.plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"auth": "key"}'])
    assert 'no key selected' in shell.output.getvalue()
    with pytest.raises(UserError, match='No Notion key is selected'):
        await shell.built()
    await shell.plugins.close('exit')


async def test_oauth_mode_ignores_a_selected_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    shell = Shell(tmp_path)
    await shell.plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"auth": "oauth"}'])
    answer(monkeypatch, keys=('enter',))
    await shell.run('/notion key')
    capability = await shell.built()
    assert capability.auth is None and isinstance(capability.client, Client)
    await shell.plugins.close('exit')


async def test_invalid_selection_fails_closed(tmp_path: Path) -> None:
    save_codex_credentials(account='notion', value='{"token": "ntn-inline"}')
    shell = Shell(tmp_path)
    await shell.plugins.enable('notion')
    with pytest.raises(UserError, match='selection is invalid'):
        await shell.built()
    await shell.plugins.close('exit')


async def test_secrets_are_not_accepted_in_settings(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(PluginError, match='extra_forbidden|Extra inputs'):
        await shell.plugins.command(['add', 'notion', 'pydantic_clai2.notion', '{"token": "ntn-token"}'])
    assert shell.plugins.capabilities() == []


async def test_logout_forgets_the_sign_in_and_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vault: Vault
) -> None:
    shell = Shell(tmp_path)
    await shell.plugins.enable('notion')
    answer(monkeypatch, 'ntn-token')
    await shell.run('/notion key')
    await notion.TOKENS.put('token', {'access_token': 'a'}, collection='mcp-oauth-token')
    [command] = [command for command in shell.commands if command.name == 'notion']
    assert list(command.complete([''])) == ['key', 'logout'] and list(command.complete(['logout', ''])) == []
    with pytest.raises(ValueError, match='Usage: /notion key'):
        await shell.run('/notion')
    assert await shell.run('/notion logout') == (
        'Signed out of Notion and cleared the selected key. The key itself stays in /keys.'
    )
    assert list(vault) == [('pydantic-clai2', 'api-keys')], 'only the named key remains'
    assert (await shell.built()).client is not None
    await shell.plugins.close('exit')

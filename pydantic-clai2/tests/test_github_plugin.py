"""The built-in `github` plugin: its token lives in `/keys`, and its settings only ever name it."""

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.github import GitHub
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.api_keys import KeyReference, SavedKey
from pydantic_clai2.commands import Commands
from pydantic_clai2.github import SETUP
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'github')


class Shell:
    """A loader plus the pieces a test inspects: typed commands, printed output, the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        self.commands = Commands()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    async def run(self, text: str) -> str:
        result = self.commands.execute(text)
        return result if isinstance(result, str) else await result


class Prompts:
    """Stands in for `prompt_api_key` and the confirmation prompt, answering from a script."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None, confirm: str | None = ''
    ) -> None:
        self.labels: list[str] = []
        self.confirm = confirm
        choose = self

        async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
            choose.labels.append(label)
            return choice

        class Session:
            async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
                choose.labels.append(label)
                if choose.confirm is None:
                    raise KeyboardInterrupt
                return choose.confirm

        monkeypatch.setattr('pydantic_clai2.github.prompt_api_key', prompt_api_key)
        monkeypatch.setattr('pydantic_clai2.github.PromptSession', Session)


def github(name: str = 'GITHUB_TOKEN', *, read_only: bool = True) -> Sequence[GitHub[None]]:
    return [GitHub[None](auth=SavedKey(name=name, setup=SETUP), read_only=read_only)]


def test_declared_disabled_with_no_settings() -> None:
    assert BUILTIN.factory == 'pydantic_clai2.github'
    assert not BUILTIN.enabled
    assert BUILTIN.settings == {}


async def test_saved_key_is_resolved_on_every_run_not_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_TOKEN', 'env-token-is-a-label-not-a-source')
    api_keys.save_key(name='GITHUB_TOKEN', value='first')
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    assert shell.loader.capabilities() == github()
    token = SavedKey(name='GITHUB_TOKEN', setup=SETUP)
    assert token(None) == 'first'
    api_keys.save_key(name='GITHUB_TOKEN', value='replaced')
    assert token(None) == 'replaced'
    api_keys.delete_key(name='GITHUB_TOKEN')
    with pytest.raises(UserError, match='GITHUB_TOKEN is missing. Run /github connect'):
        token(None)


async def test_missing_key_still_loads_with_a_warning_and_the_connect_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    assert 'GitHub has no token: GITHUB_TOKEN is not in /keys.' in shell.output.getvalue()
    assert await shell.run('/github') == 'GitHub token: GITHUB_TOKEN in /keys (missing); read-only tools.'
    prompts = Prompts(monkeypatch, ' ghp_new ')
    assert await shell.run('/github connect') == (
        'GitHub uses the saved key GITHUB_TOKEN from the next turn. Manage it in /keys.'
    )
    assert prompts.labels == ['GitHub token (saved in /keys as GITHUB_TOKEN): ']
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'ghp_new'
    assert shell.store.plugins()[0].settings == {'token': {'name': 'GITHUB_TOKEN'}, 'read_only': True}
    assert b'ghp_new' not in shell.path.read_bytes()


async def test_connect_to_an_existing_key_saves_only_its_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='WORK_GITHUB', value='work-secret')
    shell = Shell(tmp_path, {'read_only': False})
    await shell.loader.enable('github')
    Prompts(monkeypatch, KeyReference(name='WORK_GITHUB'))
    await shell.run('/github connect')
    assert await shell.run('/github') == 'GitHub token: WORK_GITHUB in /keys (saved); read and write tools.'
    [saved] = shell.store.plugins()
    assert saved.enabled
    assert saved.settings == {'token': {'name': 'WORK_GITHUB'}, 'read_only': False}
    assert b'work-secret' not in shell.path.read_bytes()
    await shell.loader.reload('github')
    assert shell.loader.capabilities() == github('WORK_GITHUB', read_only=False)


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='GITHUB_TOKEN', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    prompts = Prompts(monkeypatch, 'other', confirm='n')
    assert await shell.run('/github connect') == 'GitHub token unchanged.'
    assert prompts.labels[-1] == 'Replace GITHUB_TOKEN for every plugin and connection that uses it? [y/N]: '
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'shared'
    prompts.confirm = 'y'
    await shell.run('/github connect')
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'other'


@pytest.mark.parametrize('choice', [None, '  '])
async def test_cancelled_or_blank_connect_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str | None
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    Prompts(monkeypatch, choice)
    if choice is None:
        assert await shell.run('/github connect') == 'GitHub token unchanged.'
    else:
        with pytest.raises(ValueError, match='A GitHub token is required.'):
            await shell.run('/github connect')
    assert api_keys.load_keys() == {}
    assert shell.store.plugins()[0].settings == {}


async def test_interrupted_confirmation_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='GITHUB_TOKEN', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    Prompts(monkeypatch, 'other', confirm=None)
    assert await shell.run('/github connect') == 'GitHub token unchanged.'
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'shared'


async def test_unknown_subcommand_shows_usage(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    with pytest.raises(ValueError, match=r'Usage: /github \[connect\]'):
        await shell.run('/github login')


@pytest.mark.parametrize(
    'settings', [{'token': 'ghp_inline_secret'}, {'token': {'name': 'X', 'value': 'secret'}}, {'toolsets': ['repos']}]
)
async def test_settings_cannot_hold_a_secret_or_unknown_keys(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('github')
    assert shell.loader.capabilities() == []

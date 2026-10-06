"""Exercise onboarding at terminal, process, and credential-store boundaries."""

import io
import json
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import keyring
import pytest
from keyring.errors import NoKeyringError
from pydantic import ValidationError
from rich.text import Text

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.logfire_credentials import load_logfire_credentials, logfire_directory
from pydantic_clai2.logfire_onboarding import logfire_command, onboard_logfire
from pydantic_clai2.settings_store import SettingsStore

CREDENTIALS = json.dumps(
    {
        'token': 'test-write-token',
        'logfire_api_url': 'https://logfire-eu.pydantic.dev',
        'project_name': 'clai',
        'project_url': 'https://logfire.pydantic.dev/example/clai',
    }
)


class TerminalInput(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def connect(answers: str) -> None:
        monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
        monkeypatch.setattr(sys, 'stdin', TerminalInput(answers))
        moves = {
            '': ('enter',),
            'login': ('enter',),
            'decline': ('down', 'enter'),
            'use': ('enter',),
            'new': ('down', 'enter'),
            'escape': ('escape',),
            'cancel': ('ctrl-c',),
        }
        keys = iter(key for answer in answers.splitlines() for key in moves[answer])
        monkeypatch.setattr('pydantic_clai2.logfire_onboarding.menu_key', lambda: next(keys))

    return connect


class LogfireCLI:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.directories: list[Path] = []
        self.failure: BaseException | None = None
        self.value = CREDENTIALS

    def __call__(
        self, command: list[str], *, cwd: Path, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[:4] == [sys.executable, '-I', '-m', 'logfire']
        assert cwd.is_dir()
        if sys.platform != 'win32':
            assert cwd.stat().st_mode & 0o077 == 0
        assert check and timeout == 300
        self.commands.append(command[4:])
        self.directories.append(cwd)
        sys.stdout.write(f'Logfire {command[4]} output\n')
        if self.failure is not None:
            raise self.failure
        if command[4] == 'projects':
            assert command[6:] == ['--data-dir', str(cwd)]
            (cwd / 'logfire_credentials.json').write_text(self.value)
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def logfire_cli(monkeypatch: pytest.MonkeyPatch) -> LogfireCLI:
    cli = LogfireCLI()
    monkeypatch.setattr('pydantic_clai2.logfire_onboarding.subprocess.run', cli)
    return cli


@pytest.mark.parametrize('action', ['decline', 'escape'])
def test_decline_is_remembered_on_upgrade(
    tmp_path: Path,
    terminal: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    terminal(f'{action}\n')
    path = tmp_path / 'config.db'
    store = SettingsStore(path)
    store.set('model', 'test')
    store.save_plugin(PluginSettings(id='custom', factory='example'))
    onboard_logfire(store=store)
    output = capsys.readouterr().out
    assert 'What gets sent' in output
    assert 'Prompts, responses, tool arguments/results, and images' in output
    assert 'Log in to Logfire' in output and 'Continue without Logfire' in output
    assert output.count('\x1b[?1049h') == output.count('\x1b[?1049l') == 1
    assert '\x1b[3J' not in output
    assert 'Logfire disabled.' in output.split('\x1b[?1049l')[1]
    assert store.plugins()[1] == PluginSettings(id='logfire', factory='pydantic_clai2.logfire', enabled=False)
    monkeypatch.setattr(sys, 'stdin', TerminalInput('login\n'))
    reopened = SettingsStore(path)
    onboard_logfire(store=reopened)
    assert not capsys.readouterr().out
    assert reopened.load().model == 'test'
    assert reopened.plugins()[0].factory == 'example'


@pytest.mark.parametrize('action', ['use', 'new'])
def test_login_saves_only_project_credentials_and_cleans_temporary_files(
    tmp_path: Path,
    terminal: Callable[[str], None],
    logfire_cli: LogfireCLI,
    action: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='logfire', factory='pydantic_clai2.logfire', enabled=False))
    terminal(f'\n{action}\n')
    assert logfire_command(store, []) == ''
    assert logfire_cli.commands[0] == ['auth']
    assert logfire_cli.commands[1][:2] == ['projects', action]
    assert all(not directory.exists() for directory in logfire_cli.directories)
    credentials = load_logfire_credentials()
    assert credentials is not None and credentials.token == 'test-write-token'
    assert str(credentials.logfire_api_url) == 'https://logfire-eu.pydantic.dev/'
    assert store.plugins()[0].enabled
    assert not logfire_directory().exists()
    output = capsys.readouterr().out
    assert 'OS keyring' in output and 'test-write-token' not in output
    assert output.count('\x1b[?1049h') == output.count('\x1b[?1049l') == 1
    assert '\x1b[3J' not in output
    introduction, auth, project, selection = output.split('\x1b[2J')
    assert 'What gets sent' in introduction and 'Logfire auth output' not in introduction
    assert 'Logfire auth output' in auth and 'What gets sent' not in auth
    assert 'Use an existing project' in project and 'Logfire auth output' not in project
    assert 'Logfire projects output' in selection and 'Use an existing project' not in selection
    assert 'Logfire connected.' in output.split('\x1b[?1049l')[1]


@pytest.mark.parametrize('kind', ['token', 'saved', 'legacy', 'override', 'project', 'file', 'package'])
def test_existing_setup_is_not_prompted(
    tmp_path: Path,
    terminal: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    terminal('decline\n')
    project: tuple[PluginSettings, ...] = ()
    if kind == 'token':
        monkeypatch.setenv('LOGFIRE_TOKEN', 'configured-token')
    elif kind == 'saved':
        save_codex_credentials(account='logfire', value=CREDENTIALS)
    elif kind == 'legacy':
        directory = logfire_directory()
        directory.mkdir(parents=True)
        (directory / 'logfire_credentials.json').write_text(CREDENTIALS)
    elif kind == 'override':
        store.save_plugin(PluginSettings(id='logfire', factory='custom.tracing'))
    elif kind == 'project':
        project = (PluginSettings(id='logfire', factory='custom.tracing', enabled=False),)
    else:
        path = store.plugins_dir / ('logfire.py' if kind == 'file' else 'logfire/__init__.py')
        path.parent.mkdir(parents=True)
        path.write_text('')
    onboard_logfire(store=store, project_plugins=project)
    assert not capsys.readouterr().out


@pytest.mark.parametrize(
    'failure',
    [
        EOFError(),
        KeyboardInterrupt(),
        OSError('do-not-print'),
        NoKeyringError(),
        subprocess.CalledProcessError(1, ['logfire', 'auth']),
        subprocess.TimeoutExpired(['logfire', 'auth'], timeout=300),
    ],
)
def test_failed_or_cancelled_login_preserves_choices_and_cleans_up(
    tmp_path: Path,
    terminal: Callable[[str], None],
    logfire_cli: LogfireCLI,
    failure: BaseException,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    before = PluginSettings(id='logfire', factory='pydantic_clai2.logfire', enabled=False)
    store.save_plugin(before)
    terminal('login\nuse\n')
    logfire_cli.failure = failure
    onboard_logfire(store=store, force=True)
    assert store.plugins() == [before]
    assert load_logfire_credentials() is None
    assert all(not directory.exists() for directory in logfire_cli.directories)
    output = capsys.readouterr().out
    assert 'try again' in output.split('\x1b[?1049l')[1]
    assert 'do-not-print' not in output


@pytest.mark.parametrize('action', ['decline', 'login'])
def test_locked_database_restores_screen_and_preserves_preferences(
    tmp_path: Path,
    terminal: Callable[[str], None],
    logfire_cli: LogfireCLI,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    before = PluginSettings(id='logfire', factory='pydantic_clai2.logfire', enabled=False)
    store.save_plugin(before)
    terminal(f'{action}\nuse\n')
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute('BEGIN IMMEDIATE')
        onboard_logfire(store=store, force=True)
    assert store.plugins() == [before]
    output = capsys.readouterr().out
    assert 'did not complete' in output.split('\x1b[?1049l')[1]
    assert all(not directory.exists() for directory in logfire_cli.directories)


@pytest.mark.parametrize('enabled', [None, False, True])
def test_initial_ctrl_c_leaves_preferences_unchanged(
    tmp_path: Path, terminal: Callable[[str], None], logfire_cli: LogfireCLI, enabled: bool | None
) -> None:
    terminal('cancel\n')
    store = SettingsStore(tmp_path / 'config.db')
    before = (
        [] if enabled is None else [PluginSettings(id='logfire', factory='pydantic_clai2.logfire', enabled=enabled)]
    )
    for plugin in before:
        store.save_plugin(plugin)
    onboard_logfire(store=store, force=True)
    assert store.plugins() == before
    assert not logfire_cli.commands


def test_project_picker_cancellation_leaves_preferences_unchanged(
    tmp_path: Path, terminal: Callable[[str], None], logfire_cli: LogfireCLI
) -> None:
    terminal('login\nescape\n')
    store = SettingsStore(tmp_path / 'config.db')
    onboard_logfire(store=store)
    assert logfire_cli.commands == [['auth']]
    assert all(not directory.exists() for directory in logfire_cli.directories)
    assert not store.plugins()
    assert load_logfire_credentials() is None


def test_invalid_credentials_do_not_leak_or_enable_plugin(
    tmp_path: Path,
    terminal: Callable[[str], None],
    logfire_cli: LogfireCLI,
    capsys: pytest.CaptureFixture[str],
) -> None:
    terminal('login\nuse\n')
    logfire_cli.value = '{"token": "do-not-print"}'
    store = SettingsStore(tmp_path / 'config.db')
    onboard_logfire(store=store)
    assert not store.plugins()
    assert load_logfire_credentials() is None
    assert all(not directory.exists() for directory in logfire_cli.directories)
    output = capsys.readouterr().out
    assert 'did not complete' in output and 'do-not-print' not in output


def test_headless_skips_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    store = SettingsStore(tmp_path / 'config.db')
    onboard_logfire(store=store)
    assert not store.plugins()
    with pytest.raises(ValueError, match='interactive terminal'):
        logfire_command(store, [])
    with pytest.raises(ValueError, match='Usage: clai2 logfire'):
        logfire_command(store, ['extra'])


@pytest.mark.parametrize('source', ['saved', 'file', 'package', 'project'])
def test_explicit_setup_does_not_replace_custom_plugin(
    tmp_path: Path, terminal: Callable[[str], None], monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    terminal('decline\n')
    monkeypatch.chdir(tmp_path)
    store = SettingsStore(tmp_path / 'config.db')
    before = PluginSettings(id='logfire', factory='custom.tracing')
    if source == 'saved':
        store.save_plugin(before)
    elif source == 'project':
        project = tmp_path / '.clai/settings.json'
        project.parent.mkdir()
        project.write_text(json.dumps({'plugins': [before.model_dump()]}))
    else:
        path = store.plugins_dir / ('logfire.py' if source == 'file' else 'logfire/__init__.py')
        path.parent.mkdir(parents=True)
        path.write_text('')
    with pytest.raises(ValueError, match='has been replaced'):
        logfire_command(store, [])
    assert store.plugins() == ([before] if source == 'saved' else [])


def test_private_file_fallback_is_reported(
    tmp_path: Path,
    terminal: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
    logfire_cli: LogfireCLI,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(service: str, username: str, password: str) -> None:
        raise NoKeyringError

    monkeypatch.setattr(keyring, 'set_password', unavailable)
    terminal('login\nnew\n')
    monkeypatch.setenv('LOGFIRE_TOKEN', 'environment-token')
    logfire_command(SettingsStore(tmp_path / 'config.db'), [])
    path = logfire_directory().parent / 'credentials-logfire.json'
    if sys.platform != 'win32':
        assert path.stat().st_mode & 0o777 == 0o600
    output = ' '.join(Text.from_ansi(capsys.readouterr().out).plain.split())
    assert 'private plaintext file' in output
    assert 'LOGFIRE_TOKEN is set and takes precedence' in output
    assert 'test-write-token' not in output


@pytest.mark.parametrize('valid', [True, False])
def test_legacy_credentials_migrate_without_losing_invalid_data(valid: bool) -> None:
    directory = logfire_directory()
    directory.mkdir(parents=True)
    path = directory / 'logfire_credentials.json'
    value = CREDENTIALS if valid else '{"token":"do-not-print"}'
    path.write_text(value)
    if valid:
        credentials = load_logfire_credentials()
        assert credentials is not None and credentials.token == 'test-write-token'
        assert not path.exists()
        assert load_codex_credentials(account='logfire') is not None
        assert load_logfire_credentials() == credentials
    else:
        with pytest.raises(ValidationError) as error:
            load_logfire_credentials()
        assert 'do-not-print' not in str(error.value)
        assert path.read_text() == value


def test_migration_preserves_legacy_file_on_keyring_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(service: str, username: str, password: str) -> None:
        raise OSError('locked keyring')

    directory = logfire_directory()
    directory.mkdir(parents=True)
    path = directory / 'logfire_credentials.json'
    path.write_text(CREDENTIALS)
    monkeypatch.setattr(keyring, 'set_password', failed)
    with pytest.raises(OSError, match='locked keyring'):
        load_logfire_credentials()
    assert path.read_text() == CREDENTIALS

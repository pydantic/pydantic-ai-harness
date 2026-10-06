"""Interactive opt-in before the editor or any telemetry exporter starts."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from keyring.errors import KeyringError
from pydantic import ValidationError
from pydantic_ai.exceptions import UserError
from rich.console import Console
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]

from ._rendering import markdown_style
from .config import PluginSettings
from .field_menu import TERMINAL, Runners
from .logfire_credentials import (
    LogfireCredentials,
    load_logfire_credentials,
    logfire_directory,
    save_logfire_credentials,
)
from .menu_worker import menu_key
from .project_settings import load_project_settings
from .settings_store import SettingsStore


def build_logfire_menu(*, project: bool = False) -> Menu:
    """Paint within the setup screen, with the primary action selected and an explicit escape route."""
    items = (
        [MenuItem('Use an existing project', value='use'), MenuItem('Create a new project', value='new')]
        if project
        else [MenuItem('Log in to Logfire', value='login'), MenuItem('Continue without Logfire', value='decline')]
    )
    return (
        MenuBuilder('Choose a Logfire project' if project else 'Choose how to continue')
        .style(markdown_style())
        .items(items)
        .inline()
        .key_source(menu_key)
        .on_key('escape', lambda menu, item: MenuResult(cancelled=project, item=None if project else items[1]))
        .footer_hint('Up/Down move - Enter select - Esc ' + ('cancel' if project else 'skip'))
        .build()
    )


def logfire_command(store: SettingsStore, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Reconnect or decline without starting a chat session."""
    if args:
        raise ValueError('Usage: clai2 logfire')
    onboard_logfire(store=store, project_plugins=load_project_settings(Path.cwd()).plugins, force=True, runners=runners)
    return ''


def onboard_logfire(
    *,
    store: SettingsStore,
    project_plugins: tuple[PluginSettings, ...] = (),
    force: bool = False,
    runners: Runners = TERMINAL,
) -> None:
    """Remember login or decline as a normal plugin override, including on upgrades."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        if force:
            raise ValueError('Logfire setup requires an interactive terminal. Set LOGFIRE_TOKEN for headless use.')
        return
    saved = next((plugin for plugin in store.plugins() if plugin.id == 'logfire'), None)
    replacement = (
        saved.factory != 'pydantic_clai2.logfire' or saved.path is not None
        if saved is not None
        else any(plugin.id == 'logfire' for plugin in project_plugins)
        or (store.plugins_dir / 'logfire.py').is_file()
        or (store.plugins_dir / 'logfire' / '__init__.py').is_file()
    )
    if not force and (saved is not None or replacement):
        return
    if force and replacement:
        raise ValueError('The logfire plugin has been replaced. Remove its override before running clai2 logfire.')
    console = Console()
    try:
        if not force and (os.getenv('LOGFIRE_TOKEN') or load_logfire_credentials() is not None):
            return
        with console.screen(hide_cursor=False):
            console.print('\nConnect Logfire', style='bold')
            console.print(
                'See model calls, tool runs, token usage, and failures in your own Logfire project.',
                width=min(console.width, 72),
            )
            console.print('\n[bold]What gets sent[/bold]')
            console.print(
                'Prompts, responses, tool arguments/results, and images. '
                'These may include source code, file contents, and screenshots.',
                width=min(console.width, 72),
            )
            console.print()
            choice = runners.run_choice(build_logfire_menu())
            if choice.item is None:
                raise EOFError
            declaration = saved or PluginSettings(id='logfire', factory='pydantic_clai2.logfire')
            if choice.item.value == 'decline':
                store.save_plugin(declaration.model_copy(update={'enabled': False}))
                result = 'Logfire disabled. Run clai2 logfire to connect later.'
            else:
                result = connect_logfire(console=console, runners=runners)
                store.save_plugin(declaration.model_copy(update={'enabled': True}))
        console.print(result, markup=False, highlight=False)
    except (EOFError, KeyboardInterrupt):
        console.print('\nLogfire setup cancelled. Run clai2 logfire to try again.')
    except (OSError, subprocess.SubprocessError, sqlite3.DatabaseError, KeyringError, ValidationError, UserError):
        console.print('Logfire setup did not complete. Run clai2 logfire to try again.')


def connect_logfire(*, console: Console, runners: Runners = TERMINAL) -> str:
    """Replace each setup step, then return the result to display after restoring the terminal."""
    console.clear()
    console.print('\nLog in to Logfire', style='bold')
    console.print('Continue in your browser when prompted. Ctrl-C cancels setup.', style='dim')
    with TemporaryDirectory(prefix='clai-logfire-') as temporary:
        directory = Path(temporary)
        subprocess.run([sys.executable, '-I', '-m', 'logfire', 'auth'], cwd=directory, check=True, timeout=300)
        console.clear()
        choice = runners.run_choice(build_logfire_menu(project=True))
        if choice.item is None:
            raise EOFError
        action = 'new' if choice.item.value == 'new' else 'use'
        console.clear()
        subprocess.run(
            [sys.executable, '-I', '-m', 'logfire', 'projects', action, '--data-dir', str(directory)],
            cwd=directory,
            check=True,
            timeout=300,
        )
        path = directory / 'logfire_credentials.json'
        credentials = LogfireCredentials.model_validate_json(path.read_text(encoding='utf-8'))
        save_logfire_credentials(credentials)
    result = 'Logfire connected.\n'
    fallback = logfire_directory().parent / 'credentials-logfire.json'
    if fallback.exists():
        result += f'No OS keyring is available. Credentials are saved in a private plaintext file:\n{fallback}'
    else:
        result += 'Project credentials saved in the OS keyring.'
    if os.getenv('LOGFIRE_TOKEN'):
        result += '\nLOGFIRE_TOKEN is set and takes precedence over this saved project.'
    return result

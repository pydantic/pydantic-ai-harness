"""`.clai/settings.json`: the walk-up, the layering, and the startup report."""

import io
import json
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import chat
from pydantic_clai2.config import PluginSettings, resolve_settings
from pydantic_clai2.project_settings import PROJECT_FILE, ProjectSettings, find_project_file, load_project_settings
from pydantic_clai2.settings_store import SettingsStore


def write(directory: Path, content: dict[str, JsonValue]) -> Path:
    path = directory / PROJECT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


def test_walk_up_stops_at_the_git_root(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    nested = repo / 'src' / 'pkg'
    nested.mkdir(parents=True)
    assert find_project_file(nested) is None

    above = write(tmp_path, {'thinking': False})
    (repo / '.git').mkdir()
    assert find_project_file(nested) is None, 'a file above the git root is not this project'
    (repo / '.git').rmdir()
    assert find_project_file(nested) == above

    (repo / '.git').write_text('gitdir: elsewhere')
    inside = write(repo, {'thinking': True})
    assert find_project_file(nested) == inside
    assert find_project_file(repo) == inside
    assert find_project_file(nested / 'missing') == inside, 'the walk starts from the resolved path even if absent'


def test_load_layers_between_store_and_flags(tmp_path: Path) -> None:
    path = write(tmp_path, {'thinking': False, 'request_limit': 7, 'colour': 'mauve', 'zeta': 1})
    project = load_project_settings(tmp_path / 'deeper')
    assert project.path == path
    assert project.overrides == {'display.thinking': False, 'run.request_limit': 7}
    assert project.unknown == ('colour', 'zeta')
    assert project.plugins == ()

    store = SettingsStore(tmp_path / 'config.db')
    store.set('run.request_limit', 3)
    store.set('display.splash', False)
    layered = resolve_settings(store.overrides() | project.overrides | {'run.request_limit': 9})
    assert (layered.request_limit, layered.thinking, layered.splash) == (9, False, False)


def test_load_without_a_file_is_empty(tmp_path: Path) -> None:
    assert load_project_settings(tmp_path) == ProjectSettings()


def test_plugins_are_validated_declarations(tmp_path: Path) -> None:
    write(tmp_path, {'plugins': [{'id': 'exa', 'factory': 'pydantic_ai_harness.exa:ExaSearch', 'enabled': False}]})
    project = load_project_settings(tmp_path)
    assert project.plugins == (PluginSettings(id='exa', factory='pydantic_ai_harness.exa:ExaSearch', enabled=False),)
    assert project.overrides == {}


@pytest.mark.parametrize(
    'content',
    [
        {'request_limit': 0},
        {'model': 7},
        {'plugins': [{'id': 'nope'}]},
        {'plugins': 'exa'},
    ],
)
def test_bad_values_fail_startup(tmp_path: Path, content: dict[str, JsonValue]) -> None:
    write(tmp_path, content)
    with pytest.raises(ValueError, match='settings.json'):
        load_project_settings(tmp_path)


def test_not_an_object_fails_startup(tmp_path: Path) -> None:
    path = tmp_path / PROJECT_FILE
    path.parent.mkdir()
    path.write_text('[1, 2]')
    with pytest.raises(ValueError, match='settings.json'):
        load_project_settings(tmp_path)


async def test_startup_reports_unknown_keys_once(tmp_path: Path) -> None:
    path = write(tmp_path, {'thinking': False, 'colour': 'mauve'})
    project = load_project_settings(tmp_path)
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/set display.thinking true\n/set display.thinking\n/exit\n')
        await chat(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            console=Console(file=output, width=200),
            store=SettingsStore(tmp_path / 'config.db'),
            settings=resolve_settings(project.overrides),
            project=project,
        )
    text = output.getvalue()
    assert f'Project settings: {path}' in text
    assert text.count('Ignoring unknown settings: colour') == 1
    assert 'Saved display.thinking. Applied. The project file sets it again at next start.' in text
    assert '\nTrue\n' in text, '/set still applies to this session'


@pytest.mark.parametrize('with_file', [True, False])
async def test_startup_reports_a_clean_file_or_nothing(tmp_path: Path, with_file: bool) -> None:
    output = io.StringIO()
    project = ProjectSettings(path=tmp_path / PROJECT_FILE) if with_file else None
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\n')
        await chat(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            console=Console(file=output, width=200),
            store=SettingsStore(tmp_path / 'config.db'),
            project=project,
        )
    assert ('Project settings' in output.getvalue()) is with_file
    assert 'Ignoring' not in output.getvalue()

"""Run CLI parsing and configuration precedence without starting a model provider."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pydantic_clai2.settings_store import SettingsStore

CLI = """
import json
import runpy
from pydantic_clai2 import web

async def serve_web(*, settings, store, project, port):
    print(json.dumps({'model': settings.model, 'retries': settings.tool_retries, 'port': port}))

web.serve_web = serve_web
runpy.run_module('pydantic_clai2', run_name='__main__')
"""


@pytest.mark.parametrize('source', ['default', 'saved', 'project', 'env', 'cli'])
def test_config_precedence(tmp_path: Path, source: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    args = ['--database', str(store.path), '--web']
    env = dict(os.environ)
    env.pop('CLAI_MODEL', None)
    (tmp_path / '.git').mkdir()
    expected = 'openai-codex:gpt-6-astra'
    if source != 'default':
        store.set('model', 'saved')
        store.set('run.tool_retries', 7)
        expected = 'saved'
    if source in ('project', 'env', 'cli'):
        (tmp_path / '.clai').mkdir()
        (tmp_path / '.clai/settings.json').write_text('{"model": "project"}')
        expected = 'project'
    if source in ('env', 'cli'):
        env['CLAI_MODEL'] = 'environment'
        expected = 'environment'
    if source == 'cli':
        args += ['--model', 'test', '--port', '8765']
        expected = 'test'
    before = store.overrides()
    result = subprocess.run(
        [sys.executable, '-c', CLI, *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        'model': expected,
        'retries': 3 if source == 'default' else 7,
        'port': 8765 if source == 'cli' else 7932,
    }
    assert store.overrides() == before


@pytest.mark.parametrize(
    'args',
    [
        ['--web', '--resume'],
        ['--web', '--resume', 'session-id'],
        ['--web', 'config'],
        ['--web', 'plugins'],
        ['--port', '8765'],
        ['--web', '--port', '0'],
        ['--web', '--port', '65536'],
        ['--web', '--request-limit', '50'],
        ['--web', '--host', '0.0.0.0'],
        ['--web', '--remote'],
    ],
)
def test_reject_unsupported_options(tmp_path: Path, args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', str(tmp_path / 'config.db'), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert 'error:' in result.stderr
    assert 'Uvicorn running' not in result.stderr


@pytest.mark.parametrize('source', ['saved', 'project'])
def test_reject_explicit_request_limits(tmp_path: Path, source: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    if source == 'saved':
        store.set('run.request_limit', 10000)
    else:
        (tmp_path / '.clai').mkdir()
        (tmp_path / '.clai/settings.json').write_text('{"request_limit": 10000}')
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', str(store.path), '--web', '--model', 'test'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert 'does not support explicit request limits' in result.stderr
    assert '50 requests per run' in result.stderr
    if source == 'saved':
        assert store.overrides()['run.request_limit'] == 10000


@pytest.mark.parametrize('web', [True, False])
def test_optional_dependencies(tmp_path: Path, web: bool) -> None:
    script = "import sys, runpy; sys.modules['uvicorn'] = None; runpy.run_module('pydantic_clai2', run_name='__main__')"
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            script,
            '--database',
            str(tmp_path / 'config.db'),
            *(['--web'] if web else ['config', 'show']),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == (2 if web else 0)
    if web:
        assert 'Install pydantic-clai2[web]' in result.stderr

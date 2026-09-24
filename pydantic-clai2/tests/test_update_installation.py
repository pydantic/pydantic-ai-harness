"""Update advice follows installation provenance, not the current working directory."""

import json
import shlex
import sys
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from pydantic_clai2 import update_installation
from pydantic_clai2.update_installation import update_guidance


@pytest.mark.parametrize(
    'origin', ['{"dir_info":{"editable":true}}', '{"vcs_info":{}}', '{"archive_info":{}}', 'broken']
)
def test_direct_installs_keep_their_source(tmp_path: Path, origin: str) -> None:
    (tmp_path / 'direct_url.json').write_text(origin)
    (tmp_path / 'INSTALLER').write_text('pip\n')
    assert update_guidance(PathDistribution(tmp_path)) == (
        'Source/direct install: update from your original source, then restart CLAI2.'
    )


def test_uv_tool_receipt(tmp_path: Path) -> None:
    prefix = tmp_path / 'pydantic-clai2'
    prefix.mkdir()
    (prefix / 'uv-receipt.toml').write_text('[tool]\nrequirements = [{name = "pydantic-clai2"}]')
    assert update_guidance(PathDistribution(tmp_path), prefix=prefix) == 'Run: uv tool upgrade pydantic-clai2'
    (tmp_path / 'uv-receipt.toml').write_text('[tool]')
    assert 'original installer' in update_guidance(PathDistribution(tmp_path), prefix=tmp_path)


@pytest.mark.parametrize('suffix', ['', '-custom'])
def test_pipx_environment_including_suffix(tmp_path: Path, suffix: str) -> None:
    prefix = tmp_path / f'pydantic-clai2{suffix}'
    prefix.mkdir()
    (prefix / 'pipx_metadata.json').write_text(json.dumps({'main_package': {'package': 'pydantic-clai2'}}))
    (tmp_path / 'INSTALLER').write_text('pip\n')
    assert update_guidance(PathDistribution(tmp_path), prefix=prefix) == f'Run: pipx upgrade pydantic-clai2{suffix}'


@pytest.mark.parametrize('main', ['{"package":"other-tool"}', 'null'])
def test_injected_pipx_package_does_not_upgrade_the_host_tool(tmp_path: Path, main: str) -> None:
    (tmp_path / 'pipx_metadata.json').write_text('{"main_package":' + main + '}')
    assert 'original pipx inject command' in update_guidance(PathDistribution(tmp_path), prefix=tmp_path)


def test_broken_pipx_metadata_is_unknown(tmp_path: Path) -> None:
    (tmp_path / 'pipx_metadata.json').write_text('broken')
    assert 'original installer' in update_guidance(PathDistribution(tmp_path), prefix=tmp_path)


@pytest.mark.parametrize('windows', [False, True])
def test_pip_targets_the_running_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: bool) -> None:
    (tmp_path / 'INSTALLER').write_text('pip\n')
    monkeypatch.setattr(sys, 'executable', '/path with spaces/python')

    # Keep pathlib on the real platform while exercising command quoting on Windows.
    class Platform:
        name = 'nt' if windows else 'posix'

    monkeypatch.setattr(update_installation, 'os', Platform)
    quoted = '"/path with spaces/python"' if windows else shlex.quote(sys.executable)
    assert update_guidance(PathDistribution(tmp_path), prefix=tmp_path) == (
        f'Run: {quoted} -m pip install --upgrade pydantic-clai2'
    )


@pytest.mark.parametrize('installer', ['uv', 'unknown', ''])
def test_ambiguous_environment_does_not_guess(tmp_path: Path, installer: str) -> None:
    (tmp_path / 'INSTALLER').write_text(installer)
    result = update_guidance(PathDistribution(tmp_path), prefix=tmp_path)
    if installer == 'uv':
        assert 'uvx --upgrade --from pydantic-clai2 clai2' in result
        assert 'original uv project/environment' in result
        assert 'If using uvx' in result
    else:
        assert 'original installer' in result


def test_unreadable_installer_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(name: str) -> str:
        raise OSError('unreadable')

    installed = PathDistribution(tmp_path)
    monkeypatch.setattr(installed, 'read_text', unreadable)
    assert 'original installer' in update_guidance(installed, prefix=tmp_path)

"""Working-directory arguments are deprecated and ignored: the working directory belongs to the workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.macroscope import Macroscope
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell

from ._tool_calls import call_tool
from ._workspace import local_workspace

pytestmark = pytest.mark.anyio

_ON_THE_WORKSPACE = r"set it on the workspace, e\.g\. `LocalWorkspace\('\./repo'\)`"


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _coder(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return Coder[None](elsewhere)


def _file_system(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return FileSystem[None](cwd=elsewhere)


def _shell(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return Shell[None](cwd=elsewhere)


def _macroscope(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return Macroscope[None](cwd=elsewhere, command=str(fake_cli))


def _repo_context(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return RepoContext[None](workspace_dir=Path(elsewhere))


@pytest.mark.parametrize(
    ('build', 'warning', 'tool', 'arguments', 'expected'),
    [
        (
            _coder,
            r"`Coder\(workspace=\.\.\.\)` is deprecated and ignored: .* attach `LocalWorkspace\('elsewhere'\)`",
            'shell',
            {'command': 'ls marker.txt'},
            'marker.txt',
        ),
        (
            _file_system,
            rf'`FileSystem\(cwd=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'read_file',
            {'path': 'marker.txt'},
            'in the working directory',
        ),
        (
            _shell,
            rf'`Shell\(cwd=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'run_command',
            {'command': 'ls marker.txt'},
            'marker.txt',
        ),
        (
            _macroscope,
            rf'`Macroscope\(cwd=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'run_macroscope_review',
            {},
            "review_id='marker.txt'",
        ),
        (
            _repo_context,
            rf'`RepoContext\(workspace_dir=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'inventory_agent_context',
            {},
            "root='.claude', exists=True",
        ),
    ],
    ids=['Coder', 'FileSystem', 'Shell', 'Macroscope', 'RepoContext'],
)
async def test_working_directory_argument_warns_and_has_no_effect(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    build: Callable[[str, Path], AbstractCapability[None]],
    warning: str,
    tool: str,
    arguments: dict[str, object],
    expected: str,
) -> None:
    (tmp_path / 'elsewhere').mkdir()
    (tmp_path / 'marker.txt').write_text('in the working directory\n')
    (tmp_path / '.claude').mkdir()
    fake_cli = tmp_path_factory.mktemp('bin') / 'macroscope'
    fake_cli.write_text('#!/bin/sh\nprintf \'review_id=%s\\n\' "$(ls marker.txt)" >&2\n')
    fake_cli.chmod(0o755)

    with pytest.warns(HarnessDeprecationWarning, match=warning):
        capability = build('elsewhere', fake_cli)

    assert expected in await call_tool([capability], tool, arguments, workspace=local_workspace(tmp_path))

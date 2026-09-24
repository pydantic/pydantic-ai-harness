"""The opt-in ripgrep-backed `list_files` and `grep` tools."""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.filesystem import RIPGREP_TOOL_NAMES, FilesSearchedEvent, FileSystem, FileSystemToolset

from .._tool_calls import call_tool
from .._workspace import local_workspace

WS = local_workspace('/')
"""The workspace for direct calls; the toolsets here all have absolute roots, so its working directory is moot."""

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'app.py').write_text('import os\n\n\ndef main():\n    return os.name\n')
    (tmp_path / 'notes.txt').write_text('Import notes\nos is a module\n')
    (tmp_path / '.hidden').write_text('import os\n')
    (tmp_path / 'ignored.log').write_text('import os\n')
    (tmp_path / '.ignore').write_text('*.log\n')
    return tmp_path


def toolset(workspace: Path, **settings: object) -> FileSystemToolset[None]:
    capability = FileSystem[None](root_dir=workspace, tools=RIPGREP_TOOL_NAMES, **settings)  # pyright: ignore[reportArgumentType]
    built = capability.get_toolset()
    assert isinstance(built, FileSystemToolset)
    return built


async def call(
    workspace: Path,
    name: str,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
    **settings: object,
) -> str:
    capability = FileSystem[None](root_dir=workspace, tools=RIPGREP_TOOL_NAMES, **settings)  # pyright: ignore[reportArgumentType]
    return await call_tool([capability, *capabilities], name, arguments, workspace=WS)


class Recorder(AbstractCapability[None]):
    def __init__(self) -> None:
        self.events: list[FilesSearchedEvent] = []

    @on_event(FilesSearchedEvent)
    async def searched(self, ctx: RunContext[None], event: FilesSearchedEvent) -> None:
        self.events.append(event)


class TestRegistration:
    async def test_opt_in(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[FileSystem(root_dir=tmp_path)]).run('Inspect tools', workspace=WS)
        assert model.last_model_request_parameters is not None
        assert not {'list_files', 'grep'} & {t.name for t in model.last_model_request_parameters.function_tools}

        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[FileSystem(root_dir=tmp_path, tools=['read_file', 'grep'])]).run(
            'Inspect', workspace=WS
        )
        assert model.last_model_request_parameters is not None
        assert [t.name for t in model.last_model_request_parameters.function_tools] == ['read_file', 'grep']

    def test_unknown_tool_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='Unknown filesystem tools: bogus'):
            FileSystem(root_dir=tmp_path, tools=['bogus']).get_toolset()


class TestListFiles:
    async def test_respects_ignore_rules_and_hidden_files(self, workspace: Path) -> None:
        assert (await toolset(workspace).list_files(workspace=WS)).splitlines() == ['notes.txt', 'src/app.py']

    async def test_glob(self, workspace: Path) -> None:
        assert await toolset(workspace).list_files(glob='*.py', workspace=WS) == 'src/app.py'
        assert await toolset(workspace).list_files('src', glob='*.txt', workspace=WS) == 'No files found.'

    async def test_glob_overrides_ignore_files_but_not_hidden(self, workspace: Path) -> None:
        assert await toolset(workspace).list_files(glob='*.log', workspace=WS) == 'ignored.log'
        assert (await toolset(workspace).list_files(glob='**', workspace=WS)).splitlines() == [
            'ignored.log',
            'notes.txt',
            'src/app.py',
        ]
        assert '.hidden' not in await toolset(workspace).grep('import os', glob='**', workspace=WS)

    async def test_denied_patterns_filter_entries(self, workspace: Path) -> None:
        assert await toolset(workspace, denied_patterns=['src/*']).list_files(workspace=WS) == 'notes.txt'

    async def test_cap(self, workspace: Path) -> None:
        listed = await toolset(workspace, max_find_results=1).list_files(workspace=WS)
        assert listed.splitlines() == ['notes.txt', '[... truncated at 1 files]']

    async def test_cap_counts_permitted_entries_only(self, workspace: Path) -> None:
        listed = await toolset(workspace, max_find_results=1, denied_patterns=['notes.txt']).list_files(workspace=WS)
        assert listed == 'src/app.py'

    async def test_oversized_record_stops_the_search(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = workspace / 'bin'
        fake.mkdir()
        (fake / 'rg').write_text(f'#!{sys.executable}\nimport sys\nsys.stdout.write("x" * 2_000_000)\n')
        (fake / 'rg').chmod(0o755)
        monkeypatch.setenv('PATH', f'{fake}{os.pathsep}{os.environ["PATH"]}')
        assert await toolset(workspace).list_files(workspace=WS) == '[... truncated at 1000 files]'

    async def test_oversized_complete_record_stops_the_search(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = workspace / 'bin'
        fake.mkdir()
        (fake / 'rg').write_text(f'#!{sys.executable}\nimport sys\nsys.stdout.write("x" * 2_000_000 + "\\0")\n')
        (fake / 'rg').chmod(0o755)
        monkeypatch.setenv('PATH', f'{fake}{os.pathsep}{os.environ["PATH"]}')
        assert await toolset(workspace).list_files(workspace=WS) == '[... truncated at 1000 files]'

    async def test_search_that_dies_without_a_status_is_reported(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Killing the shell that would report `rg`'s status leaves no status line to read.
        fake = workspace / 'bin'
        fake.mkdir()
        (fake / 'rg').write_text('#!/bin/sh\nkill -9 $PPID\n')
        (fake / 'rg').chmod(0o755)
        monkeypatch.setenv('PATH', f'{fake}{os.pathsep}{os.environ["PATH"]}')
        with pytest.raises(ModelRetry, match='ripgrep failed'):
            await toolset(workspace).list_files(workspace=WS)

    @pytest.mark.parametrize('cwd_name', ['.', 'src'])
    async def test_event(self, workspace: Path, cwd_name: str) -> None:
        recorder = Recorder()
        await call(workspace, 'list_files', {'glob': '*.py'}, capabilities=[recorder], cwd=workspace / cwd_name)
        assert recorder.events[0].search == 'find'
        assert recorder.events[0].pattern == '*.py'
        assert recorder.events[0].match_count == 1
        assert recorder.events[0].path == cwd_name
        assert recorder.events[0].root_dir == str(workspace)

    @pytest.mark.parametrize('path', ['notes.txt', 'missing', '..'])
    async def test_rejects_non_directories(self, workspace: Path, path: str) -> None:
        assert await call(workspace, 'list_files', {'path': path})
        assert 'src/app.py' not in await call(workspace, 'list_files', {'path': path})


class TestGrep:
    async def test_matches_with_line_numbers(self, workspace: Path) -> None:
        assert await toolset(workspace).grep('import os', workspace=WS) == 'src/app.py:1:import os'

    async def test_options(self, workspace: Path) -> None:
        built = toolset(workspace)
        assert await built.grep('import', ignore_case=True, glob='*.txt', workspace=WS) == 'notes.txt:1:Import notes'
        assert (await built.grep('os', file_type='py', context=1, workspace=WS)).splitlines() == [
            'src/app.py:1:import os',
            'src/app.py-2-',
            'src/app.py-4-def main():',
            'src/app.py:5:    return os.name',
        ]
        assert (
            await built.grep('return os.name', literal=True, path='src', workspace=WS)
            == 'src/app.py:5:    return os.name'
        )
        assert await built.grep('nothing', workspace=WS) == 'No matches found.'

    async def test_long_lines_are_cut_by_ripgrep(self, workspace: Path) -> None:
        (workspace / 'minified.js').write_text('x' * 5000 + 'needle' + 'y' * 5000 + '\n')
        result = await toolset(workspace).grep('needle', workspace=WS)
        assert result.startswith('minified.js:1:xxxx') and result.endswith('[... omitted end of long line]')
        assert len(result) < 5000

    async def test_file_target(self, workspace: Path) -> None:
        assert await toolset(workspace).grep('os', path='notes.txt', workspace=WS) == 'notes.txt:2:os is a module'

    async def test_authorization_filters_records(self, workspace: Path) -> None:
        assert (
            await toolset(workspace, denied_patterns=['src/*']).grep('os', workspace=WS) == 'notes.txt:2:os is a module'
        )

    async def test_cap_counts_context_lines(self, workspace: Path) -> None:
        capped = await toolset(workspace, max_search_results=1).grep('os', context=2, workspace=WS)
        assert capped.splitlines() == ['notes.txt-1-Import notes', '[... truncated at 1 lines]']
        assert (await toolset(workspace, max_search_results=2).grep('os', workspace=WS)).splitlines() == [
            'notes.txt:2:os is a module',
            'src/app.py:1:import os',
            '[... truncated at 2 lines]',
        ]

    @pytest.mark.parametrize('cwd_name,path', [('.', 'src'), ('src', '.')])
    async def test_event(self, workspace: Path, cwd_name: str, path: str) -> None:
        recorder = Recorder()
        await call(
            workspace, 'grep', {'pattern': 'os', 'path': path}, capabilities=[recorder], cwd=workspace / cwd_name
        )
        assert recorder.events[0].search == 'grep'
        assert recorder.events[0].pattern == 'os'
        assert recorder.events[0].path == 'src'
        assert recorder.events[0].root_dir == str(workspace)
        assert recorder.events[0].match_count == 2

    @pytest.mark.parametrize(
        'arguments,message',
        [
            ({'pattern': '(', 'context': 0}, 'ripgrep failed'),
            ({'pattern': 'os', 'context': 21}, 'context must be between'),
            ({'pattern': 'os', 'path': 'missing'}, 'not a file or directory'),
            ({'pattern': 'os', 'path': '../outside'}, 'outside the root'),
        ],
    )
    async def test_retries(self, workspace: Path, arguments: dict[str, object], message: str) -> None:
        assert message in await call(workspace, 'grep', arguments)

    async def test_missing_ripgrep(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PATH', str(workspace))
        assert 'ripgrep (rg) is not installed' in await call(workspace, 'grep', {'pattern': 'os'})

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX symlinks')
    async def test_symlink_outside_root_is_dropped(
        self, workspace: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp('outside') / 'secret.txt'
        outside.write_text('import os\n')
        (workspace / 'link.txt').symlink_to(outside)
        listed = await toolset(workspace).list_files(workspace=WS)
        assert 'link.txt' not in listed
        assert 'link.txt' not in await toolset(workspace).grep('import os', workspace=WS)

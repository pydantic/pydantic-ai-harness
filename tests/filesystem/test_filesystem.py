"""Tests for the filesystem tools, exercised through the run's sandbox."""

from __future__ import annotations

import errno
import hashlib
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import (
    CommandResult,
    LocalSandbox,
    Sandbox,
    SandboxCommand,
    SandboxError,
    SandboxFileEntry,
    SandboxRef,
    SandboxResult,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.filesystem import READ_ONLY_TOOL_NAMES, FileSystem, FileSystemToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[Sandbox]:
    async with LocalSandbox(root=tmp_path) as backend:
        yield Sandbox.wrap(backend)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _ctx(sandbox: Sandbox | None = None) -> RunContext[None]:
    if sandbox is None:
        return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        sandbox=sandbox,
    )


def _toolset(
    root: Path,
    *,
    allowed_patterns: Sequence[str] = (),
    denied_patterns: Sequence[str] = (),
    protected_patterns: Sequence[str] = (),
    max_read_lines: int = 100,
    max_list_results: int = 100,
    max_search_results: int = 100,
    max_find_results: int = 100,
) -> FileSystemToolset[None]:
    return FileSystemToolset(
        root_dir=root,
        allowed_patterns=allowed_patterns,
        denied_patterns=denied_patterns,
        protected_patterns=protected_patterns,
        max_read_lines=max_read_lines,
        max_list_results=max_list_results,
        max_search_results=max_search_results,
        max_find_results=max_find_results,
    )


async def _call(
    toolset: FileSystemToolset[None],
    ctx: RunContext[None],
    name: str,
    tool_args: dict[str, object],
) -> str:
    tools = await toolset.get_tools(ctx)
    result: object = await toolset.call_tool(name, tool_args, ctx, tools[name])
    assert isinstance(result, str)
    return result


class _ErrorFilesystem:
    """Only the operations the tools reach before failing; `stat` guards most paths."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def read_bytes(self, path: str) -> bytes:
        raise self.error

    async def write_bytes(self, path: str, data: bytes) -> None:
        raise self.error

    async def stat(self, path: str) -> SandboxFileEntry:
        raise self.error

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        raise self.error

    async def make_dir(self, path: str) -> None:
        raise self.error

    async def remove(self, path: str) -> None:
        raise self.error

    async def exists(self, path: str) -> bool:
        raise self.error


class _ErrorBackend(_ErrorFilesystem):
    """A backend whose filesystem and commands always raise the same error."""

    ref = SandboxRef(sandbox_id='error-1')

    def __init__(self, error: Exception) -> None:
        super().__init__(error)

    async def working_dir(self) -> str:
        return '/work'

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        raise self.error


class _LocalFilesystemBackend:
    def __init__(self, backend: LocalSandbox) -> None:
        self.backend = backend

    async def read_bytes(self, path: str) -> bytes:
        return await self.backend.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.backend.write_bytes(path, data)

    async def stat(self, path: str) -> SandboxFileEntry:
        return await self.backend.stat(path)

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        return await self.backend.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self.backend.make_dir(path)

    async def remove(self, path: str) -> None:
        await self.backend.remove(path)

    async def exists(self, path: str) -> bool:
        return await self.backend.exists(path)


class _TimeoutBackend(_LocalFilesystemBackend):
    """A real local filesystem whose command execution times out."""

    def __init__(self, backend: LocalSandbox, error: SandboxTimeoutError) -> None:
        super().__init__(backend)
        self.error = error
        self.ref = backend.ref

    async def working_dir(self) -> str:
        return await self.backend.working_dir()

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        raise self.error


class _ResultBackend(_LocalFilesystemBackend):
    """A real local filesystem, with a canned result for every command."""

    ref = SandboxRef(sandbox_id='result-1')

    def __init__(self, backend: LocalSandbox, result: CommandResult) -> None:
        super().__init__(backend)
        self.result = result

    async def working_dir(self) -> str:
        return await self.backend.working_dir()

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        return self.result


async def test_error_backend_implements_the_complete_flat_filesystem() -> None:
    error = RuntimeError('filesystem failed')
    backend = _ErrorBackend(error)

    for operation in (
        backend.read_bytes('/file'),
        backend.write_bytes('/file', b'data'),
        backend.stat('/file'),
        backend.list_dir('/'),
        backend.make_dir('/dir'),
        backend.remove('/file'),
        backend.exists('/file'),
    ):
        with pytest.raises(RuntimeError, match='filesystem failed'):
            await operation


async def test_local_test_backend_delegates_the_complete_flat_filesystem(tmp_path: Path) -> None:
    async with LocalSandbox(root=tmp_path) as local:
        backend = _LocalFilesystemBackend(local)
        directory = str(tmp_path / 'nested')
        path = f'{directory}/file.txt'

        await backend.make_dir(directory)
        await backend.write_bytes(path, b'data')

        assert await backend.read_bytes(path) == b'data'
        assert (await backend.stat(path)).size == 4
        assert [entry.name for entry in await backend.list_dir(directory)] == ['file.txt']
        assert await backend.exists(path) is True

        await backend.remove(path)
        assert await backend.exists(path) is False


# --- the root directory ---


async def test_the_toolset_exposes_one_tool_per_file_operation(tmp_path: Path) -> None:
    tools = await _toolset(tmp_path).get_tools(_ctx())

    assert set(tools) == {
        'read_file',
        'write_file',
        'edit_file',
        'list_directory',
        'search_files',
        'find_files',
        'create_directory',
        'file_info',
    }


async def test_a_relative_root_resolves_against_the_sandbox_working_directory(tmp_path: Path, sandbox: Sandbox) -> None:
    content = 'nested\n'
    (tmp_path / 'workspace').mkdir()
    (tmp_path / 'workspace' / 'a.txt').write_text(content)

    result = await _call(_toolset(Path('workspace')), _ctx(sandbox), 'read_file', {'path': 'a.txt'})

    assert result == f'[a.txt | 1 lines | hash:{_hash(content)}]\n     1\tnested\n'


def test_a_home_relative_root_is_rejected() -> None:
    with pytest.raises(UserError, match='Sandbox paths do not expand `~`'):
        _toolset(Path('~/workspace'))


# --- read_file ---


async def test_read_file_numbers_lines_and_reports_the_file_hash(tmp_path: Path, sandbox: Sandbox) -> None:
    content = 'one\ntwo\n'
    (tmp_path / 'a.txt').write_text(content)

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': 'a.txt'})

    assert result == f'[a.txt | 2 lines | hash:{_hash(content)}]\n     1\tone\n     2\ttwo\n'


async def test_read_file_pages_from_a_zero_based_offset(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'lines.txt').write_text('one\ntwo\nthree\nfour\n')
    toolset = _toolset(tmp_path)

    first = await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'lines.txt', 'offset': 0, 'limit': 2})
    second = await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'lines.txt', 'offset': 2, 'limit': 2})

    # A partial window omits the hash: it would never match the whole-file hash writes verify.
    assert first == (
        '[lines.txt | lines 1-2]\n     1\tone\n     2\ttwo\n... (more lines. Use offset=2 to continue reading.)\n'
    )
    assert second == '[lines.txt | lines 3-4]\n     3\tthree\n     4\tfour\n'


async def test_read_file_stops_at_max_read_lines_without_an_explicit_limit(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('one\ntwo\nthree\n')

    result = await _call(_toolset(tmp_path, max_read_lines=2), _ctx(sandbox), 'read_file', {'path': 'a.txt'})

    assert result == (
        '[a.txt | lines 1-2]\n     1\tone\n     2\ttwo\n... (more lines. Use offset=2 to continue reading.)\n'
    )


async def test_read_file_reports_an_empty_file(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'empty.txt').write_text('')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': 'empty.txt'})

    assert result == f'[empty.txt | 0 lines | hash:{_hash("")}]\n(empty file)\n'


@pytest.mark.parametrize('path', ['lines.txt', 'empty.txt'])
async def test_read_file_rejects_an_offset_past_the_end_of_the_file(
    tmp_path: Path, sandbox: Sandbox, path: str
) -> None:
    (tmp_path / 'lines.txt').write_text('one\ntwo\n')
    (tmp_path / 'empty.txt').write_text('')

    with pytest.raises(ModelRetry, match='Offset 4 exceeds file length.'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': path, 'offset': 4, 'limit': 1})


@pytest.mark.parametrize(
    ('data', 'size'),
    [(b'hello\x00world', 11), (b'undecodable \xff\xfe\n', 15)],
    ids=['null-byte', 'undecodable'],
)
async def test_read_file_reports_binary_content_instead_of_text(
    tmp_path: Path, sandbox: Sandbox, data: bytes, size: int
) -> None:
    (tmp_path / 'binary.bin').write_bytes(data)

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': 'binary.bin'})

    assert result == f'[Binary file: {size} bytes. Use a binary-aware tool to inspect.]'


async def test_read_file_allows_a_symlink_alias_not_matching_a_denied_target_pattern(
    tmp_path: Path, sandbox: Sandbox
) -> None:
    (tmp_path / 'creds.secret').write_text('secret\n')
    (tmp_path / 'alias.txt').symlink_to(tmp_path / 'creds.secret')

    result = await _call(
        _toolset(tmp_path, denied_patterns=('*.secret',)), _ctx(sandbox), 'read_file', {'path': 'alias.txt'}
    )

    assert 'secret' in result


async def test_read_file_accepts_utf8_character_split_at_binary_sample_boundary(
    tmp_path: Path, sandbox: Sandbox
) -> None:
    (tmp_path / 'unicode.txt').write_bytes(('a' * 8191 + 'é' + 'more text').encode())

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': 'unicode.txt'})

    assert 'émore text' in result


@pytest.mark.parametrize(
    ('path', 'message'),
    [
        ('missing.txt', 'File not found: missing.txt'),
        ('directory', "'directory' is a directory, not a file."),
    ],
    ids=['missing', 'directory'],
)
async def test_read_file_reports_a_missing_path_or_a_directory(
    tmp_path: Path, sandbox: Sandbox, path: str, message: str
) -> None:
    (tmp_path / 'directory').mkdir()

    with pytest.raises(ModelRetry, match=re.escape(message)):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': path})


# --- write_file ---


async def test_write_file_writes_content_and_reports_its_hash(tmp_path: Path, sandbox: Sandbox) -> None:
    content = 'ab\ncd\n'

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'write_file', {'path': 'new.txt', 'content': content})

    assert result == f'Wrote 6 chars (2 lines) to new.txt. [hash:{_hash(content)}]'
    assert (tmp_path / 'new.txt').read_text() == content


async def test_write_file_overwrites_when_the_expected_hash_matches(tmp_path: Path, sandbox: Sandbox) -> None:
    before = 'before\n'
    (tmp_path / 'note.txt').write_text(before)

    await _call(
        _toolset(tmp_path),
        _ctx(sandbox),
        'write_file',
        {'path': 'note.txt', 'content': 'after\n', 'expected_hash': _hash(before)},
    )

    assert (tmp_path / 'note.txt').read_text() == 'after\n'


async def test_write_file_rejects_a_stale_expected_hash(tmp_path: Path, sandbox: Sandbox) -> None:
    before = 'before\n'
    (tmp_path / 'note.txt').write_text(before)

    message = f"Conflict: file 'note.txt' has changed (expected hash:stale, got hash:{_hash(before)})"
    with pytest.raises(ModelRetry, match=re.escape(message)):
        await _call(
            _toolset(tmp_path),
            _ctx(sandbox),
            'write_file',
            {'path': 'note.txt', 'content': 'after\n', 'expected_hash': 'stale'},
        )
    assert (tmp_path / 'note.txt').read_text() == before


async def test_write_file_ignores_the_expected_hash_for_a_new_file(tmp_path: Path, sandbox: Sandbox) -> None:
    await _call(
        _toolset(tmp_path),
        _ctx(sandbox),
        'write_file',
        {'path': 'new.txt', 'content': 'fresh\n', 'expected_hash': 'stale'},
    )

    assert (tmp_path / 'new.txt').read_text() == 'fresh\n'


@pytest.mark.parametrize(
    ('path', 'message'),
    [
        ('directory', "Path 'directory' exists and is not a regular file."),
        ('missing/child.txt', "Parent directory 'missing' does not exist. Use create_directory first."),
        ('parent.txt/child.txt', "Parent directory 'parent.txt' does not exist. Use create_directory first."),
    ],
    ids=['directory', 'missing-parent', 'file-as-parent'],
)
async def test_write_file_reports_a_target_it_cannot_write(
    tmp_path: Path, sandbox: Sandbox, path: str, message: str
) -> None:
    (tmp_path / 'directory').mkdir()
    (tmp_path / 'parent.txt').write_text('file')

    with pytest.raises(ModelRetry, match=re.escape(message)):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'write_file', {'path': path, 'content': 'content'})


# --- edit_file ---


async def test_edit_file_replaces_the_single_match_and_reports_the_new_hash(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'note.txt').write_text('before the end\n')
    after = 'after the end\n'

    result = await _call(
        _toolset(tmp_path),
        _ctx(sandbox),
        'edit_file',
        {'path': 'note.txt', 'old_text': 'before', 'new_text': 'after'},
    )

    assert result == f'Edited note.txt. [hash:{_hash(after)}]'
    assert (tmp_path / 'note.txt').read_text() == after


async def test_edit_file_rejects_a_stale_expected_hash(tmp_path: Path, sandbox: Sandbox) -> None:
    before = 'before\n'
    (tmp_path / 'note.txt').write_text(before)

    message = f"Conflict: file 'note.txt' has changed (expected hash:stale, got hash:{_hash(before)})"
    with pytest.raises(ModelRetry, match=re.escape(message)):
        await _call(
            _toolset(tmp_path),
            _ctx(sandbox),
            'edit_file',
            {'path': 'note.txt', 'old_text': 'before', 'new_text': 'after', 'expected_hash': 'stale'},
        )
    assert (tmp_path / 'note.txt').read_text() == before


@pytest.mark.parametrize(
    ('path', 'old_text', 'message'),
    [
        ('missing.txt', 'x', 'File not found: missing.txt'),
        ('once.txt', 'missing', 'old_text not found in once.txt.'),
        (
            'twice.txt',
            'same',
            'old_text found 2 times in twice.txt. Include more surrounding context to make the match unique.',
        ),
    ],
    ids=['missing-file', 'no-match', 'ambiguous-match'],
)
async def test_edit_file_reports_an_edit_it_cannot_apply(
    tmp_path: Path, sandbox: Sandbox, path: str, old_text: str, message: str
) -> None:
    (tmp_path / 'once.txt').write_text('once')
    (tmp_path / 'twice.txt').write_text('same same')

    with pytest.raises(ModelRetry, match=re.escape(message)):
        await _call(
            _toolset(tmp_path),
            _ctx(sandbox),
            'edit_file',
            {'path': path, 'old_text': old_text, 'new_text': 'new'},
        )


# --- list_directory ---


async def test_list_directory_reports_sizes_and_marks_directories(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('hello\n')
    (tmp_path / 'sub').mkdir()

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'list_directory', {'path': '.'})

    assert result == 'a.txt  (6 bytes)\nsub/'


async def test_list_directory_reports_entries_relative_to_the_root(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.txt').write_text('hello\n')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'list_directory', {'path': 'sub'})

    assert result == 'sub/nested.txt  (6 bytes)'


async def test_list_directory_reports_an_empty_directory(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'empty').mkdir()

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'list_directory', {'path': 'empty'})

    assert result == '(empty directory)'


async def test_list_directory_truncates_at_the_result_cap(tmp_path: Path, sandbox: Sandbox) -> None:
    for name in ('one.txt', 'three.txt', 'two.txt'):
        (tmp_path / name).write_text('x')

    result = await _call(_toolset(tmp_path, max_list_results=2), _ctx(sandbox), 'list_directory', {'path': '.'})

    assert result.splitlines() == ['one.txt  (1 bytes)', 'three.txt  (1 bytes)', '[... truncated at 2 entries]']


async def test_list_directory_at_the_result_cap_is_not_marked_truncated(tmp_path: Path, sandbox: Sandbox) -> None:
    for name in ('one.txt', 'two.txt'):
        (tmp_path / name).write_text('x')

    result = await _call(_toolset(tmp_path, max_list_results=2), _ctx(sandbox), 'list_directory', {'path': '.'})

    assert result.splitlines() == ['one.txt  (1 bytes)', 'two.txt  (1 bytes)']


@pytest.mark.parametrize('path', ['missing', 'file.txt'], ids=['missing', 'file'])
async def test_list_directory_rejects_a_path_that_is_not_a_directory(
    tmp_path: Path, sandbox: Sandbox, path: str
) -> None:
    (tmp_path / 'file.txt').write_text('file')

    with pytest.raises(ModelRetry, match=f'Not a directory: {path}'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'list_directory', {'path': path})


# --- search_files ---


async def test_search_files_reports_every_match_below_the_search_root(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('needle here\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.txt').write_text('other\nneedle there\n')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'search_files', {'pattern': 'needle'})

    assert sorted(result.splitlines()) == ['a.txt:1:needle here', 'sub/nested.txt:2:needle there']


async def test_search_files_searches_only_below_the_given_path(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('needle here\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.txt').write_text('needle there\n')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'search_files', {'pattern': 'needle', 'path': 'sub'})

    assert result == 'sub/nested.txt:1:needle there'


async def test_search_files_reports_no_matches(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('nothing here\n')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'search_files', {'pattern': 'needle'})

    assert result == 'No matches found.'


async def test_search_files_keeps_only_files_matching_the_include_glob(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'main.py').write_text('needle\n')
    (tmp_path / 'notes.md').write_text('needle\n')

    result = await _call(
        _toolset(tmp_path), _ctx(sandbox), 'search_files', {'pattern': 'needle', 'include_glob': '*.py'}
    )

    assert result == 'main.py:1:needle'


async def test_search_files_skips_binary_files(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('needle\n')
    (tmp_path / 'binary.bin').write_bytes(b'needle\x00\n')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'search_files', {'pattern': 'needle'})

    assert result == 'a.txt:1:needle'


async def test_search_files_truncates_at_the_result_cap(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'many.txt').write_text('needle\n' * 10)

    result = await _call(_toolset(tmp_path, max_search_results=2), _ctx(sandbox), 'search_files', {'pattern': 'needle'})

    assert result.splitlines() == ['many.txt:1:needle', 'many.txt:2:needle', '[... truncated at 2 matches]']


async def test_search_files_at_the_result_cap_is_not_marked_truncated(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'many.txt').write_text('needle\nneedle\n')

    result = await _call(_toolset(tmp_path, max_search_results=2), _ctx(sandbox), 'search_files', {'pattern': 'needle'})

    assert result.splitlines() == ['many.txt:1:needle', 'many.txt:2:needle']


async def test_search_files_ignores_output_lines_it_cannot_parse(tmp_path: Path) -> None:
    async with LocalSandbox(root=tmp_path) as local:
        sandbox = Sandbox.wrap(_ResultBackend(local, CommandResult(exit_code=1, stdout='malformed\n', stderr='')))
        result = await _call(_toolset(Path('.')), _ctx(sandbox), 'search_files', {'pattern': 'needle'})

    assert result == 'No matches found.'


# --- find_files ---


@pytest.mark.parametrize(
    ('pattern', 'expected'),
    [('*.txt', 'a.txt'), ('**/*.py', 'sub/nested.py'), ('sub/*.py', 'sub/nested.py')],
    ids=['one-level', 'recursive', 'path-anchored'],
)
async def test_find_files_matches_each_pattern_shape(
    tmp_path: Path, sandbox: Sandbox, pattern: str, expected: str
) -> None:
    (tmp_path / 'a.txt').write_text('')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.py').write_text('')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'find_files', {'pattern': pattern})

    assert result == expected


async def test_find_files_marks_directories_with_a_trailing_slash(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'sub').mkdir()

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'find_files', {'pattern': 'sub'})

    assert result == 'sub/'


async def test_find_files_searches_only_below_the_given_path(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.py').write_text('')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.py').write_text('')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'find_files', {'pattern': '*.py', 'path': 'sub'})

    assert result == 'sub/nested.py'


async def test_find_files_reports_no_matches(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'a.txt').write_text('')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'find_files', {'pattern': '*.py'})

    assert result == 'No matches found.'


async def test_find_files_rejects_an_absolute_pattern(tmp_path: Path, sandbox: Sandbox) -> None:
    with pytest.raises(ModelRetry, match=re.escape("Pattern '/etc/*' must be relative to the search path")):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'find_files', {'pattern': '/etc/*'})


@pytest.mark.parametrize('path', ['missing', 'file.txt'], ids=['missing', 'file'])
async def test_find_files_rejects_a_path_that_is_not_a_directory(tmp_path: Path, sandbox: Sandbox, path: str) -> None:
    (tmp_path / 'file.txt').write_text('file')

    with pytest.raises(ModelRetry, match=f'Not a directory: {path}'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'find_files', {'pattern': '*', 'path': path})


async def test_find_files_truncates_at_the_result_cap(tmp_path: Path, sandbox: Sandbox) -> None:
    for name in ('one.py', 'three.py', 'two.py'):
        (tmp_path / name).write_text('')

    result = await _call(_toolset(tmp_path, max_find_results=2), _ctx(sandbox), 'find_files', {'pattern': '*.py'})

    assert result.splitlines() == ['one.py', 'three.py', '[... truncated at 2 matches]']


async def test_find_files_at_the_result_cap_is_not_marked_truncated(tmp_path: Path, sandbox: Sandbox) -> None:
    for name in ('one.py', 'two.py'):
        (tmp_path / name).write_text('')

    result = await _call(_toolset(tmp_path, max_find_results=2), _ctx(sandbox), 'find_files', {'pattern': '*.py'})

    assert result.splitlines() == ['one.py', 'two.py']


async def test_find_files_skips_entries_deleted_mid_walk(tmp_path: Path) -> None:
    (tmp_path / 'real.py').write_text('')
    async with LocalSandbox(root=tmp_path) as local:
        listing = CommandResult(exit_code=0, stdout=f'{tmp_path}/ghost.py\n{tmp_path}/real.py\n', stderr='')
        sandbox = Sandbox.wrap(_ResultBackend(local, listing))
        result = await _call(_toolset(Path('.')), _ctx(sandbox), 'find_files', {'pattern': '*.py'})

    assert result == 'real.py'


# --- create_directory ---


async def test_create_directory_creates_missing_parents(tmp_path: Path, sandbox: Sandbox) -> None:
    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'create_directory', {'path': 'deep/nested'})

    assert result == 'Created directory: deep/nested'
    assert (tmp_path / 'deep' / 'nested').is_dir()


async def test_create_directory_accepts_a_directory_that_already_exists(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'existing').mkdir()

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'create_directory', {'path': 'existing'})

    assert result == 'Created directory: existing'


@pytest.mark.parametrize(
    ('path', 'message'),
    [
        ('file.txt', f"[Errno {errno.EEXIST}] File exists: 'file.txt'"),
        ('file.txt/nested', f"[Errno {errno.ENOTDIR}] Not a directory: 'file.txt/nested'"),
    ],
    ids=['over-a-file', 'under-a-file'],
)
async def test_create_directory_reports_a_path_blocked_by_a_file(
    tmp_path: Path, sandbox: Sandbox, path: str, message: str
) -> None:
    (tmp_path / 'file.txt').write_text('file')

    with pytest.raises(ModelRetry, match=re.escape(message)):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'create_directory', {'path': path})


# --- file_info ---


async def test_file_info_reports_size_lines_and_the_file_hash(tmp_path: Path, sandbox: Sandbox) -> None:
    content = 'first\nsecond\n'
    (tmp_path / 'info.txt').write_text(content)

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'file_info', {'path': 'info.txt'})

    assert result == f'path: info.txt\ntype: file\nsize: 13 bytes\nbinary: False\nlines: 2\nhash: {_hash(content)}'


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('directory', 'path: directory\ntype: directory\nsize: 0 bytes'),
        ('binary.bin', 'path: binary.bin\ntype: file\nsize: 12 bytes\nbinary: True'),
    ],
    ids=['directory', 'binary'],
)
async def test_file_info_omits_text_metadata_for_a_directory_or_binary_file(
    tmp_path: Path, sandbox: Sandbox, path: str, expected: str
) -> None:
    (tmp_path / 'directory').mkdir()
    (tmp_path / 'binary.bin').write_bytes(b'not utf-8: \xff')

    assert await _call(_toolset(tmp_path), _ctx(sandbox), 'file_info', {'path': path}) == expected


@pytest.mark.parametrize(('padding', 'binary'), [(8191, True), (8192, False)], ids=['inside', 'past'])
async def test_file_info_looks_for_null_bytes_only_in_the_first_8kb(
    tmp_path: Path, sandbox: Sandbox, padding: int, binary: bool
) -> None:
    (tmp_path / 'padded.bin').write_bytes(b'x' * padding + b'\x00')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), 'file_info', {'path': 'padded.bin'})

    assert f'binary: {binary}' in result


async def test_file_info_reports_a_missing_path(tmp_path: Path, sandbox: Sandbox) -> None:
    with pytest.raises(ModelRetry, match='Path not found: missing'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'file_info', {'path': 'missing'})


# --- path containment and access patterns ---


@pytest.mark.parametrize(
    'path',
    ['../outside.txt', '/etc/passwd', 'sub/../../outside.txt'],
    ids=['parent', 'absolute', 'traversal'],
)
async def test_paths_resolving_outside_the_root_are_rejected(tmp_path: Path, sandbox: Sandbox, path: str) -> None:
    (tmp_path / 'sub').mkdir()

    with pytest.raises(ModelRetry, match='resolves outside the root directory'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': path})


async def test_denied_patterns_reject_only_matching_paths(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'creds.secret').write_text('hunter2\n')
    (tmp_path / 'notes.txt').write_text('notes\n')
    toolset = _toolset(tmp_path, denied_patterns=('*.secret',))

    assert 'notes' in await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'notes.txt'})
    with pytest.raises(ModelRetry, match=re.escape("Path 'creds.secret' is denied by pattern '*.secret'.")):
        await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'creds.secret'})


async def test_allowed_patterns_reject_paths_outside_the_allow_list(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'main.py').write_text('code\n')
    (tmp_path / 'notes.md').write_text('notes\n')
    toolset = _toolset(tmp_path, allowed_patterns=('*.py',))

    assert 'code' in await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'main.py'})
    with pytest.raises(ModelRetry, match=re.escape("Path 'notes.md' does not match any allowed pattern.")):
        await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'notes.md'})


@pytest.mark.parametrize(
    ('name', 'args'),
    [
        ('write_file', {'path': '.env', 'content': 'HACKED=1\n'}),
        ('edit_file', {'path': '.env', 'old_text': 'SECRET', 'new_text': 'HACKED'}),
        ('create_directory', {'path': '.env/nested'}),
    ],
    ids=['write_file', 'edit_file', 'create_directory'],
)
async def test_protected_patterns_reject_every_writing_tool(
    tmp_path: Path, sandbox: Sandbox, name: str, args: dict[str, object]
) -> None:
    (tmp_path / '.env').write_text('SECRET=abc\n')
    toolset = _toolset(tmp_path, protected_patterns=('.env', '.env/*'))

    with pytest.raises(ModelRetry, match='is protected'):
        await _call(toolset, _ctx(sandbox), name, args)


async def test_protected_paths_stay_readable(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / '.env').write_text('SECRET=abc\n')

    result = await _call(_toolset(tmp_path, protected_patterns=('.env',)), _ctx(sandbox), 'read_file', {'path': '.env'})

    assert 'SECRET=abc' in result


async def test_protected_patterns_leave_every_other_path_writable(tmp_path: Path, sandbox: Sandbox) -> None:
    await _call(
        _toolset(tmp_path, protected_patterns=('.env',)),
        _ctx(sandbox),
        'write_file',
        {'path': 'notes.txt', 'content': 'notes\n'},
    )

    assert (tmp_path / 'notes.txt').read_text() == 'notes\n'


async def test_patterns_match_the_canonical_path(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'secret.txt').write_text('token\n')
    toolset = _toolset(tmp_path, denied_patterns=('config/secret.txt',))

    with pytest.raises(ModelRetry, match='is denied by pattern'):
        await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'config/./secret.txt'})


async def test_a_leading_double_star_pattern_also_matches_at_the_root(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'secrets.yaml').write_text('api: key\n')
    toolset = _toolset(tmp_path, protected_patterns=('**/secrets*',))

    with pytest.raises(ModelRetry, match='is protected'):
        await _call(toolset, _ctx(sandbox), 'write_file', {'path': 'secrets.yaml', 'content': 'changed\n'})


_WALKERS = [
    ('list_directory', {'path': '.'}, 'visible.txt  (7 bytes)'),
    ('search_files', {'pattern': 'marker'}, 'visible.txt:1:marker'),
    ('find_files', {'pattern': '*'}, 'visible.txt'),
]


@pytest.mark.parametrize(('name', 'args', 'expected'), _WALKERS, ids=['list', 'search', 'find'])
async def test_walkers_hide_denied_entries(
    tmp_path: Path, sandbox: Sandbox, name: str, args: dict[str, object], expected: str
) -> None:
    (tmp_path / 'visible.txt').write_text('marker\n')
    (tmp_path / 'creds.secret').write_text('marker\n')

    result = await _call(_toolset(tmp_path, denied_patterns=('*.secret',)), _ctx(sandbox), name, args)

    assert result == expected


@pytest.mark.parametrize(('name', 'args', 'expected'), _WALKERS, ids=['list', 'search', 'find'])
async def test_walkers_show_only_allowed_entries(
    tmp_path: Path, sandbox: Sandbox, name: str, args: dict[str, object], expected: str
) -> None:
    # The walk root ('.') is not required to match a file-shaped allowed pattern;
    # only the entries it yields are filtered against it.
    (tmp_path / 'visible.txt').write_text('marker\n')
    (tmp_path / 'skipped.md').write_text('marker\n')

    result = await _call(_toolset(tmp_path, allowed_patterns=('*.txt',)), _ctx(sandbox), name, args)

    assert result == expected


@pytest.mark.parametrize(('name', 'args', 'expected'), _WALKERS, ids=['list', 'search', 'find'])
async def test_walkers_show_protected_entries(
    tmp_path: Path, sandbox: Sandbox, name: str, args: dict[str, object], expected: str
) -> None:
    (tmp_path / 'visible.txt').write_text('marker\n')

    result = await _call(_toolset(tmp_path, protected_patterns=('*',)), _ctx(sandbox), name, args)

    assert result == expected


@pytest.mark.parametrize(('name', 'args', 'expected'), _WALKERS, ids=['list', 'search', 'find'])
async def test_walkers_hide_dotfiles(
    tmp_path: Path, sandbox: Sandbox, name: str, args: dict[str, object], expected: str
) -> None:
    (tmp_path / 'visible.txt').write_text('marker\n')
    (tmp_path / '.hidden.txt').write_text('marker\n')

    result = await _call(_toolset(tmp_path), _ctx(sandbox), name, args)

    assert result == expected


# --- error mapping ---


async def test_tools_report_an_unattached_sandbox_to_the_user(tmp_path: Path) -> None:
    with pytest.raises(UserError, match='No sandbox is attached'):
        await _call(_toolset(tmp_path), _ctx(), 'read_file', {'path': 'a.txt'})


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (SandboxError('temporary failure'), ModelRetry),
        (SandboxUnavailableError('gone'), SandboxUnavailableError),
        (RuntimeError('backend bug'), RuntimeError),
    ],
    ids=['sandbox-error', 'unavailable', 'runtime-error'],
)
@pytest.mark.parametrize(
    ('name', 'args'),
    [
        ('read_file', {'path': 'file.txt'}),
        ('write_file', {'path': 'file.txt', 'content': 'x'}),
        ('edit_file', {'path': 'file.txt', 'old_text': 'a', 'new_text': 'b'}),
        ('list_directory', {'path': '.'}),
        ('search_files', {'pattern': 'x'}),
        ('find_files', {'pattern': '*.py'}),
        ('create_directory', {'path': 'sub'}),
        ('file_info', {'path': 'file.txt'}),
    ],
)
async def test_backend_failures_are_recoverable_except_the_terminal_ones(
    error: Exception,
    expected: type[Exception],
    name: str,
    args: dict[str, object],
) -> None:
    with pytest.raises(expected, match=str(error)):
        await _call(_toolset(Path('.')), _ctx(Sandbox.wrap(_ErrorBackend(error))), name, args)


@pytest.mark.parametrize(
    ('name', 'args', 'exit_code', 'stderr', 'message'),
    [
        ('search_files', {'pattern': 'x'}, 2, 'grep failed', 'grep failed'),
        ('search_files', {'pattern': 'x'}, 2, '', 'grep exited with code 2.'),
        ('find_files', {'pattern': '*'}, 1, 'find failed', 'find failed'),
        ('find_files', {'pattern': '*'}, 1, '', 'find exited with code 1.'),
    ],
    ids=['grep-stderr', 'grep-silent', 'find-stderr', 'find-silent'],
)
async def test_a_failing_command_is_reported_to_the_model(
    tmp_path: Path,
    name: str,
    args: dict[str, object],
    exit_code: int,
    stderr: str,
    message: str,
) -> None:
    async with LocalSandbox(root=tmp_path) as local:
        sandbox = Sandbox.wrap(_ResultBackend(local, CommandResult(exit_code=exit_code, stdout='', stderr=stderr)))
        with pytest.raises(ModelRetry, match=re.escape(message)):
            await _call(_toolset(Path('.')), _ctx(sandbox), name, args)


@pytest.mark.parametrize(
    ('name', 'args'),
    [('search_files', {'pattern': 'x'}), ('find_files', {'pattern': '*'})],
)
async def test_command_backed_search_timeout_is_recoverable(name: str, args: dict[str, object]) -> None:
    error = SandboxTimeoutError('command timed out at /outside/root')

    async with LocalSandbox() as local:
        sandbox = Sandbox.wrap(_TimeoutBackend(local, error))
        with pytest.raises(ModelRetry, match=f'{name} timed out') as exc_info:
            await _call(_toolset(Path('.')), _ctx(sandbox), name, args)

    assert '/outside/root' not in str(exc_info.value)


async def test_a_path_name_that_is_too_long_is_recoverable(tmp_path: Path, sandbox: Sandbox) -> None:
    with pytest.raises(ModelRetry, match='The path name is too long.'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': 'x' * 300})


async def test_a_symlink_loop_is_recoverable(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'loop').symlink_to(tmp_path / 'loop')

    with pytest.raises(ModelRetry, match='The path resolves through a symlink loop.'):
        await _call(_toolset(tmp_path), _ctx(sandbox), 'read_file', {'path': 'loop'})


@pytest.mark.parametrize(
    ('error', 'expected', 'message'),
    [
        (
            OSError(errno.EILSEQ, 'Illegal byte sequence'),
            ModelRetry,
            'The path name contains a byte sequence the filesystem cannot represent.',
        ),
        (OSError(errno.EINVAL, 'Invalid argument'), OSError, 'Invalid argument'),
        (OSError(errno.ENOSPC, 'No space left on device'), OSError, 'No space left on device'),
    ],
    ids=['illegal-bytes', 'other-errno', 'no-space'],
)
async def test_os_errors_are_recoverable_only_when_the_model_can_correct_them(
    error: OSError, expected: type[Exception], message: str
) -> None:
    with pytest.raises(expected, match=re.escape(message)):
        await _call(_toolset(Path('.')), _ctx(Sandbox.wrap(_ErrorBackend(error))), 'create_directory', {'path': 'sub'})


def _permission_error(filename: object) -> PermissionError:
    error = PermissionError(errno.EACCES, 'Permission denied')
    error.filename = filename
    return error


@pytest.mark.parametrize(
    ('filename', 'expected'),
    [
        (None, f'[Errno {errno.EACCES}] Permission denied'),
        ('a.txt', f"[Errno {errno.EACCES}] Permission denied: 'a.txt'"),
        ('/work/a.txt', f"[Errno {errno.EACCES}] Permission denied: 'a.txt'"),
        ('/etc/shadow', f"[Errno {errno.EACCES}] Permission denied: '<outside-workspace>'"),
        (object(), f"[Errno {errno.EACCES}] Permission denied: '<not-a-path>'"),
    ],
    ids=['no-filename', 'relative', 'inside-the-root', 'outside-the-root', 'not-a-path'],
)
async def test_error_messages_never_expose_a_path_outside_the_root(filename: object, expected: str) -> None:
    sandbox = Sandbox.wrap(_ErrorBackend(_permission_error(filename)))

    with pytest.raises(ModelRetry, match=re.escape(expected)):
        await _call(_toolset(Path('.')), _ctx(sandbox), 'file_info', {'path': 'a.txt'})


# --- the FileSystem capability ---


def test_the_capability_defaults_to_the_sandbox_working_directory() -> None:
    capability = FileSystem[None]()

    assert capability.root_dir == '.'
    assert capability.max_read_lines == 2000
    assert capability.max_list_results == 1000
    assert capability.max_search_results == 1000
    assert capability.max_find_results == 1000


def test_the_capability_protects_secrets_and_vcs_metadata_by_default() -> None:
    assert list(FileSystem[None]().protected_patterns) == [
        '.git/*',
        '.env',
        '.env.*',
        '*.pem',
        '*.key',
        '**/secrets*',
    ]


_LIMITS: list[tuple[Callable[[int], FileSystem[None]], str]] = [
    (lambda value: FileSystem[None](max_read_lines=value), 'max_read_lines'),
    (lambda value: FileSystem[None](max_list_results=value), 'max_list_results'),
    (lambda value: FileSystem[None](max_search_results=value), 'max_search_results'),
    (lambda value: FileSystem[None](max_find_results=value), 'max_find_results'),
]


@pytest.mark.parametrize(
    ('build', 'field'), _LIMITS, ids=['read_lines', 'list_results', 'search_results', 'find_results']
)
@pytest.mark.parametrize('value', [0, -1])
def test_the_capability_requires_positive_limits(
    build: Callable[[int], FileSystem[None]], field: str, value: int
) -> None:
    with pytest.raises(ValueError, match=f'{field} must be a positive integer, got {value!r}'):
        build(value)


def test_the_capability_rejects_a_limit_that_is_not_an_integer() -> None:
    # Dataclass annotations are advisory, so a string from a config file must be rejected here.
    with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
        FileSystem[None](max_read_lines='1000')  # type: ignore[arg-type]


async def test_the_capability_configuration_reaches_the_tools(tmp_path: Path, sandbox: Sandbox) -> None:
    (tmp_path / 'creds.secret').write_text('hunter2\n')
    toolset = FileSystem[None](root_dir=str(tmp_path), denied_patterns=['*.secret']).get_toolset()
    assert isinstance(toolset, FileSystemToolset)

    with pytest.raises(ModelRetry, match='is denied by pattern'):
        await _call(toolset, _ctx(sandbox), 'read_file', {'path': 'creds.secret'})


async def test_a_read_only_capability_exposes_exactly_the_read_only_tools(tmp_path: Path) -> None:
    tools = await FileSystem[None](root_dir=tmp_path, read_only=True).get_toolset().get_tools(_ctx())

    assert set(tools) == READ_ONLY_TOOL_NAMES


async def test_filesystem_capability_runs_through_an_agent_with_a_sandbox(tmp_path: Path) -> None:
    (tmp_path / 'note.txt').write_text('hello\n')
    responses = [
        ModelResponse(parts=[ToolCallPart('read_file', {'path': 'note.txt'})]),
        ModelResponse(parts=[TextPart('done')]),
    ]

    def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return responses.pop(0)

    async with LocalSandbox(root=tmp_path) as sandbox:
        result = await Agent(FunctionModel(model), capabilities=[FileSystem()]).run('read', sandbox=sandbox)

    assert result.output == 'done'


async def test_filesystem_capability_requires_a_sandbox_on_the_public_agent_path() -> None:
    with pytest.raises(UserError, match='No sandbox is attached'):
        await Agent(TestModel(call_tools=['read_file']), capabilities=[FileSystem()]).run('read')

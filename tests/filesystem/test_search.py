"""Search on command-capable workspaces without ripgrep."""

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import CommandResult, LocalWorkspaceBackend, Workspace, WorkspaceCommand

from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset

pytestmark = pytest.mark.anyio


class CountingBackend(LocalWorkspaceBackend):
    def __init__(self, root: Path) -> None:
        super().__init__(root, env={'PATH': '/usr/bin:/bin'})
        self.commands = 0
        self.reads = 0

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self.commands += 1
        return await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def read_bytes(self, path: str) -> bytes:
        self.reads += 1
        return await super().read_bytes(path)


async def test_no_rg_uses_one_command_and_preserves_ignores(tmp_path: Path) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'visible.py').write_text('needle\n')
    (tmp_path / 'ignored').mkdir()
    (tmp_path / 'ignored' / 'secret.py').write_text('needle\n')
    (tmp_path / '.gitignore').write_text('ignored/\n')
    (tmp_path / '.hidden.py').write_text('needle\n')
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    backend = CountingBackend(tmp_path)
    workspace = Workspace(backend)
    tools = FileSystem[None](tools=['grep', 'list_files']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.grep('needle', workspace=workspace) == 'src/visible.py:1:needle'
    assert backend.commands == 2  # the first call discovers that rg is missing
    assert backend.reads == 0
    assert await tools.list_files(glob='*.py', workspace=workspace) == 'src/visible.py'
    assert backend.commands == 3
    assert backend.reads == 0


async def test_no_rg_search_files_is_one_command_and_confines_symlinks(tmp_path: Path) -> None:
    (tmp_path / 'safe.txt').write_text('needle\n')
    (tmp_path / 'secret.txt').write_text('needle\n')
    (tmp_path / 'alias.txt').symlink_to('secret.txt')
    outside = tmp_path.parent / f'{tmp_path.name}-outside'
    outside.write_text('needle\n')
    try:
        (tmp_path / 'escape.txt').symlink_to(outside)
        backend = CountingBackend(tmp_path)
        tools = FileSystem[None](root_dir=tmp_path, denied_patterns=['secret.txt']).get_toolset()
        assert isinstance(tools, FileSystemToolset)
        workspace = Workspace(backend)
        await tools.grep('absent', workspace=workspace)  # Discover the missing sandbox rg.
        backend.commands = 0
        result = await tools.search_files('needle', workspace=workspace)
        assert 'safe.txt:1:needle' in result
        assert 'secret.txt' not in result
        assert 'alias.txt' not in result
        assert 'escape.txt' not in result
        assert backend.commands <= 2
        assert backend.reads == 0
    finally:
        outside.unlink()


async def test_first_search_files_uses_command_not_walker(tmp_path: Path) -> None:
    (tmp_path / 'visible.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path)
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.search_files('needle', workspace=backend) == 'visible.txt:1:needle'
    assert backend.commands == 2  # probe rg, then search in the workspace
    assert backend.reads == 0


async def test_no_rg_ignore_and_failure_are_not_silent(tmp_path: Path) -> None:
    (tmp_path / 'visible.txt').write_text('needle\n')
    (tmp_path / 'hidden.txt').write_text('needle\n')
    (tmp_path / '.ignore').write_text('hidden.txt\n')
    backend = CountingBackend(tmp_path)
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    workspace = Workspace(backend)
    await tools.grep('absent', workspace=workspace)
    assert 'hidden.txt' not in await tools.search_files('needle', workspace=workspace)
    assert backend.reads == 0


async def test_explicit_hidden_file_glob_and_omission_count(tmp_path: Path) -> None:
    (tmp_path / '.hidden.py').write_text('needle\n')
    (tmp_path / '.another.py').write_text('needle\n')
    (tmp_path / 'visible.py').write_text('needle\n')
    backend = LocalWorkspaceBackend(tmp_path)
    tools = FileSystem[None](root_dir=tmp_path, tools=['list_files']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    listing = await tools.list_directory(workspace=backend)
    assert '2 hidden' in listing
    assert '.hidden.py' in await tools.find_files('.hidden.py', workspace=backend)
    assert '.hidden.py' in await tools.search_files('needle', include_glob='.hidden.py', workspace=backend)
    assert '.hidden.py' in await tools.list_files(glob='.hidden.py', workspace=backend)


async def test_nested_gitignore_on_posix_search(tmp_path: Path) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / '.gitignore').write_text('ignored.txt\n')
    (tmp_path / 'src' / 'ignored.txt').write_text('needle\n')
    (tmp_path / 'src' / 'visible.txt').write_text('needle\n')
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    backend = CountingBackend(tmp_path)
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.search_files('needle', workspace=backend) == 'src/visible.txt:1:needle'
    assert backend.reads == 0


async def test_no_rg_rejects_unsupported_regex(tmp_path: Path) -> None:
    (tmp_path / 'file.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path)
    tools = FileSystem[None](root_dir=tmp_path, tools=['grep']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    with pytest.raises((ModelRetry, ValueError), match='ripgrep|POSIX|unsupported'):
        await tools.grep(r'\d+', workspace=backend)

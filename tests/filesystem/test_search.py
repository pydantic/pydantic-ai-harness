"""Search on command-capable workspaces without ripgrep."""

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
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

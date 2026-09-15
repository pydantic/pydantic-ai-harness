"""The six tools exposed by Coder."""

from __future__ import annotations

import re
import subprocess
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal

import anyio
from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.coder._shell import shell
from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset
from pydantic_ai_harness.filesystem._toolset import _read_canonical_text  # pyright: ignore[reportPrivateUsage]


@dataclass
class Replacement:
    """One exact, unique replacement in an edit batch."""

    _: KW_ONLY
    old_text: str
    new_text: str


class CoderToolset(FunctionToolset[AgentDepsT]):
    """Focused local coding tools. Shell access requires a trusted workspace."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(id='coder')
        self.workspace = workspace.resolve()
        filesystem = FileSystem[AgentDepsT](self.workspace).get_toolset()
        assert isinstance(filesystem, FileSystemToolset)
        self.filesystem = filesystem
        self.add_function(self.read_file)
        self.add_function(self.write_file)
        self.add_function(self.edit_file)
        self.add_function(self.list_files)
        self.add_function(self.grep)
        self.add_function(self.shell)

    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a file with zero-based offset and one-based line numbers, without hashes."""
        result = await self.filesystem.read_file(path, offset=offset, limit=limit)
        return re.sub(r' \| hash:[0-9a-f]+(?=\])', '', result, count=1)

    async def write_file(self, path: str, content: str) -> str:
        """Write a complete file. Create missing parent directories with shell mkdir first."""
        result = await self.filesystem.write_file(path, content)
        return re.sub(r' \[hash:[0-9a-f]+\]', '', result)

    async def edit_file(
        self,
        path: str,
        *,
        old_text: str | None = None,
        new_text: str | None = None,
        replacements: list[Replacement] | None = None,
    ) -> str:
        """Apply one exact replacement or a sequential batch; each match must occur once.

        All replacements are checked in memory before the file is changed.
        """
        if replacements is None:
            if old_text is None or new_text is None:
                raise ModelRetry('Provide old_text and new_text, or a non-empty replacements list.')
            replacements = [Replacement(old_text=old_text, new_text=new_text)]
        elif old_text is not None or new_text is not None or not replacements:
            raise ModelRetry('Use either old_text/new_text or a non-empty replacements list, not both.')
        resolved = self.filesystem._safe_resolve(path, write=True)  # pyright: ignore[reportPrivateUsage]
        try:
            original = _read_canonical_text(resolved)
        except (OSError, UnicodeError) as exc:
            raise ModelRetry(f'Cannot read {path!r}: {exc}') from exc
        content = original
        for replacement in replacements:
            if not replacement.old_text or content.count(replacement.old_text) != 1:
                raise ModelRetry('Each non-empty old_text must occur exactly once; no changes were written.')
            content = content.replace(replacement.old_text, replacement.new_text, 1)
        result = await self.filesystem.edit_file(path, original, content)
        return re.sub(r' \[hash:[0-9a-f]+\]', '', result)

    def _directory(self, path: str) -> Path:
        directory = (self.workspace / path).resolve()
        if not directory.is_relative_to(self.workspace) or not directory.is_dir():
            raise ModelRetry('path must be an existing directory inside the workspace.')
        return directory

    async def _rg(self, arguments: list[str], *, path: str, limit: int) -> str:
        if not 1 <= limit <= 1000:
            raise ModelRetry('limit must be between 1 and 1000.')
        try:
            async with await anyio.open_process(
                ['rg', *arguments], cwd=self._directory(path), stderr=subprocess.DEVNULL
            ) as process:
                assert process.stdout is not None
                output = bytearray()
                truncated = False
                async for chunk in process.stdout:
                    output.extend(chunk)
                    if output.count(b'\n') > limit or len(output) > 64000:
                        truncated = True
                        process.terminate()
                        break
                await process.wait()
                if process.returncode not in (0, 1) and not truncated:
                    raise ModelRetry('ripgrep failed; check the pattern and filters.')
        except FileNotFoundError as exc:
            raise ModelRetry('Install ripgrep (rg) to use list_files and grep.') from exc
        lines = output.decode('utf-8', errors='replace').splitlines()
        result = '\n'.join(lines[:limit])[:64000]
        return result + ('\n[truncated; narrow the search]' if truncated else '')

    async def list_files(self, path: str = '.', *, glob: str | None = None, limit: int = 200) -> str:
        """List files with rg --files, respecting ignore rules and optional glob filtering."""
        arguments = ['--files', '--color=never']
        if glob is not None:
            arguments.extend(['--glob', glob])
        return await self._rg(arguments, path=path, limit=limit)

    async def grep(
        self,
        pattern: str,
        *,
        path: str = '.',
        glob: str | None = None,
        file_type: str | None = None,
        ignore_case: bool = False,
        literal: bool = False,
        context: int = 0,
        limit: int = 200,
    ) -> str:
        """Search with ripgrep, bounding returned lines including optional context."""
        if not 0 <= context <= 20:
            raise ModelRetry('context must be between 0 and 20.')
        arguments = ['--line-number', '--with-filename', '--color=never', '--context', str(context)]
        if glob is not None:
            arguments.extend(['--glob', glob])
        if file_type is not None:
            arguments.extend(['--type', file_type])
        if ignore_case:
            arguments.append('--ignore-case')
        if literal:
            arguments.append('--fixed-strings')
        arguments.extend(['--regexp', pattern, '--', '.'])
        return await self._rg(arguments, path=path, limit=limit)

    async def shell(
        self,
        command: str,
        *,
        mode: Literal['foreground', 'background'] = 'foreground',
        timeout: float = 270,
    ) -> str:
        """Run unrestricted commands. Foreground promotes after timeout (maximum 270s).

        Returns PID, output log and status file paths. Background processes survive
        agent runs; use shell to inspect logs/status and kill processes when done.
        """
        return await shell(self.workspace, command, mode=mode, timeout=timeout)

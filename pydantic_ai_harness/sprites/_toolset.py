"""Fly.io Sprites toolset: gives agents a persistent cloud computer to work in."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness._sandbox_output import render_file_window, truncate, truncate_output
from pydantic_ai_harness.sprites._session import (
    SpriteSandboxError,
    SpriteSandboxSession,
    SpriteSandboxTerminalError,
)


def _strictly_bound_output(
    text: str,
    *,
    max_lines: int,
    max_bytes: int,
    direction: Literal['head', 'tail'],
) -> str:
    """Apply a final marker-inclusive cap without adding more presentation text."""
    lines = text.split('\n')
    if len(lines) > 1 and lines[-1] == '':
        lines = lines[:-1]
    result = truncate(lines, max_lines=max_lines, max_bytes=max_bytes, direction=direction)
    return '\n'.join(result.truncated_lines)


def _render_directory_entry(name: str, *, is_dir: bool) -> str:
    """Escape line delimiters and controls so one filesystem entry stays on one output line."""
    escaped = json.dumps(name, ensure_ascii=False)[1:-1]
    return f'{escaped}/' if is_dir else escaped


class SpriteSandboxToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent a Sprite to run commands and manage files in."""

    def __init__(
        self,
        *,
        token: str | None,
        sprite_name: str | None,
        base_url: str,
        api_timeout: float,
        runtime: str | None,
        workdir: str | None,
        default_command_timeout: float,
        max_command_timeout: float,
        max_output_bytes: int,
        max_output_lines: int,
        max_read_bytes: int,
        session: SpriteSandboxSession | None = None,
        _run_scoped: bool = False,
    ) -> None:
        super().__init__()
        self._token = token
        self._sprite_name = sprite_name
        self._base_url = base_url
        self._api_timeout = api_timeout
        self._runtime = runtime
        self._workdir = workdir
        self._default_command_timeout = default_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_read_bytes = max_read_bytes
        self._external_session = session
        self._session: SpriteSandboxSession | None = None
        self._run_scoped = _run_scoped

        self.add_function(
            self.run_command,
            name='run_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.list_directory, name='list_directory')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance with one Sprite session for this agent run."""
        return SpriteSandboxToolset[AgentDepsT](
            token=self._token,
            sprite_name=self._sprite_name,
            base_url=self._base_url,
            api_timeout=self._api_timeout,
            runtime=self._runtime,
            workdir=self._workdir,
            default_command_timeout=self._default_command_timeout,
            max_command_timeout=self._max_command_timeout,
            max_output_bytes=self._max_output_bytes,
            max_output_lines=self._max_output_lines,
            max_read_bytes=self._max_read_bytes,
            session=self._external_session,
            _run_scoped=True,
        )

    async def __aenter__(self) -> Self:
        """Open a per-run Sprite, or use an already-open caller-owned session."""
        if not self._run_scoped:
            return self
        if self._external_session is not None:
            if not self._external_session.is_open:
                raise SpriteSandboxError(
                    'The injected session is not open. Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
            return self
        session = SpriteSandboxSession(
            token=self._token,
            sprite_name=self._sprite_name,
            base_url=self._base_url,
            api_timeout=self._api_timeout,
            runtime=self._runtime,
            workdir=self._workdir,
        )
        await session.__aenter__()
        self._session = session
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close the per-run session, leaving a caller-owned session open."""
        session = self._session
        if session is None:
            return
        if self._external_session is not None:
            self._session = None
            return
        await session.__aexit__(*args)
        if not session.is_open:
            self._session = None

    def _require_session(self) -> SpriteSandboxSession:
        if self._session is None:
            raise SpriteSandboxError('The Sprite session is not open.')
        return self._session

    def _command_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is not None and (
            type(timeout_seconds) is bool or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ModelRetry(f'timeout_seconds must be greater than 0, got {timeout_seconds}.')
        requested = timeout_seconds if timeout_seconds is not None else self._default_command_timeout
        return min(requested, self._max_command_timeout)

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Run a shell command in the Sprite and return its combined output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).
        """
        session = self._require_session()
        timeout = self._command_timeout(timeout_seconds)
        try:
            result = await session.exec(command, timeout=timeout, max_output_bytes=self._max_output_bytes)
        except SpriteSandboxTerminalError:
            raise
        except SpriteSandboxError as e:
            raise ModelRetry(str(e))

        output = result.output or '(no output)'
        output = truncate_output(
            output,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='tail',
            already_truncated=result.truncated,
        )
        if result.timed_out:
            output = f'{output}\n[timed out after {result.applied_timeout}s]'
        elif result.returncode:
            output = f'{output}\n[exit code: {result.returncode}]'
        return _strictly_bound_output(
            output,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='tail',
        )

    async def read_file(
        self,
        path: str,
        *,
        offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
        limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
    ) -> str:
        """Read a text file from the Sprite.

        Args:
            path: Path inside the Sprite, relative to its command working directory by default.
            offset: Line number to start reading from (1-indexed).
            limit: Maximum number of lines to read.
        """
        session = self._require_session()
        try:
            data = await session.read_bytes(path, max_bytes=self._max_read_bytes)
        except SpriteSandboxTerminalError:
            raise
        except SpriteSandboxError as e:
            raise ModelRetry(str(e))
        output = render_file_window(
            data,
            offset=offset,
            limit=limit,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
        )
        return _strictly_bound_output(
            output,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='head',
        )

    async def write_file(self, path: str, content: str) -> str:
        """Write text to a file in the Sprite, creating parent directories.

        Args:
            path: Path inside the Sprite, relative to its command working directory by default.
            content: The UTF-8 text to write.
        """
        session = self._require_session()
        try:
            data = content.encode('utf-8')
        except UnicodeEncodeError:
            raise ModelRetry('content contains characters that cannot be encoded as UTF-8 (unpaired surrogates).')
        try:
            await session.write_bytes(path, data)
        except SpriteSandboxTerminalError:
            raise
        except SpriteSandboxError as e:
            raise ModelRetry(str(e))
        return _strictly_bound_output(
            f'Wrote {len(data)} bytes to {path!r}.',
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='head',
        )

    async def list_directory(self, path: str = '.') -> str:
        """List entries in a Sprite directory, with `/` after directory names.

        Args:
            path: Directory to list, relative to the command working directory by default.
        """
        session = self._require_session()
        try:
            listing = await session.list_files(
                path,
                max_entries=self._max_output_lines,
                max_output_bytes=self._max_output_bytes,
            )
        except SpriteSandboxTerminalError:
            raise
        except SpriteSandboxError as e:
            raise ModelRetry(str(e))
        if not listing.entries and not listing.truncated:
            return _strictly_bound_output(
                '(empty)',
                max_lines=self._max_output_lines,
                max_bytes=self._max_output_bytes,
                direction='head',
            )
        names = [_render_directory_entry(name, is_dir=is_dir) for name, is_dir in sorted(listing.entries)]
        output = '\n'.join(names)
        if listing.truncated:
            output = f'[... directory listing truncated ...]\n{output}'
        else:
            output = truncate_output(
                output,
                max_lines=self._max_output_lines,
                max_bytes=self._max_output_bytes,
                direction='head',
            )
        return _strictly_bound_output(
            output,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='head',
        )

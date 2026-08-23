"""Agent tools backed by one run-scoped Daytona sandbox."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness.daytona_sandbox._session import (
    DaytonaSandboxAuthError,
    DaytonaSandboxError,
    DaytonaSandboxSession,
    DaytonaSandboxUnavailableError,
)
from pydantic_ai_harness.modal_sandbox._tool_output import guard_read_size, render_file_window, truncate_output


class DaytonaSandboxToolset(FunctionToolset[AgentDepsT]):
    """Run-scoped Daytona command and file tools."""

    def __init__(
        self,
        *,
        id: str,
        sandbox_id: str | None,
        session: DaytonaSandboxSession | None,
        snapshot: str | None,
        auto_stop_minutes: int,
        workdir: str | None,
        env: Mapping[str, str] | None,
        default_command_timeout: int,
        max_command_timeout: int,
        max_output_bytes: int,
        max_output_lines: int,
        max_read_bytes: int,
        _run_scoped: bool = False,
    ) -> None:
        super().__init__(id=id)
        self._sandbox_id = sandbox_id
        self._external_session = session
        self._snapshot = snapshot
        self._auto_stop_minutes = auto_stop_minutes
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._default_command_timeout = default_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_read_bytes = max_read_bytes
        self._run_scoped = _run_scoped
        self._session: DaytonaSandboxSession | None = None

        self.add_function(
            self.run_command,
            name='run_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.list_directory, name='list_directory')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return DaytonaSandboxToolset[AgentDepsT](
            id=self.id or 'daytona_sandbox',
            sandbox_id=self._sandbox_id,
            session=self._external_session,
            snapshot=self._snapshot,
            auto_stop_minutes=self._auto_stop_minutes,
            workdir=self._workdir,
            env=self._env,
            default_command_timeout=self._default_command_timeout,
            max_command_timeout=self._max_command_timeout,
            max_output_bytes=self._max_output_bytes,
            max_output_lines=self._max_output_lines,
            max_read_bytes=self._max_read_bytes,
            _run_scoped=True,
        )

    async def __aenter__(self) -> Self:
        if not self._run_scoped:
            return self
        if self._session is not None:
            raise DaytonaSandboxError('The Daytona sandbox session is already open.')
        if self._external_session is not None:
            if self._external_session.sandbox_id is None:
                raise DaytonaSandboxError(
                    'The injected session is not open. Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
            return self
        session = DaytonaSandboxSession(
            sandbox_id=self._sandbox_id,
            snapshot=self._snapshot,
            auto_stop_minutes=self._auto_stop_minutes,
            workdir=self._workdir,
            env=self._env,
        )
        await session.__aenter__()
        self._session = session
        return self

    async def __aexit__(self, *args: object) -> None:
        session = self._session
        self._session = None
        if session is not None and self._external_session is None:
            await session.__aexit__(*args)

    def _require_session(self) -> DaytonaSandboxSession:
        if self._session is None:
            raise DaytonaSandboxError('The Daytona sandbox session is not open.')
        return self._session

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Run a shell command in the sandbox and return its output.

        Args:
            command: Shell command to run.
            timeout_seconds: Maximum seconds to wait.
        """
        requested = self._default_command_timeout if timeout_seconds is None else timeout_seconds
        if isinstance(requested, bool) or not math.isfinite(requested) or requested <= 0:
            raise ModelRetry(f'timeout_seconds must be greater than 0, got {requested}.')
        timeout = min(math.ceil(requested), self._max_command_timeout)
        session = self._require_session()
        try:
            result = await session.exec(command, timeout=timeout)
        except (DaytonaSandboxAuthError, DaytonaSandboxUnavailableError):
            raise
        except DaytonaSandboxError as error:
            raise ModelRetry(str(error))

        output = (
            truncate_output(
                result.output,
                max_lines=self._max_output_lines,
                max_bytes=self._max_output_bytes,
                direction='tail',
            )
            or '(no output)'
        )
        if result.timed_out:
            return f'{output}\n[timed out after {timeout}s]'
        if result.returncode:
            return f'{output}\n[exit code: {result.returncode}]'
        return output

    async def read_file(
        self,
        path: str,
        *,
        offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
        limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
    ) -> str:
        """Read a UTF-8 text file from the sandbox.

        Args:
            path: File path inside the sandbox.
            offset: Line number to start reading from (1-indexed).
            limit: Maximum number of lines to read.
        """
        session = self._require_session()
        try:
            guard_read_size(await session.file_size(path), max_bytes=self._max_read_bytes)
            data = await session.read_bytes(path)
        except (DaytonaSandboxAuthError, DaytonaSandboxUnavailableError):
            raise
        except DaytonaSandboxError as error:
            raise ModelRetry(f'Could not read {path!r}: {error}')
        guard_read_size(len(data), max_bytes=self._max_read_bytes)
        return render_file_window(
            data,
            offset=offset,
            limit=limit,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
        )

    async def write_file(self, path: str, content: str) -> str:
        """Write UTF-8 text to a sandbox file, creating parent directories.

        Args:
            path: File path inside the sandbox.
            content: Text to write.
        """
        try:
            data = content.encode('utf-8')
        except UnicodeEncodeError:
            raise ModelRetry('content contains characters that cannot be encoded as UTF-8.')
        try:
            await self._require_session().write_bytes(path, data)
        except (DaytonaSandboxAuthError, DaytonaSandboxUnavailableError):
            raise
        except DaytonaSandboxError as error:
            raise ModelRetry(f'Could not write {path!r}: {error}')
        return f'Wrote {len(data)} bytes to {path!r}.'

    async def list_directory(self, path: str = '.') -> str:
        """List a sandbox directory, marking directories with `/`.

        Args:
            path: Directory path inside the sandbox.
        """
        try:
            entries = await self._require_session().list_files(path)
        except (DaytonaSandboxAuthError, DaytonaSandboxUnavailableError):
            raise
        except DaytonaSandboxError as error:
            raise ModelRetry(f'Could not list {path!r}: {error}')
        if not entries:
            return '(empty)'
        names = [f'{name}/' if is_dir else name for name, is_dir in sorted(entries)]
        return truncate_output(
            '\n'.join(names),
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='head',
        )

"""Model-facing tools backed by an Islo sandbox session."""

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

from pydantic_ai_harness._sandbox_tool_output import guard_read_size, render_file_window, truncate_output
from pydantic_ai_harness.islo_sandbox._session import (
    IsloSandboxError,
    IsloSandboxSession,
    IsloSandboxTerminalError,
)


class IsloSandboxToolset(FunctionToolset[AgentDepsT]):
    """Give one agent run command and file tools backed by Islo."""

    def __init__(
        self,
        *,
        image: str,
        sandbox_name: str | None,
        sandbox_timeout: int,
        workdir: str | None,
        env: Mapping[str, str] | None,
        vcpus: int | None,
        memory_mb: int | None,
        disk_gb: int | None,
        internet_enabled: bool | None,
        gateway_profile: str | None,
        base_url: str | None,
        compute_url: str | None,
        default_command_timeout: float,
        max_command_timeout: int | None,
        max_output_bytes: int,
        max_output_lines: int,
        max_read_bytes: int,
        poll_interval: float,
        session: IsloSandboxSession | None = None,
        _run_scoped: bool = False,
    ) -> None:
        super().__init__()
        self._image = image
        self._sandbox_name = sandbox_name
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._vcpus = vcpus
        self._memory_mb = memory_mb
        self._disk_gb = disk_gb
        self._internet_enabled = internet_enabled
        self._gateway_profile = gateway_profile
        self._base_url = base_url
        self._compute_url = compute_url
        self._default_command_timeout = default_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_read_bytes = max_read_bytes
        self._poll_interval = poll_interval
        self._external_session = session
        self._session: IsloSandboxSession | None = None
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
        """Return an isolated toolset instance for one agent run."""
        return IsloSandboxToolset[AgentDepsT](
            image=self._image,
            sandbox_name=self._sandbox_name,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            env=self._env,
            vcpus=self._vcpus,
            memory_mb=self._memory_mb,
            disk_gb=self._disk_gb,
            internet_enabled=self._internet_enabled,
            gateway_profile=self._gateway_profile,
            base_url=self._base_url,
            compute_url=self._compute_url,
            default_command_timeout=self._default_command_timeout,
            max_command_timeout=self._max_command_timeout,
            max_output_bytes=self._max_output_bytes,
            max_output_lines=self._max_output_lines,
            max_read_bytes=self._max_read_bytes,
            poll_interval=self._poll_interval,
            session=self._external_session,
            _run_scoped=True,
        )

    async def __aenter__(self) -> Self:
        """Open a per-run session or use the caller's open session."""
        if not self._run_scoped:
            return self
        if self._external_session is not None:
            if self._external_session.sandbox_name is None:
                raise IsloSandboxError(
                    'The injected session is not open. Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
            return self
        session = IsloSandboxSession(
            image=self._image,
            sandbox_name=self._sandbox_name,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            env=self._env,
            vcpus=self._vcpus,
            memory_mb=self._memory_mb,
            disk_gb=self._disk_gb,
            internet_enabled=self._internet_enabled,
            gateway_profile=self._gateway_profile,
            base_url=self._base_url,
            compute_url=self._compute_url,
            poll_interval=self._poll_interval,
        )
        await session.__aenter__()
        self._session = session
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close only the session owned by this run."""
        session = self._session
        if session is not None and self._external_session is None:
            await session.__aexit__(*args)
            if session.sandbox_name is not None:
                # One immediate retry handles a transient control-plane failure.
                # `delete_after` remains the backstop if both attempts fail.
                await session.close()
        self._session = None

    def _require_session(self) -> IsloSandboxSession:
        if self._session is None:
            raise IsloSandboxError('The Islo sandbox session is not open.')
        return self._session

    def _command_timeout(self, timeout_seconds: float | None) -> int:
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ModelRetry(f'timeout_seconds must be greater than 0, got {timeout_seconds}.')
        requested = timeout_seconds if timeout_seconds is not None else self._default_command_timeout
        ceiling = self._max_command_timeout if self._max_command_timeout is not None else self._sandbox_timeout
        return min(max(1, math.ceil(requested)), ceiling)

    def _truncate(self, text: str, *, direction: str, already_truncated: bool = False) -> str:
        if direction == 'head':
            return truncate_output(
                text,
                max_lines=self._max_output_lines,
                max_bytes=self._max_output_bytes,
                direction='head',
                already_truncated=already_truncated,
            )
        return truncate_output(
            text,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='tail',
            already_truncated=already_truncated,
        )

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Run a shell command in the sandbox and return its output.

        The command runs through `sh -c`, so pipes, redirection, `&&`, and globs work. A
        non-zero exit is reported, not raised, so you can react to it.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: the configured timeout),
                clamped to the configured ceiling.

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        session = self._require_session()
        try:
            result = await session.exec(
                ['sh', '-c', command],
                timeout=self._command_timeout(timeout_seconds),
                max_output_bytes=self._max_output_bytes,
            )
        except IsloSandboxTerminalError:
            raise
        except IsloSandboxError as e:
            raise ModelRetry(str(e))

        parts: list[str] = []
        if result.stdout:
            parts.append(
                f'[stdout]\n{self._truncate(result.stdout, direction="tail", already_truncated=result.stdout_truncated)}'
            )
        if result.stderr:
            parts.append(
                f'[stderr]\n{self._truncate(result.stderr, direction="tail", already_truncated=result.stderr_truncated)}'
            )
        output = '\n'.join(parts) if parts else '(no output)'
        if result.remote_may_be_running:
            return f'{output}\n[client wait timed out after {result.applied_timeout}s; remote command may still be running]'
        if result.timed_out:
            return f'{output}\n[timed out after {result.applied_timeout}s]'
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
        """Read a text file from the sandbox and return its contents.

        Large files are truncated to a safety cap; the result ends with the next `offset`
        to use to page through the rest. A file over the read limit is refused, with a
        suggestion to slice it with a shell command instead.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            offset: Line number to start reading from (1-indexed).
            limit: Maximum number of lines to read.
        """
        session = self._require_session()
        try:
            data = await session.read_bytes(path, max_bytes=self._max_read_bytes)
        except IsloSandboxTerminalError:
            raise
        except IsloSandboxError as e:
            raise ModelRetry(f'Could not read {path!r}: {e}')
        # `read_bytes` stops one byte past the cap, so this refuses an oversized file with
        # the same message every sandbox backend uses.
        guard_read_size(len(data), max_bytes=self._max_read_bytes)
        return render_file_window(
            data,
            offset=offset,
            limit=limit,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
        )

    async def write_file(self, path: str, content: str) -> str:
        """Write UTF-8 text to a file in the sandbox, replacing any existing contents.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            content: The text to write, encoded as UTF-8.
        """
        session = self._require_session()
        try:
            data = content.encode('utf-8')
        except UnicodeEncodeError:
            raise ModelRetry('content contains characters that cannot be encoded as UTF-8 (unpaired surrogates).')
        try:
            await session.write_bytes(path, data)
        except IsloSandboxTerminalError:
            raise
        except IsloSandboxError as e:
            raise ModelRetry(f'Could not write {path!r}: {e}')
        return f'Wrote {len(data)} bytes to {path!r}.'

    async def list_directory(self, path: str = '.') -> str:
        """List the entries of a directory in the sandbox, with directories marked by `/`.

        Args:
            path: Path to the directory inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
        """
        session = self._require_session()
        try:
            entries = await session.list_files(path)
        except IsloSandboxTerminalError:
            raise
        except IsloSandboxError as e:
            raise ModelRetry(f'Could not list {path!r}: {e}')
        if not entries:
            return '(empty)'
        names = [f'{name}/' if is_dir else name for name, is_dir in sorted(entries)]
        return self._truncate('\n'.join(names), direction='head')

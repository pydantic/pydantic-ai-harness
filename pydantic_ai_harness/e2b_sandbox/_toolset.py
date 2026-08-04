"""E2B sandbox tools and their Pydantic AI observability."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Literal

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness.e2b_sandbox._session import (
    DEFAULT_SANDBOX_TIMEOUT,
    E2BSandboxError,
    E2BSandboxSession,
    E2BSandboxTerminalError,
)
from pydantic_ai_harness.e2b_sandbox._tool_output import guard_read_size, render_file_window, truncate_output


class E2BSandboxToolset(FunctionToolset[AgentDepsT]):
    """Give an agent per-run E2B command and file tools."""

    def __init__(
        self,
        *,
        template: str | None,
        sandbox_id: str | None,
        sandbox_timeout: int,
        workdir: str,
        default_command_timeout: float,
        max_command_timeout: int | None,
        max_output_bytes: int,
        max_output_lines: int,
        max_read_bytes: int,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
        session: E2BSandboxSession | None = None,
        _tracer: Tracer | None = None,
        _run_scoped: bool = False,
    ) -> None:
        super().__init__(id='e2b_sandbox')
        self._template = template
        self._sandbox_id = sandbox_id
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._default_command_timeout = default_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_read_bytes = max_read_bytes
        self._env = dict(env) if env is not None else None
        self._metadata = dict(metadata) if metadata is not None else None
        self._allow_internet_access = allow_internet_access
        self._external_session = session
        self._tracer = _tracer
        self._session: E2BSandboxSession | None = None
        self._run_scoped = _run_scoped

        self.add_function(self.run_command, name='run_command')
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.list_directory, name='list_directory')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset so each run owns an isolated session."""
        return E2BSandboxToolset[AgentDepsT](
            template=self._template,
            sandbox_id=self._sandbox_id,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            default_command_timeout=self._default_command_timeout,
            max_command_timeout=self._max_command_timeout,
            max_output_bytes=self._max_output_bytes,
            max_output_lines=self._max_output_lines,
            max_read_bytes=self._max_read_bytes,
            env=self._env,
            metadata=self._metadata,
            allow_internet_access=self._allow_internet_access,
            session=self._external_session,
            _tracer=ctx.tracer,
            _run_scoped=True,
        )

    async def __aenter__(self) -> Self:
        """Open the run-owned session or validate an injected one."""
        if not self._run_scoped:
            return self
        if self._external_session is not None:
            if self._external_session.sandbox_id is None:
                raise E2BSandboxError(
                    'The injected session is not open. Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
            return self
        session = E2BSandboxSession(
            template=self._template,
            sandbox_id=self._sandbox_id,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            env=self._env,
            metadata=self._metadata,
            allow_internet_access=self._allow_internet_access,
            tracer=self._tracer,
        )
        await session.__aenter__()
        self._session = session
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close only a session owned by this run."""
        session = self._session
        self._session = None
        if session is not None and self._external_session is None:
            await session.__aexit__(*args)

    def _require_session(self) -> E2BSandboxSession:
        if self._session is None:
            raise E2BSandboxError('The E2B sandbox session is not open.')
        return self._session

    def _mode(self) -> Literal['owned', 'attached', 'injected']:
        if self._external_session is not None:
            return 'injected'
        return 'attached' if self._sandbox_id is not None else 'owned'

    def _operation_tracer(self) -> Tracer:
        return self._tracer or trace.get_tracer('pydantic_ai_harness.e2b_sandbox')

    def _set_span_base(self, span: Span, session: E2BSandboxSession, operation: str) -> None:
        sandbox_id = session.sandbox_id
        span.set_attribute('e2b.operation', operation)
        span.set_attribute('e2b.sandbox.mode', self._mode())
        if sandbox_id is not None:  # pragma: no branch - tools require an open session
            span.set_attribute('e2b.sandbox.id', sandbox_id)
        if session.template is not None:
            span.set_attribute('e2b.sandbox.template', session.template)

    def _command_timeout(self, timeout_seconds: float | None) -> int:
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ModelRetry(f'timeout_seconds must be greater than 0, got {timeout_seconds}.')
        requested = timeout_seconds if timeout_seconds is not None else self._default_command_timeout
        reused = self._external_session is not None or self._sandbox_id is not None
        ceiling = (
            self._max_command_timeout
            if self._max_command_timeout is not None
            else (DEFAULT_SANDBOX_TIMEOUT if reused else self._sandbox_timeout)
        )
        return min(max(1, math.ceil(requested)), ceiling)

    def _truncate_stream(self, text: str, already_truncated: bool, *, direction: Literal['head', 'tail']) -> str:
        return truncate_output(
            text,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction=direction,
            already_truncated=already_truncated,
        )

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        """Run a Bash command in the sandbox and return bounded output.

        Args:
            command: The Bash command to run.
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with timeout or non-zero exit status.
        """
        session = self._require_session()
        timeout = self._command_timeout(timeout_seconds)
        with self._operation_tracer().start_as_current_span(
            'e2b.sandbox.run_command',
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._set_span_base(span, session, 'run_command')
            span.set_attribute('e2b.command.timeout_seconds', timeout)
            try:
                result = await session.exec(command, timeout=timeout, max_output_bytes=self._max_output_bytes)
            except E2BSandboxTerminalError:
                span.set_attribute('e2b.outcome', 'terminal_error')
                raise
            except E2BSandboxError as e:
                span.set_attribute('e2b.outcome', 'retry')
                raise ModelRetry(str(e))

            span.set_attribute('e2b.command.exit_code', result.returncode)
            span.set_attribute('e2b.command.stdout_truncated', result.stdout_truncated)
            span.set_attribute('e2b.command.stderr_truncated', result.stderr_truncated)
            if result.timed_out:
                span.set_attribute('e2b.outcome', 'timeout')
            elif result.returncode:
                span.set_attribute('e2b.outcome', 'nonzero_exit')
            else:
                span.set_attribute('e2b.outcome', 'success')

        # Timed-out commands return a bounded prefix, so truncate head-first and say
        # "first" in the marker; completed commands keep the bounded tail.
        direction: Literal['head', 'tail'] = 'head' if result.timed_out else 'tail'
        parts: list[str] = []
        if result.stdout:
            parts.append(
                f'[stdout]\n{self._truncate_stream(result.stdout, result.stdout_truncated, direction=direction)}'
            )
        if result.stderr:
            parts.append(
                f'[stderr]\n{self._truncate_stream(result.stderr, result.stderr_truncated, direction=direction)}'
            )
        output = '\n'.join(parts) if parts else '(no output)'
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
        """Read a UTF-8 file, with line paging and bounded memory.

        Args:
            path: Sandbox path, relative to the configured working directory when needed.
            offset: Line number to start reading from (1-indexed).
            limit: Maximum number of lines to read.
        """
        session = self._require_session()
        with self._operation_tracer().start_as_current_span(
            'e2b.sandbox.read_file',
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._set_span_base(span, session, 'read_file')
            try:
                guard_read_size(await session.file_size(path), max_bytes=self._max_read_bytes)
                data = await session.read_bytes(path, max_bytes=self._max_read_bytes)
                guard_read_size(len(data), max_bytes=self._max_read_bytes)
                output = render_file_window(
                    data,
                    offset=offset,
                    limit=limit,
                    max_lines=self._max_output_lines,
                    max_bytes=self._max_output_bytes,
                )
            except E2BSandboxTerminalError:
                span.set_attribute('e2b.outcome', 'terminal_error')
                raise
            except ModelRetry:
                span.set_attribute('e2b.outcome', 'retry')
                raise
            except E2BSandboxError as e:
                span.set_attribute('e2b.outcome', 'retry')
                raise ModelRetry(f'Could not read {path!r}: {e}')
            span.set_attribute('e2b.outcome', 'success')
            span.set_attribute('e2b.file.size_bytes', len(data))
        return output

    async def write_file(self, path: str, content: str) -> str:
        """Write UTF-8 text to a sandbox file.

        Args:
            path: Sandbox path, relative to the configured working directory when needed.
            content: Text to write.
        """
        session = self._require_session()
        try:
            data = content.encode('utf-8')
        except UnicodeEncodeError:
            raise ModelRetry('content contains characters that cannot be encoded as UTF-8 (unpaired surrogates).')
        with self._operation_tracer().start_as_current_span(
            'e2b.sandbox.write_file',
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._set_span_base(span, session, 'write_file')
            try:
                await session.write_bytes(path, data)
            except E2BSandboxTerminalError:
                span.set_attribute('e2b.outcome', 'terminal_error')
                raise
            except E2BSandboxError as e:
                span.set_attribute('e2b.outcome', 'retry')
                raise ModelRetry(f'Could not write {path!r}: {e}')
            span.set_attribute('e2b.outcome', 'success')
            span.set_attribute('e2b.file.size_bytes', len(data))
        return f'Wrote {len(data)} bytes to {path!r}.'

    async def list_directory(self, path: str = '.') -> str:
        """List sandbox directory entries, with directories ending in `/`.

        Args:
            path: Sandbox path, relative to the configured working directory when needed.
        """
        session = self._require_session()
        with self._operation_tracer().start_as_current_span(
            'e2b.sandbox.list_directory',
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._set_span_base(span, session, 'list_directory')
            try:
                entries = await session.list_files(path)
            except E2BSandboxTerminalError:
                span.set_attribute('e2b.outcome', 'terminal_error')
                raise
            except E2BSandboxError as e:
                span.set_attribute('e2b.outcome', 'retry')
                raise ModelRetry(f'Could not list {path!r}: {e}')
            span.set_attribute('e2b.outcome', 'success')
            span.set_attribute('e2b.file.entry_count', len(entries))
        if not entries:
            return '(empty)'
        names = [f'{name}/' if is_dir else name for name, is_dir in sorted(entries)]
        return truncate_output(
            '\n'.join(names), max_lines=self._max_output_lines, max_bytes=self._max_output_bytes, direction='head'
        )

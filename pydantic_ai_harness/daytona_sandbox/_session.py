"""Small Daytona SDK boundary used by `DaytonaSandbox`.

External assumptions, verified 2026-08-23 against Daytona Python SDK 0.198.0:

- `AsyncDaytona.create`, `get`, `delete(wait=True)`, and `close` own sandbox lifecycle.
- process sessions provide bounded waits, input without echo, streamed stdout
  and stderr, exit status, and explicit deletion.
- `sandbox.fs` provides metadata, byte upload/download, and directory listing.
- `CreateSandboxFromSnapshotParams.network_block_all=True` blocks outbound traffic.

Sources:
https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
https://www.daytona.io/docs/en/python-sdk/async/async-process/
https://www.daytona.io/docs/en/python-sdk/async/async-file-system/
https://www.daytona.io/docs/en/network-limits/

Re-check those signatures against the lowest supported SDK before raising the
dependency ceiling.
"""

from __future__ import annotations

import asyncio
import posixpath
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeVar

if TYPE_CHECKING:
    from daytona import AsyncDaytona
    from daytona._async.process import AsyncProcess
    from daytona._async.sandbox import AsyncSandbox


DEFAULT_AUTO_STOP_MINUTES = 60
_DEFAULT_PROCESS_IO_TIMEOUT = 30
_DEFAULT_EXEC_OUTPUT_BYTES = 50 * 1024

ProcessOutputHandler: TypeAlias = Callable[[str], None] | Callable[[str], Awaitable[None]]
_T = TypeVar('_T')


class DaytonaSandboxError(RuntimeError):
    """A Daytona sandbox operation failed."""


class DaytonaSandboxAuthError(DaytonaSandboxError):
    """Daytona rejected the configured credentials."""


class DaytonaSandboxUnavailableError(DaytonaSandboxError):
    """The requested Daytona sandbox no longer exists."""


@dataclass(frozen=True, kw_only=True)
class DaytonaSandboxExecResult:
    """The outcome of running a command in a Daytona sandbox."""

    output: str
    """The command result text returned by Daytona's direct execution API."""

    returncode: int
    """The command exit status, or `-1` when the SDK reports a timeout."""

    timed_out: bool = False
    """Whether the Daytona SDK stopped waiting at the command deadline."""

    output_truncated: bool = False
    """Whether earlier command output was discarded to bound host memory."""


class DaytonaSandboxProcess:
    """One bounded, bidirectional command in an open Daytona sandbox session."""

    def __init__(
        self,
        *,
        process: AsyncProcess,
        process_id: str,
        command: str,
        on_stdout: ProcessOutputHandler,
        on_stderr: ProcessOutputHandler,
        max_input_bytes: int,
        io_timeout: int,
    ) -> None:
        if not process_id:
            raise ValueError('process_id must not be empty.')
        _positive_int('max_input_bytes', max_input_bytes)
        _positive_int('io_timeout', io_timeout)
        self._process = process
        self._process_id = process_id
        self._command = command
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self._max_input_bytes = max_input_bytes
        self._io_timeout = io_timeout
        self._command_id: str | None = None
        self._logs: asyncio.Task[None] | None = None

    @property
    def process_id(self) -> str:
        """The caller-supplied identity used for the Daytona process session."""
        return self._process_id

    async def __aenter__(self) -> DaytonaSandboxProcess:
        if self._command_id is not None:
            raise DaytonaSandboxError('The Daytona process is already open.')
        created = False
        try:
            await _finish_on_cancellation(
                self._process.create_session(self._process_id, request_timeout=self._io_timeout),
                on_cancel=lambda _: self._process.delete_session(self._process_id, request_timeout=self._io_timeout),
            )
            created = True
            from daytona import SessionExecuteRequest

            response = await self._process.execute_session_command(
                self._process_id,
                SessionExecuteRequest(
                    command=self._command,
                    run_async=True,
                    suppress_input_echo=True,
                ),
                timeout=self._io_timeout,
            )
        except BaseException:
            if created:
                await _finish_cleanup(self._process.delete_session(self._process_id, request_timeout=self._io_timeout))
            raise
        self._command_id = response.cmd_id
        self._logs = asyncio.create_task(
            self._process.get_session_command_logs_async(
                self._process_id,
                response.cmd_id,
                self._on_stdout,
                self._on_stderr,
            )
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def send(self, data: str, *, timeout: int = _DEFAULT_PROCESS_IO_TIMEOUT) -> None:
        """Send bounded text to the command's standard input."""
        _positive_int('timeout', timeout)
        size = len(data.encode('utf-8'))
        if size > self._max_input_bytes:
            raise ValueError(f'input is {size} bytes, over the {self._max_input_bytes}-byte limit.')
        command_id = self._require_command_id()
        try:
            await self._process.send_session_command_input(
                self._process_id,
                command_id,
                data,
                request_timeout=timeout,
            )
        except Exception as error:
            raise _translate_error(error, unavailable=True) from error

    async def wait(self, *, timeout: int) -> int:
        """Wait a bounded time for completion and return the command exit status."""
        _positive_int('timeout', timeout)
        command_id = self._require_command_id()
        logs = self._logs
        if logs is None:  # pragma: no cover - maintained with `_command_id`
            raise DaytonaSandboxError('The Daytona process log stream is unavailable.')
        try:
            await asyncio.wait_for(asyncio.shield(logs), timeout)
            command = await self._process.get_session_command(
                self._process_id,
                command_id,
                request_timeout=self._io_timeout,
            )
        except TimeoutError:
            raise
        except Exception as error:
            raise _translate_error(error, unavailable=True) from error
        if command.exit_code is None:
            raise DaytonaSandboxError('Daytona closed the process log stream before reporting an exit status.')
        return command.exit_code

    async def close(self) -> None:
        """Terminate the remote process session and its local log stream."""
        if self._command_id is None:
            return
        try:
            await _finish_cleanup(
                self._process.delete_session(self._process_id, request_timeout=self._io_timeout),
                then=self._clear,
            )
        except Exception as error:
            raise _translate_error(error, unavailable=True) from error

    def _clear(self) -> None:
        logs = self._logs
        if logs is not None and not logs.done():
            logs.cancel()
        self._command_id = None
        self._logs = None

    def _require_command_id(self) -> str:
        if self._command_id is None:
            raise DaytonaSandboxError('The Daytona process is not open.')
        return self._command_id


class DaytonaSandboxSession:
    """Async context manager that owns or attaches to one Daytona sandbox.

    A session without `sandbox_id` creates a sandbox from `snapshot` and deletes
    it on exit. A session with `sandbox_id` attaches to that sandbox and leaves it
    running. Pass an already-open session to `DaytonaSandbox(session=...)` to
    reuse one sandbox across several agent runs while retaining lifecycle
    ownership in the caller.

    ```python
    import asyncio

    from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxSession


    async def main() -> None:
        async with DaytonaSandboxSession() as session:
            result = await session.exec('python --version', timeout=30)
            print(result.output)


    asyncio.run(main())
    ```
    """

    def __init__(
        self,
        *,
        sandbox_id: str | None = None,
        snapshot: str | None = None,
        auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        network_block_all: bool = False,
    ) -> None:
        if type(auto_stop_minutes) is not int or auto_stop_minutes <= 0:
            raise ValueError(f'auto_stop_minutes must be a positive integer, got {auto_stop_minutes!r}.')
        if sandbox_id is not None and snapshot is not None:
            raise ValueError('snapshot cannot be combined with sandbox_id.')
        if sandbox_id is not None and auto_stop_minutes != DEFAULT_AUTO_STOP_MINUTES:
            raise ValueError('auto_stop_minutes cannot be combined with sandbox_id.')
        if type(network_block_all) is not bool:
            raise ValueError(f'network_block_all must be a boolean, got {network_block_all!r}.')
        if sandbox_id is not None and network_block_all:
            raise ValueError('network_block_all cannot configure an attached sandbox.')
        self._requested_id = sandbox_id
        self._snapshot = snapshot
        self._auto_stop_minutes = auto_stop_minutes
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._network_block_all = network_block_all
        self._client: AsyncDaytona | None = None
        self._sandbox: AsyncSandbox | None = None
        self._owned_sandbox_deleted = False

    @property
    def sandbox_id(self) -> str | None:
        """The ID of the open sandbox, or `None` outside the session context."""
        return self._sandbox.id if self._sandbox is not None else None

    async def __aenter__(self) -> DaytonaSandboxSession:
        if self._sandbox is not None:
            raise DaytonaSandboxError(
                'The session is already open; exit it before entering again. '
                'Use a separate session per concurrent context.'
            )
        try:
            from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
        except ImportError as error:
            raise DaytonaSandboxError(
                'The `daytona` package is required. Install it with `uv add "pydantic-ai-harness[daytona]"`.'
            ) from error

        client = AsyncDaytona()
        self._client = client
        try:
            if self._requested_id is not None:
                sandbox = await _finish_on_cancellation(client.get(self._requested_id))
            else:
                params = CreateSandboxFromSnapshotParams(
                    snapshot=self._snapshot,
                    env_vars=self._env,
                    auto_stop_interval=self._auto_stop_minutes,
                    auto_delete_interval=0,
                    network_block_all=self._network_block_all,
                )
                sandbox = await _finish_on_cancellation(
                    client.create(params),
                    on_cancel=lambda created: client.delete(created, timeout=60, wait=True),
                )
        except asyncio.CancelledError:
            if self._sandbox is not None:
                await _finish_cleanup(client.delete(self._sandbox, timeout=60, wait=True))
            await _finish_cleanup(client.close(), then=self._clear)
            raise
        except Exception as error:
            await client.close()
            self._client = None
            raise _translate_error(error, unavailable=self._requested_id is not None) from error

        self._sandbox = sandbox
        self._owned_sandbox_deleted = False
        return self

    async def __aexit__(self, *_: object) -> None:
        client = self._client
        sandbox = self._sandbox
        if client is None:
            return
        if sandbox is None:
            try:
                await _finish_cleanup(client.close(), then=self._clear)
            except Exception as error:
                raise _translate_error(error, unavailable=False) from error
            return

        if self._requested_id is None and not self._owned_sandbox_deleted:
            try:
                await _finish_cleanup(
                    client.delete(sandbox, timeout=60, wait=True),
                    then=self._mark_owned_sandbox_deleted,
                )
            except Exception as error:
                raise _translate_error(error, unavailable=False) from error
        try:
            await _finish_cleanup(client.close(), then=self._clear)
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error

    def _mark_owned_sandbox_deleted(self) -> None:
        self._owned_sandbox_deleted = True

    def _clear(self) -> None:
        self._client = None
        self._sandbox = None
        self._owned_sandbox_deleted = False

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise DaytonaSandboxError('The Daytona sandbox session is not open.')
        return self._sandbox

    def _path(self, path: str) -> str:
        if self._workdir is None or posixpath.isabs(path):
            return path
        return posixpath.join(self._workdir, path)

    def _command(self, command: str) -> str:
        if self._env:
            assignments = ' '.join(shlex.quote(f'{name}={value}') for name, value in self._env.items())
            command = f'env -- {assignments} sh -c {shlex.quote(command)}'
        if self._workdir is not None:
            command = f'cd -- {shlex.quote(self._workdir)} && {command}'
        return command

    def process(
        self,
        process_id: str,
        command: str,
        *,
        on_stdout: ProcessOutputHandler,
        on_stderr: ProcessOutputHandler,
        max_input_bytes: int,
        io_timeout: int = _DEFAULT_PROCESS_IO_TIMEOUT,
    ) -> DaytonaSandboxProcess:
        """Prepare one managed long-running command inside this open session."""
        process = self._require_sandbox().process
        return DaytonaSandboxProcess(
            process=process,
            process_id=process_id,
            command=self._command(command),
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            max_input_bytes=max_input_bytes,
            io_timeout=io_timeout,
        )

    async def exec(
        self,
        command: str,
        *,
        timeout: int,
        max_output_bytes: int = _DEFAULT_EXEC_OUTPUT_BYTES,
    ) -> DaytonaSandboxExecResult:
        """Run a command while retaining only a bounded tail of streamed output."""
        _positive_int('timeout', timeout)
        _positive_int('max_output_bytes', max_output_bytes)
        output = _TailBuffer(max_output_bytes)
        process_id = f'harness-{uuid.uuid4().hex}'
        try:
            async with self.process(
                process_id,
                command,
                on_stdout=output.append,
                on_stderr=output.append,
                max_input_bytes=1,
            ) as process:
                try:
                    returncode = await process.wait(timeout=timeout)
                except TimeoutError:
                    return DaytonaSandboxExecResult(
                        output=output.text,
                        returncode=-1,
                        timed_out=True,
                        output_truncated=output.truncated,
                    )
        except DaytonaSandboxError:
            raise
        except Exception as error:  # pragma: no cover - SDK failures are translated by the process
            raise _translate_error(error, unavailable=True) from error
        return DaytonaSandboxExecResult(
            output=output.text,
            returncode=returncode,
            output_truncated=output.truncated,
        )

    async def file_size(self, path: str) -> int:
        try:
            return (await self._require_sandbox().fs.get_file_info(self._path(path))).size
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error

    async def read_bytes(self, path: str) -> bytes:
        try:
            data = await self._require_sandbox().fs.download_file(self._path(path))
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error
        return data

    async def write_bytes(self, path: str, data: bytes) -> None:
        sandbox = self._require_sandbox()
        resolved = self._path(path)
        parent = posixpath.dirname(resolved)
        try:
            if parent not in ('', '.', '/'):
                mkdir = await sandbox.process.exec(f'mkdir -p -- {shlex.quote(parent)}', timeout=30)
                if mkdir.exit_code != 0:
                    raise DaytonaSandboxError(mkdir.result or f'Could not create {parent!r}.')
            await sandbox.fs.upload_file(data, resolved)
        except DaytonaSandboxError:
            raise
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error

    async def list_files(self, path: str) -> list[tuple[str, bool]]:
        try:
            entries = await self._require_sandbox().fs.list_files(self._path(path))
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error
        return [(entry.name, entry.is_dir) for entry in entries]


def _translate_error(error: Exception, *, unavailable: bool) -> DaytonaSandboxError:
    """Map SDK failures without leaking SDK types through the public API."""
    try:
        from daytona import (
            DaytonaAuthenticationError,
            DaytonaAuthorizationError,
            DaytonaNotFoundError,
        )
    except ImportError:  # pragma: no cover - the session already imported Daytona
        return DaytonaSandboxError(str(error))

    if isinstance(error, (DaytonaAuthenticationError, DaytonaAuthorizationError)):
        return DaytonaSandboxAuthError('Daytona rejected the credentials. Set DAYTONA_API_KEY and try again.')
    if unavailable and isinstance(error, DaytonaNotFoundError):
        return DaytonaSandboxUnavailableError('The Daytona sandbox does not exist or is no longer available.')
    return DaytonaSandboxError(str(error))


async def _finish_on_cancellation(
    operation: Awaitable[_T],
    *,
    on_cancel: Callable[[_T], Awaitable[object]] | None = None,
) -> _T:
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            result = await task
        except Exception:
            pass
        else:
            if on_cancel is not None:
                await _finish_cleanup(on_cancel(result))
        raise


async def _finish_cleanup(operation: Awaitable[object], *, then: Callable[[], None] | None = None) -> None:
    task = asyncio.ensure_future(operation)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        if then is not None:
            then()
        raise
    if then is not None:
        then()


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}.')


class _TailBuffer:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._data = bytearray()
        self.truncated = False

    @property
    def text(self) -> str:
        return bytes(self._data).decode('utf-8', errors='ignore')

    def append(self, chunk: str) -> None:
        data = chunk.encode('utf-8')
        if len(data) >= self._max_bytes:
            self.truncated = self.truncated or bool(self._data) or len(data) > self._max_bytes
            self._data[:] = data[-self._max_bytes :]
            return
        overflow = len(self._data) + len(data) - self._max_bytes
        if overflow > 0:
            del self._data[:overflow]
            self.truncated = True
        self._data.extend(data)

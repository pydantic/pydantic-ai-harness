"""A Daytona sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-08 against Daytona Python SDK 0.198.0:

* `AsyncDaytona.create`, `get`, `delete(wait=True)`, and `close` own sandbox lifecycle;
  `get` accepts a sandbox ID or name:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* process sessions provide asynchronous execution, separate stdout and stderr callbacks,
  exit status, and deletion as the per-command kill mechanism:
  https://www.daytona.io/docs/en/python-sdk/async/async-process/
* `sandbox.fs` provides metadata, byte upload/download, and directory operations:
  https://www.daytona.io/docs/en/python-sdk/async/async-file-system/
* `auto_stop_interval` and `auto_delete_interval=-1` keep an owned sandbox stopped but
  available until explicit deletion:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* `AsyncSandbox.pause` is supported only by VM sandbox classes; `stop` retains disk according
  to the sandbox's configured auto-delete policy:
  https://www.daytona.io/docs/en/python-sdk/async/async-sandbox/

Re-check those sources and the installed 0.198.0 signatures before changing lifecycle,
command, or filesystem handling.
"""

from __future__ import annotations

import asyncio
import functools
import math
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import anyio
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness._workspace_provider import absolute_path

if TYPE_CHECKING:
    from daytona import AsyncDaytona, AsyncSandbox

    # Not re-exported at the package root; typing-only, so the private path never runs.
    from daytona._async.process import AsyncProcess
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('DaytonaWorkspaceBackend',)

DEFAULT_AUTO_STOP_MINUTES = 60

try:
    import daytona
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[daytona]` to use DaytonaWorkspace.') from error

_AUTH_MESSAGE = 'Daytona rejected the credentials. Set DAYTONA_API_KEY and try again.'
# Bound sandbox acquisition so a wedged control plane cannot hang creation or connection.
_CREATE_TIMEOUT = 120
# Bound routine SDK requests so a stalled control plane cannot hang an operation.
_REQUEST_TIMEOUT = 30
# Bound provider lifecycle RPCs such as create, start, and delete.
_LIFECYCLE_TIMEOUT = 60.0
# Bound cleanup RPCs so teardown cannot wedge the caller.
_TEARDOWN_TIMEOUT = 30.0


async def cleanup_call(call: Callable[[], Awaitable[object]], *, timeout: float) -> BaseException | None:
    try:
        with anyio.fail_after(timeout):
            await call()
    except BaseException as error:
        return error
    return None


def _command_line(command: WorkspaceCommand, shell: bool) -> str:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        return command
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    if not command:
        raise TypeError('a command needs at least the program to run; the argv sequence is empty')
    return shlex.join(command)


def _command_context(command: str, cwd: str | None, env: Mapping[str, str] | None) -> str:
    """Apply command-local settings that Daytona's session request cannot represent."""
    if env:
        assignments = ' '.join(shlex.quote(f'{name}={value}') for name, value in env.items())
        command = f'env -- {assignments} sh -c {shlex.quote(command)}'
    if cwd is not None:
        command = f'cd -- {shlex.quote(cwd)} && {command}'
    return command


@dataclass(kw_only=True)
class _DaytonaProcess:
    """Output and session identity for a single command."""

    _process: AsyncProcess
    _backend: DaytonaWorkspaceBackend
    _session_id: str
    _command_id: str
    stdout: list[str]
    stderr: list[str]
    _logs: asyncio.Task[None]

    async def wait(self) -> CommandResult:
        try:
            await self._logs
            command = await self._process.get_session_command(
                self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
            )
        except Exception as error:
            raise _operation_error(error, 'Could not read the command result', unavailable=True) from error
        if command.exit_code is None:
            raise WorkspaceError('Daytona closed the command output before reporting an exit status.')
        result = CommandResult(
            exit_code=command.exit_code,
            stdout=''.join(self.stdout),
            stderr=''.join(self.stderr),
        )
        return result

    async def kill(self) -> None:
        """Delete the Daytona process session, which kills its command."""
        error = await cleanup_call(
            functools.partial(self._process.delete_session, self._session_id, request_timeout=_REQUEST_TIMEOUT),
            timeout=_TEARDOWN_TIMEOUT,
        )
        self._logs.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(self._logs, return_exceptions=True)
        if error is None or isinstance(error, daytona.DaytonaNotFoundError):
            return
        if isinstance(error, Exception):
            raise _operation_error(
                error, f'Could not kill command session {self._session_id!r}', unavailable=True
            ) from error
        raise error  # pragma: no cover - cancellation propagates after bounded session cleanup


async def _kill_quietly(process: _DaytonaProcess) -> None:
    """Best-effort kill whose failure must not mask the outcome being raised."""
    try:
        await process.kill()
    except Exception:
        pass


class DaytonaWorkspaceBackend(WorkspaceBackend, SupportsFilesystem):
    """A Daytona sandbox behind the Pydantic AI `WorkspaceBackend` protocol."""

    def __init__(
        self,
        *,
        workspace: AsyncSandbox | None = None,
        client: AsyncDaytona | None = None,
        ref: WorkspaceRef | None = None,
        name: str | None = None,
        snapshot: str | None = None,
        auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        network_block_all: bool = False,
    ) -> None:
        if ref is not None and ref.provider != 'daytona':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'daytona'")
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._workspace = workspace
        self._ref = ref if workspace is None else WorkspaceRef(provider='daytona', id=workspace.id)
        self._client = client
        self._owns_client = client is None
        self._name = name
        self._snapshot = snapshot
        self._auto_stop_minutes = auto_stop_minutes
        self._env = dict(env) if env is not None else None
        self._network_block_all = network_block_all
        self._canonical_working_dir: str | None = None
        self._working_dir = absolute_path('working_dir', working_dir)

    @property
    def workspace(self) -> Awaitable[AsyncSandbox]:
        return self._get_workspace()

    async def _get_workspace(self) -> AsyncSandbox:
        if self._workspace is None:
            async with self._lock:
                if self._workspace is None:
                    self._workspace = await self._create_or_attach(self._ref)
                    self._ref = WorkspaceRef(provider='daytona', id=self._workspace.id)
        assert self._workspace is not None
        return self._workspace

    @cached_property
    def _lock(self) -> anyio.Lock:
        return anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def _new_client(self) -> AsyncDaytona:
        if self._client is None:
            self._client = daytona.AsyncDaytona()
            self._owns_client = True
        return self._client

    async def _create_or_attach(self, ref: WorkspaceRef | None) -> AsyncSandbox:
        if ref is not None:
            return await self._attach(ref.id)
        return await self._create()

    @asynccontextmanager
    async def _translated_filesystem_error(self, path: str) -> AsyncGenerator[None]:
        try:
            yield
        except daytona.DaytonaNotFoundError as error:
            raise FileNotFoundError(f'No such file or directory in the Daytona sandbox: {path!r}') from error
        except WorkspaceError:
            raise
        except Exception as error:
            raise self._operation_error(error, f'Could not access {path!r} in the sandbox') from error

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated_filesystem_error(path):
            return await (await self.workspace).fs.download_file(path, _REQUEST_TIMEOUT)

    async def write_bytes(self, path: str, data: bytes) -> None:
        parent = posixpath.dirname(path)
        async with self._translated_filesystem_error(path):
            if parent not in ('', '.', '/'):
                mkdir = await (await self.workspace).process.exec(
                    f'mkdir -p -- {shlex.quote(parent)}', timeout=_REQUEST_TIMEOUT
                )
                if mkdir.exit_code != 0:
                    raise WorkspaceError(mkdir.result or f'Could not create {parent!r}.')
            await (await self.workspace).fs.upload_file(data, path, timeout=_REQUEST_TIMEOUT)

    async def stat(self, path: str) -> FileEntry:
        async with self._translated_filesystem_error(path):
            entry = await (await self.workspace).fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        return FileEntry(
            name=posixpath.basename(path.rstrip('/')),
            path=path,
            is_dir=entry.is_dir,
            size=None if entry.is_dir else entry.size,
        )

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._translated_filesystem_error(path):
            entries = await (await self.workspace).fs.list_files(path, request_timeout=_REQUEST_TIMEOUT)
        return [
            FileEntry(
                name=entry.name,
                path=posixpath.join(path, entry.name),
                is_dir=entry.is_dir,
                size=None if entry.is_dir else entry.size,
            )
            for entry in entries
        ]

    async def make_dir(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.workspace).fs.create_folder(path, '755', request_timeout=_REQUEST_TIMEOUT)

    async def remove(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.workspace).fs.delete_file(path, recursive=True, request_timeout=_REQUEST_TIMEOUT)

    async def exists(self, path: str) -> bool:
        try:
            await (await self.workspace).fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        except daytona.DaytonaNotFoundError:
            return False
        except WorkspaceError:
            raise
        except Exception as error:
            raise self._operation_error(error, f'Could not access {path!r} in the sandbox') from error
        return True

    async def _create(self) -> AsyncSandbox:
        client = await self._new_client()
        try:
            with anyio.fail_after(_CREATE_TIMEOUT):
                return await client.create(
                    daytona.CreateSandboxFromSnapshotParams(
                        name=self._name,
                        snapshot=self._snapshot,
                        env_vars=dict(self._env) if self._env is not None else None,
                        auto_stop_interval=self._auto_stop_minutes,
                        auto_delete_interval=-1,
                        network_block_all=self._network_block_all,
                    ),
                    timeout=_LIFECYCLE_TIMEOUT,
                )
        except BaseException as error:
            await self._close_owned_client()
            if isinstance(error, TimeoutError):
                raise WorkspaceTimeoutError(
                    f'Daytona workspace creation did not complete within {_CREATE_TIMEOUT}s.', timeout=_CREATE_TIMEOUT
                ) from error
            if isinstance(error, Exception):
                raise self._operation_error(error, 'Could not create Daytona workspace') from error
            raise

    async def _attach(self, workspace_id: str) -> AsyncSandbox:
        client = await self._new_client()
        try:
            with anyio.fail_after(_CREATE_TIMEOUT):
                sandbox = await client.get(workspace_id, request_timeout=_REQUEST_TIMEOUT)
                await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
                return sandbox
        except BaseException as error:
            await self._close_owned_client()
            if isinstance(error, TimeoutError):
                raise WorkspaceTimeoutError(
                    f'Daytona workspace connection did not complete within {_CREATE_TIMEOUT}s.', timeout=_CREATE_TIMEOUT
                ) from error
            if isinstance(error, Exception):
                raise self._operation_error(
                    error, f'Could not connect to Daytona workspace {workspace_id!r}', unavailable=True
                ) from error
            raise

    async def _close_owned_client(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    async def disconnect(self) -> None:
        async with self._lock:
            if self._client is None or not self._owns_client:
                return
            try:
                await self._client.close()
            except Exception as error:
                raise self._operation_error(error, 'Could not disconnect from Daytona workspace') from error
            self._client = None
            self._workspace = None
            self._canonical_working_dir = None

    async def working_dir(self) -> str:
        """Return the filesystem-canonical default directory inside the workspace."""
        if self._canonical_working_dir is None:
            sandbox = await self.workspace
            try:
                result = await sandbox.process.exec('pwd -P', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT)
            except Exception as error:
                raise self._operation_error(error, 'Could not determine the working directory') from error
            printed = result.result.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                assert self._ref is not None
                raise WorkspaceError(f'Could not determine the working directory of Daytona sandbox {self._ref.id}.')
            self._canonical_working_dir = printed
        return self._canonical_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        process: _DaytonaProcess | None = None
        with anyio.move_on_after(timeout) as scope:
            try:
                process = await self._start(command, shell=shell, cwd=cwd, env=env)
                return await process.wait()
            finally:
                if process is not None:
                    await _kill_quietly(process)
        assert scope.cancel_called
        raise WorkspaceTimeoutError(
            f'Command timed out after {timeout:g} seconds.',
            stdout=''.join(process.stdout) if process is not None else '',
            stderr=''.join(process.stderr) if process is not None else '',
            timeout=timeout,
        )

    async def _start(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> _DaytonaProcess:
        line = _command_context(
            _command_line(command, shell), absolute_path('cwd', cwd) if cwd is not None else self._working_dir, env
        )
        session_id = f'pydantic-ai-{uuid.uuid4().hex}'
        process = (await self.workspace).process
        created = False
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT):
                await process.create_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                created = True
                response = await process.execute_session_command(
                    session_id,
                    daytona.SessionExecuteRequest(command=line, run_async=True),
                    timeout=_REQUEST_TIMEOUT,
                )
        except BaseException as error:
            if created:
                await cleanup_call(
                    functools.partial(process.delete_session, session_id, request_timeout=_REQUEST_TIMEOUT),
                    timeout=_TEARDOWN_TIMEOUT,
                )
            if isinstance(error, TimeoutError):
                raise WorkspaceError('Daytona command session setup timed out.') from error
            if isinstance(error, Exception):
                raise self._operation_error(error, 'Could not start command', unavailable=True) from error
            raise  # pragma: no cover - cancellation propagates after bounded session cleanup
        stdout: list[str] = []
        stderr: list[str] = []
        logs = asyncio.create_task(
            process.get_session_command_logs_async(session_id, response.cmd_id, stdout.append, stderr.append)
        )
        return _DaytonaProcess(
            _process=process,
            _backend=self,
            _session_id=session_id,
            _command_id=response.cmd_id,
            stdout=stdout,
            stderr=stderr,
            _logs=logs,
        )

    @staticmethod
    def _operation_error(error: Exception, context: str, *, unavailable: bool = False) -> WorkspaceError:
        return _operation_error(error, context, unavailable=unavailable)


def _operation_error(error: Exception, context: str, *, unavailable: bool = False) -> WorkspaceError:
    if isinstance(error, (daytona.DaytonaAuthenticationError, daytona.DaytonaAuthorizationError)):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if unavailable and isinstance(error, daytona.DaytonaNotFoundError):
        return WorkspaceUnavailableError(f'{context}: the workspace does not exist or is no longer available.')
    return WorkspaceError(f'{context}: {type(error).__name__}: {error}')

"""A Daytona sandbox behind Pydantic AI's sandbox protocols.

External assumptions last verified 2026-09-08 against Daytona Python SDK 0.198.0:

* `AsyncDaytona.create`, `get`, `delete(wait=True)`, and `close` own sandbox lifecycle;
  `get` accepts a sandbox ID or name:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* process sessions provide asynchronous execution, separate stdout and stderr callbacks,
  exit status, and deletion as the per-command kill mechanism:
  https://www.daytona.io/docs/en/python-sdk/async/async-process/
* `sandbox.fs` provides metadata, byte upload/download, and directory operations:
  https://www.daytona.io/docs/en/python-sdk/async/async-file-system/
* `auto_stop_interval` together with `auto_delete_interval=0` provides the server-side
  backstop for abandoned owned sandboxes:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/

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
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
from pydantic_ai.sandboxes import (
    CommandResult,
    FileEntry,
    SandboxBackend,
    SandboxError,
    SandboxRef,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SupportsFilesystem,
)

from pydantic_ai_harness._sandbox_provider import absolute_path, cleanup_call, raise_after_cleanup

if TYPE_CHECKING:
    from daytona import AsyncDaytona, AsyncSandbox

    # Not re-exported at the package root; typing-only, so the private path never runs.
    from daytona._async.process import AsyncProcess
    from pydantic_ai.sandboxes import SandboxCommand

__all__ = (
    'DaytonaSandboxAuthError',
    'DaytonaSandboxBackend',
    'DaytonaSandboxError',
    'DaytonaSandboxUnavailableError',
)

DEFAULT_AUTO_STOP_MINUTES = 60

try:
    import daytona
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[daytona]` to use DaytonaSandbox.') from error

_AUTH_MESSAGE = 'Daytona rejected the credentials. Set DAYTONA_API_KEY and try again.'
# Bound sandbox acquisition so a wedged control plane cannot hang creation or connection.
_CREATE_TIMEOUT = 120
# Bound routine SDK requests so a stalled control plane cannot hang an operation.
_REQUEST_TIMEOUT = 30
# Bound provider lifecycle RPCs such as create, start, and delete.
_LIFECYCLE_TIMEOUT = 60.0
# Bound cleanup RPCs so teardown cannot wedge the caller.
_TEARDOWN_TIMEOUT = 30.0


class DaytonaSandboxError(SandboxError):
    """A recoverable Daytona provider operation failed."""


class DaytonaSandboxUnavailableError(DaytonaSandboxError, SandboxUnavailableError):
    """The referenced Daytona sandbox is no longer available."""


class DaytonaSandboxAuthError(DaytonaSandboxError, SandboxUnavailableError):
    """Daytona rejected the configured credentials."""


def _command_line(command: SandboxCommand, shell: bool) -> str:
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
    _backend: DaytonaSandboxBackend
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
            raise self._backend.operation_error(error, 'Could not read the command result', unavailable=True) from error
        if command.exit_code is None:
            raise DaytonaSandboxError('Daytona closed the command output before reporting an exit status.')
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
        await raise_after_cleanup(
            self._backend.operation_error(
                error, f'Could not kill command session {self._session_id!r}', unavailable=True
            ),
            cause=error,
        )


async def _kill_quietly(process: _DaytonaProcess) -> None:
    """Best-effort kill whose failure must not mask the outcome being raised."""
    try:
        await process.kill()
    except Exception:
        pass


class DaytonaSandboxBackend(SandboxBackend, SupportsFilesystem):
    """A Daytona sandbox behind Pydantic AI's `SandboxBackend` protocol.

    Building one does no I/O. It holds settings plus, optionally, the identity of a sandbox that
    already exists; the first operation creates or attaches, once, and everything after that
    reuses the same environment. Reach the live Daytona sandbox through
    [`get_sandbox`][pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend.get_sandbox], which you
    await to create or attach before using it.

    The backend owns its `AsyncDaytona` client. Nothing here deletes a sandbox on its own:
    Daytona stops an idle one after `auto_stop_minutes` and deletes it immediately after that.
    Call [`close`][pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend.close] with
    `terminate=True` to end one sooner.

    Daytona delivers output through callbacks, so complete command results are buffered while a
    command runs.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `Sandbox.wrap` call.

    Args:
        ref: Identity of an existing sandbox to attach to on first use.
        name: Daytona name to attach to on first use, creating it only if there is none. This is
            what lets several runs share one environment, and what makes a durable retry attach
            rather than provision a second sandbox. Ignored when `ref` is given.
        snapshot: Daytona snapshot a newly created sandbox starts from.
        auto_stop_minutes: How long Daytona leaves a newly created sandbox idle before stopping
            it; it is deleted immediately after.
        working_dir: Directory commands run in; the sandbox's own default when `None`.
        env: Environment variables set on a newly created sandbox.
        network_block_all: Whether a newly created sandbox is cut off from the network.
    """

    def __init__(
        self,
        *,
        ref: SandboxRef | None = None,
        name: str | None = None,
        snapshot: str | None = None,
        auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        network_block_all: bool = False,
    ) -> None:
        self._ref = ref
        self._name = name
        self._snapshot = snapshot
        self._auto_stop_minutes = auto_stop_minutes
        self._env = dict(env) if env is not None else None
        self._network_block_all = network_block_all
        self._canonical_working_dir: str | None = None
        self._working_dir = absolute_path('working_dir', working_dir)
        self._client: AsyncDaytona | None = None
        self._live: AsyncSandbox | None = None
        self._owned = False
        self._closed = False
        self._lock = anyio.Lock()

    async def get_sandbox(self) -> AsyncSandbox:
        """Create or attach on first use and return the native Daytona sandbox."""
        async with self._lock:
            if self._live is None:
                if self._ref is not None:
                    self._live = await self._attach(self._ref.sandbox_id)
                elif self._name is not None:
                    self._live = await self._create_or_attach_by_name(self._name)
                else:
                    self._live = await self._create()
                self._ref = SandboxRef(sandbox_id=self._live.id)
        return self._live

    @property
    def ref(self) -> SandboxRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _translated_filesystem_error(self, path: str) -> AsyncGenerator[None]:
        try:
            yield
        except daytona.DaytonaNotFoundError as error:
            raise FileNotFoundError(f'No such file or directory in the Daytona sandbox: {path!r}') from error
        except SandboxError:
            raise
        except Exception as error:
            raise self.operation_error(error, f'Could not access {path!r} in the sandbox') from error

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated_filesystem_error(path):
            return await (await self.get_sandbox()).fs.download_file(path, _REQUEST_TIMEOUT)

    async def write_bytes(self, path: str, data: bytes) -> None:
        parent = posixpath.dirname(path)
        async with self._translated_filesystem_error(path):
            if parent not in ('', '.', '/'):
                mkdir = await (await self.get_sandbox()).process.exec(
                    f'mkdir -p -- {shlex.quote(parent)}', timeout=_REQUEST_TIMEOUT
                )
                if mkdir.exit_code != 0:
                    raise DaytonaSandboxError(mkdir.result or f'Could not create {parent!r}.')
            await (await self.get_sandbox()).fs.upload_file(data, path, timeout=_REQUEST_TIMEOUT)

    async def stat(self, path: str) -> FileEntry:
        async with self._translated_filesystem_error(path):
            entry = await (await self.get_sandbox()).fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        return FileEntry(
            name=posixpath.basename(path.rstrip('/')),
            path=path,
            is_dir=entry.is_dir,
            size=None if entry.is_dir else entry.size,
        )

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._translated_filesystem_error(path):
            entries = await (await self.get_sandbox()).fs.list_files(path, request_timeout=_REQUEST_TIMEOUT)
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
            await (await self.get_sandbox()).fs.create_folder(path, '755', request_timeout=_REQUEST_TIMEOUT)

    async def remove(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.get_sandbox()).fs.delete_file(path, recursive=True, request_timeout=_REQUEST_TIMEOUT)

    async def exists(self, path: str) -> bool:
        try:
            await (await self.get_sandbox()).fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        except daytona.DaytonaNotFoundError:
            return False
        except SandboxError:
            raise
        except Exception as error:
            raise self.operation_error(error, f'Could not access {path!r} in the sandbox') from error
        return True

    async def _new_client(self) -> AsyncDaytona:
        client = daytona.AsyncDaytona()
        self._client = client
        return client

    async def _create(self, name: str | None = None) -> AsyncSandbox:
        """Create a sandbox with Daytona's automatic stop and delete backstop."""
        client = await self._new_client()
        try:
            # Cancellation can orphan a sandbox until the paired auto-stop and immediate
            # auto-delete settings reap it. A stable name lets a retry reconnect meanwhile.
            with anyio.fail_after(_CREATE_TIMEOUT):
                sandbox = await client.create(
                    daytona.CreateSandboxFromSnapshotParams(
                        name=name,
                        snapshot=self._snapshot,
                        env_vars=dict(self._env) if self._env is not None else None,
                        auto_stop_interval=self._auto_stop_minutes,
                        auto_delete_interval=0,
                        network_block_all=self._network_block_all,
                    ),
                    timeout=_LIFECYCLE_TIMEOUT,
                )
        except BaseException as error:
            await cleanup_call(client.close, timeout=_TEARDOWN_TIMEOUT)
            self._client = None
            if isinstance(error, TimeoutError):
                raise DaytonaSandboxError(
                    f'Daytona sandbox creation did not complete within {_CREATE_TIMEOUT}s.'
                ) from error
            if isinstance(error, Exception):
                raise self.operation_error(error, 'Could not create Daytona sandbox') from error
            raise  # pragma: no cover - cancellation propagates after bounded client cleanup
        self._owned = True
        return sandbox

    async def _attach(self, sandbox_id_or_name: str) -> AsyncSandbox:
        """Attach by Daytona sandbox ID or name, starting a stopped sandbox.

        A sandbox that is gone raises rather than resolving to a dead environment. Nothing is
        recreated in its place -- a run that expected files there must be told they are gone,
        not handed an empty workspace.
        """
        client = await self._new_client()
        try:
            with anyio.fail_after(_CREATE_TIMEOUT):
                sandbox = await client.get(sandbox_id_or_name, request_timeout=_REQUEST_TIMEOUT)
                await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
        except BaseException as error:
            await cleanup_call(client.close, timeout=_TEARDOWN_TIMEOUT)
            self._client = None
            if isinstance(error, TimeoutError):
                raise DaytonaSandboxError(
                    f'Daytona sandbox connection did not complete within {_CREATE_TIMEOUT}s.'
                ) from error
            if isinstance(error, Exception):
                raise self.operation_error(
                    error, f'Could not connect to Daytona sandbox {sandbox_id_or_name!r}', unavailable=True
                ) from error
            raise  # pragma: no cover - cancellation propagates after bounded client cleanup
        return sandbox

    async def _create_or_attach_by_name(self, name: str) -> AsyncSandbox:
        """Attach by stable name, create if absent, then attach again after a lost race."""
        try:
            return await self._attach(name)
        except DaytonaSandboxUnavailableError:
            pass
        try:
            return await self._create(name)
        except DaytonaSandboxError as create_error:
            try:
                return await self._attach(name)
            except DaytonaSandboxUnavailableError:
                raise create_error

    def _describe(self) -> str:
        """How to name this sandbox in an error.

        Every caller runs after `get_sandbox`, which sets `ref` alongside the live handle, so the
        other two spellings are only reachable if that ever stops being true. `lax no cover`
        for the same reason: they are a fallback, not a path tests should have to reach.
        """
        if self._ref is not None:
            return repr(self._ref.sandbox_id)
        return f'named {self._name!r}' if self._name is not None else 'that was never started'  # pragma: lax no cover

    async def close(self, *, terminate: bool) -> None:
        """Close the SDK client, deleting the sandbox first when this backend owns it."""
        client, sandbox = self._client, self._live
        if self._closed or sandbox is None:
            # Never used, so there is no client to close and no sandbox to delete. Resolving one
            # here just to close it would create the very sandbox being released.
            return
        reconnect = client is None
        if client is None:
            client = await self._new_client()
        deletion_error: Exception | None = None
        if terminate and self._owned:

            async def delete() -> None:
                # SDK sandbox handles retain their original client. A retry needs a fresh handle.
                target = await client.get(sandbox.id, request_timeout=_REQUEST_TIMEOUT) if reconnect else sandbox
                await client.delete(target, timeout=_LIFECYCLE_TIMEOUT, wait=True)

            deletion_error = await cleanup_call(delete, timeout=_TEARDOWN_TIMEOUT)
            if self._is_not_found(deletion_error):
                deletion_error = None
        close_error = await cleanup_call(client.close, timeout=_TEARDOWN_TIMEOUT)
        if close_error is None:
            self._client = None
        self._closed = deletion_error is None and close_error is None
        error = deletion_error or close_error
        if error is not None:
            await raise_after_cleanup(
                self.operation_error(error, f'Could not close Daytona sandbox {self._describe()}'), cause=error
            )

    @staticmethod
    async def delete_by_id(sandbox_id: str) -> None:
        """Delete a sandbox by ID without starting it."""
        client = daytona.AsyncDaytona()
        operation_error: Exception | None = None
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT):
                sandbox = await client.get(sandbox_id, request_timeout=_REQUEST_TIMEOUT)
            operation_error = await cleanup_call(
                functools.partial(client.delete, sandbox, timeout=_LIFECYCLE_TIMEOUT, wait=True),
                timeout=_TEARDOWN_TIMEOUT,
            )
        except Exception as error:
            operation_error = error
        finally:
            close_error = await cleanup_call(client.close, timeout=_TEARDOWN_TIMEOUT)
        if DaytonaSandboxBackend._is_not_found(operation_error):
            operation_error = None
        error = operation_error or close_error
        if error is not None:
            await raise_after_cleanup(
                DaytonaSandboxBackend.operation_error(error, f'Could not delete Daytona sandbox {sandbox_id!r}'),
                cause=error,
            )

    async def working_dir(self) -> str:
        """Return the filesystem-canonical default directory inside the sandbox."""
        if self._canonical_working_dir is None:
            sandbox = await self.get_sandbox()
            try:
                result = await sandbox.process.exec('pwd -P', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT)
            except Exception as error:
                raise self.operation_error(error, 'Could not determine the working directory') from error
            printed = result.result.rstrip('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise DaytonaSandboxError(
                    f'Could not determine the working directory of Daytona sandbox {self._describe()}.'
                )
            self._canonical_working_dir = printed
        return self._canonical_working_dir

    async def run(
        self,
        command: SandboxCommand,
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
        raise SandboxTimeoutError(
            f'Command timed out after {timeout:g} seconds.',
            stdout=''.join(process.stdout) if process is not None else '',
            stderr=''.join(process.stderr) if process is not None else '',
            timeout=timeout,
        )

    async def _start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> _DaytonaProcess:
        line = _command_context(
            _command_line(command, shell), absolute_path('cwd', cwd) if cwd is not None else self._working_dir, env
        )
        session_id = f'pydantic-ai-{uuid.uuid4().hex}'
        process = (await self.get_sandbox()).process
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
                raise DaytonaSandboxError('Daytona command session setup timed out.') from error
            if isinstance(error, Exception):
                raise self.operation_error(error, 'Could not start command', unavailable=True) from error
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
    def operation_error(error: Exception, context: str, *, unavailable: bool = False) -> DaytonaSandboxError:
        if isinstance(error, (daytona.DaytonaAuthenticationError, daytona.DaytonaAuthorizationError)):
            return DaytonaSandboxAuthError(_AUTH_MESSAGE)
        if unavailable and isinstance(error, daytona.DaytonaNotFoundError):
            return DaytonaSandboxUnavailableError(f'{context}: the sandbox does not exist or is no longer available.')
        return DaytonaSandboxError(f'{context}: {type(error).__name__}: {error}')

    @staticmethod
    def _is_not_found(error: Exception | None) -> bool:
        if error is None:
            return False
        return isinstance(error, daytona.DaytonaNotFoundError)

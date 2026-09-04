"""A Daytona sandbox behind Pydantic AI's sandbox protocols.

External assumptions last verified 2026-09-01 against Daytona Python SDK 0.198.0:

* `AsyncDaytona.create`, `get`, `delete(wait=True)`, and `close` own sandbox lifecycle;
  `get` accepts a sandbox ID or name:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* process sessions provide asynchronous execution, separate stdout and stderr callbacks,
  exit status, and deletion as the per-command kill mechanism:
  https://www.daytona.io/docs/en/python-sdk/async/async-process/
* `sandbox.fs` provides metadata, byte upload/download, and directory operations, while
  `sandbox.get_work_dir()` reports the configured working directory:
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
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager
from types import ModuleType
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
)

from pydantic_ai_harness._sandbox_provider import absolute_path, cleanup_call, raise_after_cleanup

if TYPE_CHECKING:
    from daytona import AsyncDaytona, AsyncSandbox

    # Not re-exported at the package root; typing-only, so the private path never runs.
    from daytona._async.process import AsyncProcess
    from pydantic_ai.sandboxes import SandboxCommand, SupportsFilesystem

__all__ = (
    'DaytonaSandboxAuthError',
    'DaytonaSandboxBackend',
    'DaytonaSandboxError',
    'DaytonaSandboxUnavailableError',
)

DEFAULT_AUTO_STOP_MINUTES = 60

_MISSING_DAYTONA = (
    'The `daytona` package is required for DaytonaSandbox. Install it with `uv add "pydantic-ai-harness[daytona]"`.'
)
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


class _DaytonaProcess:
    """A command running in a Daytona process session."""

    def __init__(
        self,
        process: AsyncProcess,
        *,
        backend: DaytonaSandboxBackend,
        session_id: str,
        command_id: str,
        stdout: list[str],
        stderr: list[str],
        logs: asyncio.Task[None],
        deadline: float | None,
        started: float,
    ) -> None:
        self._process = process
        self._backend = backend
        self._session_id = session_id
        self._command_id = command_id
        self._stdout = stdout
        self._stderr = stderr
        self._logs = logs
        self._deadline = deadline
        self._started = started

    async def wait(self) -> CommandResult:
        """Wait for the command and return its result."""
        return await self._settle()

    async def _settle(self) -> CommandResult:
        completed = False
        try:
            # `fail_after` would make its timeout indistinguishable from an SDK `TimeoutError` translated below.
            with anyio.move_on_after(self._remaining()):
                await self._logs
                completed = True
        except Exception as error:
            raise self._backend.operation_error(error, 'Could not read the command output', unavailable=True) from error
        if not completed:
            raise await self._timeout_error()
        # The exit-status RPC shares the command's deadline: a wedged status lookup after the
        # logs completed must not extend the promised bound.
        command = None
        try:
            with anyio.move_on_after(self._remaining()):
                command = await self._process.get_session_command(
                    self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
                )
        except Exception as error:
            raise self._backend.operation_error(error, 'Could not read the command result', unavailable=True) from error
        if command is None:
            raise await self._timeout_error()
        if command.exit_code is None:
            raise DaytonaSandboxError('Daytona closed the command output before reporting an exit status.')
        result = CommandResult(
            exit_code=command.exit_code,
            stdout=''.join(self._stdout),
            stderr=''.join(self._stderr),
        )
        await _kill_quietly(self)
        return result

    def _remaining(self) -> float | None:
        return None if self._deadline is None else self._deadline - (time.monotonic() - self._started)

    async def _timeout_error(self) -> SandboxTimeoutError:
        await _kill_quietly(self)
        assert self._deadline is not None
        return SandboxTimeoutError(
            f'Command timed out after {self._deadline:g} seconds and was killed.',
            stdout=''.join(self._stdout),
            stderr=''.join(self._stderr),
            timeout=self._deadline,
        )

    async def kill(self) -> None:
        """Delete the Daytona process session, which kills its command."""
        from daytona import DaytonaNotFoundError

        error = await cleanup_call(
            functools.partial(self._process.delete_session, self._session_id, request_timeout=_REQUEST_TIMEOUT),
            timeout=_TEARDOWN_TIMEOUT,
        )
        if error is None or isinstance(error, DaytonaNotFoundError):
            self._logs.cancel()
            return
        await raise_after_cleanup(
            self._backend.operation_error(
                error, f'Could not kill command session {self._session_id!r}', unavailable=True
            )
        )


async def _kill_quietly(process: _DaytonaProcess) -> None:
    """Best-effort kill whose failure must not mask the outcome being raised."""
    try:
        await process.kill()
    except Exception:
        pass


def _require_daytona() -> ModuleType:
    """Import the optional `daytona` package, or explain how to install it."""
    try:
        import daytona
    except ImportError as error:
        raise DaytonaSandboxError(_MISSING_DAYTONA) from error
    return daytona


class DaytonaSandboxBackend(SandboxBackend):
    """A Daytona sandbox behind Pydantic AI's `SandboxBackend` protocol.

    Building one does no I/O. It holds settings plus, optionally, the identity of a sandbox that
    already exists; the first operation creates or attaches, once, and everything after that
    reuses the same environment. Reach the live Daytona sandbox through
    [`sandbox`][pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend.sandbox], which you
    can only await — so no operation can run against a sandbox that does not exist yet.

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
        self._working_dir = absolute_path('working_dir', working_dir)
        self._client: AsyncDaytona | None = None
        self._live: AsyncSandbox | None = None
        self._owned = False
        self._closed = False
        self._lock = anyio.Lock()

    @property
    def sandbox(self) -> Awaitable[AsyncSandbox]:
        """The live Daytona sandbox, created or attached on first use.

        Awaitable and never a plain value: every operation has to go through the step that
        makes the sandbox exist, so none of them can skip it.
        """
        return self._resolve()

    async def _resolve(self) -> AsyncSandbox:
        async with self._lock:
            if self._live is None:
                # Guarded once, here: everything that touches Daytona runs after this.
                _require_daytona()
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
        from daytona import DaytonaNotFoundError

        try:
            yield
        except DaytonaNotFoundError as error:
            raise FileNotFoundError(f'No such file or directory in the Daytona sandbox: {path!r}') from error
        except Exception as error:
            raise self.operation_error(error, f'Could not access {path!r} in the sandbox') from error

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated_filesystem_error(path):
            return await (await self.sandbox).fs.download_file(path, _REQUEST_TIMEOUT)

    async def write_bytes(self, path: str, data: bytes) -> None:
        parent = posixpath.dirname(path)
        async with self._translated_filesystem_error(path):
            if parent not in ('', '.', '/'):
                mkdir = await (await self.sandbox).process.exec(
                    f'mkdir -p -- {shlex.quote(parent)}', timeout=_REQUEST_TIMEOUT
                )
                if mkdir.exit_code != 0:
                    raise DaytonaSandboxError(mkdir.result or f'Could not create {parent!r}.')
            await (await self.sandbox).fs.upload_file(data, path, timeout=_REQUEST_TIMEOUT)

    async def stat(self, path: str) -> FileEntry:
        async with self._translated_filesystem_error(path):
            entry = await (await self.sandbox).fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        return FileEntry(
            name=posixpath.basename(path.rstrip('/')),
            path=path,
            is_dir=entry.is_dir,
            size=None if entry.is_dir else entry.size,
        )

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._translated_filesystem_error(path):
            entries = await (await self.sandbox).fs.list_files(path, request_timeout=_REQUEST_TIMEOUT)
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
            await (await self.sandbox).fs.create_folder(path, '755', request_timeout=_REQUEST_TIMEOUT)

    async def remove(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).fs.delete_file(path, recursive=True, request_timeout=_REQUEST_TIMEOUT)

    async def exists(self, path: str) -> bool:
        from daytona import DaytonaNotFoundError

        try:
            await (await self.sandbox).fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        except DaytonaNotFoundError:
            return False
        except Exception as error:
            raise self.operation_error(error, f'Could not access {path!r} in the sandbox') from error
        return True

    async def _new_client(self) -> AsyncDaytona:
        client = _require_daytona().AsyncDaytona()
        self._client = client
        return client

    async def _create(self, name: str | None = None) -> AsyncSandbox:
        """Create a sandbox with Daytona's automatic stop and delete backstop."""
        from daytona import CreateSandboxFromSnapshotParams

        client = await self._new_client()
        try:
            # Cancellation can orphan a sandbox until the paired auto-stop and immediate
            # auto-delete settings reap it. A stable name lets a retry reconnect meanwhile.
            with anyio.fail_after(_CREATE_TIMEOUT):
                sandbox = await client.create(
                    CreateSandboxFromSnapshotParams(
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
        recreated in its place — a run that expected files there must be told they are gone,
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

        Every caller runs after `_resolve`, which sets `ref` alongside the live handle, so the
        other two spellings are only reachable if that ever stops being true. `lax no cover`
        for the same reason: they are a fallback, not a path tests should have to reach.
        """
        if self._ref is not None:
            return repr(self._ref.sandbox_id)
        return f'named {self._name!r}' if self._name is not None else 'that was never started'  # pragma: lax no cover

    async def close(self, *, terminate: bool) -> None:
        """Close the SDK client, deleting the sandbox first when this backend owns it."""
        client, sandbox = self._client, self._live
        if self._closed or client is None or sandbox is None:
            # Never used, so there is no client to close and no sandbox to delete. Resolving one
            # here just to close it would create the very sandbox being released.
            return
        deletion_error: Exception | None = None
        if terminate and self._owned:
            deletion_error = await cleanup_call(
                functools.partial(client.delete, sandbox, timeout=_LIFECYCLE_TIMEOUT, wait=True),
                timeout=_TEARDOWN_TIMEOUT,
            )
            if self._is_not_found(deletion_error):
                deletion_error = None
        close_error = await cleanup_call(client.close, timeout=_TEARDOWN_TIMEOUT)
        self._closed = close_error is None
        error = deletion_error or close_error
        if error is not None:
            await raise_after_cleanup(
                self.operation_error(error, f'Could not close Daytona sandbox {self._describe()}')
            )

    @staticmethod
    async def delete_by_id(sandbox_id: str) -> None:
        """Delete a sandbox by ID without starting it."""
        try:
            from daytona import AsyncDaytona
        except ImportError as error:
            raise DaytonaSandboxError(_MISSING_DAYTONA) from error
        client = AsyncDaytona()
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
        if DaytonaSandboxBackend._is_not_found(operation_error):
            operation_error = None
        close_error = await cleanup_call(client.close, timeout=_TEARDOWN_TIMEOUT)
        error = operation_error or close_error
        if error is not None:
            await raise_after_cleanup(
                DaytonaSandboxBackend.operation_error(error, f'Could not delete Daytona sandbox {sandbox_id!r}')
            )

    async def working_dir(self) -> str:
        """Return the sandbox's native absolute working directory."""
        # The probe is an idempotent read: overlapping first calls may each ask, get the
        # same answer, and the cache converges. No lock needed.
        if self._working_dir is None:
            try:
                sandbox = await self.sandbox
                discovered = await sandbox.get_work_dir()
            except Exception as error:
                raise self.operation_error(error, 'Could not determine the working directory') from error
            if not posixpath.isabs(discovered):
                raise DaytonaSandboxError(
                    f'Could not determine the working directory of Daytona sandbox {self._describe()}.'
                )
            self._working_dir = posixpath.normpath(discovered)
        return self._working_dir

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        process = await self._start(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
        try:
            return await process.wait()
        except SandboxTimeoutError:
            # The timeout path already killed the session, so the generic handler must not kill it twice.
            raise
        except BaseException:
            await _kill_quietly(process)
            raise

    async def _start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _DaytonaProcess:
        from daytona import SessionExecuteRequest

        line = _command_context(_command_line(command, shell), absolute_path('cwd', cwd), env)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        started = time.monotonic()
        session_id = f'pydantic-ai-{uuid.uuid4().hex}'
        process = (await self.sandbox).process
        created = False
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT):
                await process.create_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                created = True
                response = await process.execute_session_command(
                    session_id,
                    SessionExecuteRequest(command=line, run_async=True),
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
            process,
            backend=self,
            session_id=session_id,
            command_id=response.cmd_id,
            stdout=stdout,
            stderr=stderr,
            logs=logs,
            deadline=timeout,
            started=started,
        )

    @staticmethod
    def operation_error(error: Exception, context: str, *, unavailable: bool = False) -> DaytonaSandboxError:
        # Daytona's NotFound is ambiguous: the call site knows whether it asked about a file or a sandbox.
        try:
            from daytona import DaytonaAuthenticationError, DaytonaAuthorizationError, DaytonaNotFoundError
        except ImportError:  # pragma: no cover - an active backend already imported Daytona
            return DaytonaSandboxError(f'{context}: {type(error).__name__}: {error}')
        if isinstance(error, (DaytonaAuthenticationError, DaytonaAuthorizationError)):
            return DaytonaSandboxAuthError(_AUTH_MESSAGE)
        if unavailable and isinstance(error, DaytonaNotFoundError):
            return DaytonaSandboxUnavailableError(f'{context}: the sandbox does not exist or is no longer available.')
        return DaytonaSandboxError(f'{context}: {type(error).__name__}: {error}')

    @staticmethod
    def _is_not_found(error: Exception | None) -> bool:
        if error is None:
            return False
        try:
            from daytona import DaytonaNotFoundError
        except ImportError:  # pragma: no cover - an active backend already imported Daytona
            return False
        return isinstance(error, DaytonaNotFoundError)


if TYPE_CHECKING:
    _backend = DaytonaSandboxBackend()
    _backend_conforms: SandboxBackend = _backend
    _filesystem_backend_conforms: SupportsFilesystem = _backend

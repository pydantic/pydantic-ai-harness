"""A Daytona sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-08 against Daytona Python SDK 0.198.0:

* `AsyncDaytona.create`, `get`, and `close` and `AsyncSandbox.start` cover the lifecycle used
  here; `get` accepts a sandbox ID or name. The SDK lazily reopens its HTTP session after
  `AsyncDaytona.close()`. Reusing a borrowed client or sandbox after its owner closes it can
  leak an aiohttp session:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* process sessions provide asynchronous execution, separate stdout and stderr callbacks,
  exit status, and deletion as the per-command kill mechanism:
  https://www.daytona.io/docs/en/python-sdk/async/async-process/
* the session log stream ends a stream that did not end in a newline with one (`printf abc` streams
  `abc` plus a newline; observed live 2026-09-25, and the SDK passes frames through unchanged), so each command
  prints an end marker last on both streams and the output is cut at it.
* the log stream's demultiplexer (`daytona/_utils/stream.py` `_std_demux_loop`) misreads a stream prefix
  that ends a websocket frame, injecting the prefix bytes into the output and misrouting it
  (observed live 2026-09-26 above ~4 KB); the non-follow `get_session_command_logs` is exact, so a
  finished command's output comes from it.
* `sandbox.fs` provides metadata, byte upload/download, and directory operations:
  https://www.daytona.io/docs/en/python-sdk/async/async-file-system/
* `auto_stop_interval` is a creation-time setting; left unset, Daytona stops an idle sandbox after
  15 minutes, archives it after 7 days stopped, and never deletes it:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* `FileInfo` has `is_dir`, and the SDK does not say whether it follows a symlink.
* SDK errors are typed by HTTP status (`DaytonaNotFoundError` 404, `DaytonaAuthenticationError`
  401, `DaytonaAuthorizationError` 403, `DaytonaValidationError` 400, `DaytonaConflictError` 409,
  `DaytonaRateLimitError` 429); transport failures become `DaytonaConnectionError` or
  `DaytonaTimeoutError`, and anything else a plain `DaytonaError`.

Re-check those sources and the installed 0.198.0 signatures before changing lifecycle,
command, filesystem, or error handling.
"""

from __future__ import annotations

import asyncio
import logging
import math
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import anyio
import anyio.lowlevel
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness._workspace_provider import absolute_path, command_argv

if TYPE_CHECKING:
    from daytona import AsyncDaytona, AsyncSandbox

    # Not re-exported at the package root; typing-only, so the private path never runs.
    from daytona._async.process import AsyncProcess
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('DaytonaSandboxBackend',)

try:
    import daytona
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[daytona]` to use DaytonaSandbox.') from error

_AUTH_MESSAGE = (
    'Daytona rejected the credentials. Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
)
# Bound sandbox acquisition so a wedged control plane cannot hang creation or connection.
_CREATE_TIMEOUT = 120
# Bound routine SDK requests so a stalled control plane cannot hang an operation.
_REQUEST_TIMEOUT = 30
# Bound provider lifecycle RPCs such as create, start, and delete.
_LIFECYCLE_TIMEOUT = 60.0
# Bound cleanup RPCs so teardown cannot wedge the caller.
_TEARDOWN_TIMEOUT = 30.0
# Pause between attempts to delete a command session.
_RETRY_DELAY = 1.0

_logger = logging.getLogger(__name__)


async def _delete_session(process: AsyncProcess, session_id: str) -> None:
    """Delete a command session, which kills its command, even when the caller was cancelled or timed out.

    A surviving session keeps its command running in the sandbox, so a failed deletion is retried
    until `_TEARDOWN_TIMEOUT`, then logged. Its failure must not replace the outcome the caller is
    already raising or returning.
    """
    last_error: Exception | None = None
    with anyio.CancelScope(shield=True), anyio.move_on_after(_TEARDOWN_TIMEOUT):
        while True:
            try:
                await process.delete_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                return
            except daytona.DaytonaNotFoundError:
                return  # already gone, and its command with it
            except Exception as error:
                last_error = error
            await anyio.sleep(_RETRY_DELAY)
    _logger.warning(
        'Could not delete Daytona command session %s; its command may still be running.',
        session_id,
        exc_info=last_error,
    )


def _command_line(
    command: WorkspaceCommand, *, shell: bool, cwd: str | None, env: Mapping[str, str], marker: str
) -> str:
    """Build the session command: the quoted argv, with the environment and directory applied.

    The argv runs under a `sh` that prints `marker` last on stdout and stderr and exits with the
    argv's status, so `_until_marker` can drop what Daytona appends after a stream's last byte.
    `env` runs inside that `sh`: a `sh` such as dash drops variables whose names are not shell
    identifiers from the environment it passes on.
    """
    argv = command_argv(command, shell)
    if env:
        # `--` ends `env`'s options, so a name starting with `-` is not read as one.
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    prefix = f'cd -- {shlex.quote(cwd)} && ' if cwd is not None else ''
    script = f'{prefix}"$@" </dev/null; status=$?; printf %s {marker}; printf %s {marker} >&2; exit "$status"'
    return shlex.join(['sh', '-c', script, 'sh', *argv])


def _until_marker(chunks: list[str], marker: str) -> str:
    """The stream's output before `marker`; all of it when the command never printed the marker."""
    output = ''.join(chunks)
    head, found, _ = output.rpartition(marker)
    return head if found else output


@dataclass(kw_only=True)
class _DaytonaProcess:
    """Output and session identity for a single command."""

    _process: AsyncProcess
    _sandbox: AsyncSandbox
    _session_id: str
    _command_id: str
    stdout: list[str]
    stderr: list[str]
    marker: str
    _logs: asyncio.Task[None]

    async def wait(self) -> CommandResult:
        try:
            await self._logs
            command = await self._process.get_session_command(
                self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
            )
        except Exception as error:
            await _raise_failure(self._sandbox, error, 'Could not read the command result')
        if command.exit_code is None:
            raise WorkspaceError('Daytona closed the command output before reporting an exit status.')
        # The streamed copy is only for partial output on a timeout: SDK 0.198.0's stream
        # demultiplexer misreads a stream prefix split across websocket frames, corrupting output
        # over a few KB. The finished command's stored logs are exact.
        try:
            logs = await self._process.get_session_command_logs(
                self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
            )
        except Exception as error:
            await _raise_failure(self._sandbox, error, 'Could not read the command output')
        return CommandResult(
            exit_code=command.exit_code,
            stdout=_until_marker([logs.stdout or ''], self.marker),
            stderr=_until_marker([logs.stderr or ''], self.marker),
        )

    async def kill(self) -> None:
        """Delete the Daytona process session, which kills its command."""
        await _delete_session(self._process, self._session_id)
        self._logs.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(self._logs, return_exceptions=True)


class DaytonaSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A [Daytona](https://www.daytona.io) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Daytona sandbox, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox, and the typed
    `daytona.AsyncSandbox` is available through `get_client()`. The backend does not stop or
    delete the sandbox; that is the application's job, through the native handle.

    Commands run in Daytona process sessions, with complete output returned after they finish. A
    shell string runs under `/bin/sh -c`; an argv sequence is shell-quoted into the session command. The deadline is enforced
    client-side, and the session is deleted, which kills its command, when the deadline expires or
    the caller is cancelled.

    Daytona answers a request for a missing path and a request to a deleted sandbox with the same
    not-found error, so a failed request is followed by one control-plane lookup of the sandbox:
    a missing path raises `FileNotFoundError`, a deleted sandbox raises `WorkspaceUnavailableError`.
    Rejected credentials and a sandbox Daytona refuses to create (an unknown snapshot, say) raise
    `WorkspaceUnavailableError`; any other request Daytona refuses raises `WorkspaceError`;
    connection failures, rate limits, and other SDK errors propagate unchanged.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant workspace call.

    Args:
        workspace: A live `daytona.AsyncSandbox` you already have. Whoever created it owns deleting it.
        client: A `daytona.AsyncDaytona` API client to create or attach with. The caller owns closing
            it. Without one, the backend opens its own from the environment on first use and closes
            it in `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use.
        snapshot: Daytona snapshot a newly created sandbox starts from; Daytona's default when `None`.
        auto_stop_interval: Idle minutes before Daytona stops a newly created sandbox; `0` disables
            it, and `None` keeps Daytona's default (15 minutes).
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            They are also set on a newly created sandbox.
        network_block_all: Whether a newly created sandbox is blocked from outbound network access.
    """

    def __init__(
        self,
        *,
        workspace: AsyncSandbox | None = None,
        client: AsyncDaytona | None = None,
        ref: WorkspaceRef | None = None,
        snapshot: str | None = None,
        auto_stop_interval: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        network_block_all: bool = False,
    ) -> None:
        if ref is not None and ref.provider != 'daytona':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'daytona'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Daytona workspace ref id cannot be empty')
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._ref = ref if workspace is None else WorkspaceRef(provider='daytona', id=workspace.id)
        self._sandbox = workspace
        self._client = client
        self._owns_client = client is None
        self._snapshot = snapshot
        self._auto_stop_interval = auto_stop_interval
        self._env = dict(env or {})
        self._network_block_all = network_block_all
        self._working_dir = absolute_path('working_dir', working_dir)
        # `pwd -P` of `_working_dir` (or of the image default): the protocol needs a canonical absolute path.
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is always Daytona's sandbox ID. Attaching by a `ref` whose `id` is a sandbox name
        also works, since Daytona looks sandboxes up by ID or name, and the ref is then rewritten to
        the ID so it stays valid if the sandbox is renamed.
        """
        return self._ref

    async def get_client(self) -> AsyncSandbox:
        """Return the typed `daytona.AsyncSandbox`, creating or attaching to it on first use.

        This is the sandbox handle, not the `AsyncDaytona` API client passed as `client=`.

        The only place `_client` and `_sandbox` are read, so nothing can reach an
        unhydrated one: both stay optional and every other method comes through here.
        The lock serializes concurrent first uses -- two callers each creating a sandbox
        would leave the loser billed and unreferenced. A failed acquisition releases an
        API client this backend owns, so a retry starts from a clean one. Attaching by `ref`
        to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it does not
        create a replacement. Creation is not cut short by cancellation, so
        a sandbox Daytona created is always recorded in `ref` before the cancellation propagates.
        """
        async with self._lock:
            if (sandbox := self._sandbox) is not None:
                return sandbox
            ref = self._ref
            try:
                if self._client is None:
                    try:
                        # Reads the credentials, so a missing key surfaces here as an auth error.
                        self._client = daytona.AsyncDaytona()
                    except Exception as error:
                        _raise_translated(error, 'Could not configure the Daytona client')
                client = self._client
                sandbox = await self._attach(client, ref.id) if ref is not None else await self._create(client)
            except BaseException:
                await self._close_owned_client()
                raise
            self._sandbox = sandbox
            # `client.get` also accepts a name; record the ID it resolved to.
            self._ref = WorkspaceRef(provider='daytona', id=sandbox.id)
        # A caller cancelled during the shielded creation stops here, with the sandbox recorded.
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    async def aclose(self) -> None:
        """Close the `AsyncDaytona` API client this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A sandbox
        SDK handles lazily reopen their HTTP sessions after close; this backend drops its owned
        client and handle so a later operation opens a new client and attaches by `ref`. Using a
        `client=` or `workspace=` backend after its caller closes that client leaks an aiohttp session.
        `DaytonaSandbox` calls this when the run that used the backend ends.
        """
        if self._client is None or not self._owns_client:
            return
        # Shielded so a cancelled run still releases its HTTP connections.
        with anyio.CancelScope(shield=True):
            async with self._lock:
                await self._close_owned_client()

    async def _close_owned_client(self) -> None:
        """Close the API client this backend opened, bounded and shielded so a cancelled run still releases it.

        The client, and the sandbox handle and working directory learned through it, are dropped only
        once it closed, so a failed close is retried by the next `aclose()`. The failure is logged, not
        raised: it must not replace the outcome of the run being cleaned up.
        """
        client = self._client
        if client is None or not self._owns_client:
            return
        with anyio.CancelScope(shield=True), anyio.move_on_after(_TEARDOWN_TIMEOUT) as deadline:
            try:
                await client.close()
            except Exception:
                _logger.warning('Could not close the Daytona API client.', exc_info=True)
                return
        if deadline.cancelled_caught:
            _logger.warning('Closing the Daytona API client did not complete within %ss.', _TEARDOWN_TIMEOUT)
            return
        self._client = None
        self._sandbox = None
        self._resolved_working_dir = None

    async def read_bytes(self, path: str) -> bytes:
        _check_path(path)
        sandbox = await self.get_client()
        async with _translated_filesystem_error(sandbox, path):
            try:
                return await sandbox.fs.download_file(path, _REQUEST_TIMEOUT)
            except daytona.DaytonaError as error:
                if isinstance(error, daytona.DaytonaNotFoundError):
                    raise
                # The toolbox's answer for reading a directory is not a documented error type; the
                # entry type tells it apart from other failures.
                if (await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)).is_dir:
                    raise IsADirectoryError(f'Is a directory in the Daytona sandbox: {path!r}') from error
                raise

    async def write_bytes(self, path: str, data: bytes) -> None:
        _check_path(path)
        sandbox = await self.get_client()
        parent = posixpath.dirname(path)
        if parent not in ('', '.', '/'):
            try:
                mkdir = await sandbox.process.exec(f'mkdir -p -- {shlex.quote(parent)}', timeout=_REQUEST_TIMEOUT)
            except Exception as error:
                await _raise_failure(sandbox, error, f'Could not create {parent!r}')
            if mkdir.exit_code != 0:
                raise _mkdir_error(mkdir.result, parent)
        async with _translated_filesystem_error(sandbox, path):
            await sandbox.fs.upload_file(data, path, timeout=_REQUEST_TIMEOUT)

    async def stat(self, path: str) -> FileEntry:
        _check_path(path)
        sandbox = await self.get_client()
        async with _translated_filesystem_error(sandbox, path):
            entry = await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
        return FileEntry(
            name=posixpath.basename(path.rstrip('/')),
            path=path,
            is_dir=entry.is_dir,
            size=None if entry.is_dir else entry.size,
        )

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        _check_path(path)
        sandbox = await self.get_client()
        async with _translated_filesystem_error(sandbox, path):
            entries = await sandbox.fs.list_files(path, request_timeout=_REQUEST_TIMEOUT)
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
        _check_path(path)
        sandbox = await self.get_client()
        async with _translated_filesystem_error(sandbox, path):
            await sandbox.fs.create_folder(path, '755', request_timeout=_REQUEST_TIMEOUT)

    async def remove(self, path: str) -> None:
        _check_path(path)
        sandbox = await self.get_client()
        async with _translated_filesystem_error(sandbox, path):
            # Whether the toolbox rejects removing a missing path is not documented; looking it up
            # first reports that as the protocol's `FileNotFoundError` either way.
            await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
            await sandbox.fs.delete_file(path, recursive=True, request_timeout=_REQUEST_TIMEOUT)

    async def exists(self, path: str) -> bool:
        try:
            await self.stat(path)
        except FileNotFoundError:
            return False
        except WorkspaceError as error:
            if isinstance(error.__cause__, daytona.DaytonaValidationError) and (
                'Too many levels of symbolic links' in str(error.__cause__)
            ):
                return False
            raise
        return True

    async def _create(self, client: AsyncDaytona) -> AsyncSandbox:
        params = daytona.CreateSandboxFromSnapshotParams(
            snapshot=self._snapshot,
            env_vars=self._env or None,
            auto_stop_interval=self._auto_stop_interval,
            network_block_all=self._network_block_all,
            labels={'created-by': 'pydantic-ai'},
        )
        # Shielded: cancelling the request after Daytona accepted it would leave a sandbox nothing
        # names. The caller records the ref first, then a pending cancellation is delivered.
        with anyio.CancelScope(shield=True), anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                return await client.create(params, timeout=_LIFECYCLE_TIMEOUT)
            except Exception as error:
                translated = _translated(error, 'Could not start Daytona sandbox')
                if translated is error:
                    raise
                if type(translated) is WorkspaceError:
                    # Daytona refused the request (an unknown snapshot, an invalid setting): no retry or
                    # model turn can fix it, so the run ends instead of handing the model an error.
                    translated = WorkspaceUnavailableError(str(translated))
                raise translated from error
        # A stalled control plane is transient, so it is a plain `TimeoutError`, which durable engines
        # retry; `WorkspaceTimeoutError` is reserved for command deadlines.
        raise TimeoutError(f'Daytona sandbox creation did not complete within {_CREATE_TIMEOUT}s.')

    async def _attach(self, client: AsyncDaytona, workspace_id: str) -> AsyncSandbox:
        """Attach to a sandbox that already exists, starting it if it is stopped.

        `delete()` returns once Daytona accepts the request, and the sandbox stays visible in the
        `destroying` state until it is gone, so that state counts as gone here rather than being
        started.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                sandbox = await client.get(workspace_id, request_timeout=_REQUEST_TIMEOUT)
                if not _in_deleted_state(sandbox):
                    await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
            except Exception as error:
                if isinstance(error, daytona.DaytonaNotFoundError):
                    raise WorkspaceUnavailableError(
                        f'The Daytona sandbox {workspace_id!r} was not found (it was deleted, or never existed '
                        "in this Daytona organization). Pass `workspace='new'` to start a fresh sandbox."
                    ) from error
                _raise_translated(error, f'Could not attach to Daytona sandbox {workspace_id!r}')
            if _in_deleted_state(sandbox):
                raise WorkspaceUnavailableError(_unavailable_message(sandbox.id))
            return sandbox
        raise TimeoutError(
            f'Connecting to Daytona sandbox {workspace_id!r} did not complete within {_CREATE_TIMEOUT}s.'
        )

    async def working_dir(self) -> str:
        """Return the filesystem-canonical default directory inside the workspace.

        The probe runs only on the first call per connection. If the sandbox was deleted, that
        call raises `WorkspaceUnavailableError`; later calls return the cached path without probing.
        """
        if self._resolved_working_dir is None:
            sandbox = await self.get_client()
            try:
                result = await sandbox.process.exec('pwd -P', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT)
            except Exception as error:
                await _raise_failure(sandbox, error, 'Could not determine the working directory')
            printed = result.result.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise WorkspaceUnavailableError(
                    f'Could not determine the working directory of Daytona sandbox {sandbox.id}: {result.result}'
                )
            self._resolved_working_dir = printed
        return self._resolved_working_dir

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
        # Acquiring the sandbox has its own bound; the timeout is the command's alone.
        sandbox = await self.get_client()
        process: _DaytonaProcess | None = None
        with anyio.move_on_after(timeout) as scope:
            try:
                process = await self._start(sandbox, command, shell=shell, cwd=cwd, env=env)
                return await process.wait()
            finally:
                if process is not None:
                    await process.kill()
        assert scope.cancel_called
        raise WorkspaceTimeoutError(
            f'Command timed out after {timeout:g} seconds.',
            stdout=_until_marker(process.stdout, process.marker) if process is not None else '',
            stderr=_until_marker(process.stderr, process.marker) if process is not None else '',
        )

    async def _start(
        self,
        sandbox: AsyncSandbox,
        command: WorkspaceCommand,
        *,
        shell: bool,
        cwd: str | None,
        env: Mapping[str, str] | None,
    ) -> _DaytonaProcess:
        marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        line = _command_line(
            command,
            shell=shell,
            cwd=absolute_path('cwd', cwd) if cwd is not None else self._working_dir,
            env={**self._env, **(env or {})},
            marker=marker,
        )
        session_id = f'pydantic-ai-{uuid.uuid4().hex}'
        process = sandbox.process
        created = False
        with anyio.move_on_after(_REQUEST_TIMEOUT):
            try:
                await process.create_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                created = True
                response = await process.execute_session_command(
                    session_id,
                    daytona.SessionExecuteRequest(command=line, run_async=True),
                    timeout=_REQUEST_TIMEOUT,
                )
            except BaseException as error:
                if created:
                    await _delete_session(process, session_id)
                if isinstance(error, Exception):
                    await _raise_failure(sandbox, error, 'Could not start the command')
                raise
            stdout: list[str] = []
            stderr: list[str] = []
            logs = asyncio.create_task(
                process.get_session_command_logs_async(session_id, response.cmd_id, stdout.append, stderr.append)
            )
            return _DaytonaProcess(
                _process=process,
                _sandbox=sandbox,
                _session_id=session_id,
                _command_id=response.cmd_id,
                stdout=stdout,
                stderr=stderr,
                marker=marker,
                _logs=logs,
            )
        raise TimeoutError(f'Daytona command session setup did not complete within {_REQUEST_TIMEOUT}s.')


def _in_deleted_state(sandbox: AsyncSandbox) -> bool:
    return sandbox.state in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Daytona sandbox {sandbox_id!r} no longer exists (it was deleted). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _translated(error: Exception, context: str, *, sandbox_id: str | None = None, path: str | None = None) -> Exception:
    """Map a Daytona SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: connection failures, SDK timeouts,
    rate limits, server errors, and anything that is not a `DaytonaError`, which durable engines
    retry as transient. A not-found answer is a missing `path` for a path operation, the sandbox
    being gone for a call naming `sandbox_id`, and a refused request otherwise.
    """
    if path is not None and isinstance(error, daytona.DaytonaValidationError):
        # Toolbox 400s include POSIX strerror text rather than an errno field.
        for phrase, error_type in (
            ('Not a directory', NotADirectoryError),
            ('Is a directory', IsADirectoryError),
            ('Permission denied', PermissionError),
            ('File exists', FileExistsError),
        ):
            if phrase in str(error):
                return error_type(f'{phrase} in the Daytona sandbox: {path!r}')
    if path is not None and isinstance(error, daytona.DaytonaAuthorizationError):
        # Toolbox authorization is about the requested file, not the API credentials.
        return PermissionError(f'Permission denied in the Daytona sandbox: {path!r}')
    if isinstance(error, (daytona.DaytonaAuthenticationError, daytona.DaytonaAuthorizationError)):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if isinstance(error, daytona.DaytonaNotFoundError):
        if path is not None:
            return FileNotFoundError(f'No such file or directory in the Daytona sandbox: {path!r}')
        if sandbox_id is not None:
            return WorkspaceUnavailableError(_unavailable_message(sandbox_id))
    if isinstance(error, (daytona.DaytonaNotFoundError, daytona.DaytonaValidationError, daytona.DaytonaConflictError)):
        return WorkspaceError(f'{context}: {error}')
    if (
        type(error) is daytona.DaytonaError
        and error.status_code is not None
        and 400 <= error.status_code < 500  # a refusal; 5xx and status-less errors are transient
    ):
        return WorkspaceError(f'{context}: {error}')
    return error


def _raise_translated(
    error: Exception, context: str, *, sandbox_id: str | None = None, path: str | None = None
) -> NoReturn:
    translated = _translated(error, context, sandbox_id=sandbox_id, path=path)
    if translated is error:
        raise error
    raise translated from error


async def _raise_failure(sandbox: AsyncSandbox, error: Exception, context: str, *, path: str | None = None) -> NoReturn:
    """Raise the translation of a failed call on a live sandbox handle.

    Unless the error already ends the run, one control-plane lookup follows: whatever the
    toolbox answered, a deleted sandbox is reported as `WorkspaceUnavailableError`. On a live
    sandbox, a not-found answer is a missing `path`, or a refused request (such as a command
    session that no longer exists) for a call without one.
    """
    translated = _translated(error, context, path=path)
    if not isinstance(translated, WorkspaceUnavailableError) and await _is_deleted(sandbox):
        raise WorkspaceUnavailableError(_unavailable_message(sandbox.id)) from error
    if translated is error:
        raise error
    raise translated from error


async def _is_deleted(sandbox: AsyncSandbox) -> bool:
    """Ask the control plane whether the sandbox was deleted.

    Only called after a failure, so successful operations make no extra request. An inconclusive
    lookup counts as alive, leaving the original error to decide.
    """
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            await sandbox.refresh_data(request_timeout=_REQUEST_TIMEOUT)
    except daytona.DaytonaNotFoundError:
        return True
    except Exception:
        return False
    return _in_deleted_state(sandbox)


def _check_path(path: str) -> None:
    # Daytona's toolbox uses newline-delimited paths; embedded line breaks change the request's meaning.
    if '\n' in path or '\r' in path:
        raise ValueError('Daytona file paths cannot contain a newline or carriage return')


def _mkdir_error(output: str, parent: str) -> Exception:
    """Classify a failed `mkdir -p` by the `strerror` text GNU and BusyBox `mkdir` print."""
    if 'Not a directory' in output or 'File exists' in output:
        return NotADirectoryError(f'Not a directory in the Daytona sandbox: {parent!r}')
    if 'Permission denied' in output:
        return PermissionError(f'Permission denied in the Daytona sandbox: {parent!r}')
    return WorkspaceError(output or f'Could not create {parent!r}.')


@asynccontextmanager
async def _translated_filesystem_error(sandbox: AsyncSandbox, path: str) -> AsyncGenerator[None]:
    """Map Daytona's filesystem errors onto the ones the protocol promises."""
    try:
        yield
    except IsADirectoryError:
        raise
    except Exception as error:
        await _raise_failure(sandbox, error, f'Could not access {path!r} in the sandbox', path=path)

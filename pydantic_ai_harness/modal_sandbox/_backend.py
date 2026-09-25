"""A native Modal sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-09-09 against Modal Python SDK 1.5.2:

* `.aio`, `from_id`, `poll`, `terminate`, and `detach` follow the native sandbox lifecycle:
  https://modal.com/docs/guide/sandboxes
* `exec` uses whole-second deadlines and has no per-command kill operation:
  https://modal.com/docs/guide/sandbox-spawn
* the native filesystem API supplies the operations used here:
  https://modal.com/docs/sdk/py/latest/Sandbox

Re-check these SDK methods before changing the protocol integration.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import posixpath
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
from pydantic_ai.exceptions import UserError
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
    import modal
    import modal.io_streams
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('ModalSandboxBackend',)

DEFAULT_IMAGE = 'python:3.12-slim'
DEFAULT_APP_NAME = 'pydantic-ai-harness'
DEFAULT_SANDBOX_TIMEOUT = 300


_MISSING_MODAL = (
    'The \'modal\' package is required for ModalSandbox. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the workspace-create RPCs so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120


_INTERNAL_EXEC_TIMEOUT = 10

_CLIENT_DEADLINE_EXIT = -1
_SIGKILL_EXIT = 137

_RESULT_GRACE = 30


def _is_shutting_down(e: BaseException) -> bool:
    """Whether Modal refused an exec because the sandbox has been terminated.

    For up to about 30 seconds after `terminate()`, Modal still reports the sandbox as running
    from `poll()` while refusing exec with this `ConflictError`. Only the message separates it
    from a transient conflict.
    """
    import modal

    return isinstance(e, modal.exception.ConflictError) and 'shutting down' in str(e).lower()


def _translate(error: Exception, *, context: str, gone: str, path: str | None = None) -> Exception | None:
    """Map a Modal SDK exception onto the workspace protocol's typed failures.

    Returns `None` for an exception that must propagate unchanged: Modal's connection, rate-limit,
    and internal-service errors, and anything unrecognized, are transient infrastructure failures
    that a durable engine retries. `gone` is the message for a sandbox that no longer exists;
    `path` is set for a filesystem operation, whose path-level errors become the builtin ones.
    """
    import modal

    exc = modal.exception
    if isinstance(error, (exc.AuthError, exc.PermissionDeniedError)):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if isinstance(error, (exc.NotFoundError, exc.SandboxTerminatedError, exc.SandboxTimeoutError)) or (
        _is_shutting_down(error)
    ):
        return WorkspaceUnavailableError(gone)
    if path is not None:
        path_errors: tuple[tuple[type[Exception], type[OSError], str], ...] = (
            (exc.SandboxFilesystemNotFoundError, FileNotFoundError, 'No such file or directory'),
            (exc.SandboxFilesystemIsADirectoryError, IsADirectoryError, 'Is a directory'),
            (exc.SandboxFilesystemNotADirectoryError, NotADirectoryError, 'Not a directory'),
            (exc.SandboxFilesystemPermissionError, PermissionError, 'Permission denied'),
            (exc.SandboxFilesystemPathAlreadyExistsError, FileExistsError, 'File exists'),
        )
        for sdk_type, builtin, description in path_errors:
            if isinstance(error, sdk_type):
                return builtin(f'{description} in the Modal sandbox: {path!r}')
    if isinstance(
        error,
        (
            exc.InvalidError,  # includes a non-terminal `ConflictError`
            exc.AlreadyExistsError,
            exc.ExecutionError,
            exc.RequestSizeError,
            exc.SandboxFilesystemError,
            exc.FilesystemExecutionError,
        ),
    ):
        return WorkspaceError(f'{context}: {error}')
    return None


def _unwrap_filesystem_error(error: Exception) -> Exception:
    """The SDK failure Modal's filesystem layer replaced, or `error` itself.

    Modal 1.5.2 runs each filesystem operation as an exec and re-raises what that exec raised
    `from None`: connection and service failures as a `NotFoundError` ("the Sandbox is
    unavailable"), every other SDK failure -- rate limits and internal errors included -- as a
    generic `SandboxFilesystemError`. Classifying those replacements would turn a retryable
    transport failure into a terminal one, so the original, still in `__context__`, is used.
    """
    import modal

    exc = modal.exception
    original = error.__context__
    if (
        type(error) in (exc.NotFoundError, exc.SandboxFilesystemError)
        and error.__suppress_context__
        and isinstance(original, exc.Error)
    ):
        return original
    return error


def _file_entry(entry: modal.types.FileInfo, path: str) -> FileEntry:
    is_dir = entry.is_dir()
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    return FileEntry(name=entry.name, path=path, is_dir=is_dir, size=None if is_dir else entry.size)


class ModalSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A Modal sandbox implementing Pydantic AI's ``WorkspaceBackend`` protocol.

    Construction performs no I/O. The first operation creates or attaches to a sandbox, and the
    typed `modal.Sandbox` is available through `get_client()`. The backend does not terminate the
    sandbox; terminating it is the application's job.

    Modal applies whole-second command deadlines. Cancelling ``run()`` stops the local wait while
    the command may continue until its deadline or the sandbox lifetime ends.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `WorkspaceBackend` call.

    Args:
        workspace: A live `modal.Sandbox` you already have. Whoever created it owns terminating it.
        ref: Identity of an existing workspace to attach to on first use.
        name: Optional Modal name passed when creating a new sandbox. It is not used to recover a
            sandbox when `ref` is absent.
        image: Registry tag, or a `modal.Image`, a newly created workspace runs.
        app_name: Modal app a newly created workspace belongs to.
        create_app_if_missing: Create the Modal app when it does not exist yet.
        sandbox_timeout: How long Modal keeps a newly created workspace alive, in seconds.
        idle_timeout: Seconds without activity after which Modal terminates a newly created
            workspace; Modal's default (no idle limit) when `None`.
        working_dir: Absolute directory commands start in and relative paths resolve against,
            applied to every command, including in an attached workspace; the image's when `None`.
        env: Environment variables every command gets; a command's own `env` is layered on top.
    """

    def __init__(
        self,
        workspace: modal.Sandbox | None = None,
        *,
        ref: WorkspaceRef | None = None,
        name: str | None = None,
        image: str | modal.Image = DEFAULT_IMAGE,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        idle_timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'modal':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'modal'")
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._workspace: modal.Sandbox | None = workspace
        self._ref = ref if workspace is None else WorkspaceRef(provider='modal', id=workspace.object_id)
        self._name = name
        self._image = image
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._idle_timeout = idle_timeout
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env) if env is not None else {}
        self._probed_working_dir: str | None = None
        # Set once the workspace exists, so an expiry message can say which lifetime ran out.
        self._created_timeout: int | None = None
        self._lock = anyio.Lock()

    async def get_client(self) -> modal.Sandbox:
        """Return the typed `modal.Sandbox`, creating or attaching to it on first use.

        The only place `_workspace` is read, so nothing can reach an unacquired handle:
        it stays optional and every other method comes through here. The lock serializes
        concurrent first uses -- two callers each creating a sandbox would leave the loser
        billed and unreferenced. Attaching by `ref` to a sandbox that no longer exists raises
        `WorkspaceUnavailableError`; it does not create a replacement.

        Raises:
            UserError: The `modal` package is not installed.
        """
        async with self._lock:
            if (workspace := self._workspace) is not None:
                return workspace
            try:
                importlib.import_module('modal')
            except ImportError as e:
                raise UserError(_MISSING_MODAL) from e
            ref = self._ref
            workspace = await self._attach(ref.id) if ref is not None else await self._create()
            self._workspace = workspace
            self._ref = WorkspaceRef(provider='modal', id=workspace.object_id)
            return workspace

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the workspace, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _mapped_errors(self, context: str, path: str | None = None) -> AsyncGenerator[None]:
        """Raise the protocol's typed failure for a Modal exception, and let anything else through."""
        try:
            yield
        except Exception as wrapped:
            error = _unwrap_filesystem_error(wrapped) if path is not None else wrapped
            mapped = await self._failure(error, context, path)
            if mapped is None:
                raise error
            raise mapped from error

    async def read_bytes(self, path: str) -> bytes:
        workspace = await self.get_client()
        async with self._mapped_errors(f'Could not read {path!r}', path):
            return await workspace.filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        workspace = await self.get_client()
        async with self._mapped_errors(f'Could not write {path!r}', path):
            await workspace.filesystem.write_bytes.aio(data, path)

    async def stat(self, path: str) -> FileEntry:
        workspace = await self.get_client()
        async with self._mapped_errors(f'Could not stat {path!r}', path):
            return _file_entry(await workspace.filesystem.stat.aio(path), path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        workspace = await self.get_client()
        async with self._mapped_errors(f'Could not list {path!r}', path):
            entries = await workspace.filesystem.list_files.aio(path)
        return [_file_entry(entry, posixpath.join(path, entry.name)) for entry in entries]

    async def make_dir(self, path: str) -> None:
        workspace = await self.get_client()
        async with self._mapped_errors(f'Could not create directory {path!r}', path):
            await workspace.filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        workspace = await self.get_client()
        async with self._mapped_errors(f'Could not remove {path!r}', path):
            await workspace.filesystem.remove.aio(path, recursive=True)

    async def exists(self, path: str) -> bool:
        try:
            await self.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            # Modal splits "there is nothing at that path" in two: a missing entry, and a
            # non-leaf component that is a file.
            return False
        return True

    async def _create(self) -> modal.Sandbox:
        """Provision a fresh Modal sandbox."""
        import modal

        workspace: modal.Sandbox | None = None
        try:
            # Shielded so that a caller cancelled mid-create still gets the sandbox Modal made:
            # `get_client` records it before the cancellation is delivered, so `ref` names it and
            # a retry reuses it. Only the local deadline interrupts the call; a sandbox created
            # after it fires is reaped at its `sandbox_timeout`.
            with anyio.CancelScope(shield=True), anyio.move_on_after(_CREATE_TIMEOUT):
                app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
                built = (
                    modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
                    if isinstance(self._image, str)
                    else self._image
                )
                variables: dict[str, str | None] | None = dict(self._env) if self._env else None
                workspace = await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
                    app=app,
                    image=built,
                    timeout=self._sandbox_timeout,
                    idle_timeout=self._idle_timeout,
                    workdir=self._working_dir,
                    env=variables,
                    name=self._name,
                )
        except Exception as error:
            # Nothing exists yet, so "not found" here is the app or image, not a sandbox.
            mapped = _translate(
                error, context='Could not start Modal sandbox', gone=f'Could not start Modal sandbox: {error}'
            )
            if mapped is None:
                raise
            raise mapped from error
        if workspace is None:
            # A plain `TimeoutError`: an unresponsive control plane is transient, so a durable
            # engine retries it.
            raise TimeoutError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            )
        self._created_timeout = self._sandbox_timeout
        return workspace

    async def _attach(self, id: str) -> modal.Sandbox:
        """Attach to a Modal sandbox that already exists.

        Modal hands back a handle for a workspace it still knows about even after that workspace
        has terminated, so this polls: a `WorkspaceRef` must not resolve to a dead environment.
        Nothing is recreated in its place -- a run that expected files there must be told they
        are gone, not handed an empty workspace.
        """
        import modal

        try:
            workspace = await modal.Sandbox.from_id.aio(id)
            finished = await workspace.poll.aio()
        except Exception as error:
            mapped = _translate(
                error, context=f'Could not connect to Modal sandbox {id!r}', gone=_attached_gone_message(id)
            )
            if mapped is None:
                raise
            raise mapped from error
        if finished is not None:
            raise WorkspaceUnavailableError(_attached_gone_message(id))
        return workspace

    async def working_dir(self) -> str:
        """The workspace's working directory (absolute POSIX path)."""
        # Modal exposes no API for a running workspace's working directory -- it is the image's,
        # or the configured `working_dir` every command is given -- so ask the environment itself,
        # which also canonicalizes a configured path. It cannot change, so the probe is an
        # idempotent read: overlapping first calls may each run their own `pwd`, get the same
        # answer, and the cache converges. No lock needed.
        if self._probed_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            # Only an absolute path is an answer. Caching whatever else the environment
            # printed would hand every later `resolve()` a working directory that is not
            # one, mis-resolving relative paths with no error.
            if result.exit_code != 0 or not posixpath.isabs(printed):
                assert self._ref is not None
                raise WorkspaceError(
                    f'Could not determine the working directory of Modal sandbox {self._ref.id!r}: '
                    f'`pwd` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
                )
            self._probed_working_dir = printed
        return self._probed_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Execute a command and wait for it to complete.

        A shell command runs as `/bin/sh -c <command>`. Modal has no per-command kill, so a
        cancelled `run()` stops the wait but leaves the command running until its `timeout`
        deadline. Pass a finite `timeout` so an abandoned command cannot run on indefinitely.
        """
        # Modal executes argv and never a shell string, so shell interpretation is requested
        # explicitly through `/bin/sh -c`, the one shell every sandbox image carries.
        argv = command_argv(command, shell)
        # Given per command, not only at creation, so an attached sandbox honors it too.
        workdir = absolute_path('cwd', cwd) or self._working_dir
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        merged = {**self._env, **(env or {})}
        variables: dict[str, str | None] | None = dict(merged) if merged else None
        # Acquiring the sandbox has its own bound; the timeout is the command's alone.
        workspace = await self.get_client()
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        server_started_at = time.monotonic()
        try:
            with anyio.fail_after(timeout):
                async with self._mapped_errors('Command could not run in the workspace'):
                    process = await workspace.exec.aio(
                        *argv, timeout=deadline, workdir=workdir, env=variables, text=False
                    )
        except TimeoutError as error:
            raise WorkspaceTimeoutError(
                'Timed out before the command could start; the Modal process may still be running.', timeout=timeout
            ) from error

        async def read(reader: modal.io_streams.StreamReader[bytes]) -> str:
            return (await reader.read.aio()).decode('utf-8', errors='replace')

        tasks = (
            asyncio.create_task(read(process.stdout)),
            asyncio.create_task(read(process.stderr)),
            asyncio.create_task(process.wait.aio()),
        )
        gather = asyncio.gather(*tasks)
        try:
            if deadline is None:
                stdout, stderr, exit_code = await gather
            else:
                result_timeout = max(0.0, server_started_at + deadline - time.monotonic()) + _RESULT_GRACE
                stdout, stderr, exit_code = await asyncio.wait_for(gather, result_timeout)
        except BaseException as error:
            for task in tasks:
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(error, (TimeoutError, asyncio.TimeoutError)):

                def captured(task: asyncio.Task[str]) -> str:
                    if task.cancelled() or task.exception() is not None:
                        return ''
                    return task.result()

                raise WorkspaceTimeoutError(
                    f'Command timed out after {deadline} seconds.',
                    stdout=captured(tasks[0]),
                    stderr=captured(tasks[1]),
                    timeout=deadline,
                ) from error
            if isinstance(error, Exception) and (
                mapped := await self._failure(
                    error, 'Could not read the command result (the command may still run until its deadline)'
                )
            ):
                raise mapped from error
            raise

        elapsed = time.monotonic() - server_started_at
        if deadline is not None and (
            exit_code == _CLIENT_DEADLINE_EXIT or (exit_code == _SIGKILL_EXIT and elapsed >= deadline)
        ):
            raise WorkspaceTimeoutError(
                f'Command timed out after {deadline} seconds.',
                stdout=stdout,
                stderr=stderr,
                timeout=deadline,
            )
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def _gone_message(self) -> str:
        if self._created_timeout is None:
            assert self._ref is not None
            return _attached_gone_message(self._ref.id)
        assert self._ref is not None
        return (
            f'The Modal sandbox {self._ref.id!r} is no longer running (it may have reached its '
            f'sandbox_timeout of {self._created_timeout}s, or been terminated). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def _failure(self, error: Exception, context: str, path: str | None = None) -> Exception | None:
        """Translate an SDK failure on the acquired sandbox, or `None` to let it propagate.

        Modal reports two failures ambiguously -- an exec on a dead sandbox raises
        `ConflictError` (also used for transient aborts), and the filesystem layer wraps
        everything, including a dead sandbox, in a generic `SandboxFilesystemError` -- so an
        operation failure of either kind is classified by probing the sandbox before it is
        reported as one.
        """
        import modal

        mapped = _translate(error, context=context, gone=self._gone_message(), path=path)
        if type(mapped) is WorkspaceError and isinstance(
            error, (modal.exception.ConflictError, modal.exception.SandboxFilesystemError)
        ):
            return await self._probe(error) or mapped
        return mapped

    async def _probe(self, error: Exception) -> WorkspaceUnavailableError | None:
        """The terminal failure behind an ambiguous `error`, or `None` if the sandbox still runs."""
        # Probing only after an error keeps the extra round trip off successful operations.
        import modal

        try:
            workspace = await self.get_client()
            finished = await workspace.poll.aio()
            if finished is None and isinstance(error, modal.exception.SandboxFilesystemError):
                # A terminated sandbox that is still shutting down polls as running and fails
                # filesystem calls with a generic error; only exec names the state.
                with anyio.fail_after(_INTERNAL_EXEC_TIMEOUT):
                    await workspace.exec.aio('true', timeout=_INTERNAL_EXEC_TIMEOUT)
        except Exception as probe_error:
            # A probe failing for any other reason, a transport error included, leaves the
            # original error standing rather than replacing it.
            mapped = _translate(probe_error, context='', gone=self._gone_message())
            return mapped if isinstance(mapped, WorkspaceUnavailableError) else None
        if finished is not None:
            return WorkspaceUnavailableError(self._gone_message())
        return None


def _attached_gone_message(id: str) -> str:
    return (
        f'The Modal sandbox {id!r} is no longer running '
        '(it does not exist, was terminated, or expired at its configured lifetime). '
        'Attach to a live workspace, or create a new one.'
    )

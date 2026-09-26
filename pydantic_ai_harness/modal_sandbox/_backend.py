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

DEFAULT_APP_NAME = 'pydantic-ai-harness'
# Modal's maximum sandbox lifetime (24 hours). The framework never terminates a sandbox, so a
# conversation can continue in it for as long as Modal allows.
DEFAULT_SANDBOX_TIMEOUT = 86_400

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


def _translate(error: Exception, *, context: str, unavailable: str, path: str | None = None) -> Exception | None:
    """Map a Modal SDK exception onto the workspace protocol's typed failures.

    Returns `None` for an exception that must propagate unchanged: Modal's connection, rate-limit,
    and internal-service errors, and anything unrecognized, are transient infrastructure failures
    that a durable engine retries. `unavailable` is the message for a sandbox that no longer
    exists; `path` is set for a filesystem operation, whose path-level errors become the builtin
    ones.
    """
    import modal

    exc = modal.exception
    if isinstance(error, (exc.AuthError, exc.PermissionDeniedError)):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    # `SandboxTimeoutError` is the sandbox reaching its lifetime (`sandbox_timeout`), not a
    # command timing out; a command's own deadline is handled in `run()`.
    if isinstance(error, (exc.NotFoundError, exc.SandboxTerminatedError, exc.SandboxTimeoutError)) or (
        _is_shutting_down(error)
    ):
        return WorkspaceUnavailableError(unavailable)
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
            exc.InvalidError,  # `ConflictError` subclasses it; the caller probes whether the sandbox is gone
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


# Linux's limit on symlinks followed while resolving one path; a longer chain is a loop.
_MAX_SYMLINK_HOPS = 40


async def _file_entry(sandbox: modal.Sandbox, entry: modal.types.FileInfo, path: str) -> FileEntry:
    """The protocol entry for `entry` at `path`, with `is_dir` and `size` following a symlink.

    Modal's `stat` and `list_files` describe a symlink itself, so a symlink entry is resolved by
    stat-ing its target, hop by hop. A dangling or looping link is reported as a file with no size.
    """
    import modal

    target: modal.types.FileInfo | None = entry
    link, hops = path, 0
    visited = {posixpath.normpath(path)}
    while target is not None and target.is_symlink():
        hops += 1
        if hops > _MAX_SYMLINK_HOPS or target.symlink_target is None:
            target = None
            continue
        # A relative target is relative to the directory holding the link.
        link = posixpath.normpath(posixpath.join(posixpath.dirname(link), target.symlink_target))
        if link in visited:
            # Modal does not detect all cycles in its symlink metadata; skip repeated RPCs.
            target = None
            continue
        visited.add(link)
        try:
            target = await sandbox.filesystem.stat.aio(link)
        except (
            modal.exception.SandboxFilesystemNotFoundError,
            modal.exception.SandboxFilesystemNotADirectoryError,
        ):
            target = None
    is_dir = target is not None and target.is_dir()
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    size = None if target is None or is_dir else target.size
    return FileEntry(name=entry.name, path=path, is_dir=is_dir, size=size)


class ModalSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A Modal sandbox implementing Pydantic AI's `WorkspaceBackend` protocol.

    Construction performs no I/O. The first operation creates or attaches to a sandbox, and the
    typed `modal.Sandbox` is available through `get_client()`. The backend does not terminate the
    sandbox; terminating it is the application's job.

    Modal applies whole-second command deadlines. Cancelling `run()` stops the local wait while
    the command may continue until its deadline or the sandbox lifetime ends.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `WorkspaceBackend` call.

    Args:
        workspace: A live `modal.Sandbox` you already have. Whoever created it owns terminating it.
        ref: Identity of an existing sandbox to attach to on first use.
        image: Registry tag, or a `modal.Image`, a newly created sandbox runs. `None` (the default)
            is Debian slim with Python 3.12, `git`, and `ripgrep`.
        app_name: Modal app a newly created sandbox belongs to.
        create_app_if_missing: Create the Modal app when it does not exist yet.
        sandbox_timeout: Total lifetime of a newly created sandbox, in seconds (Modal's `timeout`).
            Defaults to Modal's maximum, 24 hours.
        idle_timeout: Seconds without activity after which Modal terminates a newly created
            sandbox; `None` (the default) never terminates it for being idle.
        working_dir: Absolute directory commands start in and relative paths resolve against,
            applied to every command, including in an attached sandbox; the image's when `None`.
        env: Environment variables every command gets; a command's own `env` is layered on top.
    """

    def __init__(
        self,
        workspace: modal.Sandbox | None = None,
        *,
        ref: WorkspaceRef | None = None,
        image: str | modal.Image | None = None,
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
        self._ref = ref if workspace is None else WorkspaceRef(provider='modal', id=workspace.object_id)
        self._sandbox: modal.Sandbox | None = workspace
        self._image = image
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._idle_timeout = idle_timeout
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env) if env is not None else {}
        # `_working_dir` is what was configured (`None` for the image's); the protocol needs the
        # canonical absolute path, which only `pwd -P` in the sandbox can give.
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()

    async def get_client(self) -> modal.Sandbox:
        """Return the typed `modal.Sandbox`, creating or attaching to it on first use.

        The lock serializes concurrent first uses -- two callers each creating a sandbox would
        leave the loser billed and unreferenced. Attaching by `ref` to a sandbox that no longer
        exists raises `WorkspaceUnavailableError`; it does not create a replacement.

        Raises:
            UserError: The `modal` package is not installed.
        """
        async with self._lock:
            if (sandbox := self._sandbox) is not None:
                return sandbox
            try:
                importlib.import_module('modal')
            except ImportError as e:
                raise UserError(_MISSING_MODAL) from e
            ref = self._ref
            sandbox = await self._attach(ref.id) if ref is not None else await self._create()
            self._sandbox = sandbox
            self._ref = WorkspaceRef(provider='modal', id=sandbox.object_id)
            return sandbox

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _mapped_errors(
        self, sandbox: modal.Sandbox, context: str, path: str | None = None
    ) -> AsyncGenerator[None]:
        """Raise the protocol's typed failure for a Modal exception, and let anything else through."""
        try:
            yield
        except Exception as wrapped:
            error = _unwrap_filesystem_error(wrapped) if path is not None else wrapped
            mapped = await _failure(sandbox, error, context, path)
            if mapped is None:
                raise error
            raise mapped from error

    async def read_bytes(self, path: str) -> bytes:
        absolute_path('path', path)
        sandbox = await self.get_client()
        async with self._mapped_errors(sandbox, f'Could not read {path!r}', path):
            return await sandbox.filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        absolute_path('path', path)
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        sandbox = await self.get_client()
        async with self._mapped_errors(sandbox, f'Could not write {path!r}', path):
            await sandbox.filesystem.write_bytes.aio(data, path)

    async def stat(self, path: str) -> FileEntry:
        absolute_path('path', path)
        sandbox = await self.get_client()
        async with self._mapped_errors(sandbox, f'Could not stat {path!r}', path):
            info = await sandbox.filesystem.stat.aio(path)
            entry = await _file_entry(sandbox, info, path)
            # Listings retain broken links, but stat follows them like the local backend.
            if info.is_symlink() and entry.size is None and not entry.is_dir:
                raise FileNotFoundError(path)
            return entry

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        absolute_path('path', path)
        sandbox = await self.get_client()
        async with self._mapped_errors(sandbox, f'Could not list {path!r}', path):
            entries = await sandbox.filesystem.list_files.aio(path)
            # Limit simultaneous SDK requests without making large link-heavy listings serial.
            limit = asyncio.Semaphore(8)

            async def resolve(entry: modal.types.FileInfo) -> FileEntry:
                async with limit:
                    return await _file_entry(sandbox, entry, posixpath.join(path, entry.name))

            return await asyncio.gather(*(resolve(entry) for entry in entries))

    async def make_dir(self, path: str) -> None:
        absolute_path('path', path)
        sandbox = await self.get_client()
        async with self._mapped_errors(sandbox, f'Could not create directory {path!r}', path):
            await sandbox.filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        absolute_path('path', path)
        sandbox = await self.get_client()
        async with self._mapped_errors(sandbox, f'Could not remove {path!r}', path):
            await sandbox.filesystem.remove.aio(path, recursive=True)

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

        sandbox: modal.Sandbox | None = None
        try:
            # Shielded so that a caller cancelled mid-create still gets the sandbox Modal made:
            # `get_client` records it before the cancellation is delivered, so `ref` names it and
            # a retry reuses it. Only the local deadline interrupts the call; a sandbox created
            # after it fires is reaped at its `sandbox_timeout`.
            with anyio.CancelScope(shield=True), anyio.move_on_after(_CREATE_TIMEOUT):
                app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
                if self._image is None:
                    # Built on create, not at import: Modal caches it per workspace after the first build.
                    built = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')  # pyright: ignore[reportUnknownMemberType]
                elif isinstance(self._image, str):
                    built = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
                else:
                    built = self._image
                variables: dict[str, str | None] | None = dict(self._env) if self._env else None
                sandbox = await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
                    app=app,
                    image=built,
                    workdir=self._working_dir,
                    env=variables,
                    timeout=self._sandbox_timeout,
                    idle_timeout=self._idle_timeout,
                )
        except Exception as error:
            message = f'Could not start Modal sandbox: {error}'
            mapped = _translate(error, context=message, unavailable=message)
            if mapped is None:
                raise
            # Modal refused the request itself: an unknown app or image, or an invalid argument
            # such as a `sandbox_timeout` above its limit. Retrying cannot fix that, so it ends
            # the run rather than going back to the model.
            raise (
                mapped if isinstance(mapped, WorkspaceUnavailableError) else WorkspaceUnavailableError(message)
            ) from error
        if sandbox is None:
            # A plain `TimeoutError`: an unresponsive control plane is transient, so a durable
            # engine retries it.
            raise TimeoutError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            )
        return sandbox

    async def _attach(self, sandbox_id: str) -> modal.Sandbox:
        """Attach to a Modal sandbox that already exists.

        Modal hands back a handle for a sandbox it still knows about even after that sandbox
        has terminated, so this polls: a `WorkspaceRef` must not resolve to a dead environment.
        Nothing is recreated in its place -- a run that expected files there must be told they
        are gone, not handed an empty sandbox.
        """
        import modal

        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
            finished = await sandbox.poll.aio()
        except Exception as error:
            mapped = _translate(
                error,
                context=f'Could not connect to Modal sandbox {sandbox_id!r}',
                unavailable=_unavailable_message(sandbox_id),
            )
            if mapped is None:
                raise
            # A malformed stored ref cannot become valid by retrying an attach.
            if isinstance(error, modal.exception.InvalidError):
                raise WorkspaceUnavailableError(_unavailable_message(sandbox_id)) from error
            raise mapped from error
        if finished is not None:
            raise WorkspaceUnavailableError(_unavailable_message(sandbox_id))
        return sandbox

    async def working_dir(self) -> str:
        """The sandbox's working directory (absolute POSIX path)."""
        # Modal exposes no API for a running sandbox's working directory -- it is the image's,
        # or the configured `working_dir` every command is given -- so ask the sandbox itself,
        # which also canonicalizes a configured path. It cannot change, so the probe is an
        # idempotent read: overlapping first calls may each run their own `pwd`, get the same
        # answer, and the cache converges. No lock needed.
        if self._resolved_working_dir is None:
            sandbox = await self.get_client()
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            # Only an absolute path is an answer. Caching whatever else the sandbox printed
            # would hand every later `resolve()` a working directory that is not one,
            # mis-resolving relative paths with no error.
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise WorkspaceError(
                    f'Could not determine the working directory of Modal sandbox {sandbox.object_id!r}: '
                    f'`pwd` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
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
        sandbox = await self.get_client()
        # Modal takes whole seconds and reads 0 as no deadline, so round up. Messages quote the
        # caller's `timeout`; the exception's `timeout` attribute is the deadline Modal enforced.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        timed_out = f'Command timed out after {timeout} seconds.'
        server_started_at = time.monotonic()
        try:
            with anyio.fail_after(timeout):
                async with self._mapped_errors(sandbox, 'Command could not run in the workspace'):
                    process = await sandbox.exec.aio(
                        *argv, timeout=deadline, workdir=workdir, env=variables, text=False
                    )
        except TimeoutError as error:
            raise WorkspaceTimeoutError(
                'Timed out before the command could start; the Modal process may still be running.'
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

                raise WorkspaceTimeoutError(timed_out, stdout=captured(tasks[0]), stderr=captured(tasks[1])) from error
            if isinstance(error, Exception) and (
                mapped := await _failure(
                    sandbox,
                    error,
                    'Could not read the command result (the command may still run until its deadline)',
                )
            ):
                raise mapped from error
            raise

        elapsed = time.monotonic() - server_started_at
        if deadline is not None and (
            exit_code == _CLIENT_DEADLINE_EXIT or (exit_code == _SIGKILL_EXIT and elapsed >= deadline)
        ):
            raise WorkspaceTimeoutError(timed_out, stdout=stdout, stderr=stderr)
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Modal sandbox {sandbox_id!r} is no longer running: it was terminated, or it reached its '
        "`sandbox_timeout` or `idle_timeout`. Pass `workspace='new'` to start a fresh sandbox."
    )


async def _failure(sandbox: modal.Sandbox, error: Exception, context: str, path: str | None = None) -> Exception | None:
    """Translate an SDK failure on `sandbox`, or `None` to let it propagate.

    Two Modal errors do not say whether the sandbox is gone: exec on a dead sandbox raises
    `ConflictError`, which Modal also uses for transient aborts, and the filesystem layer reports a
    dead sandbox as a generic `SandboxFilesystemError`. For those, the sandbox is probed before the
    error is reported as an ordinary operation failure.
    """
    import modal

    mapped = _translate(error, context=context, unavailable=_unavailable_message(sandbox.object_id), path=path)
    if type(mapped) is WorkspaceError and isinstance(
        error, (modal.exception.ConflictError, modal.exception.SandboxFilesystemError)
    ):
        return await _probe(sandbox, error) or mapped
    return mapped


async def _probe(sandbox: modal.Sandbox, error: Exception) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if `sandbox` has stopped running, else `None`."""
    # Probing only after an error keeps the extra round trip off successful operations.
    import modal

    unavailable = _unavailable_message(sandbox.object_id)
    try:
        finished = await sandbox.poll.aio()
        if finished is None and isinstance(error, modal.exception.SandboxFilesystemError):
            # A terminated sandbox that is still shutting down polls as running and fails
            # filesystem calls with a generic error; only exec names the state.
            with anyio.fail_after(_INTERNAL_EXEC_TIMEOUT):
                await sandbox.exec.aio('true', timeout=_INTERNAL_EXEC_TIMEOUT)
    except Exception as probe_error:
        # A probe failing for any other reason, a transport error included, leaves the
        # original error standing rather than replacing it.
        mapped = _translate(probe_error, context='', unavailable=unavailable)
        return mapped if isinstance(mapped, WorkspaceUnavailableError) else None
    if finished is not None:
        return WorkspaceUnavailableError(unavailable)
    return None

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
from collections.abc import AsyncGenerator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager
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
    import modal
    import modal.io_streams
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('ModalWorkspaceBackend',)

DEFAULT_IMAGE = 'python:3.12-slim'
DEFAULT_APP_NAME = 'pydantic-ai-harness'
DEFAULT_SANDBOX_TIMEOUT = 300


_MISSING_MODAL = (
    'The \'modal\' package is required for ModalWorkspace. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the workspace-create RPCs so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120


_INTERNAL_EXEC_TIMEOUT = 10

_CLIENT_DEADLINE_EXIT = -1
_SIGKILL_EXIT = 137

_RESULT_GRACE = 30


def _unavailable_workspace_exc_types() -> tuple[type[BaseException], ...]:
    """Modal exception types that mean the workspace itself no longer exists -- a terminal condition.

    A missing *file* is a different, recoverable error (translated to the builtin
    `FileNotFoundError`); these are the ones that say the whole workspace is unusable.
    """
    import modal

    return (
        modal.exception.NotFoundError,
        modal.exception.SandboxTerminatedError,
        modal.exception.SandboxTimeoutError,
    )


def _command_argv(command: WorkspaceCommand, shell: bool) -> Sequence[str]:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        # Modal executes argv and never a shell string, so shell interpretation is requested
        # explicitly. `/bin/sh` rather than bash: it is the one shell every sandbox image carries.
        return ['/bin/sh', '-c', command]
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    if not command:
        raise TypeError('a command needs at least the program to run; the argv sequence is empty')
    return command


def _file_entry(entry: modal.types.FileInfo, path: str) -> FileEntry:
    is_dir = entry.is_dir()
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    return FileEntry(name=entry.name, path=path, is_dir=is_dir, size=None if is_dir else entry.size)


class ModalWorkspaceBackend(WorkspaceBackend, SupportsFilesystem):
    """A Modal sandbox implementing Pydantic AI's ``WorkspaceBackend`` protocol.

    Construction performs no I/O. The first operation creates or attaches to a sandbox, and the
    native handle is available through `workspace` for application-owned SDK lifecycle.

    Modal applies whole-second command deadlines. Cancelling ``run()`` stops the local wait while
    the command may continue until its deadline or the sandbox lifetime ends.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `WorkspaceBackend` call.

    Args:
        workspace: A live `modal.Sandbox` you already have. Whoever created it owns terminating it.
        ref: Identity of an existing workspace to attach to on first use.
        name: Optional Modal name passed when creating a new sandbox. It is not used to recover a
            sandbox when `ref` is absent.
        image: Registry tag a newly created workspace runs.
        app_name: Modal app a newly created workspace belongs to.
        create_app_if_missing: Create the Modal app when it does not exist yet.
        sandbox_timeout: How long Modal keeps a newly created workspace alive, in seconds.
        workdir: Absolute directory commands start in; Modal's default when `None`.
        env: Environment variables set for the whole workspace at creation.
    """

    def __init__(
        self,
        workspace: modal.Sandbox | None = None,
        *,
        ref: WorkspaceRef | None = None,
        name: str | None = None,
        image: str = DEFAULT_IMAGE,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        workdir: str | None = None,
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
        self._workdir = absolute_path('workdir', workdir)
        self._env = dict(env) if env is not None else None
        self._working_dir: str | None = None
        # Set once the workspace exists, so an expiry message can say which lifetime ran out.
        self._created_timeout: int | None = None

    @cached_property
    def _lock(self) -> anyio.Lock:
        return anyio.Lock()

    @property
    def workspace(self) -> Awaitable[modal.Sandbox]:
        return self._get_workspace()

    async def _get_workspace(self) -> modal.Sandbox:
        """Acquire the native Modal sandbox and record its identity."""
        if self._workspace is None:
            async with self._lock:
                if self._workspace is None:
                    try:
                        importlib.import_module('modal')
                    except ImportError as e:
                        raise WorkspaceError(_MISSING_MODAL) from e
                    self._workspace = await self._create_or_attach(self._ref)
                    self._ref = WorkspaceRef(provider='modal', id=self._workspace.object_id)
        assert self._workspace is not None
        return self._workspace

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the workspace, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _translated_filesystem_error(self, path: str) -> AsyncGenerator[None]:
        """Map Modal's filesystem exceptions onto the ones the protocol promises."""
        import modal

        try:
            yield
        except modal.exception.SandboxFilesystemNotFoundError as e:
            raise FileNotFoundError(f'No such file or directory in the Modal sandbox: {path!r}') from e
        except modal.exception.SandboxFilesystemIsADirectoryError as e:
            raise IsADirectoryError(f'Is a directory in the Modal sandbox: {path!r}') from e
        except modal.exception.SandboxFilesystemNotADirectoryError as e:
            raise NotADirectoryError(f'Not a directory in the Modal sandbox: {path!r}') from e
        except modal.exception.Error as e:
            raise await self._operation_error(e, f'Could not access {path!r} in the workspace') from e

    async def read_bytes(self, path: str) -> bytes:
        workspace = await self.workspace
        async with self._translated_filesystem_error(path):
            return await workspace.filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        workspace = await self.workspace
        async with self._translated_filesystem_error(path):
            await workspace.filesystem.write_bytes.aio(data, path)

    async def stat(self, path: str) -> FileEntry:
        workspace = await self.workspace
        async with self._translated_filesystem_error(path):
            return _file_entry(await workspace.filesystem.stat.aio(path), path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        workspace = await self.workspace
        async with self._translated_filesystem_error(path):
            entries = await workspace.filesystem.list_files.aio(path)
        return [_file_entry(entry, posixpath.join(path, entry.name)) for entry in entries]

    async def make_dir(self, path: str) -> None:
        workspace = await self.workspace
        async with self._translated_filesystem_error(path):
            await workspace.filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        workspace = await self.workspace
        async with self._translated_filesystem_error(path):
            await workspace.filesystem.remove.aio(path, recursive=True)

    async def exists(self, path: str) -> bool:
        workspace = await self.workspace
        import modal

        try:
            await workspace.filesystem.stat.aio(path)
        except (
            modal.exception.SandboxFilesystemNotFoundError,
            modal.exception.SandboxFilesystemNotADirectoryError,
        ):
            return False
        except modal.exception.Error as e:
            raise await self._operation_error(e, f'Could not access {path!r} in the workspace') from e
        return True

    async def _create(self) -> modal.Sandbox:
        """Provision a fresh Modal sandbox."""
        import modal

        try:
            # Cancellation during create can orphan a workspace until `sandbox_timeout` reaps it.
            with anyio.fail_after(_CREATE_TIMEOUT):
                app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
                built = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
                variables: dict[str, str | None] | None = dict(self._env) if self._env is not None else None
                workspace = await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
                    app=app,
                    image=built,
                    timeout=self._sandbox_timeout,
                    workdir=self._workdir,
                    env=variables,
                    name=self._name,
                )
        except TimeoutError as error:
            raise WorkspaceError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            ) from error
        except modal.exception.AuthError as error:
            raise WorkspaceUnavailableError(_AUTH_MESSAGE) from error
        except modal.exception.Error as error:
            raise WorkspaceError(f'Could not start Modal sandbox: {error}') from error
        self._created_timeout = self._sandbox_timeout
        return workspace

    async def _create_or_attach(self, ref: WorkspaceRef | None) -> modal.Sandbox:
        if ref is not None:
            return await self._attach(ref.id)
        return await self._create()

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
        except modal.exception.AuthError as e:
            raise WorkspaceUnavailableError(_AUTH_MESSAGE) from e
        except _unavailable_workspace_exc_types() as e:
            raise WorkspaceUnavailableError(_attached_gone_message(repr(id))) from e
        except modal.exception.Error as e:
            raise WorkspaceError(f'Could not connect to Modal sandbox {id!r}: {e}') from e
        if finished is not None:
            raise WorkspaceUnavailableError(_attached_gone_message(repr(id)))
        return workspace

    async def working_dir(self) -> str:
        """The workspace's default working directory (absolute POSIX path)."""
        # Modal exposes no API for a running workspace's working directory -- it is the image's
        # unless `create(workdir=...)` overrode it -- so ask the environment itself. It cannot
        # change, so the probe is an idempotent read: overlapping first calls may each run
        # their own `pwd`, get the same answer, and the cache converges. No lock needed.
        if self._working_dir is None:
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
            self._working_dir = printed
        return self._working_dir

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

        Modal has no per-command kill, so a cancelled `run()` stops the wait but leaves the
        command running until its `timeout` deadline. Pass a finite `timeout` so an abandoned
        command cannot run on indefinitely.
        """
        argv = _command_argv(command, shell)
        cwd = absolute_path('cwd', cwd)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        variables: dict[str, str | None] | None = dict(env) if env is not None else None
        call_started_at = time.monotonic()
        try:
            with anyio.fail_after(timeout):
                workspace = await self.workspace
        except TimeoutError as error:
            raise WorkspaceTimeoutError('Timed out before the command could start.', timeout=timeout) from error
        import modal

        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - call_started_at))
        deadline = None if remaining is None else max(1, math.ceil(remaining))
        server_started_at = time.monotonic()
        try:
            with anyio.fail_after(remaining):
                process = await workspace.exec.aio(*argv, timeout=deadline, workdir=cwd, env=variables, text=False)
        except TimeoutError as error:
            raise WorkspaceTimeoutError(
                'Timed out before the command could start; the Modal process may still be running.', timeout=timeout
            ) from error
        except modal.exception.Error as error:
            raise await self._operation_error(error, 'Command could not run in the workspace') from error

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
            if isinstance(error, TimeoutError):

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
            if isinstance(error, Exception):
                raise await self._operation_error(
                    error, 'Could not read the command result (the command may still run until its deadline)'
                ) from error
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

    def _unavailable_message(self) -> str:
        if self._created_timeout is None:
            assert self._ref is not None
            return _attached_gone_message(repr(self._ref.id))
        return (
            'The Modal sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._created_timeout}s, or been terminated). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def _operation_error(self, e: Exception, context: str) -> WorkspaceError:
        """Translate an SDK failure into this backend's error taxonomy.

        A terminated or missing workspace and rejected credentials are terminal. Modal reports
        two failures ambiguously -- a first exec on a dead workspace raises `ConflictError`
        (also used for transient aborts), and the filesystem layer wraps everything including
        auth failures -- so those are classified by polling the workspace. Everything else stays
        a recoverable `WorkspaceError` carrying `context`, which distinguishes "the command
        never started" from "the result could not be read".
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return WorkspaceUnavailableError(_AUTH_MESSAGE)
        if isinstance(e, _unavailable_workspace_exc_types()):
            return WorkspaceUnavailableError(self._unavailable_message())
        if isinstance(e, (modal.exception.ConflictError, modal.exception.SandboxFilesystemError)):
            return await self._poll_ambiguous(e)
        if isinstance(e, modal.exception.Error):
            return WorkspaceError(f'{context}: {e}')
        return WorkspaceError(f'{context}: {type(e).__name__}: {e}')

    async def _poll_ambiguous(self, e: Exception) -> WorkspaceError:
        # Polling only after an error keeps the extra round trip off successful operations.
        import modal

        try:
            workspace = await self.workspace
            finished = await workspace.poll.aio()
        except modal.exception.AuthError:
            return WorkspaceUnavailableError(_AUTH_MESSAGE)
        except _unavailable_workspace_exc_types():
            return WorkspaceUnavailableError(self._unavailable_message())
        except Exception:
            # The classifying poll can itself fail, including with a raw transport error;
            # fall back to the original error rather than letting the probe abort the run.
            return WorkspaceError(str(e))
        if finished is not None:
            return WorkspaceUnavailableError(self._unavailable_message())
        return WorkspaceError(str(e))


def _attached_gone_message(described: str) -> str:
    return (
        f'The Modal sandbox {described} is no longer running '
        '(it does not exist, was terminated, or expired at its configured lifetime). '
        'Attach to a live workspace, or create a new one.'
    )

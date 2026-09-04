"""A Modal sandbox behind Pydantic AI's [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol.

This is the mechanism layer: every Modal-specific operation (create, connect, exec, file
access, working-directory discovery, teardown) lives here, behind the protocol the rest of
Pydantic AI already speaks. The capability in `_capability.py` owns the lifecycle; tools and
other capabilities consume the resulting `ctx.sandbox`.

External assumptions last verified 2026-08-31 against Modal Python SDK 1.5.2 (the package floor):

* asynchronous sandbox operations use the SDK's `.aio` call surface; `from_id` reconnects by
  object ID; `terminate` leaves the Python client attached, so cleanup also calls `detach`:
  https://modal.com/docs/guide/sandboxes
* `Sandbox.exec` returns a process with separate stdout and stderr readers, and its `timeout`
  bounds command execution; output is buffered unless the caller consumes the streams:
  https://modal.com/docs/guide/sandbox-spawn
* the sandbox filesystem supplies byte reads and writes, metadata, directory listing, creation,
  and removal:
  https://modal.com/docs/sdk/py/latest/Sandbox

Re-check these sources and the installed 1.5.2 signatures before changing lifecycle, command,
stream, or filesystem handling.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import posixpath
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
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
    import modal
    import modal.container_process
    import modal.io_streams
    from pydantic_ai.sandboxes import SandboxCommand, SupportsFilesystem

__all__ = (
    'ModalSandboxAuthError',
    'ModalSandboxBackend',
    'ModalSandboxError',
    'ModalSandboxUnavailableError',
)

# Defaults shared by `ModalSandboxBackend.create` and the `ModalSandbox` capability (which
# imports them), so the two cannot drift: a setting is "left at its default" iff it equals
# the constant here.
DEFAULT_IMAGE = 'python:3.12-slim'
DEFAULT_APP_NAME = 'pydantic-ai-harness'
DEFAULT_SANDBOX_TIMEOUT = 300


_MISSING_MODAL = (
    'The \'modal\' package is required for ModalSandbox. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the sandbox-create RPCs so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120

# Teardown runs shielded from cancellation, so an unreachable Modal control plane could
# otherwise hang the caller forever. Bound each teardown RPC so a stalled terminate/detach
# gives up rather than wedging the process; an owned sandbox is still reaped server-side by
# its own `sandbox_timeout`.
_TEARDOWN_TIMEOUT = 30

# Modal exposes no per-command kill, so every command carries a deadline -- including the
# internal `pwd` probe behind `working_dir()`.
_INTERNAL_EXEC_TIMEOUT = 10

# What Modal's client reports when its own copy of a command's deadline ends the wait.
_CLIENT_DEADLINE_EXIT = -1
# The exit status of a process killed by SIGKILL (128 + 9): what the server side of the same
# deadline looks like when its kill lands before the client's deadline fires.
_SIGKILL_EXIT = 137

# After a command's deadline expires, give Modal this long to report the server-side kill
# before the result wait gives up (a wedged control plane must not hang `wait()` forever).
_RESULT_GRACE = 30


class ModalSandboxError(SandboxError):
    """A recoverable Modal provider operation failed."""


class ModalSandboxUnavailableError(ModalSandboxError, SandboxUnavailableError):
    """The referenced Modal sandbox is no longer available."""


class ModalSandboxAuthError(ModalSandboxError, SandboxUnavailableError):
    """Modal rejected the configured credentials."""


class _ModalSandboxAlreadyExists(ModalSandboxError):
    """A named create lost a race to an existing sandbox."""


def _unavailable_sandbox_exc_types() -> tuple[type[BaseException], ...]:
    """Modal exception types that mean the sandbox itself no longer exists -- a terminal condition.

    A missing *file* is a different, recoverable error (translated to the builtin
    `FileNotFoundError`); these are the ones that say the whole sandbox is unusable.
    """
    import modal

    return (
        modal.exception.NotFoundError,
        modal.exception.SandboxTerminatedError,
        modal.exception.SandboxTimeoutError,
    )


def _command_argv(command: SandboxCommand, shell: bool) -> Sequence[str]:
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


class _ModalProcess:
    """Private command result helper used by `ModalSandboxBackend.run`."""

    def __init__(
        self,
        process: modal.container_process.ContainerProcess[bytes],
        *,
        backend: ModalSandboxBackend,
        deadline: int | None,
        started_at: float,
    ) -> None:
        self._process = process
        self._backend = backend
        self._deadline = deadline
        self._started_at = started_at

    async def wait(self) -> CommandResult:
        """Wait for the command and return its result."""
        return await self._settle()

    async def _settle(self) -> CommandResult:
        async def read(reader: modal.io_streams.StreamReader[bytes]) -> str:
            return (await reader.read.aio()).decode('utf-8', errors='replace')

        tasks = (
            asyncio.create_task(read(self._process.stdout)),
            asyncio.create_task(read(self._process.stderr)),
            asyncio.create_task(self._process.wait.aio()),
        )
        gather = asyncio.gather(*tasks)
        try:
            if self._deadline is None:
                stdout, stderr, exit_code = await gather
            else:
                remaining = max(0.0, self._started_at + self._deadline - time.monotonic())
                stdout, stderr, exit_code = await asyncio.wait_for(gather, remaining + _RESULT_GRACE)
        except BaseException as error:
            # Cancelling the gather cancels its children; awaiting them reaps the cancellations.
            gather.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(error, Exception):  # cancellation is BaseException-only and re-raises below
                raise await self._backend.operation_error(
                    error, 'Could not read the command result (the command may still run until its deadline)'
                ) from error
            raise

        elapsed = time.monotonic() - self._started_at
        # A wait first called long after the deadline can misdate an organic 137 as a timeout.
        if self._timed_out(exit_code, elapsed):
            assert self._deadline is not None
            raise SandboxTimeoutError(
                f'Command timed out after {self._deadline} seconds and was killed.',
                stdout=stdout,
                stderr=stderr,
                timeout=self._deadline,
            )
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def _timed_out(self, exit_code: int, elapsed: float) -> bool:
        if self._deadline is None:
            return False
        if exit_code == _CLIENT_DEADLINE_EXIT:
            return True
        # A command can exit 137 on its own account (an OOM kill, a `kill -9` it asked for), so
        # that exit only means "the deadline killed it" once the whole window has elapsed. The
        # window is measured from before the exec call, which makes it a superset of the one
        # Modal's own timer runs -- the platform starts counting when the command starts, inside
        # that round trip -- so a deadline kill always lands inside it and an earlier exit does not.
        return exit_code == _SIGKILL_EXIT and elapsed >= self._deadline


def _file_entry(entry: modal.types.FileInfo, path: str) -> FileEntry:
    is_dir = entry.is_dir()
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    return FileEntry(name=entry.name, path=path, is_dir=is_dir, size=None if is_dir else entry.size)


class ModalSandboxBackend(SandboxBackend):
    """A [Modal](https://modal.com) sandbox as a Pydantic AI [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend].

    Commands and file operations run inside a Modal container, so the host is never exposed.

    Building one does no I/O. It holds settings plus, optionally, the identity of a sandbox that
    already exists; the first operation creates or attaches, once, and everything after that
    reuses the same environment. Reach the live `modal.Sandbox` through
    [`sandbox`][pydantic_ai_harness.modal_sandbox.ModalSandboxBackend.sandbox], which you can
    only await — so no operation can run against a sandbox that does not exist yet.

    Nothing here terminates a sandbox on its own. Modal reaps one at the `sandbox_timeout` it
    was created with; call [`close`][pydantic_ai_harness.modal_sandbox.ModalSandboxBackend.close]
    with `terminate=True` to end it sooner.

    Commands run as one-shot operations, with complete output returned after they finish.
    Modal enforces `timeout=` itself, so a command is bounded by the deadline applied to its
    execution. Cancelling a `run()` stops the wait but not the command: it runs on until its
    deadline, or until the sandbox is terminated. Modal takes whole seconds, so a fractional
    `timeout=` rounds up to the deadline actually applied.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `Sandbox.wrap` call.

    Args:
        sandbox: A live `modal.Sandbox` you already have. Whoever created it owns terminating it.
        ref: Identity of an existing sandbox to attach to on first use.
        name: Modal name to create-or-attach on first use, unique among running sandboxes in the
            app. This is what lets several runs share one environment. Ignored when `sandbox` or
            `ref` is given.
        image: Registry tag a newly created sandbox runs.
        app_name: Modal app a newly created sandbox belongs to.
        create_app_if_missing: Create the Modal app when it does not exist yet.
        sandbox_timeout: How long Modal keeps a newly created sandbox alive, in seconds.
        workdir: Absolute directory commands start in; Modal's default when `None`.
        env: Environment variables set for the whole sandbox at creation.
    """

    def __init__(
        self,
        sandbox: modal.Sandbox | None = None,
        *,
        ref: SandboxRef | None = None,
        name: str | None = None,
        image: str = DEFAULT_IMAGE,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._live = sandbox
        self._ref = ref if sandbox is None else SandboxRef(sandbox_id=sandbox.object_id)
        self._name = name
        self._image = image
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = absolute_path('workdir', workdir)
        self._env = dict(env) if env is not None else None
        # Known up front only when this backend will create the sandbox with an explicit
        # `workdir`; otherwise it is the image's, discovered with `pwd` on first use.
        self._working_dir = self._workdir if (sandbox is None and ref is None) else None
        # Set once the sandbox exists, so an expiry message can say which lifetime ran out.
        self._created_timeout: int | None = None
        self._lock = anyio.Lock()

    @property
    def sandbox(self) -> Awaitable[modal.Sandbox]:
        """The live `modal.Sandbox`, created or attached on first use.

        Awaitable and never a plain value: every operation has to go through the step that
        makes the sandbox exist, so none of them can skip it.
        """
        return self._resolve()

    async def _resolve(self) -> modal.Sandbox:
        async with self._lock:
            if self._live is None:
                try:
                    # Guarded once, here: everything that touches Modal runs after this.
                    importlib.import_module('modal')
                except ImportError as e:
                    raise ModalSandboxError(_MISSING_MODAL) from e
                if self._ref is not None:
                    self._live = await self._attach(self._ref.sandbox_id)
                elif self._name is not None:
                    self._live = await self._create_or_attach_by_name(self._name)
                else:
                    self._live = await self._create()
                self._ref = SandboxRef(sandbox_id=self._live.object_id)
        return self._live

    @property
    def ref(self) -> SandboxRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _translated_filesystem_error(self, path: str) -> AsyncGenerator[None]:
        """Map Modal's filesystem exceptions onto the ones the protocol promises."""
        import modal

        try:
            yield
        except modal.exception.SandboxFilesystemNotFoundError as e:
            raise FileNotFoundError(f'No such file or directory in the Modal sandbox: {path!r}') from e
        except modal.exception.Error as e:
            raise await self.operation_error(e, f'Could not access {path!r} in the sandbox') from e

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated_filesystem_error(path):
            return await (await self.sandbox).filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).filesystem.write_bytes.aio(data, path)

    async def stat(self, path: str) -> FileEntry:
        async with self._translated_filesystem_error(path):
            return _file_entry(await (await self.sandbox).filesystem.stat.aio(path), path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._translated_filesystem_error(path):
            entries = await (await self.sandbox).filesystem.list_files.aio(path)
        return [_file_entry(entry, posixpath.join(path, entry.name)) for entry in entries]

    async def make_dir(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).filesystem.remove.aio(path, recursive=True)

    async def exists(self, path: str) -> bool:
        import modal

        try:
            await (await self.sandbox).filesystem.stat.aio(path)
        except (
            modal.exception.SandboxFilesystemNotFoundError,
            modal.exception.SandboxFilesystemNotADirectoryError,
        ):
            return False
        except modal.exception.Error as e:
            raise await self.operation_error(e, f'Could not access {path!r} in the sandbox') from e
        return True

    async def _create(self, name: str | None = None) -> modal.Sandbox:
        """Provision a fresh Modal sandbox."""
        import modal

        try:
            # Cancellation during create can orphan a sandbox until `sandbox_timeout` reaps it.
            with anyio.fail_after(_CREATE_TIMEOUT):
                app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
                built = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
                variables: dict[str, str | None] | None = dict(self._env) if self._env is not None else None
                sandbox = await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
                    app=app,
                    image=built,
                    timeout=self._sandbox_timeout,
                    workdir=self._workdir,
                    env=variables,
                    name=name,
                )
        except TimeoutError as error:
            raise ModalSandboxError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            ) from error
        except modal.exception.AlreadyExistsError as error:
            raise _ModalSandboxAlreadyExists(f'Modal sandbox named {name!r} already exists.') from error
        except modal.exception.AuthError as error:
            raise ModalSandboxAuthError(_AUTH_MESSAGE) from error
        except modal.exception.Error as error:
            raise ModalSandboxError(f'Could not start Modal sandbox: {error}') from error
        self._created_timeout = self._sandbox_timeout
        return sandbox

    async def _create_or_attach_by_name(self, name: str) -> modal.Sandbox:
        """Attach to the running named sandbox, or create it once.

        A concurrent retry may win the create race. In that case, attach by name rather than
        provisioning another environment.
        """
        try:
            return await self._attach_by_name(name)
        except ModalSandboxUnavailableError:
            pass
        try:
            return await self._create(name)
        except _ModalSandboxAlreadyExists:
            pass
        return await self._attach_by_name(name)

    async def _attach(self, sandbox_id: str) -> modal.Sandbox:
        """Attach to a Modal sandbox that already exists.

        Modal hands back a handle for a sandbox it still knows about even after that sandbox
        has terminated, so this polls: a `SandboxRef` must not resolve to a dead environment.
        Nothing is recreated in its place — a run that expected files there must be told they
        are gone, not handed an empty workspace.
        """
        import modal

        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
            finished = await sandbox.poll.aio()
        except modal.exception.AuthError as e:
            raise ModalSandboxAuthError(_AUTH_MESSAGE) from e
        except _unavailable_sandbox_exc_types() as e:
            raise ModalSandboxUnavailableError(_attached_gone_message(repr(sandbox_id))) from e
        except modal.exception.Error as e:
            raise ModalSandboxError(f'Could not connect to Modal sandbox {sandbox_id!r}: {e}') from e
        if finished is not None:
            raise ModalSandboxUnavailableError(_attached_gone_message(repr(sandbox_id)))
        return sandbox

    async def _attach_by_name(self, name: str) -> modal.Sandbox:
        """Attach to a running named sandbox without creating one."""
        import modal

        try:
            sandbox = await modal.Sandbox.from_name.aio(self._app_name, name)
            finished = await sandbox.poll.aio()
        except modal.exception.AuthError as error:
            raise ModalSandboxAuthError(_AUTH_MESSAGE) from error
        except _unavailable_sandbox_exc_types() as error:
            raise ModalSandboxUnavailableError(
                f'No running Modal sandbox named {name!r} in app {self._app_name!r}.'
            ) from error
        except modal.exception.Error as error:
            raise ModalSandboxError(
                f'Could not connect to Modal sandbox named {name!r} in app {self._app_name!r}: {error}'
            ) from error
        if finished is not None:
            raise ModalSandboxUnavailableError(
                f'Modal sandbox named {name!r} in app {self._app_name!r} is not running.'
            )
        return sandbox

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
        """Release this handle, terminating the sandbox with it when we own its lifetime.

        Each teardown call is shielded from cancellation and bounded so a stalled control
        plane cannot wedge the caller.
        """
        import modal

        sandbox = self._live
        if sandbox is None:
            # Never used, so there is nothing to terminate and nothing to detach from.
            # Resolving one here just to close it would create the very sandbox being released.
            return

        async def terminate_call() -> object:
            return await sandbox.terminate.aio(wait=True)

        async def detach_call() -> object:
            return await sandbox.detach.aio()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        calls: list[tuple[str, Callable[[], Awaitable[object]]]] = []
        if terminate:
            calls.append(('terminate', terminate_call))
        calls.append(('detach', detach_call))
        first_error: ModalSandboxError | None = None
        for operation, call in calls:
            error = await cleanup_call(call, timeout=_TEARDOWN_TIMEOUT)
            if error is None or isinstance(error, _unavailable_sandbox_exc_types()):
                continue
            if isinstance(error, modal.exception.AuthError):
                translated = ModalSandboxAuthError(_AUTH_MESSAGE)
            elif isinstance(error, TimeoutError):
                translated = ModalSandboxError(
                    f'Timed out after {_TEARDOWN_TIMEOUT}s while trying to {operation} '
                    f'Modal sandbox {self._describe()}.'
                )
            else:
                translated = ModalSandboxError(f'Could not {operation} Modal sandbox {self._describe()}: {error}')
            if first_error is None:
                first_error = translated
        if first_error is not None:
            await raise_after_cleanup(first_error)

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        # Modal exposes no API for a running sandbox's working directory -- it is the image's
        # unless `create(workdir=...)` overrode it -- so ask the environment itself. It cannot
        # change, so the probe is an idempotent read: overlapping first calls may each run
        # their own `pwd`, get the same answer, and the cache converges. No lock needed.
        if self._working_dir is None:
            result = await self.run(['pwd'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.strip()
            # Only an absolute path is an answer. Caching whatever else the environment
            # printed would hand every later `resolve()` a working directory that is not
            # one, mis-resolving relative paths with no error.
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise ModalSandboxError(
                    f'Could not determine the working directory of Modal sandbox {self._describe()}: '
                    f'`pwd` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
                )
            self._working_dir = printed
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
        """Execute a command and wait for it to complete.

        Modal has no per-command kill, so a cancelled `run()` stops the wait but leaves the
        command running until its `timeout` deadline. Pass a finite `timeout` so an abandoned
        command cannot run on indefinitely.
        """
        process = await self._start(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
        return await process.wait()

    async def _start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _ModalProcess:
        """Start the private command helper used by `run`."""
        import modal

        argv = _command_argv(command, shell)
        cwd = absolute_path('cwd', cwd)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        # Modal takes whole seconds and reads a missing deadline as "run until the sandbox
        # dies", so a sub-second deadline rounds up rather than silently becoming unbounded.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        # Stamped before the call, so the window a `137` is dated against is a superset of the
        # one Modal's own timer runs: the platform starts counting when the command starts,
        # which happens inside this round trip.
        variables: dict[str, str | None] | None = dict(env) if env is not None else None
        started_at = time.monotonic()
        try:
            # Modal's text mode decodes strictly, so read bytes and decode with replacement:
            # a command printing invalid UTF-8 must not abort the run.
            sandbox = await self.sandbox
            process = await sandbox.exec.aio(*argv, timeout=deadline, workdir=cwd, env=variables, text=False)
        except modal.exception.Error as e:
            raise await self.operation_error(e, 'Command could not run in the sandbox') from e
        return _ModalProcess(process, backend=self, deadline=deadline, started_at=started_at)

    def _unavailable_message(self) -> str:
        if self._created_timeout is None:
            return _attached_gone_message(self._describe())
        return (
            'The Modal sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._created_timeout}s, or been terminated). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def operation_error(self, e: Exception, context: str) -> ModalSandboxError:
        """Translate an SDK failure into this backend's error taxonomy.

        A terminated or missing sandbox and rejected credentials are terminal. Modal reports
        two failures ambiguously -- a first exec on a dead sandbox raises `ConflictError`
        (also used for transient aborts), and the filesystem layer wraps everything including
        auth failures -- so those are classified by polling the sandbox. Everything else stays
        a recoverable `ModalSandboxError` carrying `context`, which distinguishes "the command
        never started" from "the result could not be read".
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        if isinstance(e, _unavailable_sandbox_exc_types()):
            return ModalSandboxUnavailableError(self._unavailable_message())
        if isinstance(e, (modal.exception.ConflictError, modal.exception.SandboxFilesystemError)):
            return await self._poll_ambiguous(e)
        if isinstance(e, modal.exception.Error):
            return ModalSandboxError(f'{context}: {e}')
        return ModalSandboxError(f'{context}: {type(e).__name__}: {e}')

    async def _poll_ambiguous(self, e: Exception) -> ModalSandboxError:
        # Polling only after an error keeps the extra round trip off successful operations.
        import modal

        try:
            sandbox = await self.sandbox
            finished = await sandbox.poll.aio()
        except modal.exception.AuthError:
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        except _unavailable_sandbox_exc_types():
            return ModalSandboxUnavailableError(self._unavailable_message())
        except Exception:
            # The classifying poll can itself fail, including with a raw transport error;
            # fall back to the original error rather than letting the probe abort the run.
            return ModalSandboxError(str(e))
        if finished is not None:
            return ModalSandboxUnavailableError(self._unavailable_message())
        return ModalSandboxError(str(e))


def _attached_gone_message(described: str) -> str:
    return (
        f'The Modal sandbox {described} is no longer running '
        '(it does not exist, was terminated, or expired at its configured lifetime). '
        'Attach to a live sandbox, or create a new one.'
    )


if TYPE_CHECKING:
    # Pins full structural conformance -- signatures included -- which `isinstance` cannot
    # check. `__new__` rather than a call, because neither SDK object can be constructed
    # without a live sandbox behind it; this block never runs.
    _backend = ModalSandboxBackend()
    _backend_conforms: SandboxBackend = _backend
    _filesystem_backend_conforms: SupportsFilesystem = _backend

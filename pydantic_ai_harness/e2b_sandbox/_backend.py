"""An E2B sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-09-09 against E2B 2.46.4 and 2.34.0: `connect` resumes
paused workspaces, with a 300-second lifetime unless it is given `timeout`; commands use
`/bin/bash -l -c`, the SDK command timeout does not kill the process, and command handles
accumulate output. `create(lifecycle=...)` exists from 2.48.0, the package floor. Re-check
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py and
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command.py
before changing acquisition or deadline behavior.

Exception types verified 2026-09-25 against E2B 2.51.0
(https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/exceptions.py): a missing key or
a 401 is `AuthenticationException`; a gone sandbox is `SandboxNotFoundException` from the control
plane and a `TimeoutException` from envd (its proxy's 502); a 429 is `RateLimitException`, a
`SandboxException` subclass; a create the API refuses is a `SandboxException` carrying its 4xx
`status_code`; a 503 is `ServiceBusyException` (new in 2.48.0, hence the floor),
which is not. envd types only a missing path; its other path failures arrive as a 400
(`InvalidArgumentException`) or a 500 (`SandboxException`) whose message carries envd's wording
or Go's errno text, which `_path_error` reads.
"""

from __future__ import annotations

import math
import posixpath
import shlex
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

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
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('E2BSandboxBackend',)

try:
    import e2b
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[e2b]` to use E2BSandbox.') from error

_AUTH_MESSAGE = 'E2B rejected the credentials. Set a valid E2B_API_KEY in the environment.'

# envd's own messages and Go's errno text, most specific first: making a directory where a file
# exists says both "already exists" and "not a directory", and "is not a directory" must not
# match "is a directory".
_PATH_ERRORS: tuple[tuple[str, type[OSError]], ...] = (
    ('already exists', FileExistsError),
    ('permission denied', PermissionError),
    ('not a directory', NotADirectoryError),
    ('is a directory', IsADirectoryError),
)

# The most E2B's Hobby plan allows, so the default works on every plan; Pro plans allow 86_400.
DEFAULT_SANDBOX_TIMEOUT = 3_600

# Bound the sandbox-create call so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120

# Bounds the internal `pwd` probe behind `working_dir()` and the best-effort kills.
_INTERNAL_EXEC_TIMEOUT = 10

# E2B's own command `timeout` bounds the event stream and leaves the command running, so it is
# switched off (0 is the SDK's "no limit") and the deadline is enforced client-side instead,
# with a kill at expiry. See `E2BSandboxBackend.run`.
_SDK_STREAM_UNBOUNDED = 0


def _command_line(command: WorkspaceCommand, shell: bool) -> str:
    """The single string E2B runs: `commands.run` takes only a string, which it hands to `/bin/bash -l -c`.

    The argv is joined with `shlex.join` so each element stays one word. A `shell=True` string
    becomes `/bin/sh -c <string>`, so it runs under `sh` as on every other backend.
    """
    return shlex.join(command_argv(command, shell))


def _path_error(error: Exception, path: str) -> OSError | None:
    """The builtin path error an untyped envd failure describes, or `None` if it is not one."""
    message = str(error).lower()
    for marker, builtin in _PATH_ERRORS:
        if marker in message:
            return builtin(f'{error} (in the E2B sandbox: {path!r})')
    return None


async def _file_entry(sandbox: e2b.AsyncSandbox, entry: e2b.EntryInfo) -> FileEntry:
    """The protocol entry for `entry`, with `is_dir` and `size` following a symlink.

    envd describes a symlink with the link's own size and, as `symlink_target`, the path it
    resolves to, so a symlink entry is completed by stat-ing that target. A dangling link, which
    envd reports with itself as the target, reads as a file with no size.
    """
    target: e2b.EntryInfo | None = entry
    if entry.symlink_target is not None:
        try:
            target = await sandbox.files.get_info(posixpath.join(posixpath.dirname(entry.path), entry.symlink_target))
        except e2b.FileNotFoundException:
            target = None
        if target is not None and target.symlink_target is not None:
            target = None
    is_dir = target is not None and target.type is e2b.FileType.DIR
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    size = None if target is None or is_dir else target.size
    return FileEntry(name=entry.name, path=entry.path, is_dir=is_dir, size=size)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The E2B sandbox {sandbox_id!r} is no longer running: it was killed, or it was paused when its '
        '`sandbox_timeout` ran out (a later run that attaches to it resumes it). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _is_lifetime_refusal(error: e2b.SandboxException) -> bool:
    """Whether E2B refused the requested `sandbox_timeout`, e.g. `400: Timeout cannot be greater than 1 hours`."""
    status = error.status_code
    return status is not None and 400 <= status < 500 and 'timeout' in str(error).lower()


def _refused_message(context: str, error: e2b.SandboxException) -> str:
    message = f'{context}: {error}'
    if _is_lifetime_refusal(error):
        message += ' Hobby plans allow at most 3600 seconds; pass `E2BSandbox(sandbox_timeout=3600)`.'
    return message


class E2BSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """An [E2B](https://e2b.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside an E2B microVM, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a workspace, and the
    typed `e2b.AsyncSandbox` is available through `get_client()`. The backend does not kill the
    sandbox; killing it is the application's job.

    Commands run as one-shot operations, with complete output returned after they finish.

    Every command runs through `/bin/bash -l -c`, so an argv sequence is quoted into a single
    shell word string first and login startup files run before the command does; a
    `shell=True` string runs under `/bin/sh -c` inside that login shell. E2B's own
    command `timeout` abandons the output stream and leaves the command running, so the
    deadline is enforced client-side instead and the command is killed with SIGKILL when it
    expires or when the caller is cancelled, if E2B has returned the process ID. Cancellation
    during startup can leave the command running. That kill signals the command's own process; a
    process the command started in the background outlives it until the sandbox is torn down.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant workspace call.

    Args:
        workspace: A live `e2b.AsyncSandbox` you already have. Whoever created it owns killing it.
        ref: Identity of an existing sandbox to attach to on first use.
        template: E2B template name or ID a newly created sandbox runs; E2B's default when `None`.
            An unknown template raises `WorkspaceUnavailableError` on first use.
        sandbox_timeout: Total lifetime of the sandbox in seconds, applied when it is created and
            again when attaching to it. When it runs out, E2B pauses the sandbox rather than
            killing it, and attaching resumes it. The default, 3600, is the most E2B's Hobby plan
            allows; Pro plans allow up to 86400.
        working_dir: Absolute directory commands start in and relative paths resolve against.
            E2B has no create-time working directory, so this is applied per command, including
            on an attached sandbox; `None` uses the sandbox's own default, discovered with
            `pwd -P` on first use.
        env: Environment variables every command gets, on a created or an attached sandbox;
            per-command `env` is layered on top. Nothing is read from the host environment.
        allow_internet_access: Whether a newly created sandbox may reach the internet.
    """

    def __init__(
        self,
        workspace: e2b.AsyncSandbox | None = None,
        *,
        ref: WorkspaceRef | None = None,
        template: str | None = None,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
    ) -> None:
        if ref is not None and ref.provider != 'e2b':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'e2b'")
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._ref = ref if workspace is None else WorkspaceRef(provider='e2b', id=workspace.sandbox_id)
        self._sandbox = workspace
        self._working_dir = absolute_path('working_dir', working_dir)
        # `working_dir()` must return a canonical absolute path: the configured one, or the
        # sandbox's default, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()
        self._template = template
        self._sandbox_timeout = sandbox_timeout
        self._env = dict(env) if env is not None else None
        self._allow_internet_access = allow_internet_access

    async def get_client(self) -> e2b.AsyncSandbox:
        """Return the typed `e2b.AsyncSandbox`, creating or attaching to it on first use.

        Every operation takes the handle from here, so none reaches an unacquired one. The lock
        serializes concurrent first uses -- two callers each creating a sandbox would leave the
        loser billed and unreferenced. A caller cancelled while E2B creates the sandbox still records
        it before the cancellation propagates, so `ref` names it and a retry reuses it.
        Attaching by `ref` to a sandbox that no longer exists raises
        `WorkspaceUnavailableError`; it does not create a replacement.
        """
        async with self._lock:
            if (sandbox := self._sandbox) is not None:
                return sandbox
            ref = self._ref
            sandbox = await self._attach(ref.id) if ref is not None else await self._create()
            self._sandbox = sandbox
            self._ref = WorkspaceRef(provider='e2b', id=sandbox.sandbox_id)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _sdk_errors(self, sandbox_id: str | None, context: str, path: str | None = None) -> AsyncGenerator[None]:
        """Raise E2B's exceptions as the protocol's typed failures; see `_translate`."""
        try:
            yield
        except Exception as error:
            translated = await self._translate(error, context, path, sandbox_id)
            if translated is error:
                raise
            raise translated from error

    async def _translate(self, error: Exception, context: str, path: str | None, sandbox_id: str | None) -> Exception:
        """Map one E2B exception onto the protocol's typed failures, or return it unchanged.

        Rejected credentials and a gone sandbox end the run. E2B types an unanswered envd request
        as `TimeoutException` (the SDK's command timeout is disabled); after a liveness probe, a
        gone sandbox is unavailable, anything else is transient. It is not a
        `WorkspaceTimeoutError`, which is reserved for a command's own `timeout=`. Other SDK
        errors mean the operation failed. Rate limits, a busy service, transport failures, and
        anything E2B does not type come back unchanged to propagate as transient, for durable
        engines to retry.
        """
        if isinstance(error, e2b.AuthenticationException):
            return WorkspaceUnavailableError(_AUTH_MESSAGE)
        if isinstance(error, e2b.SandboxNotFoundException) and sandbox_id is not None:
            return WorkspaceUnavailableError(_unavailable_message(sandbox_id))
        if isinstance(error, e2b.FileNotFoundException):
            return FileNotFoundError(f'No such file or directory in the E2B sandbox: {path!r}')
        if (
            path is not None
            and type(error) in (e2b.SandboxException, e2b.InvalidArgumentException)
            and (path_error := _path_error(error, path)) is not None
        ):
            return path_error
        if isinstance(error, e2b.TimeoutException):
            sandbox = self._sandbox
            if sandbox is not None and not await _is_running(sandbox):
                return WorkspaceUnavailableError(_unavailable_message(sandbox.sandbox_id))
            return error
        if isinstance(error, e2b.SandboxException) and not isinstance(error, e2b.RateLimitException):
            return WorkspaceError(f'{context}: {error}')
        return error

    async def read_bytes(self, path: str) -> bytes:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not read {path!r}', path):
            return bytes(await sandbox.files.read(path, 'bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not write {path!r}', path):
            await sandbox.files.write(path, data)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not stat {path!r}', path):
            return await _file_entry(sandbox, await sandbox.files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not list {path!r}', path):
            # `depth=1` is E2B's non-recursive listing, as the protocol asks.
            entries = await sandbox.files.list(path, depth=1)
            return [await _file_entry(sandbox, entry) for entry in entries]

    async def make_dir(self, path: str) -> None:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not create directory {path!r}', path):
            await sandbox.files.make_dir(path)

    async def remove(self, path: str) -> None:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not remove {path!r}', path):
            # envd removes with `os.RemoveAll`, which succeeds on a missing path; the protocol
            # reports that as `FileNotFoundError`.
            if not await sandbox.files.exists(path):
                raise e2b.FileNotFoundException(path)
            await sandbox.files.remove(path)

    async def exists(self, path: str) -> bool:
        sandbox = await self.get_client()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not check {path!r}', path):
            return await sandbox.files.exists(path)

    async def _create(self) -> e2b.AsyncSandbox:
        """Provision a fresh E2B sandbox.

        The call is shielded: E2B may create the sandbox before its response arrives, and a
        cancellation then would lose the only handle to a billed sandbox. `_CREATE_TIMEOUT`
        still bounds it, so cancellation waits at most that long. A request E2B refuses (an
        unknown template, a lifetime over the plan's limit) is `WorkspaceUnavailableError`:
        retrying it cannot succeed.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT, shield=True):
            async with self._sdk_errors(None, 'Could not start E2B sandbox'):
                try:
                    return await e2b.AsyncSandbox.create(
                        template=self._template,
                        timeout=self._sandbox_timeout,
                        envs=dict(self._env) if self._env is not None else None,
                        secure=True,
                        allow_internet_access=self._allow_internet_access,
                        # Pause at the end of the lifetime instead of killing, so the files survive.
                        lifecycle={'on_timeout': 'pause'},
                    )
                except e2b.SandboxException as error:
                    refused = error.status_code is not None and 400 <= error.status_code < 500
                    if not refused or isinstance(error, e2b.RateLimitException):
                        raise
                    raise WorkspaceUnavailableError(_refused_message('Could not start E2B sandbox', error)) from error
        # A transient failure like any unreachable service: it propagates for durable engines to
        # retry, with a message that says what did not answer.
        raise TimeoutError(
            f'E2B sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
            'the E2B control plane may be unreachable.'
        )

    async def _attach(self, id: str) -> e2b.AsyncSandbox:
        """Attach to an E2B sandbox that already exists, without taking over its lifecycle.

        E2B resumes a paused sandbox on connect, so attaching to one that was paused restarts
        it; a sandbox that is gone raises `WorkspaceUnavailableError` rather than resolving to
        a dead environment. Nothing is recreated in its place; a run that expected files there
        must be told they are gone, not handed an empty workspace. A lifetime over the plan's
        limit is refused like on create, as `WorkspaceUnavailableError`.
        """
        context = f'Could not connect to E2B sandbox {id!r}'
        async with self._sdk_errors(id, context):
            try:
                # Without `timeout`, a resumed sandbox gets E2B's 300 seconds; a running one keeps
                # the longer of its current and the given lifetime.
                return await e2b.AsyncSandbox.connect(id, timeout=self._sandbox_timeout)
            except e2b.SandboxException as error:
                if type(error) is not e2b.SandboxException or not _is_lifetime_refusal(error):
                    raise
                raise WorkspaceUnavailableError(_refused_message(context, error)) from error

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                sandbox = await self.get_client()
                raise WorkspaceError(
                    f'Could not determine the working directory of E2B sandbox {sandbox.sandbox_id}: '
                    f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
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
        """Run a command, killing it on timeout, cancellation, or a failed result read."""
        line = _command_line(command, shell)
        cwd = absolute_path('cwd', cwd)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        # Acquiring the sandbox has its own bound; the timeout is the command's alone.
        sandbox = await self.get_client()
        handle: e2b.AsyncCommandHandle | None = None
        result: e2b.CommandResult | None = None
        try:
            with anyio.move_on_after(timeout):
                handle = await sandbox.commands.run(
                    line,
                    background=True,
                    # C.UTF-8 needs no locale package on the default image; libc falls back to C
                    # on images without it. Explicit caller settings take precedence.
                    envs={'LC_ALL': 'C.UTF-8', **(self._env or {}), **(env or {})},
                    cwd=cwd if cwd is not None else self._working_dir,
                    timeout=_SDK_STREAM_UNBOUNDED,
                )
                result = await handle.wait()
            if result is None:
                assert timeout is not None
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds',
                    stdout=handle.stdout if handle is not None else '',
                    stderr=handle.stderr if handle is not None else '',
                )
            return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
        except e2b.CommandExitException as error:
            return CommandResult(exit_code=error.exit_code, stdout=error.stdout, stderr=error.stderr)
        except BaseException as error:
            if handle is not None:
                # Cleanup must not replace a timeout, cancellation, or SDK failure.
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(_INTERNAL_EXEC_TIMEOUT):
                        try:
                            await sandbox.commands.kill(handle.pid)
                        except Exception:
                            pass
            if isinstance(error, Exception):
                context = (
                    'Command could not run in the E2B sandbox'
                    if handle is None
                    else 'Could not read the command result (the command may still be running)'
                )
                translated = await self._translate(error, context, None, sandbox.sandbox_id)
                if translated is not error:
                    raise translated from error
            raise


async def _is_running(sandbox: e2b.AsyncSandbox) -> bool:
    """Ask E2B's health probe whether the sandbox runs; a probe that fails counts as running.

    Probing only after an error keeps the extra round trip off successful operations, and a
    failed probe leaves the original error to propagate rather than aborting the run on a guess.
    """
    try:
        return await sandbox.is_running()
    except Exception:
        return True

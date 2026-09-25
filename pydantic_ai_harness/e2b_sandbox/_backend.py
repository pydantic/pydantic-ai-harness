"""An E2B sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-09-09 against E2B 2.46.4 and 2.34.0: `connect` resumes
paused workspaces using its default 300-second lifetime; commands use `/bin/bash -l -c`, the SDK
command timeout does not kill the process, and command handles accumulate output. Re-check
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py and
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command.py
before changing acquisition or deadline behavior.

Exception types verified 2026-09-25 against E2B 2.51.0
(https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/exceptions.py): a missing key or
a 401 is `AuthenticationException`; a gone sandbox is `SandboxNotFoundException` from the control
plane and a `TimeoutException` from envd (its proxy's 502); a 429 is `RateLimitException`, a
`SandboxException` subclass; a 503 is `ServiceBusyException` (new in 2.48.0, hence the floor),
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
DEFAULT_SANDBOX_TIMEOUT = 300

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

# Bound the sandbox-create call so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120

# Bounds the internal `pwd` probe behind `working_dir()` and the best-effort kills.
_INTERNAL_EXEC_TIMEOUT = 10

# E2B's own command `timeout` bounds the event stream and leaves the command running, so it is
# switched off (0 is the SDK's "no limit") and the deadline is enforced client-side instead,
# with a kill at expiry. See `E2BSandboxBackend.run`.
_SDK_STREAM_UNBOUNDED = 0


def _command_line(command: WorkspaceCommand, shell: bool) -> str:
    """Turn a protocol command into the single string E2B executes.

    E2B has no argv form: `commands.run` hands its string to `/bin/bash -l -c`, so the argv is
    quoted with `shlex.join` first. The shell still parses the result, but the quoting makes each
    element exactly one word, which is the guarantee argv callers rely on. A `shell=True` string
    runs as `/bin/sh -c <string>`, so it is interpreted by `sh` as on every other workspace, while
    the login shell around it still sets up `PATH`.
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
    return FileEntry(
        name=entry.name, path=entry.path, is_dir=is_dir, size=size, is_symlink=entry.symlink_target is not None
    )


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
        template: E2B template name or id a newly created sandbox runs; E2B's default when `None`.
        sandbox_timeout: How long E2B keeps a newly created sandbox alive, in seconds.
        working_dir: Absolute directory commands start in and relative paths resolve against.
            E2B has no create-time working directory, so this is applied per command, including
            on an attached sandbox; `None` uses the sandbox's own default, discovered with
            `pwd -P` on first use.
        env: Environment variables every command gets, on a created or an attached sandbox;
            per-command `env` is layered on top. Nothing is read from the host environment.
        metadata: Metadata added to a newly created workspace.
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
        metadata: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
    ) -> None:
        if ref is not None and ref.provider != 'e2b':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'e2b'")
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._workspace = workspace
        self._ref = ref if workspace is None else WorkspaceRef(provider='e2b', id=workspace.sandbox_id)
        self._template = template
        self._sandbox_timeout = sandbox_timeout
        self._env = dict(env) if env is not None else None
        self._metadata = dict(metadata) if metadata is not None else None
        self._allow_internet_access = allow_internet_access
        self._canonical_working_dir: str | None = None
        self._working_dir = absolute_path('working_dir', working_dir)
        self._created_timeout: int | None = None
        self._lock = anyio.Lock()

    async def get_client(self) -> e2b.AsyncSandbox:
        """Return the typed `e2b.AsyncSandbox`, creating or attaching to it on first use.

        The only place `_workspace` is read, so nothing can reach an unacquired handle:
        it stays optional and every other method comes through here. The lock serializes
        concurrent first uses -- two callers each creating a sandbox would leave the loser
        billed and unreferenced. A caller cancelled while E2B creates the sandbox still records
        it before the cancellation propagates, so `ref` names it and a retry reuses it.
        Attaching by `ref` to a sandbox that no longer exists raises
        `WorkspaceUnavailableError`; it does not create a replacement.
        """
        async with self._lock:
            if (workspace := self._workspace) is not None:
                return workspace
            ref = self._ref
            workspace = await self._attach(ref.id) if ref is not None else await self._create()
            self._workspace = workspace
            self._ref = WorkspaceRef(provider='e2b', id=workspace.sandbox_id)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return workspace

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _sdk_errors(self, context: str, path: str | None = None) -> AsyncGenerator[None]:
        """Raise E2B's exceptions as the protocol's typed failures; see `_translate`."""
        try:
            yield
        except Exception as error:
            translated = await self._translate(error, context, path)
            if translated is error:
                raise
            raise translated from error

    async def _translate(self, error: Exception, context: str, path: str | None) -> Exception:
        """Map one E2B exception onto the protocol's typed failures, or return it unchanged.

        Rejected credentials and a gone sandbox end the run. A `TimeoutException` is ambiguous --
        E2B raises it both for a request the sandbox never answered and for one aborted because
        the sandbox died -- so an acquired sandbox is asked whether it still runs. Other SDK
        errors mean the operation failed. Rate limits, a busy service, transport failures, and
        anything E2B does not type come back unchanged to propagate as transient, for durable
        engines to retry.
        """
        if isinstance(error, e2b.AuthenticationException):
            return WorkspaceUnavailableError(_AUTH_MESSAGE)
        if isinstance(error, e2b.SandboxNotFoundException):
            return WorkspaceUnavailableError(self._gone_message())
        if isinstance(error, e2b.FileNotFoundException):
            return FileNotFoundError(f'No such file or directory in the E2B sandbox: {path!r}')
        if (
            path is not None
            and type(error) in (e2b.SandboxException, e2b.InvalidArgumentException)
            and (path_error := _path_error(error, path)) is not None
        ):
            return path_error
        if isinstance(error, e2b.TimeoutException):
            if self._workspace is not None and not await _is_running(self._workspace):
                return WorkspaceUnavailableError(self._gone_message())
            return error
        if isinstance(error, e2b.SandboxException) and not isinstance(error, e2b.RateLimitException):
            return WorkspaceError(f'{context}: {error}')
        return error

    def _gone_message(self) -> str:
        assert self._ref is not None
        if self._created_timeout is None:
            return (
                f'The E2B sandbox {self._ref.id!r} is no longer running '
                '(it does not exist, was killed, or expired at its configured lifetime). '
                'Attach to a live sandbox, or create a new one.'
            )
        return (
            f'The E2B sandbox {self._ref.id!r} is no longer running (it may have reached its '
            f'sandbox_timeout of {self._created_timeout}s, or been killed). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def read_bytes(self, path: str) -> bytes:
        async with self._sdk_errors(f'Could not read {path!r}', path):
            return bytes(await (await self.get_client()).files.read(path, 'bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._sdk_errors(f'Could not write {path!r}', path):
            await (await self.get_client()).files.write(path, data)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        async with self._sdk_errors(f'Could not stat {path!r}', path):
            sandbox = await self.get_client()
            return await _file_entry(sandbox, await sandbox.files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._sdk_errors(f'Could not list {path!r}', path):
            sandbox = await self.get_client()
            entries = await sandbox.files.list(path, depth=1)
            return [await _file_entry(sandbox, entry) for entry in entries]

    async def make_dir(self, path: str) -> None:
        async with self._sdk_errors(f'Could not create directory {path!r}', path):
            await (await self.get_client()).files.make_dir(path)

    async def remove(self, path: str) -> None:
        async with self._sdk_errors(f'Could not remove {path!r}', path):
            sandbox = await self.get_client()
            # envd removes with `os.RemoveAll`, which succeeds on a missing path; the protocol
            # reports that as `FileNotFoundError`.
            if not await sandbox.files.exists(path):
                raise e2b.FileNotFoundException(path)
            await sandbox.files.remove(path)

    async def exists(self, path: str) -> bool:
        async with self._sdk_errors(f'Could not check {path!r}', path):
            return await (await self.get_client()).files.exists(path)

    async def _create(self) -> e2b.AsyncSandbox:
        """Provision a fresh E2B sandbox.

        The call is shielded: E2B may create the sandbox before its response arrives, and a
        cancellation then would lose the only handle to a billed sandbox. `_CREATE_TIMEOUT`
        still bounds it, so cancellation waits at most that long.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT, shield=True):
            async with self._sdk_errors('Could not start E2B sandbox'):
                sandbox = await e2b.AsyncSandbox.create(
                    template=self._template,
                    timeout=self._sandbox_timeout,
                    metadata=self._metadata,
                    envs=dict(self._env) if self._env is not None else None,
                    secure=True,
                    allow_internet_access=self._allow_internet_access,
                )
            self._created_timeout = self._sandbox_timeout
            return sandbox
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
        must be told they are gone, not handed an empty workspace.
        """
        async with self._sdk_errors(f'Could not connect to E2B sandbox {id!r}'):
            # Let E2B apply its default lifetime when connecting or resuming.
            return await e2b.AsyncSandbox.connect(id)

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        if self._canonical_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                assert self._ref is not None
                raise WorkspaceError(
                    f'Could not determine the working directory of E2B sandbox {self._ref.id}: '
                    f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
                )
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
                    envs={**(self._env or {}), **(env or {})} or None,
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
                    timeout=timeout,
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
                translated = await self._translate(error, context, None)
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

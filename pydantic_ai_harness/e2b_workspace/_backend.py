"""An E2B sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-09-09 against E2B 2.46.4 and the 2.34.0 floor: `connect` resumes
paused workspaces using its default 300-second lifetime; commands use `/bin/bash -l -c`, the SDK
command timeout does not kill the process, and command handles accumulate output. Re-check
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py and
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command.py
before changing acquisition or deadline behavior.
"""

from __future__ import annotations

import math
import posixpath
import shlex
from collections.abc import AsyncGenerator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager
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
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('E2BWorkspaceBackend',)
DEFAULT_SANDBOX_TIMEOUT = 300

try:
    import e2b
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[e2b]` to use E2BWorkspace.') from error

_AUTH_MESSAGE = 'E2B rejected the credentials. Set a valid E2B_API_KEY in the environment.'

# Bound the sandbox-create call so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120

# Bounds the internal `pwd` probe behind `working_dir()` and the best-effort kills.
_INTERNAL_EXEC_TIMEOUT = 10

# E2B's own command `timeout` bounds the event stream and leaves the command running, so it is
# switched off (0 is the SDK's "no limit") and the deadline is enforced client-side instead,
# with a kill at expiry. See `E2BWorkspaceBackend.run`.
_SDK_STREAM_UNBOUNDED = 0


def _command_line(command: WorkspaceCommand, shell: bool) -> str:
    """Turn a protocol command into the single string E2B executes.

    E2B has no argv form: `commands.run` hands its string to `/bin/bash -l -c`, so an argv
    sequence is quoted with `shlex.join` first. The shell still parses the result, but the
    quoting makes each element exactly one word, which is the guarantee argv callers rely on.
    """
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        return command
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    if not command:
        raise TypeError('a command needs at least the program to run; the argv sequence is empty')
    return shlex.join(command)


def _file_entry(entry: e2b.EntryInfo) -> FileEntry:
    is_dir = entry.type is e2b.FileType.DIR
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    return FileEntry(name=entry.name, path=entry.path, is_dir=is_dir, size=None if is_dir else entry.size)


class E2BWorkspaceBackend(WorkspaceBackend, SupportsFilesystem):
    """An [E2B](https://e2b.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside an E2B microVM, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a workspace, and the
    native `e2b.AsyncSandbox` is available by awaiting `workspace`.

    Commands run as one-shot operations, with complete output returned after they finish.

    Every command runs through `/bin/bash -l -c`, so an argv sequence is quoted into a single
    shell word string first and login startup files run before the command does. E2B's own
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
        workdir: Directory commands run in and relative paths resolve against. E2B has no
            create-time working directory, so this is applied per command; `None` uses the
            sandbox's own default, discovered with `pwd -P` on first use.
        env: Environment variables set for the whole sandbox at creation.
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
        workdir: str | None = None,
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
        self._working_dir = absolute_path('workdir', workdir)
        self._created_timeout: int | None = None
        self._lock = anyio.Lock()

    @property
    def workspace(self) -> Awaitable[e2b.AsyncSandbox]:
        return self._create_or_attach()

    async def _create_or_attach(self) -> e2b.AsyncSandbox:
        """Acquire the native E2B sandbox on first use, once, and record its identity.

        The only place `_workspace` is read, so nothing can reach an unacquired handle:
        it stays optional and every other method comes through here. The lock serializes
        concurrent first uses -- two callers each creating a sandbox would leave the loser
        billed and unreferenced.
        """
        async with self._lock:
            if (workspace := self._workspace) is not None:
                return workspace
            ref = self._ref
            workspace = await self._attach(ref.id) if ref is not None else await self._create()
            self._workspace = workspace
            self._ref = WorkspaceRef(provider='e2b', id=workspace.sandbox_id)
            return workspace

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _translated_filesystem_error(self, path: str) -> AsyncGenerator[None]:
        """Map E2B's filesystem exceptions onto the ones the protocol promises."""
        try:
            yield
        except e2b.FileNotFoundException as e:
            raise FileNotFoundError(f'No such file or directory in the E2B sandbox: {path!r}') from e
        except WorkspaceError:
            raise
        except Exception as e:
            raise await self._operation_error(e, f'Could not access {path!r} in the sandbox') from e

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated_filesystem_error(path):
            return bytes(await (await self.workspace).files.read(path, 'bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.workspace).files.write(path, data)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        async with self._translated_filesystem_error(path):
            return _file_entry(await (await self.workspace).files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._translated_filesystem_error(path):
            entries = await (await self.workspace).files.list(path, depth=1)
        return [_file_entry(entry) for entry in entries]

    async def make_dir(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.workspace).files.make_dir(path)

    async def remove(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.workspace).files.remove(path)

    async def exists(self, path: str) -> bool:
        async with self._translated_filesystem_error(path):
            return await (await self.workspace).files.exists(path)

    async def _create(self) -> e2b.AsyncSandbox:
        """Provision a fresh E2B sandbox."""
        try:
            with anyio.fail_after(_CREATE_TIMEOUT):
                sandbox = await e2b.AsyncSandbox.create(
                    template=self._template,
                    timeout=self._sandbox_timeout,
                    metadata=self._metadata,
                    envs=dict(self._env) if self._env is not None else None,
                    secure=True,
                    allow_internet_access=self._allow_internet_access,
                )
        except TimeoutError as error:
            raise WorkspaceError(
                f'E2B sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the E2B control plane may be unreachable.'
            ) from error
        except e2b.AuthenticationException as e:
            raise WorkspaceUnavailableError(_AUTH_MESSAGE) from e
        except Exception as e:
            raise WorkspaceError(f'Could not start E2B sandbox: {type(e).__name__}: {e}') from e
        self._created_timeout = self._sandbox_timeout
        return sandbox

    async def _attach(self, id: str) -> e2b.AsyncSandbox:
        """Attach to an E2B sandbox that already exists, without taking over its lifecycle.

        E2B resumes a paused sandbox on connect, so attaching to one that was paused restarts
        it; a sandbox that is gone raises `WorkspaceUnavailableError` rather than resolving to
        a dead environment. Nothing is recreated in its place; a run that expected files there
        must be told they are gone, not handed an empty workspace.
        """
        try:
            # Let E2B apply its default lifetime when connecting or resuming.
            return await e2b.AsyncSandbox.connect(id)
        except e2b.AuthenticationException as e:
            raise WorkspaceUnavailableError(_AUTH_MESSAGE) from e
        except e2b.SandboxNotFoundException as e:
            raise WorkspaceUnavailableError(_attached_gone_message(repr(id))) from e
        except Exception as e:
            raise WorkspaceError(f'Could not connect to E2B sandbox {id!r}: {type(e).__name__}: {e}') from e

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
        sandbox: e2b.AsyncSandbox | None = None
        handle: e2b.AsyncCommandHandle | None = None
        result: e2b.CommandResult | None = None
        try:
            with anyio.move_on_after(timeout):
                sandbox = await self.workspace
                handle = await sandbox.commands.run(
                    line,
                    background=True,
                    envs=dict(env) if env is not None else None,
                    cwd=cwd if cwd is not None else self._working_dir,
                    timeout=_SDK_STREAM_UNBOUNDED,
                )
                result = await handle.wait()
            if result is None:
                assert timeout is not None
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds.',
                    stdout=handle.stdout if handle is not None else '',
                    stderr=handle.stderr if handle is not None else '',
                    timeout=timeout,
                )
            return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
        except e2b.CommandExitException as error:
            return CommandResult(exit_code=error.exit_code, stdout=error.stdout, stderr=error.stderr)
        except BaseException as error:
            if handle is not None and sandbox is not None:
                # Cleanup must not replace a timeout, cancellation, or SDK failure.
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(_INTERNAL_EXEC_TIMEOUT):
                        try:
                            await sandbox.commands.kill(handle.pid)
                        except Exception:
                            pass
            if isinstance(error, WorkspaceError):
                raise
            if isinstance(error, Exception):
                context = (
                    'Command could not run in the sandbox'
                    if handle is None
                    else 'Could not read the command result (the command may still be running)'
                )
                raise await self._operation_error(error, context) from error
            raise

    def _unavailable_message(self) -> str:
        assert self._ref is not None
        if self._created_timeout is None:
            return _attached_gone_message(self._ref.id)
        return (
            'The E2B sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._created_timeout}s, or been killed). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def _operation_error(self, e: Exception, context: str) -> WorkspaceError:
        """Map an exception raised while using the sandbox.

        Rejected credentials and a sandbox E2B cannot find are terminal. A `TimeoutException`
        is ambiguous -- E2B raises it both for a request the sandbox never answered and for one
        aborted because the sandbox died -- so it is classified by asking whether the sandbox is
        still running. Everything else stays a recoverable `WorkspaceError`.
        """
        if isinstance(e, e2b.AuthenticationException):
            return WorkspaceUnavailableError(_AUTH_MESSAGE)
        if isinstance(e, e2b.SandboxNotFoundException):
            return WorkspaceUnavailableError(self._unavailable_message())
        if isinstance(e, e2b.TimeoutException):
            return await self._probe_ambiguous(e, context)
        if isinstance(e, e2b.SandboxException):
            return WorkspaceError(f'{context}: {e}')
        return WorkspaceError(f'{context}: {type(e).__name__}: {e}')

    async def _probe_ambiguous(self, e: Exception, context: str) -> WorkspaceError:
        """Classify an E2B error that may mask sandbox death by probing the sandbox.

        E2B maps an unanswered envd request to `TimeoutException` whether the sandbox is alive
        and slow or gone; its health probe recovers the distinction. Probing only after an
        error keeps the extra round trip off successful operations.
        """
        try:
            sandbox = await self.workspace
            running = await sandbox.is_running()
        except Exception:
            # The classifying probe can itself fail, including with a raw transport error; fall
            # back to the original error rather than letting the probe abort the run.
            return WorkspaceError(f'{context}: {e}')
        if not running:
            return WorkspaceUnavailableError(self._unavailable_message())
        return WorkspaceError(f'{context}: {e}')


def _attached_gone_message(described: str) -> str:
    return (
        f'The E2B sandbox {described} is no longer running '
        '(it does not exist, was killed, or expired at its configured lifetime). '
        'Attach to a live sandbox, or create a new one.'
    )

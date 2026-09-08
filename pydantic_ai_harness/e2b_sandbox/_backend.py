"""An E2B sandbox behind Pydantic AI's [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol.

This is the mechanism layer: every E2B-specific operation (create, connect, command execution,
file access, working-directory discovery, teardown) lives here, behind the protocol the rest of
Pydantic AI already speaks. The capability in `_capability.py` owns the lifecycle; tools and
other capabilities consume the resulting `ctx.sandbox`.

External assumptions last verified 2026-08-31 against E2B Python SDK 2.34.0 (the package floor):

* `AsyncSandbox.create` / `connect` / `kill` provide the owned and attached lifecycle, and
  `connect` resumes a paused sandbox and substitutes its 300-second default when `timeout=None`:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py
* `AsyncSandbox.list` accepts `query`, `limit`, and `next_token` but not `order`, and each
  `SandboxInfo` exposes `started_at`:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py
* `commands.run` starts every command as `/bin/bash -l -c <string>`, its `timeout` bounds the
  event stream rather than killing the command, and `commands.kill(pid)` sends SIGKILL:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command.py
* a command handle accumulates decoded output as it arrives and `wait()` raises
  `CommandExitException` on a non-zero exit:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command_handle.py
* the filesystem API raises `FileNotFoundException` for a missing path:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/filesystem/filesystem.py

Re-check these sources before changing lifecycle, command, or filesystem assumptions.
"""

from __future__ import annotations

import functools
import math
import posixpath
import shlex
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
from anyio.lowlevel import checkpoint
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

from pydantic_ai_harness._sandbox_provider import absolute_path

if TYPE_CHECKING:
    from pydantic_ai.sandboxes import SandboxCommand

__all__ = (
    'E2BSandboxAuthError',
    'E2BSandboxBackend',
    'E2BSandboxError',
    'E2BSandboxUnavailableError',
)

# Defaults shared by `E2BSandboxBackend.create` and the `E2BSandbox` capability (which imports
# them), so the two cannot drift: a setting is "left at its default" iff it equals the constant
# here.
DEFAULT_SANDBOX_TIMEOUT = 300

try:
    import e2b
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[e2b]` to use E2BSandbox.') from error

_AUTH_MESSAGE = 'E2B rejected the credentials. Set a valid E2B_API_KEY in the environment.'

# Bound the sandbox-create call so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120

# Teardown runs shielded from cancellation, so an unreachable E2B control plane could otherwise
# hang the caller forever. Bound the kill so a stalled request gives up rather than wedging the
# process; an owned sandbox is still reaped server-side by its own `sandbox_timeout`.
_TEARDOWN_TIMEOUT = 30

# Bounds the internal `pwd` probe behind `working_dir()` and the best-effort kills.
_INTERNAL_EXEC_TIMEOUT = 10

# E2B's own command `timeout` bounds the event stream and leaves the command running, so it is
# switched off (0 is the SDK's "no limit") and the deadline is enforced client-side instead,
# with a kill at expiry. See `E2BSandboxBackend.run`.
_SDK_STREAM_UNBOUNDED = 0


async def _kill_sandbox(sandbox_id: str, kill: Callable[[], Awaitable[object]]) -> None:
    """Run an E2B kill to completion without letting cleanup replace cancellation."""
    error: Exception | None = None
    with anyio.CancelScope(shield=True):
        try:
            with anyio.fail_after(_TEARDOWN_TIMEOUT):
                await kill()
        except e2b.SandboxNotFoundException:
            return
        except Exception as exc:
            error = exc
    if error is None:
        return
    await checkpoint()
    if isinstance(error, e2b.AuthenticationException):
        raise E2BSandboxAuthError(_AUTH_MESSAGE) from error
    if isinstance(error, TimeoutError):
        raise E2BSandboxError(
            f'Timed out after {_TEARDOWN_TIMEOUT}s while trying to kill E2B sandbox {sandbox_id!r}.'
        ) from error
    raise E2BSandboxError(f'Could not kill E2B sandbox {sandbox_id!r}: {error}') from error


class E2BSandboxError(SandboxError):
    """An E2B provider operation failed."""


class E2BSandboxUnavailableError(E2BSandboxError, SandboxUnavailableError):
    """The sandbox no longer exists: killed, or expired at its `sandbox_timeout`.

    Every later command against it would fail the same way, so it is terminal. For an owned
    sandbox this is what a run outliving the sandbox lifetime looks like; raise
    `sandbox_timeout` (or shorten the work) if runs legitimately need longer.
    """


class E2BSandboxAuthError(E2BSandboxError, SandboxUnavailableError):
    """E2B rejected the credentials, so no sandbox operation can succeed.

    Fixing this is an operator action (configure `E2B_API_KEY`), not something a retry or a
    new run can do, which is why it is terminal.
    """


def _command_line(command: SandboxCommand, shell: bool) -> str:
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


class E2BSandboxBackend(SandboxBackend, SupportsFilesystem):
    """An [E2B](https://e2b.dev) sandbox as a Pydantic AI [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend].

    Commands and file operations run inside an E2B microVM, so the host is never exposed.

    Building one does no I/O. It holds settings plus, optionally, the identity of a sandbox that
    already exists; the first operation creates or attaches, once, and everything after that
    reuses the same environment. Reach the live `e2b.AsyncSandbox` through
    [`sandbox`][pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend.sandbox], by awaiting `sandbox`.

    Nothing here kills a sandbox. E2B reaps one at the `sandbox_timeout` it was created with;
    call [`close`][pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend.close] with
    `terminate=True` to end it sooner.

    Commands run as one-shot operations, with complete output returned after they finish.

    Every command runs through `/bin/bash -l -c`, so an argv sequence is quoted into a single
    shell word string first and login startup files run before the command does. E2B's own
    command `timeout` abandons the output stream and leaves the command running, so the
    deadline is enforced client-side instead and the command is killed with SIGKILL when it
    expires or when the caller is cancelled, if E2B has returned the process ID. Cancellation
    during startup can leave the command running. That kill signals the command's own process; a
    process the command started in the background outlives it until the sandbox is torn down.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `Sandbox.wrap` call.

    Args:
        sandbox: A live `e2b.AsyncSandbox` you already have. Whoever created it owns killing it.
        ref: Identity of an existing sandbox to attach to on first use.
        identity: E2B metadata that marks one logical workspace. On first use the oldest running
            sandbox carrying it is reused, and one is created only if there is none. This is what
            lets several runs share an environment, and what makes a durable retry attach rather
            than provision a second sandbox. Ignored when `sandbox` or `ref` is given.
        template: E2B template name or id a newly created sandbox runs; E2B's default when `None`.
        sandbox_timeout: How long E2B keeps a newly created sandbox alive, in seconds.
        working_dir: Directory commands run in and relative paths resolve against. E2B has no
            create-time working directory, so this is applied per command; `None` uses the
            sandbox's own default, discovered with `pwd -P` on first use.
        env: Environment variables set for the whole sandbox at creation.
        metadata: E2B metadata recorded on a newly created sandbox.
        allow_internet_access: Whether a newly created sandbox may reach the internet.
    """

    def __init__(
        self,
        sandbox: e2b.AsyncSandbox | None = None,
        *,
        ref: SandboxRef | None = None,
        identity: Mapping[str, str] | None = None,
        template: str | None = None,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
    ) -> None:
        self._live = sandbox
        self._ref = ref if sandbox is None else SandboxRef(sandbox_id=sandbox.sandbox_id)
        self._identity = dict(identity) if identity is not None else None
        self._template = template
        self._sandbox_timeout = sandbox_timeout
        self._env = dict(env) if env is not None else None
        self._metadata = dict(metadata) if metadata is not None else None
        self._allow_internet_access = allow_internet_access
        self._canonical_working_dir: str | None = None
        self._working_dir = absolute_path('working_dir', working_dir)
        # Set once this backend creates the sandbox, so an expiry message can name the lifetime
        # that ran out rather than one this process only configured.
        self._created_timeout: int | None = None
        self._lock = anyio.Lock()

    @property
    def sandbox(self) -> Awaitable[e2b.AsyncSandbox]:
        """The native SDK sandbox, created or attached when awaited.

        Await this property before using SDK methods, including after the sandbox has been resolved.
        """
        return self._resolve()

    async def _resolve(self) -> e2b.AsyncSandbox:
        async with self._lock:
            if self._live is None:
                if self._ref is not None:
                    self._live = await self._attach(self._ref.sandbox_id)
                elif self._identity is not None:
                    self._live = await self._create_or_attach_by_identity(self._identity)
                else:
                    self._live = await self._create()
                self._ref = SandboxRef(sandbox_id=self._live.sandbox_id)
        return self._live

    @property
    def ref(self) -> SandboxRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _translated_filesystem_error(self, path: str) -> AsyncGenerator[None]:
        """Map E2B's filesystem exceptions onto the ones the protocol promises."""
        try:
            yield
        except e2b.FileNotFoundException as e:
            raise FileNotFoundError(f'No such file or directory in the E2B sandbox: {path!r}') from e
        except SandboxError:
            raise
        except Exception as e:
            raise await self.operation_error(e, f'Could not access {path!r} in the sandbox') from e

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated_filesystem_error(path):
            return bytes(await (await self.sandbox).files.read(path, 'bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).files.write(path, data)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        async with self._translated_filesystem_error(path):
            return _file_entry(await (await self.sandbox).files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._translated_filesystem_error(path):
            entries = await (await self.sandbox).files.list(path, depth=1)
        return [_file_entry(entry) for entry in entries]

    async def make_dir(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).files.make_dir(path)

    async def remove(self, path: str) -> None:
        async with self._translated_filesystem_error(path):
            await (await self.sandbox).files.remove(path)

    async def exists(self, path: str) -> bool:
        async with self._translated_filesystem_error(path):
            return await (await self.sandbox).files.exists(path)

    async def _create(self) -> e2b.AsyncSandbox:
        """Provision a fresh E2B sandbox."""
        try:
            # Cancellation can orphan a sandbox until `sandbox_timeout` reaps it. Metadata
            # search makes a durable retry reconnect to that sandbox instead of creating another.
            with anyio.fail_after(_CREATE_TIMEOUT):
                sandbox = await e2b.AsyncSandbox.create(
                    template=self._template,
                    timeout=self._sandbox_timeout,
                    metadata={**(self._metadata or {}), **(self._identity or {})} or None,
                    envs=dict(self._env) if self._env is not None else None,
                    secure=True,
                    allow_internet_access=self._allow_internet_access,
                )
        except TimeoutError as error:
            raise E2BSandboxError(
                f'E2B sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the E2B control plane may be unreachable.'
            ) from error
        except e2b.AuthenticationException as e:
            raise E2BSandboxAuthError(_AUTH_MESSAGE) from e
        except Exception as e:
            raise E2BSandboxError(f'Could not start E2B sandbox: {type(e).__name__}: {e}') from e
        self._created_timeout = self._sandbox_timeout
        return sandbox

    async def _create_or_attach_by_identity(self, identity: Mapping[str, str]) -> e2b.AsyncSandbox:
        """Attach to the oldest sandbox carrying `identity`, or create one once.

        E2B does not enforce metadata uniqueness, so a post-create query keeps the oldest match
        and kills a racing duplicate.
        """
        existing_id = await self._find_id(identity)
        if existing_id is not None:
            return await self._attach(existing_id)
        created = await self._create()
        canonical_id = await self._find_id(identity)
        if canonical_id is None or canonical_id == created.sandbox_id:
            return created
        await _kill_sandbox(created.sandbox_id, created.kill)
        self._created_timeout = None
        return await self._attach(canonical_id)

    @staticmethod
    async def _find_id(metadata: Mapping[str, str]) -> str | None:
        """Return the oldest running or paused sandbox ID matching metadata."""
        try:
            paginator = e2b.AsyncSandbox.list(query=e2b.SandboxQuery(metadata=dict(metadata)))
            oldest: e2b.SandboxInfo | None = None
            while paginator.has_next:
                for match in await paginator.next_items():
                    if oldest is None or match.started_at < oldest.started_at:
                        oldest = match
        except e2b.AuthenticationException as error:
            raise E2BSandboxAuthError(_AUTH_MESSAGE) from error
        except Exception as error:
            raise E2BSandboxError(f'Could not list E2B sandboxes: {type(error).__name__}: {error}') from error
        return oldest.sandbox_id if oldest is not None else None

    async def _attach(self, sandbox_id: str) -> e2b.AsyncSandbox:
        """Attach to an E2B sandbox that already exists, without taking over its lifecycle.

        E2B resumes a paused sandbox on connect, so attaching to one that was paused restarts
        it; a sandbox that is gone raises `E2BSandboxUnavailableError` rather than resolving to
        a dead environment. Nothing is recreated in its place — a run that expected files there
        must be told they are gone, not handed an empty workspace.
        """
        try:
            # Let E2B apply its default lifetime when connecting or resuming.
            return await e2b.AsyncSandbox.connect(sandbox_id)
        except e2b.AuthenticationException as e:
            raise E2BSandboxAuthError(_AUTH_MESSAGE) from e
        except e2b.SandboxNotFoundException as e:
            raise E2BSandboxUnavailableError(_attached_gone_message(repr(sandbox_id))) from e
        except Exception as e:
            raise E2BSandboxError(f'Could not connect to E2B sandbox {sandbox_id!r}: {type(e).__name__}: {e}') from e

    def _describe(self) -> str:
        """How to name this sandbox in an error.

        Every caller runs after `sandbox`, which sets `ref` alongside the live handle, so the
        other two spellings are only reachable if that ever stops being true. `lax no cover`
        for the same reason: they are a fallback, not a path tests should have to reach.
        """
        if self._ref is not None:
            return repr(self._ref.sandbox_id)
        return (
            f'for {self._identity!r}' if self._identity is not None else 'that was never started'
        )  # pragma: lax no cover

    async def close(self, *, terminate: bool) -> None:
        """Release this handle, killing the sandbox with it when we own its lifetime.

        Runs shielded from cancellation, since a run that is being torn down must still get its
        kill request out, bounded so a stalled control plane cannot wedge the caller. E2B has
        no client-side connection to release, so releasing an attached sandbox does nothing.
        """
        if not terminate:
            return
        sandbox = self._live
        if sandbox is None or self._ref is None:
            # Never used, so there is nothing to kill. Resolving one here just to close it
            # would create the very sandbox being released.
            return
        await _kill_sandbox(self._ref.sandbox_id, sandbox.kill)

    @staticmethod
    async def kill_by_id(sandbox_id: str) -> None:
        """Kill a sandbox by ID without reconnecting to it first.

        Applications can use this to end a sandbox explicitly; avoiding reconnect also avoids resuming a paused sandbox.
        """
        await _kill_sandbox(sandbox_id, functools.partial(e2b.AsyncSandbox.kill, sandbox_id))

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        if self._canonical_working_dir is None:
            await self.sandbox
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.strip()
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise E2BSandboxError(
                    f'Could not determine the working directory of E2B sandbox {self._describe()}: '
                    f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
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
                sandbox = await self.sandbox
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
                raise SandboxTimeoutError(
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
            if isinstance(error, SandboxError):
                raise
            if isinstance(error, Exception):
                context = (
                    'Command could not run in the sandbox'
                    if handle is None
                    else 'Could not read the command result (the command may still be running)'
                )
                raise await self.operation_error(error, context) from error
            raise

    def _unavailable_message(self) -> str:
        if self._created_timeout is None:
            return _attached_gone_message(self._describe())
        return (
            'The E2B sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._created_timeout}s, or been killed). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def operation_error(self, e: Exception, context: str) -> E2BSandboxError:
        """Map an exception raised while using the sandbox.

        Rejected credentials and a sandbox E2B cannot find are terminal. A `TimeoutException`
        is ambiguous -- E2B raises it both for a request the sandbox never answered and for one
        aborted because the sandbox died -- so it is classified by asking whether the sandbox is
        still running. Everything else stays a recoverable `E2BSandboxError`.
        """
        if isinstance(e, e2b.AuthenticationException):
            return E2BSandboxAuthError(_AUTH_MESSAGE)
        if isinstance(e, e2b.SandboxNotFoundException):
            return E2BSandboxUnavailableError(self._unavailable_message())
        if isinstance(e, e2b.TimeoutException):
            return await self._probe_ambiguous(e, context)
        if isinstance(e, e2b.SandboxException):
            return E2BSandboxError(f'{context}: {e}')
        return E2BSandboxError(f'{context}: {type(e).__name__}: {e}')

    async def _probe_ambiguous(self, e: Exception, context: str) -> E2BSandboxError:
        """Classify an E2B error that may mask sandbox death by probing the sandbox.

        E2B maps an unanswered envd request to `TimeoutException` whether the sandbox is alive
        and slow or gone; its health probe recovers the distinction. Probing only after an
        error keeps the extra round trip off successful operations.
        """
        try:
            sandbox = await self.sandbox
            running = await sandbox.is_running()
        except Exception:
            # The classifying probe can itself fail, including with a raw transport error; fall
            # back to the original error rather than letting the probe abort the run.
            return E2BSandboxError(f'{context}: {e}')
        if not running:
            return E2BSandboxUnavailableError(self._unavailable_message())
        return E2BSandboxError(f'{context}: {e}')


def _attached_gone_message(described: str) -> str:
    return (
        f'The E2B sandbox {described} is no longer running '
        '(it does not exist, was killed, or expired at its configured lifetime). '
        'Attach to a live sandbox, or create a new one.'
    )

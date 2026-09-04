"""A controllable fake `e2b` SDK for E2BSandbox tests.

Tests never reach real E2B: a fake `e2b` module is injected into `sys.modules` (via the
`fake_e2b` fixture in `conftest.py`), so the lazy `import e2b` inside the backend returns it.
The fake records calls and lets each test decide what a command returns.

Fidelity to the real SDK is the point. The exception classes, `FileType`, `CommandResult`,
`CommandExitException`, and `WriteInfo` are the real ones, imported from the installed
package, so the backend's `isinstance` checks and its unwrapping of a non-zero exit are
exercised against the types production raises. Signatures are closed, every await suspends,
a command handle accumulates output before `wait()` returns (as the SDK's event pump does),
a non-zero exit raises rather than returns, and a missing path raises E2B's own filesystem
exception. Flattering the code under test here would hide production failures.
"""

from __future__ import annotations

import posixpath
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal, Protocol

import anyio
import anyio.lowlevel
from e2b import CommandExitException, CommandResult, FileType, WriteInfo
from e2b.exceptions import (
    AuthenticationException,
    FileNotFoundException,
    SandboxException,
    SandboxNotFoundException,
    TimeoutException,
)

__all__ = (
    'FakeCommandCall',
    'FakeCreateCall',
    'FakeE2B',
    'FakeEntryInfo',
    'FakeSandbox',
)

# A responder maps (command line, deadline) to (stdout, stderr, exit_code).
Responder = Callable[[str, 'float | None'], 'tuple[str, str, int]']


def _echo_responder(command: str, timeout: float | None) -> tuple[str, str, int]:
    return f'{command}\n', '', 0


@dataclass(frozen=True)
class FakeCreateCall:
    template: str | None
    timeout: int | None
    metadata: dict[str, str] | None
    envs: dict[str, str] | None
    secure: bool
    allow_internet_access: bool


@dataclass(frozen=True)
class FakeCommandCall:
    command: str
    background: bool
    cwd: str | None
    envs: dict[str, str] | None
    timeout: float | None


@dataclass(frozen=True)
class FakeEntryInfo:
    """The `e2b.EntryInfo` members the backend reads.

    A subset rather than the real dataclass, which also carries mode, permissions, owner,
    group, and timestamps that no code here looks at. The `if TYPE_CHECKING` block below pins
    this subset against the real type so a drift in E2B's entry shape fails the type check.
    """

    name: str
    path: str
    type: FileType | None
    size: int


class FakeCommandHandle:
    """Mirrors `e2b.AsyncCommandHandle` for the members the backend uses.

    Output is accumulated at construction rather than at `wait()`, the way the SDK's event
    pump fills its chunk lists as data arrives: that is what makes `stdout` readable after a
    deadline kill cancels the wait.
    """

    def __init__(self, control: FakeE2B, *, pid: int, stdout: str, stderr: str, exit_code: int) -> None:
        self._control = control
        self._pid = pid
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def stdout(self) -> str:
        return self._stdout

    @property
    def stderr(self) -> str:
        return self._stderr

    async def wait(self) -> CommandResult:
        # A real wait suspends; yield so a test can cancel it and so concurrent tool calls
        # actually interleave.
        await anyio.lowlevel.checkpoint()
        if self._control.wait_error is not None:
            raise self._control.wait_error
        if self._control.command_hangs:
            await anyio.sleep_forever()
        if self._exit_code != 0:
            # The real SDK raises on a non-zero exit instead of returning a result.
            raise CommandExitException(
                stderr=self._stderr,
                stdout=self._stdout,
                exit_code=self._exit_code,
                error=f'exit status {self._exit_code}',
            )
        return CommandResult(stderr=self._stderr, stdout=self._stdout, exit_code=self._exit_code, error=None)


class FakeCommands:
    """Mirrors `sandbox.commands`: command execution and per-command kill."""

    def __init__(self, sandbox: FakeSandbox, control: FakeE2B) -> None:
        self._sandbox = sandbox
        self._control = control
        self.calls: list[FakeCommandCall] = []
        self.handles: list[FakeCommandHandle] = []
        self.killed_pids: list[int] = []

    async def run(
        self,
        cmd: str,
        background: bool | None = None,
        envs: dict[str, str] | None = None,
        user: str | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandHandle | CommandResult:
        # Closed signature on purpose: the real `run` rejects unknown kwargs, so the fake must
        # too, or a bad kwarg in the backend would only fail in production.
        del user
        await anyio.lowlevel.checkpoint()
        self.calls.append(FakeCommandCall(cmd, background is True, cwd, envs, timeout))
        if self._control.run_error is not None:
            raise self._control.run_error
        stdout, stderr, exit_code = self._control.responder(cmd, timeout)
        handle = FakeCommandHandle(
            self._control,
            pid=self._control.next_pid + len(self.handles),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )
        self.handles.append(handle)
        if background is True:
            return handle
        return await handle.wait()  # pragma: no cover - the backend always starts in background

    async def kill(self, pid: int, request_timeout: float | None = None) -> bool:
        del request_timeout
        await anyio.lowlevel.checkpoint()
        self.killed_pids.append(pid)
        if self._control.kill_command_error is not None:
            raise self._control.kill_command_error
        return True


class FakeFilesystem:
    """Mirrors `sandbox.files`: an in-memory tree the tests can drive and inspect."""

    def __init__(self, control: FakeE2B) -> None:
        self._control = control
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.removed: list[str] = []
        self.listed: list[str] = []

    async def read(
        self,
        path: str,
        format: Literal['bytes'],
        user: str | None = None,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> bytearray:
        del user, request_timeout, gzip
        await self._check(path)
        # The backend only asks for bytes; anything else would be a silent behavior change.
        assert format == 'bytes', f'unexpected read format {format!r}'
        if path not in self.files:
            raise FileNotFoundException(path)
        # Real E2B hands back a `bytearray` for the `bytes` format.
        return bytearray(self.files[path])

    async def write(
        self,
        path: str,
        data: str | bytes,
        user: str | None = None,
        request_timeout: float | None = None,
    ) -> WriteInfo:
        del user, request_timeout
        await self._check(path)
        self.files[path] = data.encode() if isinstance(data, str) else data
        self._add_parents(path)
        return WriteInfo(name=posixpath.basename(path), type=FileType.FILE, path=path)

    async def get_info(self, path: str, user: str | None = None, request_timeout: float | None = None) -> FakeEntryInfo:
        del user, request_timeout
        await self._check(path)
        return self._entry(path)

    async def list(
        self,
        path: str,
        depth: int | None = 1,
        user: str | None = None,
        request_timeout: float | None = None,
    ) -> list[FakeEntryInfo]:
        del user, request_timeout
        await self._check(path)
        assert depth == 1, f'unexpected list depth {depth!r}'
        self.listed.append(path)
        if not await self._exists(path):
            raise FileNotFoundException(path)
        children = {
            posixpath.join(path, name)
            for entry in (*self.files, *self.directories)
            if entry != path and entry.startswith(f'{path.rstrip("/")}/')
            for name in (entry[len(path.rstrip('/')) + 1 :].split('/')[0],)
        }
        return [self._entry(child) for child in sorted(children)]

    async def exists(self, path: str, user: str | None = None, request_timeout: float | None = None) -> bool:
        del user, request_timeout
        await self._check(path)
        return await self._exists(path)

    async def make_dir(self, path: str, user: str | None = None, request_timeout: float | None = None) -> bool:
        del user, request_timeout
        await self._check(path)
        created = path not in self.directories
        self.directories.add(path)
        self._add_parents(path)
        return created

    async def remove(self, path: str, user: str | None = None, request_timeout: float | None = None) -> None:
        del user, request_timeout
        await self._check(path)
        if not await self._exists(path):
            raise FileNotFoundException(path)
        self.removed.append(path)
        prefix = f'{path.rstrip("/")}/'
        for target in [target for target in self.files if target == path or target.startswith(prefix)]:
            del self.files[target]
        for directory in [d for d in self.directories if d == path or d.startswith(prefix)]:
            self.directories.discard(directory)

    async def _exists(self, path: str) -> bool:
        return path in self.files or path in self.directories

    def _entry(self, path: str) -> FakeEntryInfo:
        if path in self.directories:
            return FakeEntryInfo(name=posixpath.basename(path), path=path, type=FileType.DIR, size=0)
        if path not in self.files:
            raise FileNotFoundException(path)
        return FakeEntryInfo(name=posixpath.basename(path), path=path, type=FileType.FILE, size=len(self.files[path]))

    def _add_parents(self, path: str) -> None:
        parent = posixpath.dirname(path)
        while parent and parent != '/':
            self.directories.add(parent)
            parent = posixpath.dirname(parent)

    async def _check(self, path: str) -> None:
        # Real E2B's filesystem API only accepts absolute paths; assert it here so a
        # regression that let a relative path through unresolved fails in the fake the way it
        # would in prod, instead of silently keying the in-memory store on a relative path.
        assert posixpath.isabs(path), f'E2B filesystem requires an absolute path, got {path!r}'
        await anyio.lowlevel.checkpoint()
        if self._control.fs_error is not None:
            raise self._control.fs_error


class FakeSandbox:
    """Mirrors `e2b.AsyncSandbox` for the members the backend uses."""

    def __init__(
        self,
        control: FakeE2B,
        sandbox_id: str,
        metadata: dict[str, str] | None = None,
        *,
        started_at: datetime,
    ) -> None:
        self._control = control
        self.sandbox_id = sandbox_id
        self.metadata = metadata or {}
        self.started_at = started_at
        self.files = FakeFilesystem(control)
        self.commands = FakeCommands(self, control)
        self.killed = False

    async def kill(self) -> bool:
        await anyio.lowlevel.checkpoint()
        if self._control.kill_hangs:
            await anyio.sleep_forever()
        if self._control.kill_error is not None:
            raise self._control.kill_error
        self.killed = True
        return True

    async def is_running(self, request_timeout: float | None = None) -> bool:
        del request_timeout
        await anyio.lowlevel.checkpoint()
        if self._control.is_running_error is not None:
            raise self._control.is_running_error
        return self._control.sandbox_is_running and not self.killed


class FakeAsyncSandboxFactory:
    """Mirrors the `AsyncSandbox.create` / `AsyncSandbox.connect` class methods."""

    def __init__(self, control: FakeE2B) -> None:
        self._control = control

    async def create(
        self,
        template: str | None = None,
        timeout: int | None = None,
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
    ) -> FakeSandbox:
        self._control.create_calls.append(
            FakeCreateCall(template, timeout, metadata, envs, secure, allow_internet_access)
        )
        if self._control.create_hangs:
            await anyio.sleep_forever()
        await anyio.lowlevel.checkpoint()
        if self._control.create_error is not None:
            raise self._control.create_error
        return self._control.new_sandbox(f'sbx-{len(self._control.sandboxes) + 1}', metadata)

    async def connect(self, sandbox_id: str, timeout: int | None = None) -> FakeSandbox:
        self._control.connect_calls.append((sandbox_id, timeout))
        await anyio.lowlevel.checkpoint()
        if self._control.connect_error is not None:
            raise self._control.connect_error
        return self._control.new_sandbox(sandbox_id)

    async def kill(self, sandbox_id: str) -> bool:
        self._control.kill_ids.append(sandbox_id)
        self._control.kill_started = True
        if self._control.kill_gate is not None:
            await self._control.kill_gate.wait()
        if self._control.kill_hangs:
            await anyio.sleep_forever()
        await anyio.lowlevel.checkpoint()
        if self._control.kill_error is not None:
            raise self._control.kill_error
        for sandbox in self._control.sandboxes:
            if sandbox.sandbox_id == sandbox_id and not sandbox.killed:
                sandbox.killed = True
                return True
        return False

    def list(
        self,
        query: FakeSandboxQuery | None = None,
        limit: int | None = None,
        next_token: str | None = None,
    ) -> FakeSandboxPaginator:
        # E2B 2.34.0, the declared dependency floor, has no `order` parameter. Keep this
        # signature closed so use of a newer-only list option fails in tests.
        del next_token
        metadata = query.metadata if query is not None else None
        self._control.list_calls.append((metadata, limit))
        if self._control.list_error is not None:
            raise self._control.list_error
        if self._control.list_batches:
            return FakeSandboxPaginator([self._control.list_batches.pop(0)])
        matches = [
            sandbox
            for sandbox in self._control.sandboxes
            if not sandbox.killed
            and (metadata is None or all(sandbox.metadata.get(key) == value for key, value in metadata.items()))
        ]
        return FakeSandboxPaginator([matches[:limit] if limit is not None else matches])


@dataclass
class FakeSandboxQuery:
    metadata: dict[str, str] | None = None


class FakeSandboxPaginator:
    def __init__(self, pages: list[list[FakeSandbox]]) -> None:
        self._pages = pages

    @property
    def has_next(self) -> bool:
        return bool(self._pages)

    async def next_items(self) -> list[FakeSandbox]:
        await anyio.lowlevel.checkpoint()
        return self._pages.pop(0)


@dataclass
class FakeE2B:
    """Control surface for the injected fake `e2b` module."""

    responder: Responder = _echo_responder
    sandboxes: list[FakeSandbox] = field(default_factory=list[FakeSandbox])
    create_calls: list[FakeCreateCall] = field(default_factory=list[FakeCreateCall])
    connect_calls: list[tuple[str, int | None]] = field(default_factory=list[tuple[str, 'int | None']])
    kill_ids: list[str] = field(default_factory=list[str])
    list_calls: list[tuple[dict[str, str] | None, int | None]] = field(
        default_factory=list[tuple[dict[str, str] | None, int | None]]
    )
    list_batches: list[list[FakeSandbox]] = field(default_factory=list[list[FakeSandbox]])
    list_error: Exception | None = None
    create_error: Exception | None = None
    create_hangs: bool = False
    connect_error: Exception | None = None
    kill_error: Exception | None = None
    kill_hangs: bool = False
    kill_gate: anyio.Event | None = None
    kill_started: bool = False
    run_error: Exception | None = None
    wait_error: Exception | None = None
    command_hangs: bool = False
    kill_command_error: Exception | None = None
    fs_error: Exception | None = None
    is_running_error: Exception | None = None
    sandbox_is_running: bool = True
    next_pid: int = 4242

    def __post_init__(self) -> None:
        self.module = self._build_module()

    def new_sandbox(self, sandbox_id: str, metadata: dict[str, str] | None = None) -> FakeSandbox:
        started_at = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=len(self.sandboxes))
        sandbox = FakeSandbox(self, sandbox_id, metadata, started_at=started_at)
        self.sandboxes.append(sandbox)
        return sandbox

    @property
    def auth_type(self) -> type[Exception]:
        return AuthenticationException

    @property
    def sandbox_gone_type(self) -> type[Exception]:
        """E2B `SandboxNotFoundException`: the sandbox does not exist -- terminal."""
        return SandboxNotFoundException

    @property
    def ambiguous_type(self) -> type[Exception]:
        """E2B `TimeoutException`: an unanswered request, whether the sandbox is alive or gone."""
        return TimeoutException

    @property
    def error_type(self) -> type[Exception]:
        return SandboxException

    def _build_module(self) -> types.ModuleType:
        module = types.ModuleType('e2b')
        module.AsyncSandbox = FakeAsyncSandboxFactory(self)  # type: ignore[attr-defined]
        module.SandboxQuery = FakeSandboxQuery  # type: ignore[attr-defined]
        # The real classes, so the backend's `isinstance` checks and its unwrapping of a
        # non-zero exit run against exactly what production raises.
        module.AuthenticationException = AuthenticationException  # type: ignore[attr-defined]
        module.CommandExitException = CommandExitException  # type: ignore[attr-defined]
        module.CommandResult = CommandResult  # type: ignore[attr-defined]
        module.FileNotFoundException = FileNotFoundException  # type: ignore[attr-defined]
        module.FileType = FileType  # type: ignore[attr-defined]
        module.SandboxException = SandboxException  # type: ignore[attr-defined]
        module.SandboxNotFoundException = SandboxNotFoundException  # type: ignore[attr-defined]
        module.TimeoutException = TimeoutException  # type: ignore[attr-defined]
        return module


if TYPE_CHECKING:
    import e2b

    class _EntryInfoSurface(Protocol):
        """The `e2b.EntryInfo` members the backend reads.

        Pinned against both the fake and the real SDK type below, so a fake that drifts from
        E2B's own entry shape fails the type check instead of at the next live run.
        """

        @property
        def name(self) -> str: ...

        @property
        def path(self) -> str: ...

        @property
        def type(self) -> FileType | None: ...

        @property
        def size(self) -> int: ...

    class _CommandHandleSurface(Protocol):
        """The `e2b.AsyncCommandHandle` members the backend reads.

        `stdout` / `stderr` are the accumulated output the deadline path reports, so a fake
        that stopped exposing them would hide the timeout behavior entirely.
        """

        @property
        def pid(self) -> int: ...

        @property
        def stdout(self) -> str: ...

        @property
        def stderr(self) -> str: ...

        async def wait(self) -> CommandResult: ...

    _fake_entry_conforms: _EntryInfoSurface = FakeEntryInfo(name='n', path='/n', type=FileType.FILE, size=0)
    _real_entry_conforms: _EntryInfoSurface = e2b.EntryInfo.__new__(e2b.EntryInfo)
    _fake_handle_conforms: _CommandHandleSurface = FakeCommandHandle(
        FakeE2B(), pid=1, stdout='', stderr='', exit_code=0
    )
    _real_handle_conforms: _CommandHandleSurface = e2b.AsyncCommandHandle.__new__(e2b.AsyncCommandHandle)

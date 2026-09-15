"""A controllable fake `modal` SDK for ModalWorkspace tests.

Tests never reach real Modal: a fake `modal` module is injected into `sys.modules`
(via the `fake_modal` fixture in `conftest.py`), so the lazy `import modal` inside
the backend returns it. The fake records calls and lets each test decide what
`exec` returns.

Fidelity to the real SDK is the point: signatures are closed, `.aio` suspends, an
exec output reader replays from byte zero on every read, and missing paths raise
Modal's own filesystem exception. Flattering the code under test here would hide
production failures.
"""

from __future__ import annotations

import posixpath
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import anyio.lowlevel

StreamData = bytes | str
# A responder maps (argv, timeout) to (stdout, stderr, exit_code).
Responder = Callable[[list[str], 'int | None'], 'tuple[StreamData, StreamData, int]']


def _echo_responder(argv: list[str], timeout: int | None) -> tuple[bytes, bytes, int]:
    return (' '.join(argv) + '\n').encode(), b'', 0


def _stream_bytes(data: StreamData) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.encode()


@dataclass
class ExecCall:
    argv: list[str]
    timeout: int | None
    text: bool
    workdir: str | None = None
    env: dict[str, str | None] | None = None


class _AioCallable:
    """Mimics a synchronicity-wrapped Modal method: callable, plus an `.aio` async twin.

    The backend only calls `.aio`, but exposing both mirrors the real SDK shape.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - capability only uses `.aio`
        # Modal's callables work sync or async; we mirror both for fidelity, but the
        # capability drives the async `.aio` path exclusively, so this never runs in tests.
        return self._fn(*args, **kwargs)

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        # A real Modal `.aio` call suspends (it awaits gRPC); yield here so a concurrent
        # batch of tool calls actually interleaves in tests -- otherwise the sync fake would
        # run each call start-to-finish and hide races like a duplicated `pwd` probe.
        await anyio.lowlevel.checkpoint()
        return self._fn(*args, **kwargs)


class _GatedCreate(_AioCallable):
    """A create call whose control-plane response can be held for lock-race tests."""

    def __init__(self, fn: Callable[..., Any], control: FakeModal) -> None:
        super().__init__(fn)
        self._control = control

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.lowlevel.checkpoint()
        self._control.create_started = True
        if self._control.create_gate is not None:
            await self._control.create_gate.wait()
        return self._fn(*args, **kwargs)


class _HangingAioCall:
    """An `.aio` that never returns, for tests that cancel a pending call."""

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.sleep_forever()


class _FakeStream:
    """Mimics the whole-output `.read.aio()` surface used by the backend."""

    def __init__(self, data: bytes, hangs: bool = False) -> None:
        self._data = data
        self.read = _HangingAioCall() if hangs else _AioCallable(self._read)

    def _read(self) -> bytes:
        return self._data


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes,
        stderr: bytes,
        returncode: int,
        wait_error: Exception | None,
        wait_hangs: bool,
        stdout_hangs: bool = False,
    ) -> None:
        self.stdout = _FakeStream(stdout, stdout_hangs)
        self.stderr = _FakeStream(stderr)
        self._returncode = returncode
        self._wait_error = wait_error
        self.returncode: int | None = None
        self.wait: _AioCallable | _HangingAioCall = _HangingAioCall() if wait_hangs else _AioCallable(self._wait)

    def _wait(self) -> int:
        if self._wait_error is not None:
            raise self._wait_error
        self.returncode = self._returncode
        return self._returncode


class FakeModalError(Exception):
    """Stand-in for `modal.exception.Error`."""


class FakeNotFoundError(FakeModalError):
    """Stand-in for `modal.exception.NotFoundError` (the workspace itself is missing/gone)."""


class FakeAlreadyExistsError(FakeModalError):
    """Stand-in for `modal.exception.AlreadyExistsError`."""


class FakeAuthError(FakeModalError):
    """Stand-in for `modal.exception.AuthError`."""


class FakeSandboxTerminatedError(FakeModalError):
    """Stand-in for `modal.exception.SandboxTerminatedError`."""


class FakeSandboxTimeoutError(FakeModalError):
    """Stand-in for `modal.exception.SandboxTimeoutError`."""


class FakeConflictError(FakeModalError):
    """Stand-in for `modal.exception.ConflictError` (first exec on a dead workspace, or a transient abort)."""


class FakeSandboxFilesystemError(FakeModalError):
    """Stand-in for `modal.exception.SandboxFilesystemError`."""


class FakeSandboxFilesystemNotFoundError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemNotFoundError` (a missing file, recoverable)."""


class FakeSandboxFilesystemNotADirectoryError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemNotADirectoryError` (a non-directory path component)."""


class FakeSandboxFilesystemIsADirectoryError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemIsADirectoryError`."""


@dataclass
class FileInfo:
    """Minimal stand-in for `modal.types.FileInfo`, covering what the backend reads."""

    name: str
    _is_dir: bool
    size: int = 0

    def is_dir(self) -> bool:
        return self._is_dir


class _FakeFilesystem:
    """Mirrors `workspace.filesystem`: an in-memory store the tests can drive and inspect."""

    def __init__(self, workspace: FakeSandbox) -> None:
        self._workspace = workspace
        self.read_bytes = _AioCallable(self._read_bytes)
        self.write_bytes = _AioCallable(self._write_bytes)
        self.list_files = _AioCallable(self._list_files)
        self.stat = _AioCallable(self._stat)
        self.make_directory = _AioCallable(self._make_directory)
        self.remove = _AioCallable(self._remove)

    def _read_bytes(self, remote_path: str) -> bytes:
        self._check(remote_path)
        if remote_path in self._workspace.directories:
            raise FakeSandboxFilesystemIsADirectoryError(f'Is a directory: {remote_path}')
        data = self._workspace.files.get(remote_path)
        if data is None:
            raise FakeSandboxFilesystemNotFoundError(f'No such file or directory: {remote_path}')
        return data

    def _stat(self, remote_path: str) -> FileInfo:
        self._check(remote_path)
        if remote_path in self._workspace.directories:
            return FileInfo(posixpath.basename(remote_path), True)
        if remote_path not in self._workspace.files and remote_path not in self._workspace.stat_sizes:
            raise FakeSandboxFilesystemNotFoundError(f'No such file or directory: {remote_path}')
        # Size comes from the stored bytes, or an override the test set for this path.
        size = self._workspace.stat_sizes.get(remote_path, len(self._workspace.files.get(remote_path, b'')))
        # Real Modal reports the entry's basename, not the full path.
        return FileInfo(posixpath.basename(remote_path), False, size=size)

    def _write_bytes(self, data: bytes, remote_path: str) -> None:
        self._check(remote_path)
        self._workspace.files[remote_path] = data

    def _list_files(self, remote_path: str) -> list[FileInfo]:
        self._check(remote_path)
        self._workspace.list_paths.append(remote_path)
        return self._workspace.listing

    def _make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        # Closed keyword signature on purpose, like `workspace_create`: `create_parents` is the
        # real API's `mkdir -p` switch and defaults to True there too.
        self._check(remote_path)
        self._workspace.directories.add(remote_path)

    def _remove(self, remote_path: str, *, recursive: bool = False) -> None:
        self._check(remote_path)
        self._workspace.removals.append((remote_path, recursive))
        self._workspace.directories.discard(remote_path)
        self._workspace.files.pop(remote_path, None)

    def _check(self, remote_path: str) -> None:
        # Real Modal's filesystem API only accepts absolute paths; assert it here so a
        # regression that let a relative path through unresolved fails in the fake the way it
        # would in prod, instead of silently keying the in-memory store on a relative path.
        assert posixpath.isabs(remote_path), f'Modal filesystem requires an absolute path, got {remote_path!r}'
        if self._workspace.fs_error is not None:
            raise self._workspace.fs_error


class FakeSandbox:
    def __init__(self, control: FakeModal, object_id: str) -> None:
        self._control = control
        self.object_id = object_id
        self.exec_calls: list[ExecCall] = []
        self.exec = _HangingAioCall() if control.exec_hangs else _AioCallable(self._exec)
        self.poll = _AioCallable(self._poll)
        # Filesystem state the tests read and write.
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.removals: list[tuple[str, bool]] = []
        # Lets a test report a large size for a path without allocating the bytes.
        self.stat_sizes: dict[str, int] = {}
        self.list_paths: list[str] = []
        self.listing: list[FileInfo] = []
        self.fs_error: Exception | None = None
        self.poll_result: int | None = None
        self.poll_error: Exception | None = None
        self.poll_calls = 0
        self._filesystem = _FakeFilesystem(self)

    @property
    def filesystem(self) -> _FakeFilesystem:
        return self._filesystem

    def _exec(
        self,
        *args: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        text: bool = True,
    ) -> _FakeProcess:
        # Closed keyword signature on purpose: real `Sandbox.exec` rejects unknown kwargs,
        # so the fake must too, or a bad kwarg in the backend would only fail in production.
        argv = list(args)
        self.exec_calls.append(ExecCall(argv=argv, timeout=timeout, text=text, workdir=workdir, env=env))
        if self._control.exec_error is not None:
            raise self._control.exec_error
        stdout, stderr, code = self._control.responder(argv, timeout)
        return _FakeProcess(
            _stream_bytes(stdout),
            _stream_bytes(stderr),
            code,
            self._control.wait_error,
            self._control.wait_hangs,
            self._control.stdout_hangs,
        )

    def _poll(self) -> int | None:
        self.poll_calls += 1
        if self.poll_error is not None:
            raise self.poll_error
        if self.poll_result is not None:
            return self.poll_result
        return None


class FakeModal:
    """Control surface for the injected fake `modal` module."""

    def __init__(self) -> None:
        self.responder: Responder = _echo_responder
        self.sandboxes: list[FakeSandbox] = []
        self.create_kwargs: list[dict[str, object]] = []
        self.app_lookups: list[dict[str, object]] = []
        # The marker objects `App.lookup` returned, so a test can assert the looked-up app
        # is the one passed to `Sandbox.create`.
        self.apps: list[object] = []
        self.image_tags: list[str] = []
        self.attach_ids: list[str] = []
        self.owned_creates = 0
        self.create_error: Exception | None = None
        self.create_gate: anyio.Event | None = None
        self.create_started = False
        self.attach_error: Exception | None = None
        self.attach_poll_result: int | None = None
        self.exec_error: Exception | None = None
        self.exec_hangs = False
        self.wait_error: Exception | None = None
        self.wait_hangs = False
        self.stdout_hangs = False
        self.module = self._build_module()

    @property
    def error_type(self) -> type[Exception]:
        return FakeModalError

    @property
    def filesystem_error_type(self) -> type[Exception]:
        return FakeSandboxFilesystemError

    @property
    def unavailable_type(self) -> type[Exception]:
        """A missing/terminated *workspace* (Modal `NotFoundError`) -- terminal, not retried."""
        return FakeNotFoundError

    @property
    def auth_type(self) -> type[Exception]:
        return FakeAuthError

    @property
    def workspace_terminated_type(self) -> type[Exception]:
        return FakeSandboxTerminatedError

    @property
    def workspace_timeout_type(self) -> type[Exception]:
        """Modal `SandboxTimeoutError`: the sandbox expired at its `sandbox_timeout` -- terminal."""
        return FakeSandboxTimeoutError

    @property
    def conflict_type(self) -> type[Exception]:
        """Modal `ConflictError`: ambiguous (dead workspace on first exec, or a transient abort)."""
        return FakeConflictError

    def _build_module(self) -> types.ModuleType:
        control = self
        module = types.ModuleType('modal')

        def app_lookup(name: str, *, create_if_missing: bool = False) -> object:
            control.app_lookups.append({'name': name, 'create_if_missing': create_if_missing})
            app = object()
            control.apps.append(app)
            return app

        def image_from_registry(tag: str) -> object:
            # Closed signature on purpose, like `workspace_create` below: signature drift in
            # the backend should fail here, not only in production.
            control.image_tags.append(tag)
            return object()

        def workspace_create(
            *,
            app: object,
            image: object,
            timeout: int | None = None,
            workdir: str | None = None,
            env: dict[str, str | None] | None = None,
            name: str | None = None,
        ) -> FakeSandbox:
            if control.create_error is not None:
                raise control.create_error
            control.create_kwargs.append(
                {'app': app, 'image': image, 'timeout': timeout, 'workdir': workdir, 'env': env, 'name': name}
            )
            control.owned_creates += 1
            suffix = '' if control.owned_creates == 1 else f'-{control.owned_creates}'
            workspace = FakeSandbox(control, f'sb-owned{suffix}')
            control.sandboxes.append(workspace)
            return workspace

        def workspace_from_id(id: str) -> FakeSandbox:
            control.attach_ids.append(id)
            if control.attach_error is not None:
                raise control.attach_error
            existing = next((s for s in control.sandboxes if s.object_id == id), None)
            if existing is not None:
                return existing
            workspace = FakeSandbox(control, id)
            workspace.poll_result = control.attach_poll_result
            control.sandboxes.append(workspace)
            return workspace

        class App:
            lookup = _AioCallable(app_lookup)

        class Image:
            from_registry = staticmethod(image_from_registry)

        class Sandbox:
            create = _GatedCreate(workspace_create, control)
            from_id = _AioCallable(workspace_from_id)

        module.App = App  # type: ignore[attr-defined]
        module.Image = Image  # type: ignore[attr-defined]
        module.Sandbox = Sandbox  # type: ignore[attr-defined]
        module.exception = types.SimpleNamespace(  # type: ignore[attr-defined]
            Error=FakeModalError,
            AlreadyExistsError=FakeAlreadyExistsError,
            NotFoundError=FakeNotFoundError,
            AuthError=FakeAuthError,
            ConflictError=FakeConflictError,
            SandboxTerminatedError=FakeSandboxTerminatedError,
            SandboxTimeoutError=FakeSandboxTimeoutError,
            SandboxFilesystemError=FakeSandboxFilesystemError,
            SandboxFilesystemNotFoundError=FakeSandboxFilesystemNotFoundError,
            SandboxFilesystemNotADirectoryError=FakeSandboxFilesystemNotADirectoryError,
            SandboxFilesystemIsADirectoryError=FakeSandboxFilesystemIsADirectoryError,
        )
        return module


if TYPE_CHECKING:
    import modal.types

    class _FileInfoSurface(Protocol):
        """The `modal.types.FileInfo` members the backend reads.

        Pinned against both the fake and the real SDK type below, so a fake that drifts from
        Modal's own entry shape fails the type check instead of at the next live run.
        """

        @property
        def name(self) -> str: ...

        @property
        def size(self) -> int: ...

        def is_dir(self) -> bool: ...

    _fake_file_info_conforms: _FileInfoSurface = FileInfo('name', False)
    _real_file_info_conforms: _FileInfoSurface = modal.types.FileInfo.__new__(modal.types.FileInfo)

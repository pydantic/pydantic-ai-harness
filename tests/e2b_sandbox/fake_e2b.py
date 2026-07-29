"""In-memory E2B async SDK fake used by sandbox capability tests."""

from __future__ import annotations

import posixpath
import re
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

import anyio
from typing_extensions import Self


class AuthenticationException(Exception):
    """Fake E2B authentication failure."""


class SandboxException(Exception):
    """Fake generic E2B failure."""


class SandboxNotFoundException(SandboxException):
    """Fake missing E2B sandbox failure."""


class FileNotFoundException(SandboxException):
    """Fake missing E2B file failure."""


class TimeoutException(SandboxException):
    """Fake command timeout."""


class CommandExitException(SandboxException):
    """Fake non-zero command result."""

    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.error = f'exit status {exit_code}'
        super().__init__(self.error)


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
    timeout: float | None


@dataclass(frozen=True)
class FakeCommandResult:
    stdout: str
    stderr: str
    exit_code: int
    error: str | None = None


@dataclass(frozen=True)
class FakeFileType:
    value: str


@dataclass(frozen=True)
class FakeEntryInfo:
    name: str
    path: str
    type: FakeFileType | None
    size: int


class FakeFileStream:
    """Async byte stream whose chunking is configurable."""

    def __init__(self, data: bytes, chunk_size: int) -> None:
        self._data = data
        self._chunk_size = chunk_size
        self._index = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._data):
            raise StopAsyncIteration
        chunk = self._data[self._index : self._index + self._chunk_size]
        self._index += self._chunk_size
        return chunk

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeCommandHandle:
    """Background command handle matching the portion Harness consumes."""

    def __init__(
        self,
        control: FakeE2B,
        *,
        result: FakeCommandResult,
        wait_error: Exception | None,
    ) -> None:
        self._control = control
        self._result = result
        self._wait_error = wait_error
        self.pid = 4242
        self.killed = False

    async def wait(self) -> FakeCommandResult:
        if self._wait_error is not None:
            raise self._wait_error
        if self._result.exit_code:
            raise CommandExitException(
                self._control.sdk_stdout,
                self._control.sdk_stderr,
                self._result.exit_code,
            )
        return self._result

    async def kill(self) -> bool:
        self.killed = True
        if self._control.handle_kill_error is not None:
            raise self._control.handle_kill_error
        return True


class FakeFilesystem:
    """Dictionary-backed E2B filesystem."""

    def __init__(self, control: FakeE2B) -> None:
        self._control = control
        self.files: dict[str, bytes] = {}
        self.stat_sizes: dict[str, int] = {}
        self.listings: dict[str, list[FakeEntryInfo]] = {}
        self.removed: list[str] = []

    async def get_info(self, path: str) -> FakeEntryInfo:
        if self._control.info_error is not None:
            raise self._control.info_error
        if path not in self.files and path not in self.stat_sizes:
            raise FileNotFoundException(path)
        size = self.stat_sizes.get(path, len(self.files.get(path, b'')))
        return FakeEntryInfo(posixpath.basename(path), path, FakeFileType('file'), size)

    @overload
    async def read(self, path: str, format: Literal['bytes']) -> bytearray: ...  # pragma: no cover

    @overload
    async def read(
        self,
        path: str,
        format: Literal['stream'],
        *,
        stream_idle_timeout: float | None = None,
    ) -> FakeFileStream: ...  # pragma: no cover

    async def read(
        self,
        path: str,
        format: Literal['bytes', 'stream'],
        *,
        stream_idle_timeout: float | None = None,
    ) -> bytearray | FakeFileStream:
        del stream_idle_timeout
        if self._control.read_error is not None:
            raise self._control.read_error
        try:
            data = self.files[path]
        except KeyError as e:
            raise FileNotFoundException(path) from e
        if format == 'bytes':
            return bytearray(data)
        return FakeFileStream(data, self._control.file_chunk_size)

    async def write(self, path: str, data: str | bytes) -> object:
        if self._control.write_error is not None:
            raise self._control.write_error
        self.files[path] = data.encode() if isinstance(data, str) else data
        return object()

    async def list(self, path: str, depth: int | None = 1) -> list[FakeEntryInfo]:
        del depth
        if self._control.list_error is not None:
            raise self._control.list_error
        return list(self.listings.get(path, []))

    async def remove(self, path: str) -> None:
        self.removed.append(path)
        if self._control.remove_error is not None:
            raise self._control.remove_error
        for target in [target for target in self.files if target == path or target.startswith(f'{path}/')]:
            del self.files[target]


class FakeCommands:
    """Command manager that simulates Harness's capture wrapper."""

    def __init__(self, sandbox: FakeSandbox, control: FakeE2B) -> None:
        self._sandbox = sandbox
        self._control = control
        self.calls: list[FakeCommandCall] = []
        self.handles: list[FakeCommandHandle] = []

    @overload
    async def run(
        self,
        command: str,
        *,
        background: Literal[True],
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandHandle: ...  # pragma: no cover

    @overload
    async def run(
        self,
        command: str,
        *,
        background: Literal[False] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandResult: ...  # pragma: no cover

    async def run(
        self,
        command: str,
        *,
        background: bool | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandHandle | FakeCommandResult:
        is_background = background is True
        self.calls.append(FakeCommandCall(command, is_background, cwd, timeout))
        if self._control.run_error is not None:
            raise self._control.run_error
        if not is_background:
            if self._control.kill_process_error is not None:
                raise self._control.kill_process_error
            return FakeCommandResult('', '', 0)

        wrapper_path = command.rsplit(' ', 1)[-1]
        temp_dir = posixpath.dirname(wrapper_path)
        command_text = self._sandbox.files.files[f'{temp_dir}/command.sh'].decode()
        wrapper = self._sandbox.files.files[wrapper_path].decode()
        match = re.search(r'tail -c (\d+)', wrapper)
        if match is None:  # pragma: no cover - asserts Harness's private wrapper contract
            raise AssertionError('Harness capture wrapper did not contain a byte limit')
        byte_limit = int(match.group(1))
        stdout, stderr, exit_code = self._control.responder(command_text, int(timeout or 0))
        if not self._control.omit_capture:
            self._write_capture(temp_dir, 'stdout', stdout, byte_limit)
            self._write_capture(temp_dir, 'stderr', stderr, byte_limit)
        result = FakeCommandResult(
            self._control.sdk_stdout,
            self._control.sdk_stderr,
            exit_code,
            None if exit_code == 0 else f'exit status {exit_code}',
        )
        handle = FakeCommandHandle(self._control, result=result, wait_error=self._control.wait_error)
        self.handles.append(handle)
        return handle

    def _write_capture(self, temp_dir: str, stream: str, text: str, byte_limit: int) -> None:
        data = text.encode()
        self._sandbox.files.files[f'{temp_dir}/{stream}'] = data[-byte_limit:]
        count = b'invalid' if self._control.invalid_count else str(len(data)).encode()
        if not self._control.omit_count:
            self._sandbox.files.files[f'{temp_dir}/{stream}.count'] = count


class FakeSandbox:
    """Fake async sandbox instance."""

    def __init__(self, control: FakeE2B, sandbox_id: str) -> None:
        self._control = control
        self.sandbox_id = sandbox_id
        self.files = FakeFilesystem(control)
        self.commands = FakeCommands(self, control)
        self.kill_calls = 0

    async def kill(self) -> bool:
        self.kill_calls += 1
        if self._control.kill_hangs:
            await anyio.sleep_forever()
        if self._control.kill_error is not None:
            raise self._control.kill_error
        return True


class FakeAsyncSandboxFactory:
    """Fake `AsyncSandbox` class methods."""

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
        if self._control.create_error is not None:
            raise self._control.create_error
        sandbox = FakeSandbox(self._control, f'sbx-{len(self._control.sandboxes) + 1}')
        self._control.sandboxes.append(sandbox)
        return sandbox

    async def connect(self, sandbox_id: str, timeout: int | None = None) -> FakeSandbox:
        self._control.connect_calls.append((sandbox_id, timeout))
        if self._control.connect_error is not None:
            raise self._control.connect_error
        sandbox = FakeSandbox(self._control, sandbox_id)
        self._control.sandboxes.append(sandbox)
        return sandbox


class FakeE2B:
    """Control surface plus a module-shaped E2B SDK fake."""

    def __init__(self) -> None:
        self.sandboxes: list[FakeSandbox] = []
        self.create_calls: list[FakeCreateCall] = []
        self.connect_calls: list[tuple[str, int | None]] = []
        self.responder: Callable[[str, int], tuple[str, str, int]] = lambda command, timeout: ('', '', 0)
        self.create_error: Exception | None = None
        self.connect_error: Exception | None = None
        self.kill_error: Exception | None = None
        self.kill_hangs = False
        self.create_hangs = False
        self.run_error: Exception | None = None
        self.kill_process_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.handle_kill_error: Exception | None = None
        self.write_error: Exception | None = None
        self.read_error: Exception | None = None
        self.info_error: Exception | None = None
        self.list_error: Exception | None = None
        self.remove_error: Exception | None = None
        self.omit_capture = False
        self.omit_count = False
        self.invalid_count = False
        self.sdk_stdout = ''
        self.sdk_stderr = ''
        self.file_chunk_size = 4

        self.module = types.ModuleType('e2b')
        setattr(self.module, 'AsyncSandbox', FakeAsyncSandboxFactory(self))
        setattr(self.module, 'AuthenticationException', AuthenticationException)
        setattr(self.module, 'CommandExitException', CommandExitException)
        setattr(self.module, 'FileNotFoundException', FileNotFoundException)
        setattr(self.module, 'SandboxException', SandboxException)
        setattr(self.module, 'SandboxNotFoundException', SandboxNotFoundException)
        setattr(self.module, 'TimeoutException', TimeoutException)

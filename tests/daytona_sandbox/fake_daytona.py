"""Controllable fake for the Daytona SDK boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Protocol

from daytona import DaytonaNotFoundError


class CreateParams(Protocol):
    name: str | None
    snapshot: str | None
    auto_stop_interval: int | None
    auto_delete_interval: int | None
    env_vars: dict[str, str] | None
    network_block_all: bool | None


class FakeProcess:
    def __init__(self, owner: FakeSandbox) -> None:
        self.owner = owner

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        if command == 'pwd -P':
            self.owner.workdir_calls += 1
            self.owner.workdir_started.set()
            if self.owner.workdir_gate is not None:
                await self.owner.workdir_gate.wait()
            if self.owner.workdir_error is not None:
                raise self.owner.workdir_error
            return SimpleNamespace(result=(cwd or self.owner.workdir) + '\n', exit_code=0)
        assert command.startswith('mkdir -p -- ')
        return SimpleNamespace(result='', exit_code=self.owner.mkdir_exit_code)

    async def create_session(self, session_id: str, request_timeout: float | None = None) -> None:
        if self.owner.process_create_gate is not None:
            await self.owner.process_create_gate.wait()
        self.owner.process_sessions.add(session_id)

    async def execute_session_command(
        self,
        session_id: str,
        request: object,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        if self.owner.exec_error is not None:
            raise self.owner.exec_error
        self.owner.process_command = getattr(request, 'command')
        if not self.owner.process_stdout and not self.owner.process_stderr and not self.owner.process_hangs:
            output, self.owner.process_exit_code = self.owner.responder(self.owner.process_command, timeout)
            self.owner.process_stdout = [output]
        return SimpleNamespace(cmd_id='cmd-1')

    async def get_session_command_logs_async(
        self,
        session_id: str,
        command_id: str,
        on_stdout: Callable[[str], None],
        on_stderr: Callable[[str], None],
    ) -> None:
        self.owner.process_logs_started.set()
        if self.owner.process_logs_error is not None:
            raise self.owner.process_logs_error
        for handler, chunks in (
            (on_stdout, self.owner.process_stdout),
            (on_stderr, self.owner.process_stderr),
        ):
            for chunk in chunks:
                handler(chunk)
        if self.owner.process_hangs:
            await asyncio.Event().wait()

    async def get_session_command(
        self,
        session_id: str,
        command_id: str,
        request_timeout: float | None = None,
    ) -> SimpleNamespace:
        if self.owner.process_status_gate is not None:
            await self.owner.process_status_gate.wait()
        if self.owner.process_status_error is not None:
            raise self.owner.process_status_error
        return SimpleNamespace(exit_code=self.owner.process_exit_code)

    async def delete_session(self, session_id: str, request_timeout: float | None = None) -> None:
        if self.owner.process_delete_error is not None:
            raise self.owner.process_delete_error
        self.owner.process_sessions.discard(session_id)


class FakeFileSystem:
    def __init__(self, owner: FakeSandbox) -> None:
        self.owner = owner

    async def get_file_info(self, path: str, request_timeout: float | None = None) -> SimpleNamespace:
        self._raise_if_needed()
        if path in self.owner.directories:
            return SimpleNamespace(size=0, is_dir=True)
        data = self.owner.files.get(path)
        if data is None:
            raise DaytonaNotFoundError(f'no file: {path}')
        return SimpleNamespace(size=len(data), is_dir=False)

    async def download_file(self, path: str, timeout: int | None = None) -> bytes:
        self._raise_if_needed()
        data = self.owner.files.get(path)
        if data is None:
            raise DaytonaNotFoundError(f'no file: {path}')
        return data

    async def upload_file(self, data: bytes, path: str, timeout: int = 1800) -> None:
        self._raise_if_needed()
        self.owner.files[path] = data

    async def list_files(
        self, path: str, depth: int | None = None, request_timeout: float | None = None
    ) -> list[SimpleNamespace]:
        self._raise_if_needed()
        prefix = '' if path in ('', '.') else path.rstrip('/') + '/'
        entries: dict[str, bool] = {}
        # File keys never end with '/', so the first segment under the prefix is never empty.
        for candidate in self.owner.files:
            if candidate.startswith(prefix):
                name, separator, _ = candidate[len(prefix) :].partition('/')
                entries[name] = bool(separator) or entries.get(name, False)
        for directory in self.owner.directories:
            if directory.startswith(prefix):
                name = directory[len(prefix) :].partition('/')[0]
                entries[name] = True
        return [
            SimpleNamespace(
                name=name,
                is_dir=is_dir,
                size=0 if is_dir else len(self.owner.files[prefix + name]),
            )
            for name, is_dir in entries.items()
        ]

    async def create_folder(self, path: str, mode: str, request_timeout: float | None = None) -> None:
        self._raise_if_needed()
        assert mode == '755'
        self.owner.directories.add(path)

    async def delete_file(self, path: str, recursive: bool = False, request_timeout: float | None = None) -> None:
        self._raise_if_needed()
        self.owner.files.pop(path, None)
        self.owner.directories.discard(path)
        if recursive:  # pragma: no branch - the sandbox protocol always requests recursive removal
            prefix = path.rstrip('/') + '/'
            self.owner.files = {key: value for key, value in self.owner.files.items() if not key.startswith(prefix)}
            self.owner.directories = {key for key in self.owner.directories if not key.startswith(prefix)}

    def _raise_if_needed(self) -> None:
        if self.owner.fs_error is not None:
            raise self.owner.fs_error


class FakeSandbox:
    def __init__(self, sandbox_id: str, name: str | None = None) -> None:
        self.client: FakeClient | None = None
        self.id = sandbox_id
        self.name = name or sandbox_id
        self.deleted = False
        self.started = False
        self.start_calls: list[float | None] = []
        self.start_gate: asyncio.Event | None = None
        self.paused = False
        self.pause_calls: list[float | None] = []
        self.pause_error: Exception | None = None
        self.stop_calls: list[float | None] = []
        self.auto_delete_interval: int | None = -1
        self.remote_auto_delete_interval: int | None = -1
        self.refresh_data_calls = 0
        self.refresh_error: Exception | None = None
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.exec_error: Exception | None = None
        self.fs_error: Exception | None = None
        self.workdir = '/srv/repo'
        self.workdir_error: Exception | None = None
        self.workdir_calls = 0
        self.workdir_gate: asyncio.Event | None = None
        self.workdir_started = asyncio.Event()
        self.mkdir_exit_code = 0
        self.responder: Callable[[str, int | None], tuple[str, int]] = lambda command, timeout: ('', 0)
        self.process_sessions: set[str] = set()
        self.process_command = ''
        self.process_stdout: list[str] = []
        self.process_stderr: list[str] = []
        self.process_hangs = False
        self.process_exit_code: int | None = 0
        self.process_delete_error: Exception | None = None
        self.process_status_error: Exception | None = None
        self.process_status_gate: asyncio.Event | None = None
        self.process_logs_error: Exception | None = None
        self.process_create_gate: asyncio.Event | None = None
        self.process_logs_started = asyncio.Event()
        self.process = FakeProcess(self)
        self.fs = FakeFileSystem(self)

    async def start(self, timeout: float | None = 60) -> None:
        self.start_calls.append(timeout)
        if self.start_gate is not None:
            await self.start_gate.wait()
        self.started = True

    async def pause(self, timeout: float = 60) -> None:
        self.pause_calls.append(timeout)
        if self.pause_error is not None:
            raise self.pause_error
        self.paused = True

    async def stop(self, timeout: float | None = 60, force: bool = False) -> None:
        del force
        self.stop_calls.append(timeout)
        self.started = False

    async def refresh_data(self, request_timeout: float | None = None) -> None:
        del request_timeout
        self.refresh_data_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error
        self.auto_delete_interval = self.remote_auto_delete_interval


class FakeClient:
    def __init__(self, owner: FakeDaytona) -> None:
        self.owner = owner
        self.closed = False

    async def create(self, params: CreateParams, *, timeout: float = 60) -> FakeSandbox:
        if self.owner.create_gate is not None:
            await self.owner.create_gate.wait()
        if self.owner.create_error is not None:
            raise self.owner.create_error
        sandbox = FakeSandbox(f'sb-{len(self.owner.sandboxes) + 1}', params.name)
        sandbox.client = self
        self.owner.sandboxes.append(sandbox)
        self.owner.create_params.append(params)
        return sandbox

    async def get(self, sandbox_id: str, request_timeout: float | None = None) -> FakeSandbox:
        if self.owner.get_gate is not None:
            await self.owner.get_gate.wait()
        if self.owner.get_error is not None:
            raise self.owner.get_error
        for sandbox in self.owner.sandboxes:
            if (sandbox.id == sandbox_id or sandbox.name == sandbox_id) and not sandbox.deleted:
                sandbox.client = self
                return sandbox
        raise DaytonaNotFoundError(f'no sandbox: {sandbox_id}')

    async def delete(self, sandbox: FakeSandbox, timeout: float, wait: bool) -> None:
        assert sandbox.client is not None and not sandbox.client.closed
        if self.owner.delete_error is not None:
            raise self.owner.delete_error
        sandbox.deleted = True

    async def close(self) -> None:
        self.owner.close_calls += 1
        if self.owner.close_error is not None:
            raise self.owner.close_error
        self.closed = True
        self.owner.closed_clients += 1


class FakeDaytona:
    def __init__(self) -> None:
        self.sandboxes: list[FakeSandbox] = []
        self.create_params: list[CreateParams] = []
        self.closed_clients = 0
        self.create_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.close_error: Exception | None = None
        self.close_calls = 0
        self.create_gate: asyncio.Event | None = None
        self.get_gate: asyncio.Event | None = None
        self.get_error: Exception | None = None

    def client(self) -> FakeClient:
        return FakeClient(self)

    def sandbox(self, sandbox_id: str = 'sb-existing') -> FakeSandbox:
        sandbox = FakeSandbox(sandbox_id)
        self.sandboxes.append(sandbox)
        return sandbox

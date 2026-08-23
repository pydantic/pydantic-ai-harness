"""Controllable fake for the Daytona SDK boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

from daytona import DaytonaNotFoundError


class CreateParams(Protocol):
    snapshot: str | None
    auto_stop_interval: int | None
    auto_delete_interval: int | None
    env_vars: dict[str, str] | None
    network_block_all: bool | None


@dataclass
class ExecCall:
    command: str
    cwd: str | None
    env: dict[str, str] | None
    timeout: int | None


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
        self.owner.exec_calls.append(ExecCall(command, cwd, env, timeout))
        if self.owner.exec_error is not None:
            raise self.owner.exec_error
        if command.startswith('mkdir -p -- '):
            return SimpleNamespace(result='', exit_code=self.owner.mkdir_exit_code)
        output, exit_code = self.owner.responder(command, timeout)
        return SimpleNamespace(result=output, exit_code=exit_code)


class FakeFileSystem:
    def __init__(self, owner: FakeSandbox) -> None:
        self.owner = owner

    async def get_file_info(self, path: str) -> SimpleNamespace:
        self._raise_if_needed()
        data = self.owner.files.get(path)
        if data is None:
            raise DaytonaNotFoundError(f'no file: {path}')
        size = self.owner.reported_sizes.get(path, len(data))
        return SimpleNamespace(size=size)

    async def download_file(self, path: str) -> bytes:
        self._raise_if_needed()
        data = self.owner.files.get(path)
        if data is None:
            raise DaytonaNotFoundError(f'no file: {path}')
        return data

    async def upload_file(self, data: bytes, path: str) -> None:
        self._raise_if_needed()
        self.owner.files[path] = data

    async def list_files(self, path: str) -> list[SimpleNamespace]:
        self._raise_if_needed()
        prefix = '' if path in ('', '.') else path.rstrip('/') + '/'
        entries: dict[str, bool] = {}
        for candidate in self.owner.files:
            if not candidate.startswith(prefix):
                continue
            relative = candidate[len(prefix) :]
            name, separator, _ = relative.partition('/')
            if name:
                entries[name] = bool(separator) or entries.get(name, False)
        return [SimpleNamespace(name=name, is_dir=is_dir) for name, is_dir in entries.items()]

    def _raise_if_needed(self) -> None:
        if self.owner.fs_error is not None:
            raise self.owner.fs_error


class FakeSandbox:
    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id
        self.deleted = False
        self.files: dict[str, bytes] = {}
        self.reported_sizes: dict[str, int] = {}
        self.exec_calls: list[ExecCall] = []
        self.exec_error: Exception | None = None
        self.fs_error: Exception | None = None
        self.mkdir_exit_code = 0
        self.responder: Callable[[str, int | None], tuple[str, int]] = lambda command, timeout: ('', 0)
        self.process = FakeProcess(self)
        self.fs = FakeFileSystem(self)


class FakeClient:
    def __init__(self, owner: FakeDaytona) -> None:
        self.owner = owner
        self.closed = False

    async def create(self, params: CreateParams) -> FakeSandbox:
        if self.owner.create_error is not None:
            raise self.owner.create_error
        sandbox = FakeSandbox(f'sb-{len(self.owner.sandboxes) + 1}')
        self.owner.sandboxes.append(sandbox)
        self.owner.create_params.append(params)
        return sandbox

    async def get(self, sandbox_id: str) -> FakeSandbox:
        if self.owner.get_error is not None:
            raise self.owner.get_error
        for sandbox in self.owner.sandboxes:
            if sandbox.id == sandbox_id:
                return sandbox
        raise DaytonaNotFoundError(f'no sandbox: {sandbox_id}')

    async def delete(self, sandbox: FakeSandbox, timeout: float, wait: bool) -> None:
        self.owner.delete_calls.append((sandbox.id, timeout, wait))
        if self.owner.delete_error is not None:
            raise self.owner.delete_error
        sandbox.deleted = True

    async def close(self) -> None:
        self.closed = True
        self.owner.closed_clients += 1


class FakeDaytona:
    def __init__(self) -> None:
        self.sandboxes: list[FakeSandbox] = []
        self.create_params: list[CreateParams] = []
        self.delete_calls: list[tuple[str, float, bool]] = []
        self.closed_clients = 0
        self.create_error: Exception | None = None
        self.get_error: Exception | None = None
        self.delete_error: Exception | None = None

    def client(self) -> FakeClient:
        return FakeClient(self)

    def sandbox(self, sandbox_id: str = 'sb-existing') -> FakeSandbox:
        sandbox = FakeSandbox(sandbox_id)
        self.sandboxes.append(sandbox)
        return sandbox

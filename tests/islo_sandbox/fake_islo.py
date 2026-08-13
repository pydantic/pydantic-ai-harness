"""A controllable, in-memory stand-in for the public Islo SDK."""

from __future__ import annotations

import posixpath
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio


class FakeApiError(Exception):
    """Stand-in for `islo.core.api_error.ApiError`."""

    def __init__(self, status_code: int, body: object) -> None:
        super().__init__(f'HTTP {status_code}: {body}')
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class FakeLifecyclePolicy:
    """Subset of Islo's lifecycle request used by the integration."""

    delete_after: int


@dataclass
class FakeSandboxResponse:
    """Subset of a sandbox response consumed by the integration."""

    name: str = 'sandbox-owned'
    id: str = 'sb-owned'
    status: str = 'running'
    workdir: str | None = '/workspace'


@dataclass(frozen=True)
class FakeExecStarted:
    """Minimal response from starting a command."""

    exec_id: str


@dataclass(frozen=True)
class FakeExecResult:
    """Minimal response from polling a command."""

    status: str = 'completed'
    stdout: str | None = ''
    stderr: str | None = ''
    exit_code: int | None = 0
    truncated: bool = False


@dataclass(frozen=True)
class ExecCall:
    """One recorded Islo command start."""

    sandbox_name: str
    command: list[str]
    workdir: str | None
    timeout_secs: int | None


Responder = Callable[[ExecCall], FakeExecResult | list[FakeExecResult]]


def _default_responder(call: ExecCall) -> FakeExecResult:
    return FakeExecResult(stdout=' '.join(call.command) + '\n')


class FakeSandboxesClient:
    """Records calls while simulating sandbox, exec, and file endpoints."""

    def __init__(self, control: FakeIslo) -> None:
        self._control = control
        self.create_calls: list[dict[str, object]] = []
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.exec_calls: list[ExecCall] = []
        self.exec_result_calls: list[tuple[str, str]] = []
        self.download_calls: list[tuple[str, str]] = []
        self.download_closed: list[tuple[str, str]] = []
        self.upload_calls: list[tuple[str, str, tuple[str, bytes, str]]] = []
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = {'/workspace'}

        self.create_response = FakeSandboxResponse()
        self.attach_response = FakeSandboxResponse(name='sandbox-attached', id='sb-attached')
        self.ready_responses: list[FakeSandboxResponse] = []
        self.create_error: Exception | None = None
        self.get_error: Exception | None = None
        self.delete_errors: list[Exception] = []
        self.exec_error: Exception | None = None
        self.poll_error: Exception | None = None
        self.download_error: Exception | None = None
        self.upload_error: Exception | None = None
        self.responder: Responder = _default_responder
        self.create_delay = 0.0
        self.exec_delay = 0.0
        self.poll_delays: list[float] = []
        self._next_exec = 0
        self._exec_results: dict[str, list[FakeExecResult]] = {}

    async def create_sandbox(self, **kwargs: object) -> FakeSandboxResponse:
        self.create_calls.append(dict(kwargs))
        if self.create_delay:
            await anyio.sleep(self.create_delay)
        if self.create_error is not None:
            raise self.create_error
        return self.create_response

    async def get_sandbox(self, name: str) -> FakeSandboxResponse:
        self.get_calls.append(name)
        if self.get_error is not None:
            raise self.get_error
        if name == self.attach_response.name and self.get_calls.count(name) == 1:
            response = self.attach_response
        elif self.ready_responses:
            return self.ready_responses.pop(0)
        else:
            response = self.attach_response if name == self.attach_response.name else self.create_response
        return FakeSandboxResponse(name=name, id=response.id, status=response.status, workdir=response.workdir)

    async def delete_sandbox(self, name: str) -> None:
        self.delete_calls.append(name)
        if self.delete_errors:
            raise self.delete_errors.pop(0)

    async def exec_in_sandbox(
        self,
        sandbox_name: str,
        *,
        command: list[str],
        workdir: str | None = None,
        timeout_secs: int | None = None,
    ) -> FakeExecStarted:
        call = ExecCall(sandbox_name, command, workdir, timeout_secs)
        self.exec_calls.append(call)
        if self.exec_delay:
            await anyio.sleep(self.exec_delay)
        if self.exec_error is not None:
            raise self.exec_error

        self._next_exec += 1
        exec_id = f'exec-{self._next_exec}'
        if len(command) >= 5 and command[:2] == ['sh', '-c'] and command[3] == 'islo-list-directory':
            response: FakeExecResult | list[FakeExecResult] = self._directory_result(command[4])
        else:
            response = self.responder(call)
        self._exec_results[exec_id] = list(response) if isinstance(response, list) else [response]
        return FakeExecStarted(exec_id)

    async def get_exec_result(self, sandbox_name: str, exec_id: str) -> FakeExecResult:
        self.exec_result_calls.append((sandbox_name, exec_id))
        if self.poll_delays:
            await anyio.sleep(self.poll_delays.pop(0))
        if self.poll_error is not None:
            raise self.poll_error
        responses = self._exec_results[exec_id]
        if len(responses) > 1:
            return responses.pop(0)
        return responses[0]

    async def download_file(self, sandbox_name: str, *, path: str):  # type: ignore[no-untyped-def]
        self.download_calls.append((sandbox_name, path))
        try:
            if self.download_error is not None:
                raise self.download_error
            data = self.files[path]
            midpoint = max(1, len(data) // 2)
            for chunk in (data[:midpoint], data[midpoint:]):
                if chunk:
                    yield chunk
        finally:
            self.download_closed.append((sandbox_name, path))

    async def upload_file(
        self,
        sandbox_name: str,
        *,
        path: str,
        file: tuple[str, bytes, str],
    ) -> None:
        self.upload_calls.append((sandbox_name, path, file))
        if self.upload_error is not None:
            raise self.upload_error
        self.files[path] = file[1]
        self._add_parent_directories(path)

    def _directory_result(self, target: str) -> FakeExecResult:
        target = posixpath.normpath(target)
        if target not in self.directories:
            return FakeExecResult(stderr=f'not an accessible directory: {target}\n', exit_code=1)
        prefix = target.rstrip('/') + '/'
        entries: dict[str, bool] = {}
        for path in self.files:
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix) :]
            if not relative:
                continue
            name, separator, _ = relative.partition('/')
            entries[name] = bool(separator) or entries.get(name, False)
        stdout = ''.join(f'{"d" if is_dir else "f"}\t{name}\n' for name, is_dir in entries.items())
        return FakeExecResult(stdout=stdout)

    def _add_parent_directories(self, path: str) -> None:
        parent = posixpath.dirname(path)
        while parent and parent != '/':
            self.directories.add(parent)
            parent = posixpath.dirname(parent)


class FakeAsyncIslo:
    """SDK root client exposed by the fake `islo` module."""

    control: FakeIslo

    def __init__(self, **kwargs: object) -> None:
        self.control.client_init_calls.append(dict(kwargs))
        self.sandboxes = self.control.sandboxes


class FakeIslo:
    """Control surface and module tree installed into `sys.modules`."""

    def __init__(self) -> None:
        self.client_init_calls: list[dict[str, object]] = []
        self.sandboxes = FakeSandboxesClient(self)
        FakeAsyncIslo.control = self

        module = types.ModuleType('islo')
        module.__path__ = []  # type: ignore[attr-defined]
        module.AsyncIslo = FakeAsyncIslo  # type: ignore[attr-defined]
        self.module = module

        types_module = types.ModuleType('islo.types')
        types_module.LifecyclePolicy = FakeLifecyclePolicy  # type: ignore[attr-defined]
        self.types_module = types_module

        core_module = types.ModuleType('islo.core')
        core_module.__path__ = []  # type: ignore[attr-defined]
        self.core_module = core_module

        api_error_module = types.ModuleType('islo.core.api_error')
        api_error_module.ApiError = FakeApiError  # type: ignore[attr-defined]
        self.api_error_module = api_error_module

    def install(self, modules: dict[str, Any]) -> None:
        """Install the complete lazy-import surface into a module mapping."""
        modules['islo'] = self.module
        modules['islo.types'] = self.types_module
        modules['islo.core'] = self.core_module
        modules['islo.core.api_error'] = self.api_error_module

    def put_file(self, path: str, data: bytes) -> None:
        """Seed a file at a normalized absolute path."""
        normalized = posixpath.normpath(path)
        self.sandboxes.files[normalized] = data
        self.sandboxes._add_parent_directories(normalized)

    def add_directory(self, path: str) -> None:
        """Seed an empty directory and all of its parents."""
        normalized = posixpath.normpath(path)
        self.sandboxes.directories.add(normalized)
        self.sandboxes._add_parent_directories(f'{normalized}/child')

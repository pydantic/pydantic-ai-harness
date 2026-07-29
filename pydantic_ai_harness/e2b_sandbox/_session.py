"""Lifecycle and I/O for an E2B sandbox.

External assumptions last verified 2026-07-24:

* `AsyncSandbox.create`, `connect`, and `kill` provide the owned/attached lifecycle:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py
* foreground command results buffer all output, so bounded command capture must happen
  inside the sandbox before the SDK receives it:
  https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command_handle.py
* the default `base` template contains Bash, GNU coreutils, and `setsid` (util-linux),
  which the bounded capture wrapper uses:
  https://github.com/e2b-dev/E2B/blob/main/templates/base/e2b.Dockerfile

Re-check these sources before changing lifecycle, command, or template assumptions.
"""

from __future__ import annotations

import importlib
import posixpath
import shlex
import warnings
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Protocol, overload, runtime_checkable
from uuid import uuid4

import anyio
import anyio.lowlevel
from opentelemetry import trace
from opentelemetry.trace import Tracer
from typing_extensions import Self

DEFAULT_SANDBOX_TIMEOUT = 300
DEFAULT_WORKDIR = '/home/user'

_CREATE_TIMEOUT = 120
_TEARDOWN_TIMEOUT = 30
_INTERNAL_COMMAND_TIMEOUT = 10
_TIMEOUT_EXIT_CODE = 124

_MISSING_E2B = (
    "The 'e2b' package is required for E2BSandbox. Install it alongside Harness with "
    '`uv add pydantic-ai-harness "e2b>=2.34.0"`.'
)
_AUTH_MESSAGE = 'E2B rejected the credentials. Set a valid E2B_API_KEY in the environment.'


class _CommandResult(Protocol):  # pragma: no cover - structural typing only
    stdout: str
    stderr: str
    exit_code: int
    error: str | None


@runtime_checkable
class _CommandFailure(Protocol):  # pragma: no cover - structural typing only
    stdout: str
    stderr: str
    exit_code: int
    error: str | None


class _CommandHandle(Protocol):  # pragma: no cover - structural typing only
    pid: int

    async def wait(self) -> _CommandResult: ...

    async def kill(self) -> bool: ...


class _Commands(Protocol):  # pragma: no cover - structural typing only
    @overload
    async def run(
        self,
        command: str,
        *,
        background: Literal[True],
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> _CommandHandle: ...

    @overload
    async def run(
        self,
        command: str,
        *,
        background: Literal[False] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> _CommandResult: ...


class _FileType(Protocol):  # pragma: no cover - structural typing only
    value: str


class _EntryInfo(Protocol):  # pragma: no cover - structural typing only
    name: str
    path: str
    type: _FileType | None
    size: int


class _AsyncFileStream(Protocol):  # pragma: no cover - structural typing only
    def __aiter__(self) -> AsyncIterator[bytes]: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *args: object) -> None: ...


class _Filesystem(Protocol):  # pragma: no cover - structural typing only
    async def get_info(self, path: str) -> _EntryInfo: ...

    @overload
    async def read(self, path: str, format: Literal['bytes']) -> bytearray: ...

    @overload
    async def read(
        self,
        path: str,
        format: Literal['stream'],
        *,
        stream_idle_timeout: float | None = None,
    ) -> _AsyncFileStream: ...

    async def write(self, path: str, data: str | bytes) -> object: ...

    async def list(self, path: str, depth: int | None = 1) -> list[_EntryInfo]: ...

    async def remove(self, path: str) -> None: ...


class _AsyncSandbox(Protocol):  # pragma: no cover - structural typing only
    sandbox_id: str
    commands: _Commands
    files: _Filesystem

    async def kill(self) -> bool: ...


class _AsyncSandboxFactory(Protocol):  # pragma: no cover - structural typing only
    async def create(
        self,
        template: str | None = None,
        timeout: int | None = None,
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
    ) -> _AsyncSandbox: ...

    async def connect(self, sandbox_id: str, timeout: int | None = None) -> _AsyncSandbox: ...


@runtime_checkable
class _E2BModule(Protocol):  # pragma: no cover - structural typing only
    AsyncSandbox: _AsyncSandboxFactory
    AuthenticationException: type[Exception]
    CommandExitException: type[Exception]
    FileNotFoundException: type[Exception]
    SandboxException: type[Exception]
    SandboxNotFoundException: type[Exception]
    TimeoutException: type[Exception]


def _load_e2b() -> _E2BModule:
    try:
        module: ModuleType = importlib.import_module('e2b')
    except ImportError as e:
        raise E2BSandboxError(_MISSING_E2B) from e
    if not isinstance(module, _E2BModule):  # pragma: no cover - incompatible third-party package
        raise E2BSandboxError('The installed `e2b` package does not expose the expected async sandbox API.')
    return module


class E2BSandboxError(RuntimeError):
    """Base class for failures reported by the E2B sandbox integration."""


class E2BSandboxTerminalError(E2BSandboxError):
    """A sandbox failure that a model retry cannot repair."""


class E2BSandboxUnavailableError(E2BSandboxTerminalError):
    """The E2B sandbox does not exist or is no longer running."""


class E2BSandboxAuthError(E2BSandboxTerminalError):
    """E2B rejected the configured credentials."""


class _E2BSandboxReadTooLargeError(E2BSandboxError):
    def __init__(self, *, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(f'File grew beyond the {max_bytes}-byte read limit.')


@dataclass(frozen=True, kw_only=True)
class E2BSandboxExecResult:
    """The bounded outcome of running a command in an E2B sandbox."""

    stdout: str
    stderr: str
    returncode: int
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    applied_timeout: int | None = None


def _capture_wrapper(temp_dir: str, max_output_bytes: int) -> str:
    """Build a Bash wrapper that retains a bounded tail and counts total bytes."""
    command_path = shlex.quote(posixpath.join(temp_dir, 'command.sh'))
    stdout_path = shlex.quote(posixpath.join(temp_dir, 'stdout'))
    stderr_path = shlex.quote(posixpath.join(temp_dir, 'stderr'))
    stdout_stream = shlex.quote(posixpath.join(temp_dir, 'stdout.stream'))
    stderr_stream = shlex.quote(posixpath.join(temp_dir, 'stderr.stream'))
    stdout_count_stream = shlex.quote(posixpath.join(temp_dir, 'stdout.count.stream'))
    stderr_count_stream = shlex.quote(posixpath.join(temp_dir, 'stderr.count.stream'))
    stdout_count = shlex.quote(posixpath.join(temp_dir, 'stdout.count'))
    stderr_count = shlex.quote(posixpath.join(temp_dir, 'stderr.count'))
    limit = str(max_output_bytes)
    return f"""#!/bin/bash
set +e
mkfifo {stdout_stream} {stderr_stream} {stdout_count_stream} {stderr_count_stream} || exit 125
wc -c < {stdout_count_stream} > {stdout_count} &
stdout_count_pid=$!
tee {stdout_count_stream} < {stdout_stream} | tail -c {limit} > {stdout_path} &
stdout_capture_pid=$!
wc -c < {stderr_count_stream} > {stderr_count} &
stderr_count_pid=$!
tee {stderr_count_stream} < {stderr_stream} | tail -c {limit} > {stderr_path} &
stderr_capture_pid=$!
/bin/bash {command_path} > {stdout_stream} 2> {stderr_stream}
command_status=$?
wait "$stdout_capture_pid" "$stderr_capture_pid" "$stdout_count_pid" "$stderr_count_pid"
capture_status=$?
if [ "$capture_status" -ne 0 ]; then
  exit 125
fi
exit "$command_status"
"""


class E2BSandboxSession:
    """Async context manager that creates or attaches to an E2B sandbox.

    Owned sessions create a sandbox on enter and kill it on exit. Attached
    sessions connect to `sandbox_id` and leave it running. The E2B async SDK uses
    `asyncio.create_task` for command handles, so this session requires asyncio.
    """

    def __init__(
        self,
        *,
        template: str | None = None,
        sandbox_id: str | None = None,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        workdir: str = DEFAULT_WORKDIR,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        if type(sandbox_timeout) is not int or sandbox_timeout <= 0:
            raise ValueError(f'sandbox_timeout must be a positive integer, got {sandbox_timeout!r}.')
        if not workdir or not posixpath.isabs(workdir):
            raise ValueError(f'workdir must be an absolute sandbox path, got {workdir!r}.')
        if type(allow_internet_access) is not bool:
            raise ValueError(f'allow_internet_access must be a boolean, got {allow_internet_access!r}.')
        if sandbox_id is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('template', template, None),
                    ('sandbox_timeout', sandbox_timeout, DEFAULT_SANDBOX_TIMEOUT),
                    ('env', env, None),
                    ('metadata', metadata, None),
                    ('allow_internet_access', allow_internet_access, True),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_id` attaches '
                    'to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
                )
        self._template = template
        self._sandbox_id = sandbox_id
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._metadata = dict(metadata) if metadata is not None else None
        self._allow_internet_access = allow_internet_access
        self._tracer = tracer or trace.get_tracer('pydantic_ai_harness.e2b_sandbox')
        self._sandbox: _AsyncSandbox | None = None

    @property
    def sandbox_id(self) -> str | None:
        """The running sandbox id, or None while the session is closed."""
        return self._sandbox.sandbox_id if self._sandbox is not None else None

    @property
    def template(self) -> str | None:
        """The configured template for an owned sandbox."""
        return self._template

    @property
    def mode(self) -> Literal['owned', 'attached']:
        """Whether this session owns the sandbox lifecycle."""
        return 'attached' if self._sandbox_id is not None else 'owned'

    @property
    def workdir(self) -> str:
        """Working directory shared by command and relative file operations."""
        return self._workdir

    async def __aenter__(self) -> Self:
        """Create or connect to the configured sandbox."""
        if self._sandbox is not None:
            raise E2BSandboxError(
                'The session is already open; exit it before entering again. '
                'Use a separate session per concurrent context.'
            )
        operation = 'connect' if self._sandbox_id is not None else 'create'
        with self._tracer.start_as_current_span(f'e2b.sandbox.{operation}') as span:
            span.set_attribute('e2b.sandbox.mode', self.mode)
            if self._template is not None:
                span.set_attribute('e2b.sandbox.template', self._template)
            try:
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(_CREATE_TIMEOUT):
                        self._sandbox = await self._open_sandbox()
            except Exception as e:
                error = self._operation_error(e, f'Could not {operation} E2B sandbox')
                span.set_attribute('e2b.outcome', 'error')
                span.set_attribute('e2b.exception_type', type(e).__name__)
                raise error from e
            if self._sandbox is None:
                span.set_attribute('e2b.outcome', 'timeout')
                raise E2BSandboxError(
                    f'E2B sandbox {operation} did not complete within {_CREATE_TIMEOUT}s; '
                    'the E2B control plane may be unreachable.'
                )
            span.set_attribute('e2b.sandbox.id', self._sandbox.sandbox_id)
            span.set_attribute('e2b.outcome', 'success')
        try:
            await anyio.lowlevel.checkpoint()
        except BaseException as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            raise
        return self

    async def _open_sandbox(self) -> _AsyncSandbox:
        e2b = _load_e2b()
        if self._sandbox_id is not None:
            return await e2b.AsyncSandbox.connect(self._sandbox_id)
        return await e2b.AsyncSandbox.create(
            template=self._template,
            timeout=self._sandbox_timeout,
            metadata=self._metadata,
            envs=self._env,
            secure=True,
            allow_internet_access=self._allow_internet_access,
        )

    async def __aexit__(self, *args: object) -> None:
        """Kill an owned sandbox; leave an attached sandbox running."""
        body_failed = bool(args and args[0] is not None)
        try:
            await self.close()
        except E2BSandboxError as e:
            if not body_failed:
                raise
            warnings.warn(f'Could not clean up the owned E2B sandbox: {e}', RuntimeWarning, stacklevel=2)

    async def close(self) -> None:
        """Close the session, preserving sandbox identity when owned cleanup fails."""
        sandbox = self._sandbox
        if sandbox is None:
            return
        if self._sandbox_id is not None:
            self._sandbox = None
            return
        with self._tracer.start_as_current_span('e2b.sandbox.kill') as span:
            span.set_attribute('e2b.sandbox.id', sandbox.sandbox_id)
            span.set_attribute('e2b.sandbox.mode', 'owned')
            try:
                with anyio.CancelScope(shield=True):
                    with anyio.fail_after(_TEARDOWN_TIMEOUT):
                        await sandbox.kill()
            except Exception as e:
                span.set_attribute('e2b.outcome', 'error')
                span.set_attribute('e2b.exception_type', type(e).__name__)
                raise self._operation_error(e, 'Could not kill the owned E2B sandbox') from e
            span.set_attribute('e2b.outcome', 'success')
            self._sandbox = None

    def _require_sandbox(self) -> _AsyncSandbox:
        if self._sandbox is None:
            raise E2BSandboxError('The E2B sandbox session is not open.')
        return self._sandbox

    def _operation_error(self, e: Exception, context: str) -> E2BSandboxError:
        if isinstance(e, E2BSandboxError):
            return e
        e2b = _load_e2b()
        if isinstance(e, e2b.AuthenticationException):
            return E2BSandboxAuthError(_AUTH_MESSAGE)
        if isinstance(e, e2b.SandboxNotFoundException):
            sandbox_id = self.sandbox_id or self._sandbox_id
            suffix = f' {sandbox_id!r}' if sandbox_id is not None else ''
            return E2BSandboxUnavailableError(f'The E2B sandbox{suffix} is no longer available.')
        return E2BSandboxError(f'{context}: {type(e).__name__}: {e}')

    def _resolve(self, path: str) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(self._workdir, path))

    async def exec(
        self,
        command: str,
        *,
        timeout: int,
        max_output_bytes: int,
    ) -> E2BSandboxExecResult:
        """Run a command with bounded stdout/stderr capture inside the sandbox."""
        if type(timeout) is not int or timeout <= 0:
            raise ValueError(f'timeout must be a positive integer, got {timeout!r}.')
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError(f'max_output_bytes must be a positive integer, got {max_output_bytes!r}.')
        try:
            command.encode('utf-8')
        except UnicodeEncodeError as e:
            raise E2BSandboxError('Command contains characters that cannot be encoded as UTF-8.') from e

        sandbox = self._require_sandbox()
        e2b = _load_e2b()
        temp_dir = f'/tmp/pydantic-ai-harness-{uuid4().hex}'
        command_path = posixpath.join(temp_dir, 'command.sh')
        wrapper_path = posixpath.join(temp_dir, 'capture.sh')
        handle: _CommandHandle | None = None
        sdk_stdout = ''
        sdk_stderr = ''
        returncode = 0
        timed_out = False

        try:
            await sandbox.files.write(command_path, command)
            await sandbox.files.write(wrapper_path, _capture_wrapper(temp_dir, max_output_bytes))
            run = f'exec setsid /bin/bash {shlex.quote(wrapper_path)}'
            handle = await sandbox.commands.run(run, background=True, cwd=self._workdir, timeout=timeout)
            try:
                result = await handle.wait()
                returncode = result.exit_code
                sdk_stdout = result.stdout
                sdk_stderr = result.stderr
            except Exception as e:
                if isinstance(e, e2b.CommandExitException):
                    if not isinstance(e, _CommandFailure):  # pragma: no cover - incompatible SDK exception
                        raise E2BSandboxError('E2B returned an incomplete command failure.') from e
                    returncode = e.exit_code
                    sdk_stdout = e.stdout
                    sdk_stderr = e.stderr
                elif isinstance(e, e2b.TimeoutException):
                    timed_out = True
                    returncode = _TIMEOUT_EXIT_CODE
                    await self._kill_command(handle)
                else:
                    raise self._operation_error(e, 'Could not read the E2B command result') from e
        except BaseException as e:
            if handle is not None and not timed_out:
                await self._kill_command(handle)
            with anyio.CancelScope(shield=True):
                await self._remove_temp_dir(temp_dir)
            if isinstance(e, Exception) and not isinstance(e, E2BSandboxError):
                raise self._operation_error(e, 'Could not run E2B sandbox command') from e
            raise

        try:
            stdout, stdout_truncated = await self._read_capture(
                temp_dir,
                'stdout',
                max_output_bytes,
                incomplete_ok=timed_out,
            )
            stderr, stderr_truncated = await self._read_capture(
                temp_dir,
                'stderr',
                max_output_bytes,
                incomplete_ok=timed_out,
            )
        except Exception as e:
            raise self._operation_error(e, 'Could not read bounded E2B command output') from e
        finally:
            await self._remove_temp_dir(temp_dir)

        if sdk_stdout:
            stdout = f'{stdout}\n{sdk_stdout}' if stdout else sdk_stdout
        if sdk_stderr:
            stderr = f'{stderr}\n{sdk_stderr}' if stderr else sdk_stderr
        return E2BSandboxExecResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            applied_timeout=timeout,
        )

    async def _kill_command(self, handle: _CommandHandle) -> None:
        sandbox = self._require_sandbox()
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(_INTERNAL_COMMAND_TIMEOUT):
                try:
                    await sandbox.commands.run(
                        f'kill -KILL -- -{handle.pid}',
                        cwd=self._workdir,
                        timeout=_INTERNAL_COMMAND_TIMEOUT,
                    )
                except Exception:
                    pass
            with anyio.move_on_after(_INTERNAL_COMMAND_TIMEOUT):
                try:
                    await handle.kill()
                except Exception:
                    pass

    async def _read_capture(
        self,
        temp_dir: str,
        stream: str,
        max_output_bytes: int,
        *,
        incomplete_ok: bool,
    ) -> tuple[str, bool]:
        sandbox = self._require_sandbox()
        e2b = _load_e2b()
        try:
            data = bytes(await sandbox.files.read(posixpath.join(temp_dir, stream), 'bytes'))
        except Exception as e:
            if incomplete_ok and isinstance(e, e2b.FileNotFoundException):
                return '', False
            raise
        bounded = data[-max_output_bytes:]
        decoded = bounded.decode('utf-8', errors='replace')
        try:
            count_data = bytes(await sandbox.files.read(posixpath.join(temp_dir, f'{stream}.count'), 'bytes'))
        except Exception as e:
            if incomplete_ok and isinstance(e, e2b.FileNotFoundException):
                return decoded, len(data) >= max_output_bytes
            raise
        try:
            total_bytes = int(count_data.decode().strip())
        except (UnicodeDecodeError, ValueError) as e:
            if incomplete_ok:
                return decoded, len(data) >= max_output_bytes
            raise E2BSandboxError(f'E2B command {stream} byte count was invalid.') from e
        return decoded, total_bytes > len(bounded)

    async def _remove_temp_dir(self, temp_dir: str) -> None:
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(_INTERNAL_COMMAND_TIMEOUT):
                try:
                    await self._require_sandbox().files.remove(temp_dir)
                except Exception:
                    pass

    async def file_size(self, path: str) -> int:
        """Return the size of a sandbox file without reading it."""
        try:
            return (await self._require_sandbox().files.get_info(self._resolve(path))).size
        except Exception as e:
            raise self._operation_error(e, f'Could not inspect sandbox file {path!r}') from e

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes:
        """Stream a file into a bounded buffer."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError(f'max_bytes must be a positive integer, got {max_bytes!r}.')
        target = self._resolve(path)
        try:
            stream = await self._require_sandbox().files.read(
                target,
                'stream',
                stream_idle_timeout=_INTERNAL_COMMAND_TIMEOUT,
            )
            data = bytearray()
            async with stream:
                async for chunk in stream:
                    remaining = max_bytes + 1 - len(data)
                    if remaining > 0:  # pragma: no branch - we raise as soon as max_bytes + 1 is retained
                        data.extend(chunk[:remaining])
                    if len(data) > max_bytes:
                        raise _E2BSandboxReadTooLargeError(size_bytes=len(data), max_bytes=max_bytes)
            return bytes(data)
        except _E2BSandboxReadTooLargeError:
            raise
        except Exception as e:
            raise self._operation_error(e, f'Could not read sandbox file {path!r}') from e

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a sandbox file, creating parent directories."""
        try:
            await self._require_sandbox().files.write(self._resolve(path), data)
        except Exception as e:
            raise self._operation_error(e, f'Could not write sandbox file {path!r}') from e

    async def list_files(self, path: str) -> list[tuple[str, bool]]:
        """List a sandbox directory as `(name, is_dir)` pairs."""
        try:
            entries = await self._require_sandbox().files.list(self._resolve(path), depth=1)
        except Exception as e:
            raise self._operation_error(e, f'Could not list sandbox directory {path!r}') from e
        return [(entry.name, entry.type is not None and entry.type.value == 'dir') for entry in entries]

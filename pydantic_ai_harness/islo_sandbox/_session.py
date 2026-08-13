"""Lifecycle and I/O for an Islo sandbox.

External assumptions verified 2026-08-13:

* `AsyncIslo` exposes create/get/delete, async exec polling, and streaming file
  download/upload. Re-check against https://github.com/islo-labs/python-sdk.
* Sandbox lifecycle `delete_after` is enforced by the Islo control plane and is
  used as the cleanup backstop. Re-check against
  https://docs.islo.dev/concepts/sandbox-lifecycle.
* `timeout_secs` is currently a compatibility hint, not a documented
  server-enforced command deadline. A client timeout is therefore reported as
  potentially leaving the remote process running. Re-check against
  https://docs.islo.dev/api-reference/sandboxes/exec-in-sandbox.
* The command-result API reports one provider-level `truncated` flag for both
  streams and does not document its transport retention limit. Re-check against
  https://docs.islo.dev/api-reference/sandboxes/get-exec-result.
"""

from __future__ import annotations

import math
import posixpath
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import anyio
import anyio.lowlevel
import httpx
from typing_extensions import NotRequired, Self, TypedDict

if TYPE_CHECKING:
    from islo import AsyncIslo
    from islo.sandboxes.client import AsyncSandboxesClient
    from islo.types import LifecyclePolicy

DEFAULT_IMAGE = 'ghcr.io/islo-labs/islo-runner:latest'
DEFAULT_SANDBOX_TIMEOUT = 900
DEFAULT_WORKDIR = '/workspace'

_CREATE_TIMEOUT = 180.0
_TEARDOWN_TIMEOUT = 30.0
_INTERNAL_COMMAND_TIMEOUT = 10.0
_TERMINAL_EXEC_STATUSES = frozenset({'completed', 'failed', 'timeout'})
_TERMINAL_SANDBOX_STATUSES = frozenset({'deleted', 'failed', 'paused', 'stopped'})
_MISSING_ISLO = (
    'The \'islo\' package is required for IsloSandbox. Install it with `uv add "pydantic-ai-harness[islo]"`.'
)
_AUTH_MESSAGE = 'Islo rejected the credentials. Set ISLO_API_KEY to a valid Islo API key.'


@runtime_checkable
class _ClosableAsyncIterator(Protocol):
    async def aclose(self) -> None: ...  # pragma: no cover - structural typing only


class _CreateSandboxKwargs(TypedDict):
    """Create fields with unset values absent instead of encoded as JSON null."""

    image: str
    lifecycle: LifecyclePolicy
    env: NotRequired[dict[str, str | None]]
    workdir: NotRequired[str]
    vcpus: NotRequired[int]
    memory_mb: NotRequired[int]
    disk_gb: NotRequired[int]
    internet_enabled: NotRequired[bool]
    gateway_profile: NotRequired[str]


class IsloSandboxError(RuntimeError):
    """Base class for failures from the Islo sandbox integration."""


class IsloSandboxTerminalError(IsloSandboxError):
    """An Islo failure that retrying the same tool call cannot repair."""


class IsloSandboxUnavailableError(IsloSandboxTerminalError):
    """The target sandbox no longer exists or cannot execute further work."""


class IsloSandboxAuthError(IsloSandboxTerminalError):
    """Islo rejected the configured credentials."""


@dataclass(frozen=True, kw_only=True)
class IsloSandboxExecResult:
    """The normalized outcome of one Islo sandbox command.

    `status` preserves Islo's terminal state, or is `client_timeout` when the
    Harness deadline expires. The truncation flags combine provider and local
    limits. `remote_may_be_running` distinguishes a client deadline from a
    confirmed provider timeout because Islo does not currently expose command
    cancellation.
    """

    stdout: str
    stderr: str
    returncode: int
    status: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    applied_timeout: int | None = None
    remote_may_be_running: bool = False


def _tail_utf8(text: str, max_bytes: int | None) -> tuple[str, bool]:
    """Retain a UTF-8 byte suffix without returning a partial leading character."""
    if max_bytes is None:
        return text, False
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[-max_bytes:].decode('utf-8', errors='ignore'), True


class IsloSandboxSession:
    """Async context manager that creates or attaches to an Islo sandbox.

    With no `sandbox_name`, entering creates a fresh sandbox and exiting deletes
    it. `delete_after` is also configured as a server-side cleanup backstop. Set
    `sandbox_name` to attach to a sandbox managed elsewhere; attached sandboxes
    are left running.

    The public Islo SDK uses asyncio for token refresh, so this session requires
    an asyncio event loop.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        sandbox_name: str | None = None,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        workdir: str | None = DEFAULT_WORKDIR,
        env: Mapping[str, str] | None = None,
        vcpus: int | None = None,
        memory_mb: int | None = None,
        disk_gb: int | None = None,
        internet_enabled: bool | None = None,
        gateway_profile: str | None = None,
        base_url: str | None = None,
        compute_url: str | None = None,
        client: AsyncIslo | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        if type(sandbox_timeout) is not int or sandbox_timeout <= 0:
            raise ValueError(f'sandbox_timeout must be a positive integer, got {sandbox_timeout!r}.')
        if type(poll_interval) not in {int, float} or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError(f'poll_interval must be a positive finite number, got {poll_interval!r}.')
        self._image = image
        self._attach_name = sandbox_name
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._env: dict[str, str | None] | None = dict(env) if env is not None else None
        self._vcpus = vcpus
        self._memory_mb = memory_mb
        self._disk_gb = disk_gb
        self._internet_enabled = internet_enabled
        self._gateway_profile = gateway_profile
        self._base_url = base_url
        self._compute_url = compute_url
        self._injected_client = client
        self._poll_interval = poll_interval

        self._client: AsyncIslo | None = None
        self._sandboxes: AsyncSandboxesClient | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._sandbox_name: str | None = None
        self._sandbox_id: str | None = None
        self._cwd: str | None = None
        self._owned = False

    @property
    def sandbox_name(self) -> str | None:
        """The active sandbox name, or None while the session is closed."""
        return self._sandbox_name

    @property
    def sandbox_id(self) -> str | None:
        """The active sandbox public ID, or None while the session is closed."""
        return self._sandbox_id

    async def __aenter__(self) -> Self:
        """Create or attach to the configured sandbox."""
        if self._sandboxes is not None:
            raise IsloSandboxError('The session is already open; exit it before entering again.')

        try:
            with anyio.CancelScope(shield=True):
                with anyio.fail_after(_CREATE_TIMEOUT):
                    await self._open()
            await anyio.lowlevel.checkpoint_if_cancelled()
        except BaseException as e:
            with anyio.CancelScope(shield=True):
                await self.close()
            if isinstance(e, TimeoutError):
                raise IsloSandboxError(
                    f'Islo sandbox creation or attachment did not become ready within {_CREATE_TIMEOUT:g}s.'
                ) from e
            raise
        return self

    async def _open(self) -> None:
        client, sandboxes = self._open_client()
        self._client = client
        self._sandboxes = sandboxes
        try:
            if self._attach_name is not None:
                response = await sandboxes.get_sandbox(self._attach_name)
                self._set_active(response.name, response.id, response.workdir)
                await self._wait_until_ready(response.status)
                return

            from islo.types import LifecyclePolicy

            lifecycle = LifecyclePolicy(delete_after=self._sandbox_timeout)
            create_kwargs = _CreateSandboxKwargs(image=self._image, lifecycle=lifecycle)
            if self._env is not None:
                create_kwargs['env'] = self._env
            if self._workdir is not None:
                create_kwargs['workdir'] = self._workdir
            if self._vcpus is not None:
                create_kwargs['vcpus'] = self._vcpus
            if self._memory_mb is not None:
                create_kwargs['memory_mb'] = self._memory_mb
            if self._disk_gb is not None:
                create_kwargs['disk_gb'] = self._disk_gb
            if self._internet_enabled is not None:
                create_kwargs['internet_enabled'] = self._internet_enabled
            if self._gateway_profile is not None:
                create_kwargs['gateway_profile'] = self._gateway_profile
            response = await sandboxes.create_sandbox(**create_kwargs)
            self._owned = True
            self._set_active(response.name, response.id, response.workdir)
            await self._wait_until_ready(response.status)
        except Exception as e:
            raise self._map_error(e, 'Could not open the Islo sandbox', unavailable_on_404=True) from e

    def _open_client(self) -> tuple[AsyncIslo, AsyncSandboxesClient]:
        try:
            from islo import AsyncIslo
        except ImportError as e:
            raise IsloSandboxError(_MISSING_ISLO) from e

        if self._injected_client is not None:
            client = self._injected_client
        else:
            self._http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
            client = AsyncIslo(
                base_url=self._base_url,
                compute_url=self._compute_url,
                httpx_client=self._http_client,
            )
        sandboxes: AsyncSandboxesClient = client.sandboxes
        return client, sandboxes

    def _set_active(self, name: str, sandbox_id: str, workdir: str | None) -> None:
        self._sandbox_name = name
        self._sandbox_id = sandbox_id
        self._cwd = workdir or self._workdir or DEFAULT_WORKDIR

    async def _wait_until_ready(self, initial_status: str) -> None:
        if initial_status == 'running':
            return
        self._ensure_sandbox_usable(initial_status)
        sandboxes, name = self._require_open()
        while True:
            await anyio.sleep(self._poll_interval)
            response = await sandboxes.get_sandbox(name)
            self._ensure_sandbox_usable(response.status)
            if response.status == 'running':
                self._cwd = response.workdir or self._cwd
                return

    @staticmethod
    def _ensure_sandbox_usable(status: str) -> None:
        if status in _TERMINAL_SANDBOX_STATUSES:
            raise IsloSandboxUnavailableError(f'Islo sandbox is {status!r} and cannot execute commands.')

    async def __aexit__(self, *args: object) -> None:
        """Delete an owned sandbox and release client resources."""
        await self.close()

    async def close(self) -> None:
        """Close the session, retrying owned cleanup when a prior call failed."""
        sandboxes = self._sandboxes
        name = self._sandbox_name
        cleanup_succeeded = not self._owned or sandboxes is None or name is None

        if self._owned and sandboxes is not None and name is not None:
            try:
                with anyio.CancelScope(shield=True):
                    with anyio.fail_after(_TEARDOWN_TIMEOUT):
                        await sandboxes.delete_sandbox(name)
                cleanup_succeeded = True
            except Exception as e:
                mapped = self._map_error(e, 'Could not delete the owned Islo sandbox', unavailable_on_404=True)
                if isinstance(mapped, IsloSandboxUnavailableError):
                    cleanup_succeeded = True
                else:
                    warnings.warn(
                        f'{mapped} Sandbox name retained for cleanup retry: {name}.', RuntimeWarning, stacklevel=2
                    )

        if not cleanup_succeeded:
            # Keep the control-plane handle and its transport alive so a later
            # `close()` can actually retry deletion. The server-side
            # `delete_after` remains the final cleanup backstop if the caller
            # does not retry.
            return

        http_client = self._http_client
        self._http_client = None
        if http_client is not None:
            with anyio.CancelScope(shield=True):
                try:
                    await http_client.aclose()
                except Exception as e:  # pragma: no cover - httpx close is local and non-failing in supported versions
                    warnings.warn(f'Could not close the Islo HTTP client: {e}', RuntimeWarning, stacklevel=2)

        self._client = None
        self._sandboxes = None
        self._cwd = None
        self._sandbox_name = None
        self._sandbox_id = None
        self._owned = False

    def _require_open(self) -> tuple[AsyncSandboxesClient, str]:
        if self._sandboxes is None or self._sandbox_name is None:
            raise IsloSandboxError('The Islo sandbox session is not open.')
        return self._sandboxes, self._sandbox_name

    def _resolve(self, path: str) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        cwd = self._cwd or DEFAULT_WORKDIR
        return posixpath.normpath(posixpath.join(cwd, path))

    async def exec(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        max_output_bytes: int | None = None,
    ) -> IsloSandboxExecResult:
        """Run an argument vector and poll until it reaches a terminal status."""
        if isinstance(argv, str):
            raise TypeError(f'argv must be a sequence of arguments, not a string; got {argv!r}.')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f'timeout must be a positive finite number, got {timeout!r}.')
        if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes <= 0):
            raise ValueError(f'max_output_bytes must be a positive integer or None, got {max_output_bytes!r}.')

        sandboxes, name = self._require_open()
        applied_timeout = max(1, math.ceil(timeout))
        deadline = time.monotonic() + timeout
        try:
            try:
                with anyio.fail_after(max(0.0, deadline - time.monotonic())):
                    started = await sandboxes.exec_in_sandbox(
                        name,
                        command=list(argv),
                        workdir=self._cwd,
                        timeout_secs=applied_timeout,
                    )
            except TimeoutError:
                return IsloSandboxExecResult(
                    stdout='',
                    stderr='',
                    returncode=-1,
                    status='client_timeout',
                    timed_out=True,
                    applied_timeout=applied_timeout,
                    remote_may_be_running=True,
                )
            last_stdout = ''
            last_stderr = ''
            provider_truncated = False
            while time.monotonic() < deadline:
                try:
                    with anyio.fail_after(max(0.0, deadline - time.monotonic())):
                        result = await sandboxes.get_exec_result(name, started.exec_id)
                except TimeoutError:
                    break
                last_stdout = result.stdout or ''
                last_stderr = result.stderr or ''
                provider_truncated = result.truncated
                if result.status in _TERMINAL_EXEC_STATUSES:
                    stdout, stdout_cut = _tail_utf8(last_stdout, max_output_bytes)
                    stderr, stderr_cut = _tail_utf8(last_stderr, max_output_bytes)
                    return IsloSandboxExecResult(
                        stdout=stdout,
                        stderr=stderr,
                        returncode=result.exit_code if result.exit_code is not None else -1,
                        status=result.status,
                        stdout_truncated=provider_truncated or stdout_cut,
                        stderr_truncated=provider_truncated or stderr_cut,
                        timed_out=result.status == 'timeout',
                        applied_timeout=applied_timeout,
                    )
                await anyio.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))

            stdout, stdout_cut = _tail_utf8(last_stdout, max_output_bytes)
            stderr, stderr_cut = _tail_utf8(last_stderr, max_output_bytes)
            return IsloSandboxExecResult(
                stdout=stdout,
                stderr=stderr,
                returncode=-1,
                status='client_timeout',
                stdout_truncated=provider_truncated or stdout_cut,
                stderr_truncated=provider_truncated or stderr_cut,
                timed_out=True,
                applied_timeout=applied_timeout,
                remote_may_be_running=True,
            )
        except Exception as e:
            raise self._map_error(e, 'Command could not run in the Islo sandbox', unavailable_on_404=True) from e

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes:
        """Read at most `max_bytes` plus one sentinel byte from a sandbox file."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError(f'max_bytes must be a positive integer, got {max_bytes!r}.')
        sandboxes, name = self._require_open()
        target = self._resolve(path)
        data = bytearray()
        chunks = sandboxes.download_file(name, path=target)
        try:
            async for chunk in chunks:
                remaining = max_bytes + 1 - len(data)
                data.extend(chunk[:remaining])
                if len(data) > max_bytes:
                    raise IsloSandboxError(f'File exceeds the {max_bytes}-byte read limit.')
            return bytes(data)
        except IsloSandboxError:
            raise
        except Exception as e:
            raise self._map_error(e, f'Could not read {target!r}', unavailable_on_404=False) from e
        finally:
            if isinstance(chunks, _ClosableAsyncIterator):  # pragma: no branch - SDK returns an async generator
                with anyio.CancelScope(shield=True):
                    await chunks.aclose()

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a sandbox file through Islo's file API."""
        sandboxes, name = self._require_open()
        target = self._resolve(path)
        try:
            await sandboxes.upload_file(
                name,
                path=target,
                file=(posixpath.basename(target), data, 'application/octet-stream'),
            )
        except Exception as e:
            raise self._map_error(e, f'Could not write {target!r}', unavailable_on_404=False) from e

    async def list_files(self, path: str) -> list[tuple[str, bool]]:
        """List one directory using the POSIX shell available in Islo runner images."""
        target = self._resolve(path)
        script = (
            'if [ ! -d "$1" ] || [ ! -r "$1" ] || [ ! -x "$1" ]; then '
            'printf "not an accessible directory: %s\\n" "$1" >&2; exit 1; fi; '
            'for entry in "$1"/.[!.]* "$1"/..?* "$1"/*; do '
            '[ -e "$entry" ] || [ -L "$entry" ] || continue; '
            'name=${entry##*/}; '
            'if [ -d "$entry" ]; then printf "d\\t%s\\n" "$name"; '
            'else printf "f\\t%s\\n" "$name"; fi; done'
        )
        result = await self.exec(
            ['sh', '-c', script, 'islo-list-directory', target],
            timeout=_INTERNAL_COMMAND_TIMEOUT,
            max_output_bytes=5 * 1024 * 1024,
        )
        if result.timed_out:
            raise IsloSandboxError(f'Listing {target!r} timed out.')
        if result.returncode != 0:
            detail = result.stderr.strip() or f'exit code {result.returncode}'
            raise IsloSandboxError(f'Could not list {target!r}: {detail}.')
        if result.stdout_truncated:
            raise IsloSandboxError(f'Directory listing for {target!r} exceeded the provider output limit.')
        entries: list[tuple[str, bool]] = []
        for line in result.stdout.splitlines():
            kind, separator, name = line.partition('\t')
            if separator != '\t' or kind not in {'d', 'f'}:
                raise IsloSandboxError(f'Could not parse directory entry returned by Islo: {line!r}.')
            entries.append((name, kind == 'd'))
        return entries

    @staticmethod
    def _map_error(e: Exception, context: str, *, unavailable_on_404: bool) -> IsloSandboxError:
        if isinstance(e, IsloSandboxError):
            return e
        try:
            from islo.core.api_error import ApiError
        except ImportError:
            return IsloSandboxError(_MISSING_ISLO)
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in {401, 403}:
            return IsloSandboxAuthError(_AUTH_MESSAGE)
        if isinstance(e, ApiError):
            if e.status_code in {401, 403}:
                return IsloSandboxAuthError(_AUTH_MESSAGE)
            if e.status_code == 404 and unavailable_on_404:
                return IsloSandboxUnavailableError(f'{context}: the sandbox was not found.')
            return IsloSandboxError(f'{context}: Islo API returned HTTP {e.status_code}: {e.body}')
        return IsloSandboxError(f'{context}: {type(e).__name__}: {e}')

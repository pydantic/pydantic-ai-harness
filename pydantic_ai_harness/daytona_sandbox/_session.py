"""Small Daytona SDK boundary used by `DaytonaSandbox`.

External assumptions, verified 2026-08-23 against Daytona Python SDK 0.198.0:

- `AsyncDaytona.create`, `get`, `delete(wait=True)`, and `close` own sandbox lifecycle.
- `sandbox.process.exec` returns `exit_code` and text in `result`.
- `sandbox.fs` provides metadata, byte upload/download, and directory listing.

Sources:
https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
https://www.daytona.io/docs/en/python-sdk/async/async-process/
https://www.daytona.io/docs/en/python-sdk/async/async-file-system/

Re-check those signatures against the lowest supported SDK before raising the
dependency ceiling.
"""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daytona import AsyncDaytona
    from daytona._async.sandbox import AsyncSandbox


DEFAULT_AUTO_STOP_MINUTES = 60


class DaytonaSandboxError(RuntimeError):
    """A Daytona sandbox operation failed."""


class DaytonaSandboxAuthError(DaytonaSandboxError):
    """Daytona rejected the configured credentials."""


class DaytonaSandboxUnavailableError(DaytonaSandboxError):
    """The requested Daytona sandbox no longer exists."""


@dataclass(frozen=True, kw_only=True)
class DaytonaSandboxExecResult:
    """The outcome of running a command in a Daytona sandbox."""

    output: str
    """The command result text returned by Daytona's direct execution API."""

    returncode: int
    """The command exit status, or `-1` when the SDK reports a timeout."""

    timed_out: bool = False
    """Whether the Daytona SDK stopped waiting at the command deadline."""


class DaytonaSandboxSession:
    """Async context manager that owns or attaches to one Daytona sandbox.

    A session without `sandbox_id` creates a sandbox from `snapshot` and deletes
    it on exit. A session with `sandbox_id` attaches to that sandbox and leaves it
    running. Pass an already-open session to `DaytonaSandbox(session=...)` to
    reuse one sandbox across several agent runs while retaining lifecycle
    ownership in the caller.

    ```python
    import asyncio

    from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxSession


    async def main() -> None:
        async with DaytonaSandboxSession() as session:
            result = await session.exec('python --version', timeout=30)
            print(result.output)


    asyncio.run(main())
    ```
    """

    def __init__(
        self,
        *,
        sandbox_id: str | None = None,
        snapshot: str | None = None,
        auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if type(auto_stop_minutes) is not int or auto_stop_minutes <= 0:
            raise ValueError(f'auto_stop_minutes must be a positive integer, got {auto_stop_minutes!r}.')
        if sandbox_id is not None and snapshot is not None:
            raise ValueError('snapshot cannot be combined with sandbox_id.')
        self._requested_id = sandbox_id
        self._snapshot = snapshot
        self._auto_stop_minutes = auto_stop_minutes
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._client: AsyncDaytona | None = None
        self._sandbox: AsyncSandbox | None = None

    @property
    def sandbox_id(self) -> str | None:
        """The ID of the open sandbox, or `None` outside the session context."""
        return self._sandbox.id if self._sandbox is not None else None

    async def __aenter__(self) -> DaytonaSandboxSession:
        if self._sandbox is not None:
            raise DaytonaSandboxError(
                'The session is already open; exit it before entering again. '
                'Use a separate session per concurrent context.'
            )
        try:
            from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
        except ImportError as error:
            raise DaytonaSandboxError(
                'The `daytona` package is required. Install it with `uv add "pydantic-ai-harness[daytona]"`.'
            ) from error

        client = AsyncDaytona()
        try:
            if self._requested_id is not None:
                sandbox = await client.get(self._requested_id)
            else:
                params = CreateSandboxFromSnapshotParams(
                    snapshot=self._snapshot,
                    env_vars=self._env,
                    auto_stop_interval=self._auto_stop_minutes,
                    auto_delete_interval=0,
                )
                sandbox = await client.create(params)
        except Exception as error:
            await client.close()
            raise _translate_error(error, unavailable=self._requested_id is not None) from error

        self._client = client
        self._sandbox = sandbox
        return self

    async def __aexit__(self, *_: object) -> None:
        client = self._client
        sandbox = self._sandbox
        if client is None or sandbox is None:
            return

        try:
            if self._requested_id is None:
                await client.delete(sandbox, timeout=60, wait=True)
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error
        finally:
            await client.close()
            self._client = None
            self._sandbox = None

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise DaytonaSandboxError('The Daytona sandbox session is not open.')
        return self._sandbox

    def _path(self, path: str) -> str:
        if self._workdir is None or posixpath.isabs(path):
            return path
        return posixpath.join(self._workdir, path)

    async def exec(self, command: str, *, timeout: int) -> DaytonaSandboxExecResult:
        """Run a command with a finite whole-second timeout and return the raw SDK result.

        This lower-level method does not apply the capability's output limits.
        """
        if type(timeout) is not int or timeout <= 0:
            raise ValueError(f'timeout must be a positive integer, got {timeout!r}.')
        sandbox = self._require_sandbox()
        try:
            response = await sandbox.process.exec(
                command,
                cwd=self._workdir,
                env=self._env,
                timeout=timeout,
            )
        except Exception as error:
            try:
                from daytona import DaytonaTimeoutError
            except ImportError:  # pragma: no cover - the session already imported Daytona
                raise DaytonaSandboxError('The Daytona SDK became unavailable.') from error
            if isinstance(error, DaytonaTimeoutError):
                return DaytonaSandboxExecResult(output='', returncode=-1, timed_out=True)
            raise _translate_error(error, unavailable=True) from error
        return DaytonaSandboxExecResult(output=response.result, returncode=response.exit_code)

    async def file_size(self, path: str) -> int:
        try:
            return (await self._require_sandbox().fs.get_file_info(self._path(path))).size
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error

    async def read_bytes(self, path: str) -> bytes:
        try:
            data = await self._require_sandbox().fs.download_file(self._path(path))
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error
        return data

    async def write_bytes(self, path: str, data: bytes) -> None:
        sandbox = self._require_sandbox()
        resolved = self._path(path)
        parent = posixpath.dirname(resolved)
        try:
            if parent not in ('', '.', '/'):
                mkdir = await sandbox.process.exec(f'mkdir -p -- {shlex.quote(parent)}', timeout=30)
                if mkdir.exit_code != 0:
                    raise DaytonaSandboxError(mkdir.result or f'Could not create {parent!r}.')
            await sandbox.fs.upload_file(data, resolved)
        except DaytonaSandboxError:
            raise
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error

    async def list_files(self, path: str) -> list[tuple[str, bool]]:
        try:
            entries = await self._require_sandbox().fs.list_files(self._path(path))
        except Exception as error:
            raise _translate_error(error, unavailable=False) from error
        return [(entry.name, entry.is_dir) for entry in entries]


def _translate_error(error: Exception, *, unavailable: bool) -> DaytonaSandboxError:
    """Map SDK failures without leaking SDK types through the public API."""
    try:
        from daytona import (
            DaytonaAuthenticationError,
            DaytonaAuthorizationError,
            DaytonaNotFoundError,
        )
    except ImportError:  # pragma: no cover - the session already imported Daytona
        return DaytonaSandboxError(str(error))

    if isinstance(error, (DaytonaAuthenticationError, DaytonaAuthorizationError)):
        return DaytonaSandboxAuthError('Daytona rejected the credentials. Set DAYTONA_API_KEY and try again.')
    if unavailable and isinstance(error, DaytonaNotFoundError):
        return DaytonaSandboxUnavailableError('The Daytona sandbox does not exist or is no longer available.')
    return DaytonaSandboxError(str(error))

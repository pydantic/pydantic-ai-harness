"""Lifecycle and execution boundary for the embedded Belgie runtime."""

from __future__ import annotations

import asyncio
import math
import sys
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import TYPE_CHECKING

import anyio
from typing_extensions import Self

if TYPE_CHECKING:
    from belgie import JsonOutput, Runtime, RuntimePermissions  # pyright: ignore[reportMissingTypeStubs]
    from belgie._core import AsyncRuntime  # pyright: ignore[reportMissingTypeStubs]

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_OLD_GENERATION_SIZE_MB = 128
DEFAULT_RENDER_SPECIFIER = 'npm:@belgie/render'

_MISSING_BELGIE = (
    'Belgie Sandbox requires Belgie and Python 3.12-3.14. Install it with `uv add "pydantic-ai-harness[belgie]"`.'
)
_DEFAULT_RENDER_SYS_PERMISSIONS = ('homedir', 'uid', 'gid', 'cpus', 'osRelease', 'systemMemoryInfo')
_DEFAULT_RENDER_READ_PATHS = (
    ()
    if sys.platform == 'win32'
    else (
        '/etc',
        '/proc',
        '/usr/bin/ldd',
    )
)


class BelgieSandboxError(RuntimeError):
    """Base error for Belgie Sandbox configuration and lifecycle failures."""


class BelgieSandboxExecutionError(BelgieSandboxError):
    """A script failed inside the Belgie runtime."""


class BelgieSandboxTimeoutError(BelgieSandboxExecutionError):
    """A script exceeded its configured execution timeout."""


class BelgieSandboxUnavailableError(BelgieSandboxError):
    """The Belgie runtime could not be imported or started."""


async def _drain_cancelled_task(task: asyncio.Task[JsonOutput]) -> None:
    task.cancel()
    with suppress(BaseException):
        await task


class BelgieSandboxSession:
    """An async context manager around one Belgie environment and runtime.

    The default session owns a temporary dependency environment and a restricted
    Deno worker. It can execute multiple independent `Script` bindings while open;
    runtime-global JavaScript state can therefore persist between calls.

    Pass an existing `belgie.Runtime` to take full control of its environment and
    permissions. The session enters and exits that runtime but does not otherwise
    alter its configuration.
    """

    def __init__(
        self,
        *,
        allow_package_imports: bool = False,
        allow_network: bool = False,
        enable_rendering: bool = False,
        max_old_generation_size_mb: int | None = DEFAULT_MAX_OLD_GENERATION_SIZE_MB,
        runtime: Runtime | None = None,
    ) -> None:
        for name, value in (
            ('allow_package_imports', allow_package_imports),
            ('allow_network', allow_network),
            ('enable_rendering', enable_rendering),
        ):
            if type(value) is not bool:
                raise ValueError(f'{name} must be a bool, got {value!r}.')
        if max_old_generation_size_mb is not None and (
            type(max_old_generation_size_mb) is not int or max_old_generation_size_mb <= 0
        ):
            raise ValueError(
                f'max_old_generation_size_mb must be a positive integer or None, got {max_old_generation_size_mb!r}.'
            )
        if runtime is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('allow_package_imports', allow_package_imports, False),
                    ('allow_network', allow_network, False),
                    ('enable_rendering', enable_rendering, False),
                    (
                        'max_old_generation_size_mb',
                        max_old_generation_size_mb,
                        DEFAULT_MAX_OLD_GENERATION_SIZE_MB,
                    ),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} cannot be combined with `runtime`, which already defines '
                    'the Belgie environment and runtime options.'
                )

        self._allow_package_imports = allow_package_imports
        self._allow_network = allow_network
        self._enable_rendering = enable_rendering
        self._max_old_generation_size_mb = max_old_generation_size_mb
        self._runtime = runtime
        self._active_runtime: AsyncRuntime | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._workspace: Path | None = None

    @property
    def is_open(self) -> bool:
        """Whether the session has an active Belgie runtime."""
        return self._active_runtime is not None

    @property
    def workspace(self) -> Path | None:
        """The owned temporary workspace while open, or None for a custom runtime."""
        return self._workspace

    async def __aenter__(self) -> Self:
        """Create and enter the configured Belgie runtime."""
        if self._exit_stack is not None:
            raise BelgieSandboxError(
                'The session is already open; exit it before entering again. '
                'Use a separate session per concurrent context.'
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError as error:
            raise BelgieSandboxError('Belgie Sandbox requires an asyncio event loop.') from error

        try:
            from belgie import (  # pyright: ignore[reportMissingTypeStubs]
                Environment,
                EnvironmentOptions,
                Runtime,
                RuntimeOptions,
            )
        except ImportError as error:
            raise BelgieSandboxUnavailableError(_MISSING_BELGIE) from error

        stack = AsyncExitStack()
        try:
            if self._runtime is not None:
                active_runtime = await stack.enter_async_context(self._runtime)
            else:
                workspace = Path(stack.enter_context(TemporaryDirectory(prefix='belgie-sandbox-'))).resolve()
                self._workspace = workspace
                packages_enabled = self._allow_package_imports or self._enable_rendering
                dependencies = {'@belgie/render': DEFAULT_RENDER_SPECIFIER} if self._enable_rendering else None
                environment = Environment(
                    dependencies,
                    path=workspace,
                    options=EnvironmentOptions(allow_remote=packages_enabled, no_npm=not packages_enabled),
                )
                active_environment = await stack.enter_async_context(environment)
                if self._enable_rendering:
                    await active_environment.install()
                runtime = Runtime(
                    env=active_environment,
                    options=RuntimeOptions(
                        max_old_generation_size_mb=self._max_old_generation_size_mb,
                        permissions=self._runtime_permissions(workspace),
                    ),
                )
                active_runtime = await stack.enter_async_context(runtime)
        except BaseException as error:
            self._workspace = None
            with anyio.CancelScope(shield=True):
                await stack.aclose()
            if isinstance(error, Exception):
                raise BelgieSandboxUnavailableError(f'Could not start the Belgie sandbox: {error}') from error
            raise

        self._active_runtime = active_runtime
        self._exit_stack = stack
        return self

    def _runtime_permissions(self, workspace: Path) -> RuntimePermissions:
        from belgie import RuntimePermissions  # pyright: ignore[reportMissingTypeStubs]

        return RuntimePermissions(
            allow_read=([str(workspace), *_DEFAULT_RENDER_READ_PATHS] if self._enable_rendering else [str(workspace)]),
            allow_net=[] if self._allow_network else None,
            allow_ffi=[str(workspace / 'node_modules')] if self._enable_rendering else None,
            allow_sys=_DEFAULT_RENDER_SYS_PERMISSIONS if self._enable_rendering else None,
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the runtime, environment, and owned temporary workspace."""
        stack = self._exit_stack
        self._exit_stack = None
        self._active_runtime = None
        self._workspace = None
        if stack is None:
            return
        with anyio.CancelScope(shield=True):
            await stack.__aexit__(exc_type, exc, traceback)

    async def close(self) -> None:
        """Close the session without an active exception context."""
        await self.__aexit__(None, None, None)

    async def run_script(self, source: str, *, timeout: float = DEFAULT_TIMEOUT) -> JsonOutput:
        """Run a JavaScript, TypeScript, or TSX module and return its JSON value."""
        active_runtime = self._active_runtime
        if active_runtime is None:
            raise BelgieSandboxError('The Belgie sandbox session is not open.')
        if type(source) is not str:
            raise TypeError(f'source must be a string, got {type(source).__name__}.')
        if type(timeout) is bool or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f'timeout must be a positive finite number, got {timeout!r}.')

        try:
            from belgie import Script  # pyright: ignore[reportMissingTypeStubs]
            from belgie.errors import BelgieError  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as error:  # pragma: no cover - an open session proves the dependency exists
            raise BelgieSandboxUnavailableError(_MISSING_BELGIE) from error

        try:
            script: Script[[], JsonOutput] = Script(source)
            task: asyncio.Task[JsonOutput] = asyncio.create_task(active_runtime(script)())
            try:
                return await asyncio.wait_for(task, timeout=float(timeout))
            except TimeoutError as error:
                await _drain_cancelled_task(task)
                raise BelgieSandboxTimeoutError(
                    f'Belgie script execution timed out after {timeout} seconds.'
                ) from error
            except asyncio.CancelledError:
                await _drain_cancelled_task(task)
                raise
        except BelgieSandboxTimeoutError:
            raise
        except BelgieError as error:
            raise BelgieSandboxExecutionError(f'Belgie script execution failed:\n{error}') from error
        except (TypeError, ValueError) as error:
            raise BelgieSandboxExecutionError(f'Belgie script returned an invalid JSON value:\n{error}') from error

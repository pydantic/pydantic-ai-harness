"""Controllable in-process stand-in for Belgie's public runtime API."""

from __future__ import annotations

import asyncio
import types
from collections.abc import Awaitable, Callable
from pathlib import Path


class FakeBelgie:
    """State and module objects used by Belgie Sandbox unit tests."""

    def __init__(self) -> None:
        self.result: object = {'ok': True}
        self.script_error: Exception | None = None
        self.start_error: BaseException | None = None
        self.environment_exit_error: BaseException | None = None
        self.runtime_exit_error: BaseException | None = None
        self.hang = False
        self.cancelled = False
        self.environments: list[_Environment] = []
        self.runtimes: list[_Runtime] = []
        self.scripts: list[str] = []

        module = _BelgieModule('belgie')
        errors = _ErrorsModule('belgie.errors')

        module.Environment = _EnvironmentFactory(self)
        module.EnvironmentOptions = _EnvironmentOptions
        module.Runtime = _RuntimeFactory(self)
        module.RuntimeOptions = _RuntimeOptions
        module.RuntimePermissions = _RuntimePermissions
        module.Script = _Script
        errors.BelgieError = BelgieError
        errors.BelgieRuntimeError = BelgieRuntimeError
        errors.BelgieModuleError = BelgieModuleError
        errors.BelgieJavaScriptError = BelgieJavaScriptError
        module.errors = errors

        self.module = module
        self.errors_module = errors


class BelgieError(Exception):
    """Fake upstream base error."""


class BelgieRuntimeError(BelgieError):
    """Fake upstream runtime error."""


class BelgieModuleError(BelgieError):
    """Fake upstream module error."""


class BelgieJavaScriptError(BelgieError):
    """Fake upstream JavaScript error."""


class _EnvironmentOptions:
    def __init__(self, *, allow_remote: bool = True, no_npm: bool = False) -> None:
        self.allow_remote = allow_remote
        self.no_npm = no_npm


class _RuntimePermissions:
    def __init__(
        self,
        *,
        allow_read: list[str] | None = None,
        allow_net: list[str] | None = None,
    ) -> None:
        self.kwargs: dict[str, object] = {'allow_read': allow_read}
        if allow_net is not None:
            self.kwargs['allow_net'] = allow_net


class _RuntimeOptions:
    def __init__(
        self,
        *,
        max_old_generation_size_mb: int | None = None,
        permissions: _RuntimePermissions | None = None,
    ) -> None:
        self.max_old_generation_size_mb = max_old_generation_size_mb
        self.permissions = permissions


class _Environment:
    def __init__(
        self,
        control: FakeBelgie,
        dependencies: dict[str, str] | None,
        *,
        path: str | Path | None,
        options: _EnvironmentOptions | None,
    ) -> None:
        self.control = control
        self.dependencies = dependencies
        self.workspace = Path(path) if path is not None else Path.cwd()
        self.options = options
        self.install_calls = 0
        self.entered = False
        self.exited = False
        self.exit_calls = 0
        control.environments.append(self)

    async def __aenter__(self) -> _Environment:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exit_calls += 1
        if self.control.environment_exit_error is not None:
            raise self.control.environment_exit_error
        self.exited = True

    async def install(self) -> object:  # pragma: no cover - retained to match the upstream environment protocol
        self.install_calls += 1
        return object()


class _Script:
    def __init__(self, content: str) -> None:
        self.content = content


class _ActiveRuntime:
    def __init__(self, control: FakeBelgie) -> None:
        self.control = control

    def __call__(self, script: _Script) -> Callable[[], Awaitable[object]]:
        control = self.control

        async def run() -> object:
            control.scripts.append(script.content)
            if control.hang:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    control.cancelled = True
                    raise
            if control.script_error is not None:
                raise control.script_error
            return control.result

        return run


class _Runtime:
    def __init__(
        self,
        control: FakeBelgie,
        *,
        env: _Environment | None,
        options: _RuntimeOptions | None,
    ) -> None:
        self.control = control
        self.env = env
        self.options = options
        self.entered = False
        self.exited = False
        self.exit_calls = 0
        control.runtimes.append(self)

    async def __aenter__(self) -> _ActiveRuntime:
        if self.control.start_error is not None:
            raise self.control.start_error
        self.entered = True
        return _ActiveRuntime(self.control)

    async def __aexit__(self, *args: object) -> None:
        self.exit_calls += 1
        if self.control.runtime_exit_error is not None:
            raise self.control.runtime_exit_error
        self.exited = True


class _EnvironmentFactory:
    def __init__(self, control: FakeBelgie) -> None:
        self.control = control

    def __call__(
        self,
        dependencies: dict[str, str] | None = None,
        *,
        path: str | Path | None = None,
        options: _EnvironmentOptions | None = None,
    ) -> _Environment:
        return _Environment(self.control, dependencies, path=path, options=options)


class _RuntimeFactory:
    def __init__(self, control: FakeBelgie) -> None:
        self.control = control

    def __call__(
        self,
        *,
        env: _Environment | None = None,
        options: _RuntimeOptions | None = None,
    ) -> _Runtime:
        return _Runtime(self.control, env=env, options=options)


class _ErrorsModule(types.ModuleType):
    BelgieError: type[BelgieError]
    BelgieRuntimeError: type[BelgieRuntimeError]
    BelgieModuleError: type[BelgieModuleError]
    BelgieJavaScriptError: type[BelgieJavaScriptError]


class _BelgieModule(types.ModuleType):
    Environment: _EnvironmentFactory
    EnvironmentOptions: type[_EnvironmentOptions]
    Runtime: _RuntimeFactory
    RuntimeOptions: type[_RuntimeOptions]
    RuntimePermissions: type[_RuntimePermissions]
    Script: type[_Script]
    errors: _ErrorsModule

"""Reusable conformance checks for sandbox provider backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from pydantic_ai.sandboxes import Sandbox, SandboxBackend, SandboxError, SandboxTimeoutError

BackendFactory = Callable[[], Awaitable[SandboxBackend]]


async def check_command_validation(factory: BackendFactory) -> None:
    backend = await factory()
    with pytest.raises(TypeError):
        await backend.run(['echo'], shell=True)
    with pytest.raises(TypeError):
        await backend.run('echo')
    with pytest.raises(TypeError):
        await backend.run([])
    with pytest.raises(ValueError):
        await backend.run(['pwd'], cwd='relative')


async def check_missing_file(factory: BackendFactory) -> None:
    sandbox = Sandbox(await factory())
    with pytest.raises(FileNotFoundError):
        await sandbox.read_bytes('/missing')


async def check_timeout(factory: BackendFactory) -> None:
    backend = await factory()
    with pytest.raises(SandboxTimeoutError) as exc_info:
        await backend.run(['sleep', '10'], timeout=1)
    assert isinstance(exc_info.value, TimeoutError)
    assert isinstance(exc_info.value, SandboxError)

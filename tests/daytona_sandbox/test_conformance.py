"""Pydantic AI's workspace backend conformance suite, run against `DaytonaSandboxBackend`.

`TestFakeDaytonaSandboxBackend` runs everywhere, over the fake Daytona SDK in host mode, where
commands and file operations act on a temporary host directory. `TestLiveDaytonaSandboxBackend`
runs the same rules against a real sandbox and is gated like `test_daytona_live.py`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.testing import WorkspaceBackendSuite

from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend

from .conftest import LIVE_AUTO_STOP_INTERVAL
from .fake_daytona import FakeDaytona

_live_enabled = os.getenv('PYDANTIC_AI_HARNESS_DAYTONA_LIVE') == '1'


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return DaytonaSandboxBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, DaytonaSandboxBackend)
    sandbox = await backend.get_client()
    await sandbox.delete()


async def _delete_if_live(sandbox_id: str) -> None:  # pragma: no cover - live tier runs without coverage
    import daytona  # noqa: PLC0415 - optional extra, absent on slim installs

    async with daytona.AsyncDaytona() as client:
        try:
            sandbox = await client.get(sandbox_id)
        except daytona.DaytonaNotFoundError:
            return
        # The destroy rule normally deleted it already; a second delete is not needed.
        if sandbox.state not in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED):
            await sandbox.delete()


class TestFakeDaytonaSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_daytona: FakeDaytona, tmp_path: Path) -> DaytonaSandboxBackend:
        # Resolved so the working directory `pwd -P` reports matches it on hosts whose temp
        # directory sits behind a symlink.
        fake_daytona.host_root = tmp_path.resolve()
        return DaytonaSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, fake_daytona: FakeDaytona, tmp_path: Path) -> Callable[[], WorkspaceBackend]:
        fake_daytona.host_root = tmp_path.resolve()
        return DaytonaSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.daytona_live
@pytest.mark.skipif(not _live_enabled, reason='requires PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1')
class TestLiveDaytonaSandboxBackend(WorkspaceBackendSuite):  # pragma: no cover - live tier runs without coverage
    # One event loop for the class: the backend's `AsyncDaytona` client holds an HTTP session bound
    # to the loop it was first used on. A class-scoped async fixture is what holds that loop open
    # between tests; a class-scoped `anyio_backend` alone does not.
    @pytest.fixture(scope='class')
    @classmethod
    def anyio_backend(cls) -> str:
        return 'asyncio'

    @pytest.fixture
    def destructive_backend(self) -> Callable[[], WorkspaceBackend]:
        # Destructive rules need their own sandbox, not the shared class-scoped fixture.
        return lambda: DaytonaSandboxBackend(auto_stop_interval=LIVE_AUTO_STOP_INTERVAL)

    # Class-scoped so the rules share one sandbox instead of starting one each; the suite runs
    # its destroy rule last.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[DaytonaSandboxBackend]:
        backend = DaytonaSandboxBackend(auto_stop_interval=LIVE_AUTO_STOP_INTERVAL)
        try:
            await backend.get_client()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await _delete_if_live(backend.ref.id)
            finally:
                await backend.aclose()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete

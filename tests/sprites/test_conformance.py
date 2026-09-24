"""Pydantic AI's workspace backend conformance suite, run against `SpriteWorkspaceBackend`.

`TestFakeSpriteWorkspaceBackend` runs everywhere, over the fake Sprites SDK, whose commands run in
local subprocesses under a temporary host directory. `TestLiveSpriteWorkspaceBackend` runs the same
rules against a real Sprite and is gated like `test_sprites_live.py`.

The backend implements `SupportsCommands` only; the suite derives the filesystem operations, and
with them its reattach and destroy rules, through `Workspace`, so those rules run here too.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.testing import WorkspaceBackendSuite

from pydantic_ai_harness.sprites import SpriteWorkspaceBackend

from .fake_sprites import SpriteTransport


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return SpriteWorkspaceBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, SpriteWorkspaceBackend)
    sprite = await backend.get_client()
    await sprite.delete()


class TestFakeSpriteWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, transport: SpriteTransport) -> SpriteWorkspaceBackend:
        del transport
        return SpriteWorkspaceBackend()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.sprites_live
@pytest.mark.usefixtures('sprites_token')
class TestLiveSpriteWorkspaceBackend(WorkspaceBackendSuite):  # pragma: no cover - live tier runs without coverage
    # One event loop for the class: the backend's `AsyncSpritesClient` holds an `httpx.AsyncClient`
    # whose pooled connections are bound to the loop they were opened on. A class-scoped async
    # fixture is what holds that loop open between tests.
    @pytest.fixture(scope='class')
    @classmethod
    def anyio_backend(cls) -> str:
        return 'asyncio'

    # Class-scoped so the rules share one Sprite instead of creating one each; the suite runs its
    # destroy rule last.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls, sprites_token: str) -> AsyncIterator[SpriteWorkspaceBackend]:
        backend = SpriteWorkspaceBackend(token=sprites_token)
        yield backend
        if backend.ref is not None:
            from sprites.exceptions import NotFoundError  # noqa: PLC0415 - optional extra, absent on slim installs

            try:
                await _delete(backend)
            except NotFoundError:
                pass
        await backend.aclose()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete

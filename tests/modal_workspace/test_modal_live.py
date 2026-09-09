"""Opt-in integration tests for a real Modal workspace."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.workspaces import Workspace, WorkspaceTimeoutError

from pydantic_ai_harness.modal_workspace import ModalWorkspaceBackend

_enabled = os.getenv('PYDANTIC_AI_HARNESS_MODAL_LIVE') == '1'
_credentials = (os.getenv('MODAL_TOKEN_ID') is not None and os.getenv('MODAL_TOKEN_SECRET') is not None) or Path(
    '~/.modal.toml'
).expanduser().exists()

pytestmark = [
    pytest.mark.anyio(backends=['asyncio']),
    pytest.mark.modal_live,
    pytest.mark.skipif(
        not _enabled or not _credentials,
        reason='requires PYDANTIC_AI_HARNESS_MODAL_LIVE=1 and Modal credentials',
    ),
]


@asynccontextmanager
async def owned_backend() -> AsyncGenerator[ModalWorkspaceBackend, None]:
    backend = ModalWorkspaceBackend(image='python:3.12-slim')
    native = await backend.workspace
    try:
        yield backend
    finally:
        try:
            await native.terminate.aio()
        finally:
            # Modal 1.5.2 leaves the return type of `detach.aio()` unspecified.
            await native.detach.aio()  # pyright: ignore[reportUnknownMemberType]


async def test_real_command_and_filesystem() -> None:
    async with owned_backend() as backend:
        result = await backend.run(['sh', '-c', 'printf out; printf err >&2; exit 3'], timeout=30)
        assert (result.stdout, result.stderr, result.exit_code) == ('out', 'err', 3)
        workspace = Workspace(backend)
        await workspace.write_text('/tmp/modal-workspace.txt', 'content')
        assert await workspace.read_text('/tmp/modal-workspace.txt') == 'content'


async def test_real_timeout_retains_output() -> None:
    async with owned_backend() as backend:
        await backend.run(['true'], timeout=30)
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sh', '-c', 'echo diagnostic; sleep 30'], timeout=2)
        assert 'diagnostic' in (exc_info.value.stdout or '')

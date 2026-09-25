"""Opt-in integration tests for a real Modal workspace.

Skipped unless `PYDANTIC_AI_HARNESS_MODAL_LIVE=1` and Modal credentials are present; CI also
sets `MODAL_REQUIRE_LIVE`, which turns that skip into a failure (see `conftest.py`).

Run locally:
`PYDANTIC_AI_HARNESS_MODAL_LIVE=1 uv run pytest -m modal_live tests/modal_sandbox`
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import pytest
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend

from .._tool_calls import call_tools
from .conftest import LIVE_IDLE_TIMEOUT, LIVE_SANDBOX_TIMEOUT

pytestmark = [pytest.mark.anyio(backends=['asyncio']), pytest.mark.modal_live]


@asynccontextmanager
async def owned_backend(**settings: object) -> AsyncGenerator[ModalSandboxBackend, None]:
    limits = {'sandbox_timeout': LIVE_SANDBOX_TIMEOUT, 'idle_timeout': LIVE_IDLE_TIMEOUT}
    backend = ModalSandboxBackend(image='python:3.12-slim', **(limits | settings))  # type: ignore[arg-type]
    native = await backend.get_client()
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
        await workspace.write_text('/tmp/modal-sandbox.txt', 'content')
        assert await workspace.read_text('/tmp/modal-sandbox.txt') == 'content'


async def test_real_timeout_retains_output() -> None:
    async with owned_backend() as backend:
        await backend.run(['true'], timeout=30)
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sh', '-c', 'echo diagnostic; sleep 30'], timeout=2)
        assert 'diagnostic' in (exc_info.value.stdout or '')


async def test_reattach_sees_state_and_applies_working_dir_and_env() -> None:
    """An attached backend reuses the sandbox's files, and its own `working_dir` and `env` reach commands."""
    marker = f'/tmp/{uuid.uuid4().hex}.txt'
    async with owned_backend() as owner:
        await owner.write_bytes(marker, b'shared')
        attached = ModalSandboxBackend(ref=owner.ref, working_dir='/tmp', env={'PROBE': 'attached'})
        assert await attached.read_bytes(marker) == b'shared'
        result = await attached.run('printf "%s %s" "$(pwd)" "$PROBE"', shell=True, timeout=30)
        assert result.stdout == '/tmp attached'


async def test_a_terminated_sandbox_is_unavailable() -> None:
    async with owned_backend() as owner:
        await (await owner.get_client()).terminate.aio()
        with pytest.raises(WorkspaceUnavailableError):
            await ModalSandboxBackend(ref=owner.ref).run(['true'], timeout=30)
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_an_expired_sandbox_is_unavailable() -> None:
    """A sandbox past its `sandbox_timeout` is gone, both to its owner and to a reattach."""
    async with owned_backend(sandbox_timeout=10) as owner:
        await anyio.sleep(30)
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await owner.run(['true'], timeout=30)
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await ModalSandboxBackend(ref=owner.ref).run(['true'], timeout=30)


async def test_coder_tools_run_in_the_sandbox_modal_sandbox_supplies() -> None:
    """`ModalSandbox` supplies the workspace, and `Coder`'s tools run in it."""
    import modal  # noqa: PLC0415 - optional extra, absent on slim installs

    supplied: list[ModalSandboxBackend] = []

    class RecordingModalSandbox(ModalSandbox[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            backend = super().get_workspace(ctx, ref=ref)
            assert isinstance(backend, ModalSandboxBackend)
            supplied.append(backend)
            return backend

    image = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')
    try:
        results = await call_tools(
            [
                RecordingModalSandbox(
                    image=image,
                    working_dir='/tmp',
                    sandbox_timeout=LIVE_SANDBOX_TIMEOUT,
                    idle_timeout=LIVE_IDLE_TIMEOUT,
                ),
                Coder(),
            ],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hello from modal')\n"}),
                ('shell', {'command': 'python hello.py && git --version'}),
                ('grep', {'pattern': 'hello from modal'}),
            ],
        )
        assert 'hello from modal' in results[1]
        assert 'git version' in results[1]
        assert 'hello.py' in results[2]
        # Core may ask for the workspace again after `for_run` only to compare it; that backend is
        # discarded unused, so exactly one of them created a sandbox.
        assert sum(backend.ref is not None for backend in supplied) == 1
    finally:
        for backend in supplied:
            if backend.ref is not None:
                await (await backend.get_client()).terminate.aio()

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
from typing import Any

import anyio
import pytest
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import (
    WorkspaceBackend,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pytest_examples import CodeExample

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from .conftest import LIVE_IDLE_TIMEOUT, LIVE_SANDBOX_TIMEOUT

pytestmark = [pytest.mark.anyio(backends=['asyncio']), pytest.mark.modal_live]


@asynccontextmanager
async def owned_backend(**settings: Any) -> AsyncGenerator[ModalSandboxBackend, None]:
    defaults: dict[str, Any] = {
        'sandbox_timeout': LIVE_SANDBOX_TIMEOUT,
        'idle_timeout': LIVE_IDLE_TIMEOUT,
    }
    backend = ModalSandboxBackend(**(defaults | settings))
    native = await backend.get_client()
    try:
        yield backend
    finally:
        try:
            await native.terminate.aio()
        finally:
            # Modal 1.5.2 leaves the return type of `detach.aio()` unspecified.
            await native.detach.aio()  # pyright: ignore[reportUnknownMemberType]


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
    supplied: list[ModalSandboxBackend] = []

    class RecordingModalSandbox(ModalSandbox[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            backend = super().get_workspace(ctx, ref=ref)
            assert isinstance(backend, ModalSandboxBackend)
            supplied.append(backend)
            return backend

    try:
        results = await call_tools(
            [
                RecordingModalSandbox(
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


async def test_commands_get_a_usable_environment() -> None:
    """With no `image` or `env`, a command sees `PATH` and `HOME` and finds git and ripgrep; `env=` adds to them."""
    async with owned_backend() as backend:
        result = await backend.run('echo "$PATH"; echo "$HOME"; git --version; rg --version', shell=True, timeout=60)
        path, home, git, rg, *_ = result.stdout.splitlines()
        assert path and home and git.startswith('git version') and rg.startswith('ripgrep')

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=30)
        assert result.stdout.splitlines() == ['1', path]


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/modal-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    A follow-up run, from the message history or a stored ref, works in the first run's sandbox.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'terminate_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs

"""Integration tests that require a real Daytona sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution, a client-owned deadline over a real process, one filesystem shared by
Daytona's file API and its process sessions, and real lifecycle state in Daytona's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Daytona behavior.

Gating:
  * `daytona_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1` opts in explicitly.
  * also requires a non-empty `DAYTONA_API_KEY`: without one the tests skip, or fail where
    `DAYTONA_REQUIRE_LIVE` is set, as CI does.
  * the module-scoped async `client` fixture keeps the shared Daytona client on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1 uv run pytest -m daytona_live tests/daytona_sandbox/test_daytona_live.py`
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import daytona
import pytest
from pydantic_ai.workspaces import WorkspaceTimeoutError, WorkspaceUnavailableError

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend

from .._tool_calls import call_tools
from .conftest import require_live_credentials

_live_enabled = os.getenv('PYDANTIC_AI_HARNESS_DAYTONA_LIVE') == '1'

pytestmark = [
    pytest.mark.daytona_live,
    pytest.mark.skipif(not _live_enabled, reason='requires PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1'),
]


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def client() -> AsyncIterator[daytona.AsyncDaytona]:
    """One caller-owned API client for every backend in this module, closed at the end."""
    # Module-scoped fixtures set up before the function-scoped credential check, so check here too.
    require_live_credentials()
    async with daytona.AsyncDaytona() as client:
        yield client


@asynccontextmanager
async def _owned(client: daytona.AsyncDaytona) -> AsyncGenerator[DaytonaSandboxBackend]:
    """Create a sandbox and delete it on the way out, even when the test deleted it already."""
    backend = DaytonaSandboxBackend(client=client)
    native = await backend.get_client()
    try:
        yield backend
    finally:
        try:
            await native.refresh_data()
            if native.state not in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED):
                await native.delete()
        except daytona.DaytonaNotFoundError:
            pass


async def test_creates_a_fresh_sandbox_and_runs_a_command(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a session command reports real output and exit code."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)

        assert backend.ref is not None
        assert (result.stdout.strip(), result.stderr.strip(), result.exit_code) == ('out', 'err', 3)


async def test_reattach_by_ref_reads_a_file_the_first_backend_wrote(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that `client.get` plus `start` reaches the same sandbox."""
    path = f'/tmp/{_unique("reattach")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'shared')
        assert owner.ref is not None

        attached = DaytonaSandboxBackend(client=client, ref=owner.ref)

        assert await attached.read_bytes(path) == b'shared'
        assert attached.ref == owner.ref


async def test_command_and_file_api_share_one_filesystem(client: daytona.AsyncDaytona) -> None:
    """Validates the protocol's one-environment contract: process sessions and `sandbox.fs` see one tree."""
    root = f'/tmp/{_unique("roundtrip")}'
    async with _owned(client) as backend:
        await backend.write_bytes(f'{root}/nested/in.txt', b'from-file-api\n')
        result = await backend.run(
            ['sh', '-c', 'cat "$1/nested/in.txt" && printf from-shell > "$1/out.txt"', 'sh', root], timeout=60
        )

        assert (result.exit_code, result.stdout) == (0, 'from-file-api\n')
        assert await backend.read_bytes(f'{root}/out.txt') == b'from-shell'


async def test_timeout_raises_with_the_partial_output(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that session logs stream before the command ends.

    The deadline is enforced client-side, so the output printed before it expired must reach
    the `WorkspaceTimeoutError`, and deleting the session must stop the command.
    """
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo DIAGNOSTIC; sleep 20; touch {marker}', shell=True, timeout=5)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        assert exc_info.value.timeout == 5
        assert (await backend.run(['sleep', '20'], timeout=60)).exit_code == 0
        assert await backend.exists(marker) is False


async def test_reattach_to_a_deleted_sandbox_is_unavailable(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption about deletion: `delete()` returns while the sandbox is
    still `destroying`, and a later attach or a call on the held handle is reported as gone.

    If Daytona answers either differently, a deleted sandbox would surface as a retryable error
    or, worse, as a missing file.
    """
    path = f'/tmp/{_unique("deleted")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'before-delete')
        assert owner.ref is not None
        await (await owner.get_client()).delete()

        with pytest.raises(WorkspaceUnavailableError):
            await DaytonaSandboxBackend(client=client, ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await owner.read_bytes(path)
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_coder_tools_run_in_the_sandbox(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a relative path the file API writes is where a
    session command started in the working directory finds it, so `Coder`'s tools agree."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from daytona')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )

        assert 'hi from daytona' in results[1]

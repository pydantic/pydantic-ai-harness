"""Integration tests that require a real Fly.io Sprite.

The fake-backed suites cover the harness-owned logic: remote process supervision, deadline
handling, and exception mapping. This live tier admits only what a correctly written fake could
not catch: real process execution in a Sprite, output reaching the client before a command ends,
cancellation of the remote process group, and deletion as the Sprites control plane reports it.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Sprites behavior.

Gating:
  * `sprites_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_SPRITES_LIVE=1` opts in explicitly.
  * also requires a non-empty `SPRITE_TOKEN`; CI sets `SPRITES_REQUIRE_LIVE`, so there a
    missing token fails instead of skipping.
  * a module-scoped `anyio_backend` fixture keeps the shared Sprites client on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_SPRITES_LIVE=1 uv run pytest -m sprites_live tests/sprites/test_sprites_live.py`
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.workspaces import Workspace, WorkspaceTimeoutError, WorkspaceUnavailableError
from sprites import AsyncSpritesClient
from sprites.exceptions import NotFoundError

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites import SpriteWorkspace, SpriteWorkspaceBackend

pytestmark = pytest.mark.sprites_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def client(sprites_token: str) -> AsyncIterator[AsyncSpritesClient]:
    """One caller-owned API client for every backend in this module, closed at the end."""
    async with AsyncSpritesClient(token=sprites_token) as client:
        yield client


@asynccontextmanager
async def _owned(client: AsyncSpritesClient) -> AsyncGenerator[SpriteWorkspaceBackend]:
    """Create a Sprite and delete it on the way out, even when the test deleted it already."""
    backend = SpriteWorkspaceBackend(client=client, name=_unique('pydantic-ai-live'))
    native = await backend.get_client()
    try:
        yield backend
    finally:
        try:
            await native.delete()
        except NotFoundError:
            pass


async def test_creates_a_fresh_sprite_and_runs_a_command(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that the control exec reports real output and exit code."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)

        assert backend.ref is not None
        assert (result.stdout.strip(), result.stderr.strip(), result.exit_code) == ('out', 'err', 3)


async def test_reattach_by_ref_reads_a_file_the_first_backend_wrote(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that `get_sprite(ref.id)` reaches the same Sprite."""
    path = f'/tmp/{_unique("reattach")}.txt'
    async with _owned(client) as owner:
        assert (await owner.run(['sh', '-c', 'printf shared > "$1"', 'sh', path], timeout=60)).exit_code == 0
        assert owner.ref is not None

        attached = SpriteWorkspaceBackend(client=client, ref=owner.ref)
        result = await attached.run(['cat', path], timeout=60)

        assert (result.exit_code, result.stdout) == (0, 'shared')
        assert attached.ref == owner.ref


async def test_command_and_file_round_trip(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that binary data survives the command-backed filesystem.

    The backend has no file API of its own, so `Workspace` moves file contents through commands;
    the fake runs those on the host, which cannot show what the Sprite's shell and tools do with them.
    """
    root = f'/tmp/{_unique("roundtrip")}'
    async with _owned(client) as backend:
        workspace = Workspace(backend)
        await workspace.write_bytes(f'{root}/nested/in.txt', b'from-file-api\n')
        await workspace.write_bytes(f'{root}/binary.bin', b'\x00\xff\n')
        result = await backend.run(
            ['sh', '-c', 'cat "$1/nested/in.txt" && printf from-shell > "$1/out.txt"', 'sh', root], timeout=60
        )

        assert (result.exit_code, result.stdout) == (0, 'from-file-api\n')
        assert await workspace.read_bytes(f'{root}/out.txt') == b'from-shell'
        assert await workspace.read_bytes(f'{root}/binary.bin') == b'\x00\xff\n'


async def test_timeout_raises_with_the_partial_output(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that exec output streams before the command ends.

    The deadline is enforced client-side, so the output printed before it expired must reach the
    `WorkspaceTimeoutError`, and the cancel payload must stop the remote process group.
    """
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo DIAGNOSTIC; sleep 20; touch {marker}', shell=True, timeout=5)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        assert exc_info.value.timeout == 5
        assert (await backend.run(['sleep', '20'], timeout=60)).exit_code == 0
        assert (await backend.run(['test', '-e', marker], timeout=60)).exit_code == 1


async def test_reattach_to_a_deleted_sprite_is_unavailable(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption about deletion: once `delete()` returns, `get_sprite`
    raises `NotFoundError` and a control connection to the Sprite fails its handshake with HTTP 404.

    If Sprites answers either differently, a deleted Sprite would surface as a retryable
    `WorkspaceError` or, worse, as a missing file.
    """
    async with _owned(client) as owner:
        assert (await owner.run(['true'], timeout=60)).exit_code == 0
        assert owner.ref is not None
        await (await owner.get_client()).delete()

        with pytest.raises(WorkspaceUnavailableError):
            await SpriteWorkspaceBackend(client=client, ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=60)
        with pytest.raises(WorkspaceUnavailableError):
            await Workspace(owner).read_bytes('/tmp/anything')


async def test_coder_shell_and_file_tools_run_in_the_sprite(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that Coder's tools work against a real Sprite.

    The fake runs `shell` on the host, whose `sh`, `git`, and file tools are not the Sprite's.
    """

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            yield {0: DeltaToolCall('write_file', json.dumps({'path': 'notes.txt', 'content': 'hello'}))}
        elif len(returns) == 1:
            yield {0: DeltaToolCall('shell', json.dumps({'command': 'git --version && cat notes.txt'}))}
        else:
            yield str(returns[-1].content)

    async with _owned(client) as backend:
        agent = Agent(FunctionModel(stream_function=model), capabilities=[SpriteWorkspace(client=client), Coder()])
        result = await agent.run('go', workspace=backend.ref)

    assert 'git version' in result.output
    assert 'hello' in result.output

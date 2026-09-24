"""Test the shared hosted MCP helpers: per-run credentials and read-only selection."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass

import anyio
import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import MCPAuth, credential, is_read_only, per_run


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# The `mcp` SDK's server, not FastMCP's: the slim install has only the FastMCP client.
_SERVER = """
import asyncio, socket, uvicorn
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP('whoami', stateless_http=True)

@mcp.tool()
async def whoami(ctx: Context) -> str:
    await asyncio.sleep(0.05)  # keep concurrent runs overlapping
    return ctx.request_context.request.headers['authorization']

server_socket = socket.create_server(('127.0.0.1', 0))
print(server_socket.getsockname()[1], flush=True)
uvicorn.run(mcp.streamable_http_app(), fd=server_socket.fileno(), log_level='warning')
"""


@pytest.fixture
async def server_url() -> AsyncIterator[str]:
    """A streamable HTTP MCP server whose tool reports the caller's `Authorization` header."""
    async with await anyio.open_process([sys.executable, '-c', _SERVER], stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        port = int((await process.stdout.receive()).decode().strip())
        yield f'http://127.0.0.1:{port}/mcp'
        process.terminate()


@dataclass
class User:
    token: str | None


def connect(url: str) -> Callable[[MCPAuth | None], AbstractToolset[User]]:
    """Connect as the given credential, or as `WHOAMI_TOKEN` when there is none."""

    def build(auth: MCPAuth | None) -> AbstractToolset[User]:
        return MCPToolset(url, id='whoami', auth=credential(auth, env='WHOAMI_TOKEN', service='whoami'))

    return build


def token(ctx: RunContext[User]) -> str | None:
    return ctx.deps.token


async def test_concurrent_runs_use_their_own_credentials(server_url: str) -> None:
    agent = Agent(
        TestModel(),
        deps_type=User,
        toolsets=[per_run(token, connect(server_url), id='whoami')],
    )
    alice, bob = await asyncio.gather(
        agent.run('Who am I?', deps=User('alice-token')), agent.run('Who am I?', deps=User('bob-token'))
    )
    assert (alice.output, bob.output) == ('{"whoami":"Bearer alice-token"}', '{"whoami":"Bearer bob-token"}')


async def test_missing_credential_omits_tools_instead_of_falling_back(server_url: str) -> None:
    agent = Agent(TestModel(), deps_type=User, toolsets=[per_run(token, connect(server_url), id='whoami')])
    result = await agent.run('Who am I?', deps=User(None))
    assert result.output == 'success (no tool calls)'


async def test_async_provider(server_url: str) -> None:
    async def bearer(ctx: RunContext[User]) -> httpx.Auth | None:
        return None if ctx.deps.token is None else _Bearer(ctx.deps.token)

    agent = Agent(TestModel(), deps_type=User, toolsets=[per_run(bearer, connect(server_url), id='whoami')])
    result = await agent.run('Who am I?', deps=User('alice-token'))
    assert result.output == '{"whoami":"Bearer alice-token"}'


async def test_environment_credential_is_shared(server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', 'deployment-token')
    toolset = per_run(None, connect(server_url), id='whoami')
    assert isinstance(toolset, MCPToolset)
    result = await Agent(TestModel(), deps_type=User, toolsets=[toolset]).run('Who am I?', deps=User('ignored'))
    assert result.output == '{"whoami":"Bearer deployment-token"}'


async def test_callable_auth_object_is_fixed(server_url: str) -> None:
    class CallableBearer(_Bearer):
        def __call__(self, ctx: RunContext[User]) -> str:
            raise AssertionError('a fixed `httpx.Auth` is never called with the run context')  # pragma: no cover

    toolset = per_run(CallableBearer('deployment-token'), connect(server_url), id='whoami')
    result = await Agent(TestModel(), deps_type=User, toolsets=[toolset]).run('Who am I?', deps=User('ignored'))
    assert result.output == '{"whoami":"Bearer deployment-token"}'


async def test_auth_function_cannot_return_oauth(server_url: str) -> None:
    agent = Agent(TestModel(), deps_type=User, toolsets=[per_run(token, connect(server_url), id='whoami')])
    with pytest.raises(UserError, match='Browser OAuth is not supported'):
        await agent.run('Who am I?', deps=User('oauth'))


@pytest.mark.parametrize('env', ['WHOAMI_TOKEN', None])
@pytest.mark.parametrize('auth', [None, ''])
def test_missing_credential_raises(auth: str | None, env: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', '')
    with pytest.raises(UserError, match='to connect to whoami'):
        credential(auth, env=env, service='whoami')


def test_oauth_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(UserError, match='Browser OAuth is not supported'):
        per_run('oauth', connect('https://example.com/mcp'), id='whoami')
    monkeypatch.setenv('WHOAMI_TOKEN', 'oauth')
    with pytest.raises(UserError, match='Browser OAuth is not supported'):
        credential(None, env='WHOAMI_TOKEN', service='whoami')


class _Bearer(httpx.Auth):
    def __init__(self, token: str) -> None:
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers['Authorization'] = f'Bearer {self.token}'
        yield request


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'annotations': {'readOnlyHint': True}}, True),
        ({'annotations': {'readOnlyHint': False}}, False),
        ({'annotations': {}}, False),
        (None, False),
    ],
)
def test_is_read_only_requires_an_explicit_hint(metadata: dict[str, object] | None, expected: bool) -> None:
    assert is_read_only(ToolDefinition(name='tool', metadata=metadata)) is expected

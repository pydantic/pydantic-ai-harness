"""Slack credentials resolved through native capability factories."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from mcp import types
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.slack import Slack
from tests.slack.conftest import OfflineMCP  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class Deps:
    user_id: str


async def test_async_factory_exposes_tools_with_each_users_credentials(offline_mcp: OfflineMCP) -> None:
    offline_mcp.tools = [types.Tool(name='lookup', inputSchema={'type': 'object', 'properties': {}})]
    tokens = {'U1': 'first-token', 'U2': 'second-token'}

    async def slack_for_user(ctx: RunContext[Deps]) -> Slack[Deps]:
        await asyncio.sleep(0)
        return Slack(token=tokens[ctx.deps.user_id])

    agent = Agent(TestModel(), deps_type=Deps, capabilities=[slack_for_user])
    await asyncio.gather(
        agent.run('lookup', deps=Deps(user_id='U1')),
        agent.run('lookup', deps=Deps(user_id='U2')),
    )

    assert {(call.name, call.token) for call in offline_mcp.calls} == {
        ('lookup', 'Bearer first-token'),
        ('lookup', 'Bearer second-token'),
    }

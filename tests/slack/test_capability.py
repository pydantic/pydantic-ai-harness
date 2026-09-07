"""Tests for the Slack capability: which token a run uses and what it hands to the MCP toolset."""

from __future__ import annotations

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.slack import Slack

pytestmark = pytest.mark.anyio

SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def run_context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)


@pytest.fixture(autouse=True)
def no_slack_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)


async def resolved_transport(capability: Slack[None], run_context: RunContext[None]) -> StreamableHttpTransport:
    """Resolve the token as a run would and return the HTTP transport the toolset was built with."""
    resolved = await capability.for_run(run_context)
    toolset = resolved.get_toolset()
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestSlack:
    async def test_token_becomes_the_bearer_header_for_slack_mcp(self, run_context: RunContext[None]) -> None:
        transport = await resolved_transport(Slack(token='xoxp-explicit'), run_context)
        assert transport.url == SLACK_MCP_URL
        assert transport.headers == {'Authorization': 'Bearer xoxp-explicit'}

    async def test_explicit_token_wins_over_environment(
        self, monkeypatch: pytest.MonkeyPatch, run_context: RunContext[None]
    ) -> None:
        monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-environment')
        transport = await resolved_transport(Slack(token='xoxp-explicit'), run_context)
        assert transport.headers == {'Authorization': 'Bearer xoxp-explicit'}

    async def test_environment_token_is_used_when_none_is_passed(
        self, monkeypatch: pytest.MonkeyPatch, run_context: RunContext[None]
    ) -> None:
        monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-environment')
        transport = await resolved_transport(Slack(), run_context)
        assert transport.headers == {'Authorization': 'Bearer xoxp-environment'}

    async def test_missing_token_is_reported_at_run_time_not_construction(self, run_context: RunContext[None]) -> None:
        capability = Slack()
        assert capability.get_toolset() is None
        with pytest.raises(UserError, match=r'Pass Slack\(token=...\) or set SLACK_USER_TOKEN'):
            await capability.for_run(run_context)

    async def test_bot_token_is_rejected_because_slack_mcp_needs_a_user_token(
        self, run_context: RunContext[None]
    ) -> None:
        with pytest.raises(UserError, match='accepts user tokens only, not bot tokens'):
            await Slack(token='xoxb-bot').for_run(run_context)

    async def test_server_instructions_are_forwarded_to_the_agent(self, run_context: RunContext[None]) -> None:
        resolved = await Slack(token='xoxp-user').for_run(run_context)
        toolset = resolved.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.include_instructions is True
        assert toolset.id == 'slack-mcp'

    def test_combine_rejects_different_tokens_and_merges_equal_or_missing_tokens(self) -> None:
        with pytest.raises(UserError, match='different credentials cannot be combined'):
            Slack.combine([Slack(token='xoxp-one'), Slack(token='xoxp-two')])

        equal = Slack.combine([Slack(token='xoxp-one'), Slack(token='xoxp-one')])
        missing = Slack.combine([Slack(), Slack()])
        assert isinstance(equal, Slack) and equal.token == 'xoxp-one'
        assert isinstance(missing, Slack) and missing.token is None

    def test_token_is_kept_out_of_repr(self) -> None:
        capability = Slack(token='xoxp-user', description='Slack access', defer_loading=True)
        assert 'xoxp-user' not in repr(capability)
        assert capability.id == 'slack'
        assert capability.description == 'Slack access'
        assert capability.defer_loading is True

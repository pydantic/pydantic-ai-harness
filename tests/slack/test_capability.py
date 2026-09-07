"""Tests for the public Slack capability contract."""

from __future__ import annotations

import anyio
import pytest
from mcp import types
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.slack import Slack
from tests.slack.conftest import OfflineMCP  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestSlack:
    @pytest.mark.parametrize(
        ('explicit', 'expected'),
        [('xoxp-explicit', 'xoxp-explicit'), (None, 'xoxp-user')],
        ids=['explicit-over-environment', 'environment'],
    )
    async def test_token_resolution_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        offline_mcp: OfflineMCP,
        explicit: str | None,
        expected: str,
    ) -> None:
        offline_mcp.tools = [types.Tool(name='lookup', inputSchema={'type': 'object', 'properties': {}})]
        monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-user')

        agent = Agent(TestModel(call_tools=['lookup']), capabilities=[Slack(token=explicit)])
        await agent.run('lookup')

        assert offline_mcp.authorization_headers[-1] == f'Bearer {expected}'

    async def test_missing_token_is_only_reported_when_a_run_needs_slack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)
        capability = Slack()
        assert capability.token is None
        agent = Agent(TestModel(), capabilities=[capability])

        async with agent:
            pass

        with pytest.raises(UserError) as exc_info:
            await agent.run('use Slack')
        assert str(exc_info.value) == 'Slack tools need a user token. Pass Slack(token=...) or set SLACK_USER_TOKEN.'

    async def test_bot_token_is_rejected_because_slack_mcp_needs_a_user_token(self) -> None:
        agent = Agent(TestModel(), capabilities=[Slack(token='xoxb-bot')])
        with pytest.raises(UserError, match='accepts user tokens only, not bot tokens'):
            await agent.run('use Slack')

    async def test_user_token_uses_the_hosted_mcp_server(self, offline_mcp: OfflineMCP) -> None:
        offline_mcp.tools = [types.Tool(name='lookup', inputSchema={'type': 'object', 'properties': {}})]
        agent = Agent(TestModel(call_tools=['lookup']), capabilities=[Slack(token='xoxp-hosted')])

        await agent.run('lookup')

        assert offline_mcp.authorization_headers == ['Bearer xoxp-hosted']
        assert [call.name for call in offline_mcp.calls] == ['lookup']

    async def test_capability_factory_gives_each_run_its_own_token(self, offline_mcp: OfflineMCP) -> None:
        offline_mcp.tools = [types.Tool(name='lookup', inputSchema={'type': 'object', 'properties': {}})]
        tokens = {'U1': 'xoxp-first', 'U2': 'xoxp-second'}

        def slack_for_user(ctx: RunContext[str]) -> Slack[str]:
            return Slack(token=tokens[ctx.deps])

        agent = Agent(TestModel(call_tools=['lookup']), deps_type=str, capabilities=[slack_for_user])

        async def run_as(user_id: str) -> None:
            await agent.run('lookup', deps=user_id)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_as, 'U1')
            task_group.start_soon(run_as, 'U2')

        assert set(offline_mcp.authorization_headers) == {'Bearer xoxp-first', 'Bearer xoxp-second'}

    def test_combine_rejects_different_tokens_and_merges_equal_or_missing_tokens(self) -> None:
        with pytest.raises(
            UserError, match='Multiple Slack capabilities with different credentials cannot be combined.'
        ):
            Slack.combine([Slack(token='xoxp-one'), Slack(token='xoxp-two')])

        equal = Slack.combine([Slack(token='xoxp-one'), Slack(token='xoxp-one')])
        missing = Slack.combine([Slack(), Slack()])
        assert isinstance(equal, Slack)
        assert isinstance(missing, Slack)
        assert equal.token == 'xoxp-one'
        assert missing.token is None

    def test_capability_metadata_and_optional_token(self) -> None:
        capability = Slack(token='xoxp-user', description='Slack access', defer_loading=True)
        assert capability.id == 'slack'
        assert capability.description == 'Slack access'
        assert capability.defer_loading is True
        assert 'xoxp-user' not in repr(capability)
        assert Slack().get_toolset() is None

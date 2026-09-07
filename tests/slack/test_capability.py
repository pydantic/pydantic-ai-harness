"""Tests for the public Slack capability contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import anyio
import pytest
from mcp import types
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from slack_sdk.errors import SlackApiError

from pydantic_ai_harness.slack import Slack
from tests.slack.conftest import OfflineMCP  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class FakeResponse:
    data: dict[str, object]

    def __getitem__(self, key: str) -> object:
        return self.data[key]


class FakeAsyncWebClient:
    instances: list[FakeAsyncWebClient] = []
    raise_method: str | None = None

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.__class__.instances.append(self)

    def _response(self, method: str, kwargs: Mapping[str, object], data: dict[str, object]) -> FakeResponse:
        self.calls.append((method, dict(kwargs)))
        if self.raise_method == method:
            raise SlackApiError('request failed', FakeResponse({'ok': False}))
        return FakeResponse(data)

    async def chat_postMessage(self, **kwargs: str) -> FakeResponse:
        return self._response('chat_postMessage', kwargs, {'ok': True, 'ts': '2.1'})

    async def reactions_add(self, **kwargs: str) -> FakeResponse:
        return self._response('reactions_add', kwargs, {'ok': True})

    async def conversations_replies(self, **kwargs: object) -> FakeResponse:
        return self._response(
            'conversations_replies',
            kwargs,
            {'ok': True, 'messages': [{'user': 'U1', 'ts': '1.1', 'text': 'hello'}, {'ts': '2.1'}]},
        )


class TestSlack:
    @pytest.mark.parametrize(
        ('explicit', 'user_token', 'bot_token', 'expected'),
        [
            ('xoxp-explicit', 'xoxp-user', 'xoxb-bot', 'xoxp-explicit'),
            (None, 'xoxp-user', 'xoxb-bot', 'xoxp-user'),
        ],
        ids=['explicit-over-environment', 'user-environment-over-bot-environment'],
    )
    async def test_token_resolution_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        offline_mcp: OfflineMCP,
        explicit: str | None,
        user_token: str | None,
        bot_token: str | None,
        expected: str,
    ) -> None:
        offline_mcp.tools = [types.Tool(name='lookup', inputSchema={'type': 'object', 'properties': {}})]
        if user_token is None:
            monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)
        else:
            monkeypatch.setenv('SLACK_USER_TOKEN', user_token)
        if bot_token is None:
            monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        else:
            monkeypatch.setenv('SLACK_BOT_TOKEN', bot_token)

        agent = Agent(TestModel(call_tools=['lookup']), capabilities=[Slack(token=explicit)])
        await agent.run('lookup')

        assert offline_mcp.authorization_headers[-1] == f'Bearer {expected}'

    async def test_missing_token_is_only_reported_when_a_run_needs_slack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        capability = Slack()
        assert capability.token is None
        agent = Agent(TestModel(), capabilities=[capability])

        async with agent:
            pass

        with pytest.raises(UserError) as exc_info:
            await agent.run('use Slack')
        assert str(exc_info.value) == (
            'Slack tools need a token. Pass Slack(token=...) or set SLACK_USER_TOKEN or SLACK_BOT_TOKEN.'
        )

    async def test_bot_token_exposes_and_calls_the_three_slack_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeAsyncWebClient.instances.clear()
        monkeypatch.setattr('pydantic_ai_harness.slack._capability.AsyncWebClient', FakeAsyncWebClient)

        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Slack(token='xoxb-test')]).run('list Slack tools')
        assert model.last_model_request_parameters is not None
        assert {tool.name for tool in model.last_model_request_parameters.function_tools} == {
            'send_message',
            'add_reaction',
            'read_thread',
        }

        async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(
                parts=[
                    ToolCallPart('send_message', {'channel': 'C1', 'text': 'Hello', 'thread_ts': '1.2'}),
                    ToolCallPart('add_reaction', {'channel': 'C1', 'timestamp': '1.2', 'name': 'thumbsup'}),
                    ToolCallPart('read_thread', {'channel': 'C1', 'thread_ts': '1.1', 'limit': 250}),
                ]
            )

        await Agent(FunctionModel(respond), capabilities=[Slack(token='xoxb-test')]).run('use Slack')
        client = FakeAsyncWebClient.instances[-1]
        assert client.token == 'xoxb-test'
        assert ('chat_postMessage', {'channel': 'C1', 'thread_ts': '1.2', 'markdown_text': 'Hello'}) in client.calls
        assert ('reactions_add', {'channel': 'C1', 'timestamp': '1.2', 'name': 'thumbsup'}) in client.calls
        assert ('conversations_replies', {'channel': 'C1', 'ts': '1.1', 'limit': 200}) in client.calls

    @pytest.mark.parametrize(
        ('method', 'tool_name', 'arguments'),
        [
            ('chat_postMessage', 'send_message', {'channel': 'C1', 'text': 'Hello'}),
            ('reactions_add', 'add_reaction', {'channel': 'C1', 'timestamp': '1.2', 'name': 'thumbsup'}),
            ('conversations_replies', 'read_thread', {'channel': 'C1', 'thread_ts': '1.1'}),
        ],
        ids=['send-message', 'add-reaction', 'read-thread'],
    )
    async def test_bot_api_errors_are_returned_to_the_model_as_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        tool_name: str,
        arguments: dict[str, str],
    ) -> None:
        FakeAsyncWebClient.instances.clear()
        FakeAsyncWebClient.raise_method = method
        monkeypatch.setattr('pydantic_ai_harness.slack._capability.AsyncWebClient', FakeAsyncWebClient)
        retry_messages: list[ModelMessage] = []

        async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts):
                retry_messages.extend(messages)
                return ModelResponse(parts=[TextPart('recovered')])
            return ModelResponse(parts=[ToolCallPart(tool_name, arguments)])

        await Agent(FunctionModel(respond), capabilities=[Slack(token='xoxb-test')]).run('use Slack')
        assert any(
            isinstance(part, RetryPromptPart) and 'Slack API error:' in part.content
            for message in retry_messages
            for part in message.parts
        )
        FakeAsyncWebClient.raise_method = None

    async def test_non_bot_token_uses_the_hosted_mcp_server(self, offline_mcp: OfflineMCP) -> None:
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

"""Tests for the native Slack capability contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from functools import partial
from typing import Protocol

import anyio
import httpx
import pytest
from mcp import types
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.authorization.authorize_result import AuthorizeResult
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.web.async_client import AsyncWebClient

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.slack import Slack, SlackApp, current_slack_context

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


_SCHEMA: dict[str, object] = {
    'type': 'object',
    'properties': {
        'payload': {
            'type': 'object',
            'properties': {
                'items': {'type': 'array', 'items': {'type': 'integer'}},
                'enabled': {'type': 'boolean'},
            },
            'required': ['items', 'enabled'],
        },
        'label': {'type': 'string'},
    },
    'required': ['payload', 'label'],
}


def _tool(*, output_schema: dict[str, object] | None = None) -> types.Tool:
    return types.Tool(
        name='future_slack_tool',
        description='Future nested Slack tool',
        inputSchema=_SCHEMA,
        outputSchema=output_schema,
    )


class OfflineMCPProtocol(Protocol):
    tools: list[types.Tool]
    result: types.CallToolResult | None
    error: Exception | None
    error_once: bool
    block_calls: bool
    call_started: anyio.Event
    call_release: anyio.Event
    calls: list[MCPCallProtocol]
    authorization_headers: list[str]
    session_starts: list[str]
    http_clients: list[httpx.AsyncClient]


class MCPCallProtocol(Protocol):
    token: str
    name: str
    arguments: dict[str, object]


def _event(*, user: str = 'U1', team: str = 'T1', channel: str = 'C1', ts: str = '1.1') -> dict[str, object]:
    return {
        'type': 'app_mention',
        'team': team,
        'user': user,
        'channel': channel,
        'text': '<@UBOT> hello',
        'ts': ts,
    }


async def _dispatch(
    monkeypatch: pytest.MonkeyPatch,
    agent: Agent[None, str],
    *,
    token: str | None = 'xoxp-user',
    user: str = 'U1',
    team: str = 'T1',
    channel: str = 'C1',
    ts: str = '1.1',
) -> list[Mapping[str, object]]:
    posts: list[Mapping[str, object]] = []

    async def authorize(team_id: str | None, user_id: str | None) -> AuthorizeResult:
        assert team_id == team
        assert user_id == user
        return AuthorizeResult(
            enterprise_id='E1',
            team_id=team,
            user_id=user,
            user_token=token,
            bot_user_id='UBOT',
        )

    async def post_message(self: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
        del self
        posts.append(kwargs)
        return {'ok': True}

    async def set_status(self: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
        del self, kwargs
        return {'ok': True}

    monkeypatch.setattr(AsyncWebClient, 'chat_postMessage', post_message)
    monkeypatch.setattr(AsyncWebClient, 'agents_sessions_setStatus', set_status)
    bolt_app = AsyncApp(
        authorize=authorize,
        client=AsyncWebClient(token='xoxb-test'),
        process_before_response=True,
        request_verification_enabled=False,
        ignoring_self_events_enabled=False,
    )
    SlackApp(agent, app=bolt_app, allowed_users={team: {user}})
    response = await bolt_app.async_dispatch(
        AsyncBoltRequest(
            body={
                'type': 'event_callback',
                'team_id': team,
                'event': _event(user=user, team=team, channel=channel, ts=ts),
            },
            mode='socket_mode',
        )
    )
    assert response.status == 200
    return posts


class TestSlack:
    def test_metadata_and_old_arguments(self) -> None:
        capability = Slack(description='native Slack', defer_loading=True)
        assert capability.id == 'slack'
        assert capability.description == 'native Slack'
        assert capability.defer_loading is True
        assert {field.name for field in fields(Slack)} == {'id', 'description', 'defer_loading', '_dynamic_toolset'}
        with pytest.raises(TypeError):
            Slack(tools=[])  # type: ignore[call-arg]

    def test_instructions_are_static_and_do_not_claim_authorization(self) -> None:
        instructions = Slack().get_instructions()
        assert isinstance(instructions, str)
        assert 'authorization boundary' in instructions
        assert 'security boundary' not in instructions

    async def test_native_mcp_schema_args_result_instructions_and_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        offline_mcp.result = types.CallToolResult(
            content=[types.TextContent(type='text', text='text-result')],
            structuredContent={'nested': {'number': 7}},
        )
        model = TestModel(call_tools=['future_slack_tool'])
        posts = await _dispatch(monkeypatch, Agent(model, capabilities=[Slack()]), token='xoxp-a')

        assert len(offline_mcp.calls) == 1
        assert offline_mcp.calls[0].name == 'future_slack_tool'
        assert offline_mcp.calls[0].arguments == {'payload': {'items': [0], 'enabled': False}, 'label': 'a'}
        assert 'number' in str(posts[-1])
        assert model.last_model_request_parameters is not None
        tool_def = model.last_model_request_parameters.function_tools[0]
        assert tool_def.name == 'future_slack_tool'
        assert tool_def.description == 'Future nested Slack tool'
        assert tool_def.parameters_json_schema == _SCHEMA
        assert offline_mcp.session_starts == ['Bearer xoxp-a']
        assert offline_mcp.http_clients
        assert all(client.is_closed for client in offline_mcp.http_clients)

    async def test_server_instructions_reach_model(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        captured: list[ModelMessage] = []

        async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            captured.extend(messages)
            return ModelResponse(parts=[TextPart('done')])

        await _dispatch(monkeypatch, Agent(FunctionModel(respond), capabilities=[Slack()]))
        assert any(
            isinstance(message, ModelRequest)
            and message.instructions is not None
            and 'Offline MCP server instructions.' in message.instructions
            for message in captured
        )

    async def test_native_mcp_text_result_reaches_model(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        offline_mcp.result = types.CallToolResult(content=[types.TextContent(type='text', text='text-result')])
        posts = await _dispatch(
            monkeypatch,
            Agent(TestModel(call_tools=['future_slack_tool']), capabilities=[Slack()]),
            token='xoxp-text',
        )
        assert 'text-result' in str(posts[-1])

    async def test_prepare_tools_hides_native_slack_tool_from_model(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        observed: list[str] = []

        async def prepare_tools(_ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            observed.extend(tool.name for tool in tool_defs)
            return [tool for tool in tool_defs if tool.name != 'future_slack_tool']

        offered: list[str] = []

        async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.extend(tool.name for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        await _dispatch(
            monkeypatch,
            Agent[None, str](  # pyright: ignore[reportArgumentType, reportCallIssue]
                FunctionModel(respond), capabilities=[Slack[None](), PrepareTools[None](prepare_tools)]
            ),
            token='xoxp-prepared',
        )
        assert 'future_slack_tool' in observed
        assert 'future_slack_tool' not in offered
        assert offline_mcp.http_clients
        assert all(client.is_closed for client in offline_mcp.http_clients)

    async def test_code_mode_calls_native_slack_tool_and_propagates_result(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool(output_schema={'type': 'object'})]
        offline_mcp.result = types.CallToolResult(
            content=[types.TextContent(type='text', text='provider-result')],
            structuredContent={'answer': 'provider-result'},
        )
        responses = 0

        async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal responses
            responses += 1
            if responses == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'run_code',
                            {
                                'code': (
                                    'result = await future_slack_tool('
                                    "payload={'items': [4, 5], 'enabled': True}, label='from-code')\nresult"
                                )
                            },
                        )
                    ]
                )
            tool_return = next(
                part
                for message in messages
                for part in getattr(message, 'parts', ())
                if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
            )
            return ModelResponse(parts=[TextPart(f'provider saw {tool_return.content}')])

        posts = await _dispatch(
            monkeypatch,
            Agent(FunctionModel(respond), capabilities=[Slack(), CodeMode()]),
            token='xoxp-code-mode',
        )
        assert responses == 2
        assert len(offline_mcp.calls) == 1
        assert offline_mcp.calls[0].name == 'future_slack_tool'
        assert offline_mcp.calls[0].arguments == {
            'payload': {'items': [4, 5], 'enabled': True},
            'label': 'from-code',
        }
        assert 'provider-result' in str(posts[-1])
        assert offline_mcp.http_clients
        assert all(client.is_closed for client in offline_mcp.http_clients)

    async def test_distinct_overlapping_users_get_distinct_credentials_and_missing_identity_does_not_reuse(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        offline_mcp.block_calls = True
        slack = Slack()
        agent = Agent(TestModel(call_tools=['future_slack_tool']), capabilities=[slack])
        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(_dispatch, monkeypatch, agent, token='xoxp-first', user='U1', ts='1.1'))
            await offline_mcp.call_started.wait()
            tg.start_soon(partial(_dispatch, monkeypatch, agent, token='xoxp-second', user='U2', ts='2.1'))
            with anyio.fail_after(2):
                while len(offline_mcp.calls) < 2:
                    await anyio.sleep(0)
            assert {'Bearer xoxp-first', 'Bearer xoxp-second'} <= set(offline_mcp.authorization_headers)
            offline_mcp.call_release.set()
        posts = await _dispatch(monkeypatch, agent, token=None, user='U3', ts='3.1')
        assert posts[-1]['markdown_text'] == 'Connect your Slack account before using this agent.'
        assert offline_mcp.authorization_headers.count('Bearer xoxp-first') == 1
        assert offline_mcp.authorization_headers.count('Bearer xoxp-second') == 1
        assert len(offline_mcp.authorization_headers) == 2
        assert all(client.is_closed for client in offline_mcp.http_clients)

    async def test_server_model_retry_reaches_core_and_model_corrects_args(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        attempts = 0

        async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('future_slack_tool', {'payload': {'items': [1], 'enabled': True}, 'label': 'ok'})
                    ]
                )
            if attempts == 2:
                assert any(
                    isinstance(part, RetryPromptPart) for message in messages for part in getattr(message, 'parts', ())
                )
                offline_mcp.error = None
                return ModelResponse(
                    parts=[
                        ToolCallPart('future_slack_tool', {'payload': {'items': [2], 'enabled': True}, 'label': 'ok'})
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        offline_mcp.error = ModelRetry('server rejected the first call')
        offline_mcp.error_once = True
        posts = await _dispatch(monkeypatch, Agent(FunctionModel(respond), capabilities=[Slack()]))
        assert attempts == 3
        assert len(offline_mcp.calls) == 2
        assert offline_mcp.calls[-1].arguments['label'] == 'ok'
        assert posts[-1]['markdown_text']

    def test_agent_from_spec_defaults_and_legacy_keys_fail(self) -> None:
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'Slack': {'description': 'from spec'}}]},
            custom_capability_types=[Slack],
        )
        assert agent.name is None
        with pytest.raises((AttributeError, TypeError, ValueError)):
            Slack.from_spec(tools=['search'])  # type: ignore[call-arg]

    async def test_agent_from_spec_runs_native_slack(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        model = TestModel(call_tools=['future_slack_tool'])
        agent = Agent.from_spec(
            {'capabilities': [{'Slack': {'description': 'from spec'}}]},
            custom_capability_types=[Slack],
            model=model,
        )
        await _dispatch(monkeypatch, agent, token='xoxp-spec')  # pyright: ignore[reportArgumentType]
        assert len(offline_mcp.calls) == 1
        assert offline_mcp.calls[0].token == 'Bearer xoxp-spec'
        assert offline_mcp.http_clients
        assert all(client.is_closed for client in offline_mcp.http_clients)

    def test_two_defaults_combine_and_defer_loading_is_metadata(self) -> None:
        first = Slack()
        second = Slack()
        Agent(TestModel(), capabilities=[first, second])
        assert first.defer_loading is False

    async def test_defer_loading_exposes_capability_loader_until_revealed(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        model = TestModel(call_tools=[])
        await _dispatch(
            monkeypatch,
            Agent(model, capabilities=[Slack(defer_loading=True)]),
            token='xoxp-deferred',
        )
        assert model.last_model_request_parameters is not None
        assert [tool.name for tool in model.last_model_request_parameters.function_tools] == [
            'load_capability',
            'future_slack_tool',
        ]
        assert model.last_model_request_parameters.function_tools[1].defer_loading is True
        assert offline_mcp.calls == []

    async def test_context_is_restored_after_success_and_exception(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        contexts: list[object] = []

        async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            contexts.append(current_slack_context())
            return ModelResponse(parts=[TextPart('done')])

        await _dispatch(monkeypatch, Agent(FunctionModel(respond), capabilities=[Slack()]))
        assert contexts and contexts[-1] is not None
        assert current_slack_context() is None
        assert offline_mcp.http_clients
        assert all(client.is_closed for client in offline_mcp.http_clients)

        async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            contexts.append(current_slack_context())
            raise RuntimeError('boom')

        await _dispatch(monkeypatch, Agent(FunctionModel(fail), capabilities=[Slack()]), ts='1.2')
        assert contexts[-1] is not None
        assert current_slack_context() is None
        assert all(client.is_closed for client in offline_mcp.http_clients)

    async def test_nested_public_hosts_restore_outer_context(
        self, monkeypatch: pytest.MonkeyPatch, offline_mcp: OfflineMCPProtocol
    ) -> None:
        offline_mcp.tools = [_tool()]
        contexts: list[tuple[str, str]] = []

        async def inner_respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            context = current_slack_context()
            assert context is not None
            contexts.append(('inner', context.user_id))
            return ModelResponse(parts=[TextPart('inner done')])

        inner = Agent(FunctionModel(inner_respond), capabilities=[Slack()])

        async def outer_respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            context = current_slack_context()
            assert context is not None
            contexts.append(('outer-before', context.user_id))
            await _dispatch(monkeypatch, inner, user='U2', channel='C2', ts='2.1', token='xoxp-inner')
            context = current_slack_context()
            assert context is not None
            contexts.append(('outer-after', context.user_id))
            return ModelResponse(parts=[TextPart('outer done')])

        await _dispatch(
            monkeypatch,
            Agent(FunctionModel(outer_respond), capabilities=[Slack()]),
            token='xoxp-outer',
        )
        assert contexts == [('outer-before', 'U1'), ('inner', 'U2'), ('outer-after', 'U1')]
        assert current_slack_context() is None
        assert all(client.is_closed for client in offline_mcp.http_clients)

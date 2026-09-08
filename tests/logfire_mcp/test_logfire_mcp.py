"""Tests for Logfire MCP through its public capability surface."""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire_mcp import LogfireMCP

pytestmark = pytest.mark.anyio


def tool_returns(messages: list[ModelMessage]) -> dict[str, dict[str, Any]]:
    """Map each executed tool to the structured content it returned."""
    returns: dict[str, dict[str, Any]] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                content = part.content  # pyright: ignore[reportUnknownMemberType]
                assert isinstance(content, dict), content
                returns[part.tool_name] = content  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
    return returns


def test_default_endpoint_is_the_us_region():
    toolset = LogfireMCP[None](auth='logfire-key').get_toolset()
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == 'https://logfire-us.pydantic.dev/mcp'


async def test_url_and_credential_reach_the_server(logfire_url: str):
    capability = LogfireMCP(url=logfire_url, auth='logfire-key')
    result = await Agent(TestModel(call_tools=['query_run']), capabilities=[capability]).run('Count errors')
    assert tool_returns(result.all_messages())['query_run']['authorization'] == 'Bearer logfire-key'


async def test_credential_falls_back_to_the_environment(logfire_url: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('LOGFIRE_MCP_TOKEN', 'env-key')
    capability = LogfireMCP(url=logfire_url)
    result = await Agent(TestModel(call_tools=['query_run']), capabilities=[capability]).run('Count errors')
    assert tool_returns(result.all_messages())['query_run']['authorization'] == 'Bearer env-key'


def test_credential_stays_out_of_repr():
    assert 'logfire-key' not in repr(LogfireMCP[None](auth='logfire-key'))


async def test_default_exposes_every_tool(logfire_url: str):
    capability = LogfireMCP(url=logfire_url, auth='logfire-key')
    result = await Agent(TestModel(), capabilities=[capability]).run('Set up a dashboard')
    assert set(tool_returns(result.all_messages())) == {'query_run', 'dashboard_create', 'unannotated_tool'}


async def test_read_only_exposes_only_tools_logfire_marks_read_only(logfire_url: str):
    capability = LogfireMCP(url=logfire_url, auth='logfire-key', read_only=True)
    result = await Agent(TestModel(), capabilities=[capability]).run('Set up a dashboard')
    assert set(tool_returns(result.all_messages())) == {'query_run'}


def test_agent_spec_round_trips(logfire_url: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('LOGFIRE_MCP_TOKEN', 'env-key')
    properties = AgentSpec.model_json_schema_with_capabilities([LogfireMCP])['$defs']['spec_params_LogfireMCP'][
        'properties'
    ]
    assert set(properties) == {'id', 'description', 'defer_loading', 'url', 'read_only'}
    agent = Agent.from_spec(
        {'capabilities': [{'LogfireMCP': {'url': logfire_url, 'read_only': True}}]},
        custom_capability_types=[LogfireMCP],
        model=TestModel(),
    )
    assert isinstance(agent, Agent)


def test_agent_spec_rejects_the_credential():
    with pytest.raises(ValueError, match="unexpected keyword argument 'auth'"):
        Agent.from_spec(
            {'capabilities': [{'LogfireMCP': {'auth': 'logfire-key'}}]},
            custom_capability_types=[LogfireMCP],
            model=TestModel(),
        )

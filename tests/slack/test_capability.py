"""Tests for the connection `Slack` hands to `MCPToolset`."""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp.client.auth import BearerAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset

from pydantic_ai_harness._mcp import is_read_only
from pydantic_ai_harness.slack import Slack


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own `SLACK_USER_TOKEN` must not decide what these tests assert."""
    monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)


def _transport(toolset: AbstractToolset[Any]) -> StreamableHttpTransport:
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def _tool(name: str, annotations: dict[str, bool] | None) -> ToolDefinition:
    return ToolDefinition(name=name, metadata={'annotations': annotations})


def test_endpoint_and_token_reach_the_transport():
    toolset = Slack(auth='xoxp-user').get_toolset()
    # The default hands back every tool Slack serves: no approval gate, no filter.
    assert type(toolset) is MCPToolset
    transport = _transport(toolset)

    assert transport.url == 'https://mcp.slack.com/mcp'
    assert isinstance(transport.auth, BearerAuth)
    assert transport.auth.token.get_secret_value() == 'xoxp-user'


def test_token_stays_out_of_repr():
    assert 'xoxp-user' not in repr(Slack(auth='xoxp-user'))


def test_token_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-from-env')

    auth = _transport(Slack().get_toolset()).auth

    assert isinstance(auth, BearerAuth)
    assert auth.token.get_secret_value() == 'xoxp-from-env'


def test_missing_token_raises_user_error():
    with pytest.raises(UserError, match='SLACK_USER_TOKEN'):
        Slack().get_toolset()


def test_read_only_keeps_only_the_tools_slack_marks_read_only():
    # Slack publishes no read-only endpoint, so the annotation filter is the whole mechanism.
    assert isinstance(Slack(auth='xoxp-user', read_only=True).get_toolset(), FilteredToolset)
    tools = [
        _tool('search', {'readOnlyHint': True}),
        _tool('post', {'readOnlyHint': False}),
        _tool('legacy', None),
        ToolDefinition(name='no_metadata'),
    ]

    assert [tool.name for tool in tools if is_read_only(tool)] == ['search']


def test_toolset_id_defaults_to_slack_and_follows_capability_id():
    assert Slack(auth='xoxp-user').get_toolset().id == 'slack'
    assert Slack(auth='xoxp-user', id='acme-slack').get_toolset().id == 'acme-slack'


def test_spec_round_trip_rebuilds_the_capability(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-from-env')

    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'Slack': {'id': 'acme-slack', 'read_only': True}}]},
        custom_capability_types=[Slack],
    )
    (capability,) = [c for c in agent.root_capability.capabilities if isinstance(c, Slack)]

    assert capability.id == 'acme-slack'
    assert capability.read_only is True


def test_a_spec_cannot_carry_the_token():
    with pytest.raises(ValueError, match='auth'):
        Agent.from_spec(
            {'model': 'test', 'capabilities': [{'Slack': {'auth': 'xoxp-secret'}}]},
            custom_capability_types=[Slack],
        )

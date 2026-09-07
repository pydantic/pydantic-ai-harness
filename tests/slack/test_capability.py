"""Tests for the Slack capability: which token it uses and what it hands to the MCP toolset."""

from __future__ import annotations

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FilteredToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.slack import Slack

SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@pytest.fixture(autouse=True)
def no_slack_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLACK_USER_TOKEN', raising=False)


def transport_of(capability: Slack[None]) -> StreamableHttpTransport:
    """Return the HTTP transport the capability's toolset was built with."""
    toolset = capability.get_toolset()
    if isinstance(toolset, FilteredToolset):
        toolset = toolset.wrapped
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestSlack:
    def test_token_becomes_the_bearer_header_for_slack_mcp(self) -> None:
        transport = transport_of(Slack(token='xoxp-explicit'))
        assert transport.url == SLACK_MCP_URL
        assert transport.headers == {'Authorization': 'Bearer xoxp-explicit'}

    def test_explicit_token_wins_over_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-environment')
        assert transport_of(Slack(token='xoxp-explicit')).headers == {'Authorization': 'Bearer xoxp-explicit'}

    def test_environment_token_is_used_when_none_is_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-environment')
        assert transport_of(Slack()).headers == {'Authorization': 'Bearer xoxp-environment'}

    @pytest.mark.parametrize('token', [None, ''], ids=['omitted', 'empty'])
    def test_missing_token_fails_at_construction(self, token: str | None) -> None:
        with pytest.raises(UserError, match=r'Pass Slack\(token=...\) or set SLACK_USER_TOKEN'):
            Slack(token=token)

    def test_empty_token_does_not_fall_back_to_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_USER_TOKEN', 'xoxp-environment')
        with pytest.raises(UserError):
            Slack(token='')

    def test_server_instructions_are_forwarded_if_slack_sends_any(self) -> None:
        toolset = Slack(token='xoxp-user').get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.include_instructions is True
        assert toolset.id == 'slack-mcp'

    def test_all_tools_are_exposed_by_default(self) -> None:
        assert isinstance(Slack(token='xoxp-user').get_toolset(), MCPToolset)

    def test_read_only_keeps_only_the_tools_slack_marks_read_only(self) -> None:
        toolset = Slack(token='xoxp-user', read_only=True).get_toolset()
        assert isinstance(toolset, FilteredToolset)
        assert transport_of(Slack(token='xoxp-user', read_only=True)).headers == {'Authorization': 'Bearer xoxp-user'}

        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
        schema: dict[str, object] = {'type': 'object', 'properties': {}}

        def tool(name: str, annotations: dict[str, bool] | None) -> ToolDefinition:
            return ToolDefinition(name=name, parameters_json_schema=schema, metadata={'annotations': annotations})

        keeps = toolset.filter_func
        assert keeps(ctx, tool('slack_read_thread', {'readOnlyHint': True})) is True
        assert keeps(ctx, tool('slack_send_message', {'readOnlyHint': False})) is False
        assert keeps(ctx, tool('unannotated', None)) is False

    def test_combine_rejects_different_tokens_and_merges_equal_ones(self) -> None:
        with pytest.raises(UserError, match='different credentials cannot be combined'):
            Slack.combine([Slack(token='xoxp-one'), Slack(token='xoxp-two')])

        merged = Slack.combine([Slack(token='xoxp-one'), Slack(token='xoxp-one')])
        assert isinstance(merged, Slack) and merged.token == 'xoxp-one'

    def test_token_is_kept_out_of_repr(self) -> None:
        capability = Slack(token='xoxp-user', description='Slack access', defer_loading=True)
        assert 'xoxp-user' not in repr(capability)
        assert capability.id == 'slack'
        assert capability.description == 'Slack access'
        assert capability.defer_loading is True

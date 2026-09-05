"""The Notion page-search/update example runs without live credentials."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ._support import NotionState

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _load_example() -> ModuleType:
    path = Path(__file__).parents[2] / 'examples' / 'notion_page_update.py'
    spec = importlib.util.spec_from_file_location('examples_notion_page_update_test', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def test_page_search_update_flow_uses_fake_server_and_approval(
    notion_server: FastMCP, notion_state: NotionState
) -> None:
    example = _load_example()
    step = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        calls = [
            ToolCallPart('notion-ai-search', {'query': 'launch plan'}, 'search'),
            ToolCallPart('notion-fetch', {'id': 'page-1'}, 'page'),
            ToolCallPart(
                'notion-update-page',
                {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Shipped'},
                'update',
            ),
        ]
        if step < len(calls):
            response = ModelResponse(parts=[calls[step]])
            step += 1
            return response
        return ModelResponse(parts=[TextPart('Updated https://notion.so/page-1')])

    approval_requests: list[ToolCallPart] = []

    attributions: list[str] = []

    def approve(call: ToolCallPart, attribution: str) -> bool:
        approval_requests.append(call)
        attributions.append(attribution)
        return True

    result = await example.run_task(
        'Find the launch plan and replace its content with: Shipped',
        model=FunctionModel(model),
        client=notion_server,
        approve=approve,
    )

    assert result == 'Updated https://notion.so/page-1'
    assert [call.tool_name for call in approval_requests] == ['notion-update-page']
    assert 'workspace-1' in attributions[0]
    assert 'user-1' in attributions[0]
    assert notion_state['page_content'] == 'Shipped'
    assert notion_state['calls'][:3] == [
        ('notion-fetch', {'id': 'self'}),
        ('notion-fetch', {'id': 'self'}),
        ('notion-ai-search', {'query': 'launch plan'}),
    ]


async def test_page_update_denial_leaves_page_unchanged(notion_server: FastMCP, notion_state: NotionState) -> None:
    example = _load_example()
    step = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'notion-update-page',
                        {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Rejected'},
                        'update',
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('Update was not approved.')])

    def deny(_call: ToolCallPart, _attribution: str) -> bool:
        return False

    result = await example.run_task(
        'Replace the launch plan',
        model=FunctionModel(model),
        client=notion_server,
        approve=deny,
    )

    assert result == 'Update was not approved.'
    assert notion_state['page_content'] == 'Old launch plan'
    assert all(name != 'notion-update-page' for name, _args in notion_state['calls'])


def test_terminal_approval_shows_identity_and_exact_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _load_example()
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return 'yes'

    monkeypatch.setattr('builtins.input', answer)
    call = ToolCallPart(
        'notion-update-page',
        {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Shipped'},
        'update',
    )

    assert example.terminal_approval(call, '{"workspace":"Acme","user":"Ada"}') is True
    assert 'Connected Notion identity' in prompts[0]
    assert '"workspace":"Acme"' in prompts[0]
    assert 'notion-update-page' in prompts[0]
    assert '"page_id": "page-1"' in prompts[0]
    assert '"new_str": "Shipped"' in prompts[0]


def test_terminal_approval_defaults_to_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _load_example()

    def answer(_prompt: str) -> str:
        return ''

    monkeypatch.setattr('builtins.input', answer)

    assert example.terminal_approval(ToolCallPart('notion-update-page', {}, 'update'), '{}') is False

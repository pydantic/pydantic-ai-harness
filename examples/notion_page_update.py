"""Search Notion and update one page after explicit terminal approval.

The first run opens Notion's OAuth flow. Set `PYDANTIC_AI_MODEL` and the model
provider's API key before running:

    uv run python examples/notion_page_update.py "Find the launch plan and replace its content with: Shipped"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable

from fastmcp import Client
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.mcp import MCPToolsetClient
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.tools import DeferredToolApprovalResult

from pydantic_ai_harness.notion import NOTION_MCP_URL, NotionToolset

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')


def build_agent(
    model: Model | str = DEFAULT_MODEL,
    *,
    client: MCPToolsetClient,
) -> Agent[None, str | DeferredToolRequests]:
    """Build an agent whose only write tool is `notion-update-page`."""
    agent, _ = _build_agent_and_toolset(model, client=client)
    return agent


def _build_agent_and_toolset(
    model: Model | str,
    *,
    client: MCPToolsetClient,
) -> tuple[Agent[None, str | DeferredToolRequests], NotionToolset[None]]:
    notion = NotionToolset[None](client=client, mutations='notion-update-page')
    notion_with_approval = notion.approval_required(
        lambda _ctx, tool_def, _args: (tool_def.metadata or {}).get('notion_mutation') is True
    )
    agent = Agent(
        model,
        deps_type=type(None),
        instructions=(
            'Use the connection attribution supplied by the Notion toolset. Search for the requested page, '
            'fetch the selected result, then propose the smallest page update. '
            'Preserve the page URL in the final response.'
        ),
        toolsets=[notion_with_approval],
        output_type=[str, DeferredToolRequests],
    )
    return agent, notion


async def run_task(
    prompt: str,
    *,
    model: Model | str = DEFAULT_MODEL,
    client: MCPToolsetClient,
    approve: Callable[[ToolCallPart, str], bool],
) -> str:
    """Run a Notion task and resolve mutation approvals with `approve`."""
    agent, notion = _build_agent_and_toolset(model, client=client)
    result = await agent.run(prompt)
    while isinstance(result.output, DeferredToolRequests):
        attribution = notion.attribution
        approvals: dict[str, bool | DeferredToolApprovalResult] = {
            call.tool_call_id: approve(call, attribution) for call in result.output.approvals
        }
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )
    return result.output


def terminal_approval(call: ToolCallPart, attribution: str) -> bool:
    """Show the exact proposed mutation and ask the terminal user to approve it."""
    arguments = json.dumps(call.args_as_dict(), indent=2, sort_keys=True)
    answer = input(f'Connected Notion identity:\n{attribution}\n\nApprove {call.tool_name}?\n{arguments}\n[y/N] ')
    return answer.strip().lower() in {'y', 'yes'}


def main() -> None:
    """Run one search/update task."""
    if len(sys.argv) != 2:
        raise SystemExit('Usage: notion_page_update.py "<search and update request>"')
    client = Client(NOTION_MCP_URL, auth='oauth')
    print(asyncio.run(run_task(sys.argv[1], client=client, approve=terminal_approval)))


if __name__ == '__main__':
    main()

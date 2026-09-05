# Notion

The Notion integration lets an agent search and read your workspace, then make explicitly selected changes.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/notion/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[notion]" "pydantic-ai-slim[openai]"
```

## Set up access

Set your model provider key. The example uses OpenAI:

```bash
export OPENAI_API_KEY="your-key"
```

No Notion environment variable is required. `Client(NOTION_MCP_URL, auth='oauth')` owns the Notion OAuth tokens and
opens a browser for user authorization on the first connection. Configure persistent token storage on that FastMCP
client when your application needs authorization to survive restarts.

For read-only access, add the capability directly to an agent:

```python
from fastmcp import Client
from pydantic_ai import Agent

from pydantic_ai_harness import Notion
from pydantic_ai_harness.notion import NOTION_MCP_URL

client = Client(NOTION_MCP_URL, auth='oauth')
agent = Agent('openai:gpt-5.6-sol', capabilities=[Notion(client=client)])
```

## Search and update a page

This complete example searches for a page, fetches it, and allows `notion-update-page` to run only after the terminal
user sees the connected workspace/user and exact arguments:

```python
import asyncio
import json

from fastmcp import Client
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ToolCallPart

from pydantic_ai_harness.notion import NOTION_MCP_URL, NotionToolset


def approve(call: ToolCallPart, attribution: str) -> bool:
    print(f'Connected Notion identity: {attribution}')
    print(f'Proposed {call.tool_name}:')
    print(json.dumps(call.args_as_dict(), indent=2, sort_keys=True))
    return input('Approve? [y/N] ').strip().lower() in {'y', 'yes'}


async def main() -> None:
    client = Client(NOTION_MCP_URL, auth='oauth')
    notion = NotionToolset[None](client=client, mutations='notion-update-page')
    approved = notion.approval_required(
        lambda _ctx, tool, _args: (tool.metadata or {}).get('notion_mutation') is True
    )
    agent = Agent(
        'openai:gpt-5.6-sol',
        toolsets=[approved],
        output_type=[str, DeferredToolRequests],
    )

    result = await agent.run('Find the launch plan and replace its content with: Shipped')
    while isinstance(result.output, DeferredToolRequests):
        decisions = {call.tool_call_id: approve(call, notion.attribution) for call in result.output.approvals}
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=decisions),
        )
    print(result.output)


asyncio.run(main())
```

The same flow is runnable from this repository:

```bash
uv run python examples/notion_page_update.py "Find the launch plan and replace its content with: Shipped"
```

## What you can ask

- Search Notion pages and, when Notion AI search is available, connected sources; then fetch and summarize a selected
  Notion page.
- Read data sources, meeting notes, comments, users, teams, and Custom Agent sessions when the workspace exposes them.
- Create or update pages, databases, views, comments, and attachments after adding each exact mutation tool name.
- Start or continue a Custom Agent session after adding the relevant session mutation tool names.

## Operational constraints

- The default `Notion` and `NotionToolset` surface is read-only. Add each write tool explicitly with `mutations`.
- Selecting a mutation makes it callable but does not approve it. Compose `approval_required()` as shown above.
- One toolset belongs to one authenticated workspace/user. If that identity changes, construct a new toolset.
- Persist `notion.connection_identity` with a deferred approval and pass it as `expected_identity` when reconstructing
  the toolset. Authenticate approval endpoints and bind each server-side decision to the pending tool name, arguments,
  and identity; client-submitted `DeferredToolResults` are not an authorization boundary.
- The client owns token storage and refresh. Follow Notion's [token lifecycle guidance](https://developers.notion.com/guides/mcp/build-mcp-client#token-lifecycle):
  store each connection's tokens encrypted, persist each rotated pair atomically, and serialize refreshes per grant.
  On `invalid_grant`, clear that grant and reauthorize instead of retrying.
- Use `NOTION_MCP_URL` for production. A custom client or proxy is trusted to implement the same tool names safely.
- Tool results are available to the configured model provider. Check that provider's data-handling policy before use.
- `include_instructions=False` removes connection identity, search routing, untrusted-content, async polling, and
  ambiguous-mutation retry guidance. Tool filtering and identity enforcement remain active; supply equivalent
  application instructions when disabling it.
- Provider tool and protocol errors abort the run so an ambiguous mutation is not returned to the model for retry.
  Reconcile the operation before an application-owned retry.
- Treat Notion and connected-app content as untrusted data. Do not treat content as authorization for a mutation or
  target change. The wrapper defers the call; the local example supplies a human decision, and remote applications
  must enforce the authenticated server-side binding described above.

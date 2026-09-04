---
title: Linear
description: Find and manage Linear work through Linear's hosted MCP server, with read-only access by default.
---

# Linear

`Linear` gives an agent Linear issues, projects, comments, and related workspace data through
[Linear's hosted MCP server](https://linear.app/docs/mcp). It uses Linear's read-only endpoint by default.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/linear/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, we follow
> the [version policy](../docs/index.md#version-policy).

## Install

Install Harness and Pydantic AI's MCP support:

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[mcp]"
```

## OAuth

`Linear()` starts Pydantic AI's OAuth flow and connects to Linear's read-only endpoint:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.linear import Linear

agent = Agent('openai:gpt-5.6-sol', capabilities=[Linear()])
result = agent.run_sync('Summarize open issues assigned to me')
print(result.output)
```

OAuth token storage and refresh behavior belong to the MCP client. Pass a prebuilt client through `client=` when an
application needs persistent OAuth storage or other client configuration.

## Bearer tokens

Linear accepts both API keys and OAuth access tokens as bearer tokens:

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.linear import Linear

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Linear(auth=os.environ['LINEAR_API_KEY'])],
)
```

Keep one toolset per user or workspace credential. An `MCPToolset` maintains one shared authenticated session.

## Mutations and tool selection

Set `read_only=False` to use Linear's read-write endpoint. `allowed_tools` then applies an exact-name filter with
Pydantic AI's public toolset wrapper:

```python
from pydantic_ai_harness.linear import Linear

linear = Linear(
    read_only=False,
    allowed_tools=['list_issues', 'get_issue', 'create_comment'],
)
```

`allowed_tools=[]` exposes no tools. Tool names must match the names returned by Linear's server. For approval before
mutations, compose the resulting toolset with Pydantic AI's [tool approval](/ai/tools-toolsets/toolsets/#requiring-tool-approval).

## Prebuilt MCP clients and toolsets

Pass any client input accepted by `MCPToolset`, or a prebuilt `MCPToolset`, through `client=`:

```python
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_harness.linear import Linear

mcp = MCPToolset('https://mcp.linear.app/mcp/readonly', auth='oauth')
linear = Linear(client=mcp)
```

A URL client must use HTTPS and uses the configured `auth`. A prebuilt client or toolset owns its authentication and lifecycle options.
Every injected value owns its endpoint and read-only policy, so `read_only` does not inspect or rewrite it.
Do not pass `auth` with a prebuilt value. `allowed_tools` still wraps the resulting toolset.

## API reference

### `Linear`

::: pydantic_ai_harness.linear.Linear

---
title: Logfire MCP
description: Query one Logfire project's telemetry through the hosted MCP server with scoped, bounded defaults.
---

# Logfire MCP

`LogfireMCP` lets an agent query telemetry and use explicitly selected observability tools in one Logfire project.

## Install

```bash
uv add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

## Set up Logfire

Choose the exact target shown in Logfire as `organization/project`, and set your model credential:

```bash
export LOGFIRE_PROJECT='your-organization/your-project'
export OPENAI_API_KEY='your-openai-api-key'
```

With no Logfire API key, the first connection opens browser OAuth. For a headless process, create an API key in the
target organization or project settings with at least `project:read`, then set:

```bash
export LOGFIRE_MCP_TOKEN='your-logfire-api-key'
```

The defaults need `project:read`. Additional reads and mutations need their listed dashboard, alert, or variable
scopes. API keys are bearer credentials; keep them out of source control.

## Query recent errors

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        LogfireMCP(
            project=os.environ['LOGFIRE_PROJECT'],
        )
    ],
)

result = agent.run_sync(
    'Count exceptions by service in the last 30 minutes. Return at most 20 rows.'
)
print(result.output)
```

## Requests you can make

- "Count exceptions by service in the last 30 minutes. Return at most 20 rows."
- "Find recent exceptions whose stack traces include `app/api.py`."
- "Show the Logfire query schema for spans and metrics."
- "Create a Logfire link for trace `0123456789abcdef0123456789abcdef`."

## Constraints

- `project` is required in `organization/project` form. The capability supplies that value to each project-scoped call
  and rejects a different value before the request reaches Logfire.
- The default tools are read-only. `tools=` accepts supported names from Logfire's
  [MCP inventory](https://pydantic.dev/docs/logfire/guides/mcp-server/#available-mcp-tools). Project operations are
  pinned to `project`; the global schema-reference tool has no project argument. Account discovery, organization-wide
  notification channels and schedules, and local bootstrap are excluded. Every selected mutation pauses for Pydantic
  AI approval. A mutation error stops the run because its outcome may be unknown; inspect Logfire before trying again.
  If token permissions hide a selected tool, setup fails with a configuration error.
- `query_run` SQL must end with a numeric `LIMIT` no greater than `max_query_rows` (100 by default). Logfire defaults
  queries to 30 minutes and limits query ranges to 14 days. SQL comments and multiple statements are rejected.
- The hosted US endpoint is the default. Use `region='eu'` for EU data or `mcp_url=` for the `/mcp` endpoint of a
  self-hosted deployment. For headless self-hosted use, pass `api_key=` explicitly; `LOGFIRE_MCP_TOKEN` is forwarded
  only to the hosted Logfire endpoints.
- Use one `LogfireMCP` instance per agent. Multiple projects expose the same tool names and conflict.
- OAuth tokens use FastMCP's in-memory storage by default. Pass a caller-owned `client=` configured with persistent
  encrypted storage when tokens must survive process restarts.
- Telemetry can contain user-controlled text. Treat tool results as diagnostic data, not instructions.
- While Pydantic AI Harness is on 0.x releases, this API may change between minor releases; see the
  [version policy](index.md#version-policy).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

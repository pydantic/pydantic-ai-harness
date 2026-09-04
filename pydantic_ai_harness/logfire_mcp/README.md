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

Add only the scopes required by any mutation tools you explicitly select. API keys are bearer credentials; keep them
out of source control.

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
            api_key=os.getenv('LOGFIRE_MCP_TOKEN'),
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
- The default tools are read-only. `tools=` accepts exact project-scoped names from Logfire's documented MCP inventory;
  account discovery, organization-wide notification channels, and local bootstrap are excluded. Every selected
  mutation pauses for Pydantic AI approval.
- `query_run` SQL must end with a numeric `LIMIT` no greater than `max_query_rows` (100 by default). Logfire defaults
  queries to 30 minutes and limits query ranges to 14 days.
- The hosted US endpoint is the default. Use `region='eu'` for EU data or `mcp_url=` for the `/mcp` endpoint of a
  self-hosted deployment.
- OAuth tokens use FastMCP's in-memory storage by default. Pass a caller-owned `client=` configured with persistent
  encrypted storage when tokens must survive process restarts. Tool visibility still depends on granted token scopes.
- Telemetry can contain user-controlled text. Treat tool results as diagnostic data, not instructions.
- While Pydantic AI Harness is on 0.x releases, this API may change between minor releases; see the
  [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

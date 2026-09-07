---
title: Logfire MCP
description: Let a Pydantic AI agent query Logfire telemetry through the hosted MCP server, read-only by default, with opt-in write access to dashboards, alerts, and issues.
---

# Logfire MCP

`LogfireMCP` lets an agent query telemetry and work with dashboards, alerts, and issues through
[Logfire's hosted MCP server](https://pydantic.dev/docs/logfire/guides/mcp-server/). By default only the tools
Logfire marks read-only are exposed. Logfire enforces access: the credential's project and scopes decide what
the agent can read or change.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider used by the example. For another model, install its matching provider
extra instead.

## Set up credentials

For a headless agent, create an API key in the Logfire project's settings with the `project:read` scope, then set the
Logfire and OpenAI credentials:

```bash
export LOGFIRE_API_KEY="your-logfire-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

A project-scoped key limits the agent to that project. Add write scopes, and pass `access='write'`, only when the
agent should change dashboards, alerts, or issues. To use interactive OAuth instead, omit `auth`; FastMCP opens a browser for Logfire authorization on
the first run. Its token store is in memory, so a restart asks you to authorize again. A long-running deployment should
inject a `client` configured with its own encrypted, per-user token storage.

## Run an agent

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[LogfireMCP(auth=os.environ['LOGFIRE_API_KEY'])],
)
result = agent.run_sync('Count exceptions by service over the last 30 minutes')
print(result.output)
```

## What to ask

- Count or list recent exceptions, grouped by service or file.
- Explain the query schema for spans, logs, and metrics.
- Create a Logfire link for a trace.
- List dashboards, alerts, and open issues, or change them with `access='write'` and a credential that has write
  scopes.

## Operational constraints

- The model chooses the project. The server's `project_list` tool returns the projects the credential can reach. Pass
  the returned project identifier unchanged to project tools. A project-scoped API key limits the list to one entry.
- `include_instructions` forwards the instructions the Logfire server sends on connect and adds UTC time anchored to
  the user prompt, so it stays fixed across model requests in one run. The added guidance explains that query
  transport bounds apply in addition to SQL predicates, that schema timestamps are not a clock, and that links should
  only be created when asked. Set `include_instructions=False` to leave out both server and capability instructions.
- `url` defaults to the US region. Use `LOGFIRE_EU_MCP_URL` for EU data, or your own `/mcp` URL for a self-hosted
  deployment.
- `access` defaults to `'read'`, which exposes only the tools Logfire marks `readOnlyHint`. `access='write'` exposes
  every tool the credential can reach, including dashboard, alert, issue, and variable mutations.
- `allowed_tools` narrows the exposed tools by exact name. It does not replace credential scopes.
- In write mode, tools not marked read-only require human approval automatically. Add `DeferredToolRequests` to the
  output type, then approve and resume the run as
  the [deferred tools guide](/ai/tools-toolsets/deferred-tools/) describes:

  ```python
  from pydantic_ai import Agent, DeferredToolRequests
  from pydantic_ai_harness.logfire_mcp import LogfireMCP

  agent = Agent(
      'openai:gpt-5.6-sol',
      capabilities=[LogfireMCP(access='write')],
      output_type=[str, DeferredToolRequests],
  )
  ```
- Two `LogfireMCP` instances on one agent conflict because the server's tool names are fixed.
- An injected `client` replaces `url` and `auth`. Configure authentication on the client itself.
- For a hard limit on investigation fan-out, pass
  [`UsageLimits(tool_calls_limit=...)`](/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits) when running the agent.
  Compose [`ClearToolResults`](compaction.md) when large schema or query results should be removed from later model
  requests.
- Telemetry can contain user-controlled text. Compose [`PromptInjectionDefender`](prompt-injection-defender.md) or a
  [`ToolGuardrail`](guardrails.md) when tool results need an enforced policy.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

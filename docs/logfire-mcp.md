---
title: Logfire MCP
description: Let a Pydantic AI agent query Logfire telemetry and manage dashboards, alerts, and issues through the hosted MCP server.
---

# Logfire MCP

`LogfireMCP` lets an agent query your telemetry and work with dashboards, alerts, and issues through
[Logfire's hosted MCP server](https://pydantic.dev/docs/logfire/guides/mcp-server/). The default exposes the write
tools as well; the credential's project and scopes are the real boundary.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider used by the example. For another model, install its matching provider
extra instead.

## Set up credentials

Create an API key with at least the `project:read` scope in the Logfire project's settings, and export it alongside
your model provider's credential:

```bash
export LOGFIRE_MCP_TOKEN="your-logfire-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

With no key set, the capability authorizes through the browser on first use, which suits local work but not a
deployed agent. The scopes on the key govern what Logfire will serve it. The token is only ever read from
`LOGFIRE_MCP_TOKEN` or passed as `auth=`; an agent spec file has no `auth` field, so it cannot hold one.

## Run an agent

```python
from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent('openai:gpt-5.6-sol', capabilities=[LogfireMCP()])
result = agent.run_sync('Count exceptions by service over the last 30 minutes')
print(result.output)
```

## Operational constraints

- `read_only=True` exposes only the tools Logfire marks read-only; a tool it does not annotate counts as a write and
  stays hidden.
- `url` defaults to the US region. Use `https://logfire-eu.pydantic.dev/mcp` for EU data, or your own `/mcp` URL for a
  self-hosted deployment.
- To hold writes for human confirmation, apply the
  [tool approval](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/#requiring-tool-approval)
  to the toolset.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

## API reference

::: pydantic_ai_harness.logfire_mcp.LogfireMCP

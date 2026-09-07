# Logfire MCP

`LogfireMCP` lets an agent query telemetry and work with dashboards, alerts, and issues through
[Logfire's hosted MCP server](https://pydantic.dev/docs/logfire/guides/mcp-server/). Logfire enforces access:
the credential's project and scopes decide what the agent can read or change.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

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

A project-scoped key limits the agent to that project. Add write scopes only when the agent should change dashboards,
alerts, or issues. To use interactive OAuth instead, omit `auth`; FastMCP opens a browser for Logfire authorization on
the first run. Its token store is in memory, so a restart asks you to authorize again. A long-running deployment should
inject a `client` configured with its own encrypted, per-user token storage.

## Run an agent

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[LogfireMCP(project='acme/production', auth=os.environ['LOGFIRE_API_KEY'])],
)
result = agent.run_sync('Count exceptions by service in the last 30 minutes')
print(result.output)
```

## What to ask

- Count or list recent exceptions, grouped by service or file.
- Explain the query schema for spans, logs, and metrics.
- Create a Logfire link for a trace.
- List dashboards, alerts, and open issues, or change them when the credential has write scopes.

## Operational constraints

- `project` is removed from the tool schemas the model sees and added to every call that takes one, so the model can
  neither name nor change the project. Leave it unset to let the model choose among the projects the credential can
  reach.
- `url` defaults to the US region. Use `LOGFIRE_EU_MCP_URL` for EU data, or your own `/mcp` URL for a self-hosted
  deployment.
- `allowed_tools` narrows the exposed tools by exact name. It does not replace credential scopes.
- Mutation tools do not require human approval automatically. When a person must approve calls, register the wrapped
  toolset instead of the capability:

  ```python
  from pydantic_ai import Agent
  from pydantic_ai_harness.logfire_mcp import LogfireMCP

  logfire = LogfireMCP(project='acme/production')
  agent = Agent(
      'openai:gpt-5.6-sol',
      instructions=logfire.get_instructions(),
      toolsets=[logfire.get_toolset().approval_required()],
  )
  ```
- Two `LogfireMCP` instances on one agent conflict because the server's tool names are fixed.
- An injected `client` replaces `url`. `auth` still applies when the client is a URL, and is ignored for an
  in-process server or a prebuilt FastMCP client.
- Telemetry can contain user-controlled text. Treat tool results as data, not instructions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

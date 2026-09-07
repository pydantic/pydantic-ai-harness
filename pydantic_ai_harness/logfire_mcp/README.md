# Logfire MCP

`LogfireMCP` lets an agent query telemetry and work with dashboards, alerts, and issues through
[Logfire's hosted MCP server](https://pydantic.dev/docs/logfire/guides/mcp-server/). By default only the tools
Logfire marks read-only are exposed. Logfire enforces access: the credential's project and scopes decide what
the agent can read or change.

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
result = agent.run_sync('Count exceptions by service in acme/production over the last 30 minutes')
print(result.output)
```

## What to ask

- Count or list recent exceptions, grouped by service or file.
- Explain the query schema for spans, logs, and metrics.
- Create a Logfire link for a trace.
- List dashboards, alerts, and open issues, or change them with `access='write'` and a credential that has write
  scopes.

## Operational constraints

- The model chooses the project. The server's `project_list` tool returns the projects the credential can reach, and
  a project-scoped API key limits that list to one entry.
- The capability adds no instructions of its own. `include_instructions` forwards the instructions the Logfire server
  sends on connect; set it to `False` to leave them out.
- `url` defaults to the US region. Use `LOGFIRE_EU_MCP_URL` for EU data, or your own `/mcp` URL for a self-hosted
  deployment.
- `access` defaults to `'read'`, which exposes only the tools Logfire marks `readOnlyHint`. `access='write'` exposes
  every tool the credential can reach, including dashboard, alert, issue, and variable mutations.
- `allowed_tools` narrows the exposed tools by exact name. It does not replace credential scopes.
- In write mode, mutation tools do not require human approval automatically. When a person must approve calls,
  register the wrapped toolset instead of the capability, add `DeferredToolRequests` to the output type, then approve
  and resume the run as
  the [deferred tools guide](https://ai.pydantic.dev/deferred-tools/) describes:

  ```python
  from pydantic_ai import Agent, DeferredToolRequests
  from pydantic_ai_harness.logfire_mcp import LogfireMCP

  agent = Agent(
      'openai:gpt-5.6-sol',
      toolsets=[LogfireMCP(access='write').get_toolset().approval_required()],
      output_type=[str, DeferredToolRequests],
  )
  ```
- Two `LogfireMCP` instances on one agent conflict because the server's tool names are fixed.
- An injected `client` replaces `url` and `auth`. Configure authentication on the client itself.
- Telemetry can contain user-controlled text. Add a guard when tool results must not influence other actions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

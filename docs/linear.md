---
title: Linear
description: Let a Pydantic AI agent read or update Linear work through Linear's hosted MCP server.
---

# Linear

`Linear` lets an agent read or update issues, projects, comments, and teams through Linear's hosted MCP server.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[linear]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider used by the example. For another model, install its matching provider
extra instead.

## Set up credentials

For bearer authentication, create a Linear API key under **Settings > Account > Security & Access > Personal API
keys**, then set the Linear and OpenAI credentials:

```bash
export LINEAR_ACCESS_TOKEN="your-linear-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

Give the Linear key the `Read` permission for the default read-only connection. To use interactive OAuth instead, pass
`auth='oauth'` and omit `LINEAR_ACCESS_TOKEN`. FastMCP handles OAuth and opens a browser for Linear authorization on the
first run. For a headless job, use a bearer token. To reuse OAuth authorization across runs, inject a FastMCP client
configured with encrypted persistent token storage that is independently namespaced for each user and workspace;
FastMCP's default token storage is in memory.

## Run an agent

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.linear import Linear

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Linear(auth=os.environ['LINEAR_ACCESS_TOKEN'])],
)
result = agent.run_sync('Summarize my assigned issues that were updated this week')
print(result.output)
```

## What to ask

- Summarize an issue and its latest comments.
- List issues assigned to you or updated in a date range.
- Find projects, teams, or related issues.
- Create or update issues when read-write access is enabled.

## Operational constraints

- `read_only=True` is the default and selects Linear's server-enforced read-only endpoint. Set `read_only=False` and use
  a credential with write access to enable updates.
- `allowed_tools` narrows the exposed tools by exact name. It does not replace endpoint or credential permissions.
- Mutation tools do not require human approval automatically. Use `linear.get_toolset().approval_required()` when a
  person must approve calls.
- Use a separate Linear capability or preconfigured client for each user or workspace. Reconnecting an OAuth client
  does not switch workspaces.
- An injected FastMCP client or `MCPToolset` owns its endpoint and authentication. Do not also pass `auth`.
- URL clients that receive `auth` directly must use HTTPS.
- Treat issue text, comments, and tool errors as untrusted model input. Limit other agent tools or add a guard when that
  content must not influence external actions or data disclosure.
- Approval does not make writes idempotent. After an uncertain mutation error, reconcile the Linear record before retrying
  or provide application-level deduplication.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/linear/)

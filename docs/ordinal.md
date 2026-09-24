---
title: Ordinal
description: Connect a Pydantic AI agent to Ordinal's hosted MCP server to draft, schedule, and analyze social posts.
---

# Ordinal

`Ordinal` connects an agent to [Ordinal](https://www.tryordinal.com)'s hosted MCP server so it can draft, schedule, and analyze social posts in the signed-in user's workspaces.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/ordinal/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Before you start

Ordinal MCP needs the Pro plan or higher, the same as the REST API. You need an Ordinal account with access to at least one workspace. Ordinal MCP takes an OAuth access token; there is no API key to copy.

If you set up the old server at `https://app.tryordinal.com/api/mcp` with a workspace API key, remove it. Do not use the old and new servers together, because their duplicate tool names confuse the agent.

## Installation

```bash
pip/uv-add "pydantic-ai-harness[ordinal]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider the example uses. For another model, install that provider's extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Ordinal

agent = Agent('openai:gpt-5.6-sol', capabilities=[Ordinal()])
result = agent.run_sync('List my Ordinal workspaces')
print(result.output)
```

Set `ORDINAL_ACCESS_TOKEN` to an Ordinal access token, or pass `auth=` a token or an `httpx.Auth`. To serve several users from one agent, pass a function instead (see [Per-user credentials](#per-user-credentials)). Then start by listing workspaces. Every other Ordinal tool needs a `workspaceSlug` from `ordinal_get_workspace_context`.

`Ordinal` has no settings for custom clients, tool filtering, or approval. For those, use Pydantic AI's generic [`MCP`](/ai/capabilities/mcp/) capability or [`MCPToolset`](/ai/mcp/) directly.

## Per-user credentials

A fixed token or `ORDINAL_ACCESS_TOKEN` connects every run as the same account.

When one agent serves several users, pass a function as `auth` that returns the current user's Ordinal access token:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import Ordinal


@dataclass
class Deps:
    ordinal_token: str | None


def ordinal_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.ordinal_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[Ordinal(auth=ordinal_token)])
```

The function is called at the start of each run, so each run connects as its own user. It can be async, and it can return a token or an `httpx.Auth`. If it returns `None`, that run has no Ordinal tools; it never falls back to `ORDINAL_ACCESS_TOKEN`.

Your application is responsible for getting each user's token, storing it, and refreshing it, for example with a "Connect Ordinal" button in your web app. The function only reads the current token.

With durable execution such as Temporal, read the token from the run's deps rather than from a global, since the function may run in another process. To add more than one `Ordinal` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same.

## Define the agent in YAML or JSON

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - Ordinal: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Ordinal

agent = Agent.from_file('agent.yaml', custom_capability_types=[Ordinal])
```

Pass `custom_capability_types` so the loader can create `Ordinal` from the file.

## API reference

::: pydantic_ai_harness.ordinal.Ordinal

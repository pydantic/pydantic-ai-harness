---
title: Ordinal
description: Connect a Pydantic AI agent to Ordinal's hosted MCP server to draft, schedule, and analyze social posts.
---

# Ordinal

`Ordinal` connects an agent to [Ordinal](https://www.tryordinal.com)'s hosted MCP server so it can draft, schedule, and analyze social posts in the signed-in user's workspaces.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/ordinal/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Before you start

Ordinal MCP is available on the Pro plan or higher, same as the REST API. You need an Ordinal account with access to at least one workspace. Sign-in is OAuth -- there is no API key to copy.

If you previously configured the old server at `https://app.tryordinal.com/api/mcp` with a workspace API key, remove it. The old and new servers cannot coexist: duplicate tool names confuse the agent.

## Installation

```bash
pip/uv-add "pydantic-ai-harness[ordinal]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider used by the example. For another model, install its matching provider extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Ordinal

agent = Agent('openai:gpt-5', capabilities=[Ordinal()])
result = agent.run_sync('List my Ordinal workspaces')
print(result.output)
```

The first Ordinal tool call opens a browser. Sign in and approve access. After that, start by listing workspaces -- every other Ordinal tool needs a `workspaceSlug` from `ordinal_get_workspace_context`.

`Ordinal` only configures the connection. For custom clients, tool filtering, or approval policy, use Pydantic AI's generic [`MCP`](/ai/capabilities/mcp/) capability or [`MCPToolset`](/ai/mcp/) directly.

## Define the agent in YAML or JSON

```yaml
# agent.yaml
model: openai:gpt-5
capabilities:
  - Ordinal: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Ordinal

agent = Agent.from_file('agent.yaml', custom_capability_types=[Ordinal])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `Ordinal`.

## API reference

::: pydantic_ai_harness.ordinal.Ordinal

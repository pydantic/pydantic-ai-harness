---
title: Supabase
description: Inspect one non-production Supabase project through its official hosted MCP server.
---

# Supabase

`Supabase` lets an agent inspect one Supabase development or test project through Supabase's hosted MCP server.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[supabase]" "pydantic-ai-slim[openai]"
```

## Set up Supabase

Create or choose a non-production project, then copy its project reference from the Supabase Dashboard project
settings. Create a scoped personal access token in Supabase Account Settings > Access Tokens. Limit it to this project
and the read permissions the selected feature groups need. Then set:

```bash
export SUPABASE_PROJECT_REF="your-project-ref"
export SUPABASE_ACCESS_TOKEN="sbp_fc..."
export OPENAI_API_KEY="your-openai-api-key"
```

Scoped personal access tokens are Public Alpha and are rolling out gradually. If scoped tokens are unavailable for
your account, a classic token grants access to every organization and project available to that account.

Omitting `access_token` selects browser OAuth through Pydantic AI. OAuth is not currently usable with Supabase because
the released MCP client cannot complete Supabase's token exchange. Track
[pydantic/pydantic-ai#8123](https://github.com/pydantic/pydantic-ai/issues/8123). When fixed, the first connection will
open a browser for sign-in and consent; headless environments will still need a PAT.

## Run

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.supabase import Supabase

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        Supabase(
            project_ref=os.environ['SUPABASE_PROJECT_REF'],
            access_token=os.environ['SUPABASE_ACCESS_TOKEN'],
        )
    ],
)
result = agent.run_sync('List the public tables and report any security advisor findings')
print(result.output)
```

You can ask the agent to:

- list tables, extensions, and migrations;
- run read-only SQL queries;
- inspect security and performance advisors or query project logs;
- get the project URL and publishable keys;
- generate TypeScript database types; or
- search Supabase documentation.

## Operational constraints

- The MCP server is Public Alpha. Use this integration only with development or test data, and do not expose it to
  end users.
- `project_ref` is required. Account-wide tools are not exposed.
- One capability session is one authenticated identity. Create a separate capability and agent session for each user.
- Multiple projects can share an agent only when their selected feature groups expose disjoint tool names. Overlapping
  groups, including two default configurations, fail before the model runs.
- The defaults are `read_only=True` and the `database`, `debugging`, `development`, and `docs` feature groups.
- You can explicitly select any non-empty combination of those groups plus `functions`, `storage`, and `branching`.
  The Storage MCP group is disabled by default. Storage configuration updates and Branching require a paid plan;
  Branching is experimental. Branch creation is not exposed because the project-scoped server cannot complete its
  required cost confirmation without account access or an interactive form handler.
- `read_only=False` enables mutation tools, but every SQL, schema, data, Edge Function, Storage, or Branching mutation
  still requires Pydantic AI tool approval. Include `DeferredToolRequests` in the agent output types and approve or
  deny each request before resuming the run.
- Treat rows and logs as untrusted content. Review each tool call and keep credential permissions narrow.
- SQL, log, and advisor results can be large. Add
  [`ToolOutputLimits`](tool-output-limits.md) to the agent capabilities when result size can exceed the model context.

[Supabase MCP reference](https://supabase.com/docs/guides/ai-tools/mcp) | [Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/supabase/)

## API reference

::: pydantic_ai_harness.supabase.Supabase

---
title: Cloudflare
description: Use official Cloudflare managed MCP servers from a Pydantic AI agent.
---

# Cloudflare

Cloudflare lets an agent read Cloudflare documentation and account data, then run approved platform changes through Cloudflare's official managed MCP servers.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[cloudflare]" "pydantic-ai-slim[openai]"
```

## Set up Cloudflare

The public Cloudflare documentation, Agents SDK documentation, Developer Stack, blog, and demo servers need no Cloudflare credentials.
Authenticated servers use OAuth when `api_token` is omitted. Pydantic AI's MCP client handles OAuth, and the first
authorization requires browser interaction. For CI or another non-interactive process, create a least-privilege
Cloudflare API token and pass it as `api_token`, usually from `CLOUDFLARE_API_TOKEN`.
The default OAuth token store is process memory, so restarting the process starts authorization again. Pass a prebuilt
MCP client with persistent OAuth storage when the application must retain authorization across restarts.

The example below uses OAuth with a multi-account user grant. Set the model credential and the Cloudflare account and
zone to query:

```bash
export OPENAI_API_KEY="your OpenAI API key"
export CLOUDFLARE_ACCOUNT_ID="your Cloudflare account ID"
export CLOUDFLARE_ZONE_ID="your Cloudflare zone ID"
```

## Example

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.cloudflare import Cloudflare, CloudflareServer

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        Cloudflare(
            server=CloudflareServer.DNS_ANALYTICS,
            account_id=os.environ['CLOUDFLARE_ACCOUNT_ID'],
            zone_id=os.environ['CLOUDFLARE_ZONE_ID'],
        )
    ],
)
result = agent.run_sync('Show details for the configured zone')
print(result.output)
```

## What an agent can do

- Search Cloudflare developer and Agents SDK documentation.
- Inspect the Cloudflare API schema with `CloudflareServer.API`.
- Read data from focused servers for DNS analytics, Workers, observability, containers, Logpush, AI Gateway,
  audit logs, DEX, CASB, Developer Stack, browser tasks, the Cloudflare blog, and Demo Day.
- Run create, update, or delete operations after `allow_mutations=True` exposes them and the application approves each
  call.

## Operational constraints

- One `Cloudflare` instance selects one `CloudflareServer`. The default is the public documentation server.
- Focused servers expose only tools marked read-only by Cloudflare unless `allow_mutations=True`. The official API
  server exposes only `docs` and its network-isolated OpenAPI `search` tool by default. MCP safety annotations state
  server intent, so credentials should still have only the permissions required for the selected server.
- Mutation-capable tools raise Pydantic AI's standard deferred approval request before the MCP request runs. Configured
  account and zone IDs are required in the mutation tool call so the approval shows the exact target; these IDs enter
  model context, but credentials do not. Resume with `DeferredToolResults` after the application or user approves.
- On focused servers, `account_id` keeps only tools with an explicit account argument and supplies that value to read
  calls. Use it with a multi-account user credential. If the token already pins one account, omit `account_id`; the
  token is the account boundary. `zone_id` applies the same policy to explicit zone arguments. Scoped instances do not
  forward remote server instructions because those instructions can include other accessible account IDs.
- Code Mode `execute` accepts arbitrary JavaScript, so `CloudflareServer.API` cannot combine mutation access with an
  enforced `account_id` or `zone_id`. Use a focused server when either boundary is required.
- `max_results` bounds recognized top-level pagination fields and nested fields when their parent object is supplied.
  `max_output_bytes` and `max_output_lines` bound each model-facing result. Oversized structured or binary results are
  replaced rather than returned partially.
- A prebuilt `client` owns authentication and account selection, so it cannot be combined with `api_token` or
  `account_id`. Custom clients treat every tool as mutation-capable unless `trust_server_annotations=True`; even then,
  contradictory safety annotations remain mutation-capable. Remote instructions from custom clients are not
  forwarded. Prebuilt clients do not receive the official managed-server safe-name exceptions, even when their URL
  matches an official endpoint.
- Run tools through an `Agent`. `CloudflareToolset.direct_call_tool()` is disabled because direct MCP calls have no
  agent approval or resource-boundary context.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/cloudflare/) |
[Cloudflare managed MCP servers](https://github.com/cloudflare/mcp-server-cloudflare) |
[Cloudflare Code Mode](https://github.com/cloudflare/mcp)

## API reference

::: pydantic_ai_harness.cloudflare.Cloudflare

::: pydantic_ai_harness.cloudflare.CloudflareServer

::: pydantic_ai_harness.cloudflare.CloudflareToolset

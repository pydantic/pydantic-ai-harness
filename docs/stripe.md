# Stripe

Read and change Stripe resources through its hosted MCP tools. `Stripe` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[stripe]" "pydantic-ai-slim[openai]"
```

Set `STRIPE_API_KEY` to a Stripe restricted API key, or pass `auth=...`. When neither is supplied, the connection starts browser OAuth. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://docs.stripe.com/mcp).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.stripe import Stripe

agent = Agent('openai:gpt-5.6-sol', capabilities=[Stripe()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

The credential determines sandbox or live mode and which resources can be changed. For a Connect account, set `connected_account='acct_...'` with a platform restricted API key; Stripe receives the native `Stripe-Account` header. Connected-account access does not support OAuth. Grant the restricted key only the resource permissions the agent needs.

## Tool selection and approval

For application-level filtering or approval, compose the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.stripe import Stripe

capability = Stripe()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](tool-output-limits.md).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/stripe/)

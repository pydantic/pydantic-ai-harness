# Stripe

`Stripe` lets an agent read one Stripe platform or connected account and request approval for opt-in writes through
Stripe's hosted MCP server.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/stripe/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[stripe]" "pydantic-ai-slim[openai]"
```

## Set up Stripe and your model

In the Stripe Dashboard, create a restricted API key and set `Customers` to `Read` for the example below. Grant only
the other read permissions the agent needs. Export that key and your model-provider key:

```bash
export STRIPE_API_KEY='rk_test_...'
export OPENAI_API_KEY='...'
```

The capability sends `STRIPE_API_KEY` directly to Stripe as a bearer token. It does not run OAuth or open a browser.
Do not use an unrestricted `sk_...` key.

## Run an agent

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.stripe import Stripe

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Stripe(api_key=os.environ['STRIPE_API_KEY'])],
)
result = agent.run_sync('List the five most recent customers')
print(result.output)
```

You can ask the agent to:

- search for Stripe API methods and inspect their parameters;
- retrieve account information;
- read customers, payments, refunds, invoices, subscriptions, and other methods supported by Stripe MCP;
- search Stripe documentation;
- request supported API writes when writes are enabled.

## Operational constraints

- The default tool allowlist is `get_stripe_account_info`, `search_stripe_documentation`, `stripe_api_details`,
  `stripe_api_read`, and `stripe_api_search`. `stripe_api_write` is the only tool added by `enable_writes=True`.
- Access is read-only by default. `enable_writes=True` exposes `stripe_api_write`; every call returns a
  `DeferredToolRequests` approval request before Stripe receives the write. Preserve the request metadata when
  resuming. The restricted key must grant write permission for each resource the agent may change. An approved result
  remains replayable for the same tool call ID, arguments, and account scope, so persist and consume it atomically.
  Approval is not idempotency; after a timeout or unknown response, verify the resource before retrying a write. If an
  API or UI accepts approval decisions, it must authenticate the caller and authorize that caller for the exact
  operation and account scope before accepting one.
- Stripe list reads can be paginated. Follow the pagination fields returned by Stripe when complete results are
  required.
- `mode='sandbox'` accepts `rk_test_...` keys. Set `mode='live'` explicitly for an `rk_live_...` key.
- Set `connected_account='acct_...'` to send every request to one Connect account. Connected-account access requires
  a restricted platform key with the needed connected-account permissions and does not support OAuth.
- Use a separate agent for each platform or connected-account scope. Two `Stripe` capabilities on one agent expose
  the same tool names and are rejected instead of merging their access.
- `include_instructions=True` adds the Stripe usage guidance shown to the model. Set it to `False` when supplying
  equivalent instructions elsewhere; the code-enforced allowlist and approval checks remain active.
- Treat Stripe object fields and tool results as untrusted model input. When enabled, the built-in guidance tells the
  model not to follow instructions in that content, but it cannot prevent disclosure through another tool. Apply host
  policy and approval before combining Stripe with unrelated outbound or mutation tools.
- The capability sends requests only to `https://mcp.stripe.com` and exposes an exact tool allowlist. Stripe labels
  the MCP server Public preview. Confirm that preview services meet your requirements before using live mode.

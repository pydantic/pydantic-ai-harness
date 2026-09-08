# Slack

Give an agent Slack messages, channels, and canvas tools. `Slack` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[slack]" "pydantic-ai-slim[openai]"
```

Set `SLACK_USER_TOKEN` to a Slack user token, or pass `auth=...`. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://docs.slack.dev/ai/slack-mcp-server/).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.slack import Slack

agent = Agent('openai:gpt-5.6-sol', capabilities=[Slack()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

Slack's hosted MCP server accepts user tokens (`xoxp-`), not bot tokens (`xoxb-`). Register the app and grant its user-token scopes as described in Slack's MCP documentation. Automatic OAuth client registration is not supported; a configured OAuth client can be supplied through `client`.

The tools act as the token's user, including when posting messages or editing canvases. This capability supplies tools to ordinary agent runs; receiving Slack messages is a separate application concern.

## Tool selection and approval

`read_only=True` keeps only tools explicitly marked `readOnlyHint: true`; unmarked tools are omitted. This can leave no tools when a server does not annotate its read operations. Credentials remain the access-control boundary.

For application-level filtering or approval, compose the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.slack import Slack

capability = Slack()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](tool-output-limits.md).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability. `read_only=True` applies the same annotation filter to custom clients.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

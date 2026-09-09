# Google Workspace

Use Gmail, Calendar, Drive, and other Google Workspace tools. `GoogleWorkspace` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[google-workspace]" "pydantic-ai-slim[openai]"
```

Set `GOOGLE_ACCESS_TOKEN` to a Google OAuth access token, or pass `auth=...`. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://developers.google.com/workspace/guides/configure-mcp-servers).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.google_workspace import GoogleWorkspace

agent = Agent('openai:gpt-5.6-sol', capabilities=[GoogleWorkspace(services=['gmail', 'calendar'])])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

`services` selects one product or a list: `gmail`, `drive`, `docs`, `sheets`, `slides`, `calendar`, `chat`, or `people`. Each product gets its own MCP connection and tool prefix, such as `gmail_search_threads`. Register a Google OAuth client and request the scopes needed for the selected products; Google does not support automatic client registration. `auth` can supply a refresh-capable `httpx.Auth`.

## Tool selection and approval

`read_only=True` keeps only tools explicitly marked `readOnlyHint: true`; unmarked tools are omitted. This can leave no tools when a server does not annotate its read operations. Credentials remain the access-control boundary.

For application-level filtering or approval, compose the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.google_workspace import GoogleWorkspace

capability = GoogleWorkspace(services=['gmail', 'calendar'])
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](tool-output-limits.md).

## Connection customization

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/google_workspace/)

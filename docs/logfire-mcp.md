# Logfire MCP

Query Logfire telemetry and manage observability resources. `LogfireMCP` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

Set `LOGFIRE_API_KEY` to a Logfire API key, or pass `auth=...`. When neither is supplied, the connection starts browser OAuth. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://pydantic.dev/docs/logfire/guides/mcp-server/).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent('openai:gpt-5.6-sol', capabilities=[LogfireMCP()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

The default endpoint is `https://logfire-us.pydantic.dev/mcp`. Set `url=LOGFIRE_EU_MCP_URL` for EU data, or provide a self-hosted MCP URL. API-key scopes determine access to projects and operations.

The capability supplies the current UTC time and brief query guidance: schema timestamps are not a clock, transport time bounds also constrain SQL, and links are created only when requested. `include_instructions=False` disables both this guidance and server instructions. Logfire owns query semantics, time windows, and result schemas.

## Tool selection and approval

`read_only=True` keeps only tools explicitly marked `readOnlyHint: true`; unmarked tools are omitted. This can leave no tools when a server does not annotate its read operations. Credentials remain the access-control boundary.

For application-level filtering or approval, compose the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.logfire_mcp import LogfireMCP

capability = LogfireMCP()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    instructions=capability.get_instructions(),
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](tool-output-limits.md).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability. `read_only=True` applies the same annotation filter to custom clients.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

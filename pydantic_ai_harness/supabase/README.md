# Supabase

Use Supabase project and account tools. `Supabase` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[supabase]" "pydantic-ai-slim[openai]"
```

Set `SUPABASE_ACCESS_TOKEN` to a Supabase personal access token, or pass `auth=...`. When neither is supplied, the connection starts browser OAuth. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://supabase.com/docs/guides/ai-tools/mcp).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.supabase import Supabase

agent = Agent('openai:gpt-5.6-sol', capabilities=[Supabase(project_ref='your-project-ref')])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

`project_ref` selects a project through Supabase's native URL parameter; omit it to retain account-level tools. `features=['database', 'docs']` selects native feature groups; omitting it keeps server defaults. `read_only=True` sends `read_only=true`, including Supabase's read-only SQL execution mode. These settings belong to Supabase and need no local tool catalog. Follow Supabase's current guidance when selecting a development or production project.

## Tool selection and approval

For application-level filtering or approval, compose the existing [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.supabase import Supabase

capability = Supabase(project_ref='your-project-ref')
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability. With a custom client, `read_only=True` filters annotations rather than configuring the remote server.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/supabase/)

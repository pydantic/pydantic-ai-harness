# Composio

Give an agent access to connected applications through a Composio session. Composio handles tool discovery, app authorization, and execution; this capability connects the session to your agent.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/index.md#version-policy).

## Install and connect

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[mcp,openai]" "composio>=0.21.1"
```

Set `COMPOSIO_API_KEY` for Composio and `OPENAI_API_KEY` for the model. Create a session for a stable user ID from your application, and pass both connection values returned by Composio:

```python
from composio import Composio as ComposioClient
from pydantic_ai import Agent
from pydantic_ai_harness.composio import Composio

composio = ComposioClient()
session = composio.sessions.create(user_id='user_123', mcp=True)
agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Composio(url=session.mcp.url, headers=session.mcp.headers)],
)
result = agent.run_sync('Find the applications I can connect to')
print(result.output)
```

For later turns, reuse the session. Store `session.session_id` and restore it with `composio.use(session_id, mcp=True)` when needed. Session creation and restoration belong to your application, outside the agent's run loop.

## Session settings

Choose toolkits, connected accounts, and tool restrictions when configuring the Composio session. By default, its discovery and execution tools let the agent find app actions as needed. Composio also offers a direct-tools preset when you want a fixed set of actions. Follow [Composio's session guide](https://docs.composio.dev/docs/sessions-via-mcp) for configuration and app authorization.

The capability forwards server instructions by default; set `include_instructions=False` to omit them. For a custom transport, pass `client=...`; that client owns its connection settings and overrides `url` and `headers`.

The agent can use every tool exposed by the session, including writes. Configure access in Composio. For application-level approval or filtering, use Pydantic AI's [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/) on `capability.get_toolset()`.

Composio's hosted MCP endpoint does not run local SDK tool-call modifiers or expose in-process custom tools. Those features require Composio's native SDK execution path.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/composio/)

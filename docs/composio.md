# Composio

Let an agent use the applications a user has connected through Composio. `Composio` gives the agent the tools of one Composio session. Composio finds the right app actions, asks the user to authorize apps, and runs the actions.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[composio]" "pydantic-ai-slim[openai]"
```

The `composio` extra installs the Composio SDK. Install your model provider separately.

Set `COMPOSIO_API_KEY` for Composio and `OPENAI_API_KEY` for the model. Create a session for a user ID that stays the same for that user, and pass the session's URL and headers:

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

Reuse the session for later turns: store `session.session_id` and restore it with `composio.sessions.use(session_id, mcp=True)`. Your application creates and restores sessions. For an agent that serves several users, do this in a `client` function (see [Per-user sessions](#per-user-sessions)).

## Per-user sessions

A fixed `url` and `headers`, or a fixed `client`, connect every run to the same Composio session, so every run acts as that session's user. When one agent serves several users, pass a function as `client` that returns a transport for the current user's session:

```python
import asyncio
from dataclasses import dataclass

from composio import Composio as ComposioClient
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.composio import Composio

composio = ComposioClient()


@dataclass
class Deps:
    user_id: str
    composio_session_id: str | None = None


def session_transport(deps: Deps) -> StreamableHttpTransport:
    if deps.composio_session_id is None:
        session = composio.sessions.create(user_id=deps.user_id, mcp=True)
        # Store session.session_id for this user so later runs restore it.
    else:
        session = composio.sessions.use(deps.composio_session_id, mcp=True)
    headers = {key: value for key, value in (session.mcp.headers or {}).items() if value is not None}
    return StreamableHttpTransport(session.mcp.url, headers=headers)


async def composio_session(ctx: RunContext[Deps]) -> StreamableHttpTransport:
    return await asyncio.to_thread(session_transport, ctx.deps)


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[Composio(client=composio_session)])
```

The function is called at the start of each run, so each run connects to its own user's session. It can be async, and it can return any MCP client or transport. If it returns `None`, that run has no Composio tools; it never falls back to `url` and `headers`.

Composio's SDK is synchronous and calls Composio's API, so run it in a thread as above to keep the agent responsive. `session.mcp.headers` can contain unset values; drop them before building the transport. Your application is responsible for mapping each user to a Composio user ID and storing their session ID.

With durable execution such as Temporal, read the session from the run's deps rather than from a global, since the function may run in another process. To add more than one `Composio` to an agent, give each a distinct `id`.

## Session settings

Choose toolkits, connected accounts, and tool restrictions when you create the Composio session. By default, the session gives the agent tools to search for app actions and run them. Composio also has a preset that gives the agent a fixed set of actions instead. See [Composio's session guide](https://docs.composio.dev/docs/sessions-via-mcp) for settings and app authorization.

The server's instructions reach the agent by default; `include_instructions=False` turns them off. To use your own transport, pass `client`. It then owns the connection settings, and `url` and `headers` are ignored. A fixed `client` is one connection shared by every run; see [Per-user sessions](#per-user-sessions) to connect each user separately.

The agent can use every tool the session offers, including tools that make changes. Control access in Composio. To filter tools or require approval in your application, wrap `capability.get_toolset()` with Pydantic AI's [toolset wrappers](/ai/tools-toolsets/toolsets/).

Composio's tool-call modifiers and custom tools defined in your own code do not work through this capability. They need Composio's own SDK to run the tools.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/composio/)

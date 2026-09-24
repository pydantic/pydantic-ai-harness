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

Reuse the session for later turns: store `session.session_id` and restore it with `composio.sessions.use(session_id, mcp=True)`. Your application creates and restores sessions. For an agent that serves several users, build the whole capability for each run with a [dynamic capability](/ai/capabilities/custom/#dynamically-building-a-capability) (see [Per-user sessions](#per-user-sessions)).

## Per-user sessions

A fixed `url` and `headers`, or a fixed `client`, connect every run to the same Composio session, so every run acts as that session's user. When one agent serves several users, build the whole capability for each run with a [dynamic capability](/ai/capabilities/custom/#dynamically-building-a-capability):

```python
import asyncio
import os
from dataclasses import dataclass

from composio import Composio as ComposioClient
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai_harness.composio import Composio

COMPOSIO_API_KEY = os.environ['COMPOSIO_API_KEY']
composio = ComposioClient()
session_ids: dict[str, str] = {}  # Your database: each user's Composio session ID.


@dataclass
class Deps:
    composio_url: str | None


def composio_session(ctx: RunContext[Deps]) -> Composio[Deps] | None:
    if not ctx.deps.composio_url:
        return None
    return Composio(url=ctx.deps.composio_url, headers={'x-api-key': COMPOSIO_API_KEY})


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[DynamicCapability(composio_session, id='composio')])


async def run_for_user(prompt: str, user_id: str) -> str:
    session_id = session_ids.get(user_id)
    if session_id is None:
        session = await asyncio.to_thread(composio.sessions.create, user_id=user_id, mcp=True)
        session_ids[user_id] = session.session_id
    else:
        session = await asyncio.to_thread(composio.sessions.use, session_id, mcp=True)
    result = await agent.run(prompt, deps=Deps(composio_url=session.mcp.url))
    return result.output
```

Open the user's session before the run, save its ID so later runs reuse it, and put its URL in the deps; the function only reads it, so each run connects to its own user's session. If it returns `None`, that run has no Composio tools. Composio's SDK is synchronous and calls Composio's API, so run it in a thread as above to keep the agent responsive.

The session's headers carry only your Composio API key, which is the same for every user, so the function adds it from your configuration instead of the deps. Deps can be stored, for example in a durable execution's history, and should not hold secrets that are not the user's own. Your application is responsible for mapping each user to a Composio user ID.

With durable execution such as Temporal, the function may run again when a run is replayed, so it must not call Composio's API itself. Opening the session before the run, as above, keeps that call out of the replay. The capability's `id` defaults to `composio`, so `defer_loading=True` works without one. To add more than one `Composio` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same; two that share an `id` but differ raise an error.

## Session settings

Choose toolkits, connected accounts, and tool restrictions when you create the Composio session. By default, the session gives the agent tools to search for app actions and run them. Composio also has a preset that gives the agent a fixed set of actions instead. See [Composio's session guide](https://docs.composio.dev/docs/sessions-via-mcp) for settings and app authorization.

The server's instructions reach the agent by default; `include_instructions=False` turns them off. To use your own transport, pass `client`. It then owns the connection, so passing `client` together with `url` or `headers` raises an error. A `client` is one connection shared by every run; see [Per-user sessions](#per-user-sessions) to connect each user separately.

The agent can use every tool the session offers, including tools that make changes. Control access in Composio. To filter tools or require approval in your application, wrap `capability.get_toolset()` with Pydantic AI's [toolset wrappers](/ai/tools-toolsets/toolsets/).

Composio's tool-call modifiers and custom tools defined in your own code do not work through this capability. They need Composio's own SDK to run the tools.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/composio/)

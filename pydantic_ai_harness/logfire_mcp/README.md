# Logfire MCP

Let an agent query Logfire telemetry and manage Logfire projects. `LogfireMCP` gives the agent every tool Logfire's hosted MCP server offers, including tools that make changes. The credential you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install and connect

uv:

```bash
uv add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

Set `LOGFIRE_API_KEY`, or pass `auth=` an API key or an `httpx.Auth`. With neither, the agent opens a browser so you can log in to Logfire, which only works when you run it on your own machine. See the [provider setup](https://pydantic.dev/docs/logfire/guides/mcp-server/).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent('openai:gpt-5.6-sol', capabilities=[LogfireMCP()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

An API key, `LOGFIRE_API_KEY`, and browser login all connect every run as the same account. When one agent serves several users, pass a function that returns the current user's credential instead:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.logfire_mcp import LogfireMCP


@dataclass
class Deps:
    logfire_token: str | None
    logfire_region: str = 'us'


def logfire_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.logfire_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[LogfireMCP(auth=logfire_token)])
```

The function is called at the start of each run, so each run connects as its own user. It can be async, and it can return a token or an `httpx.Auth`. If it returns `None`, that run has no Logfire tools; it never falls back to `LOGFIRE_API_KEY` or browser login.

Your application is responsible for getting each user's token, storing it, and refreshing it, for example with a "Connect Logfire" OAuth flow in your web app. The function only reads the current token. Returning `'oauth'` from it raises an error, because browser login would open on the server rather than for the user.

`client` also accepts a function, for when users differ in more than their credential, such as users whose data is in the EU region:

```python
from dataclasses import dataclass

from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import RunContext
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP


@dataclass
class Deps:
    logfire_token: str | None
    logfire_region: str = 'us'


def logfire_client(ctx: RunContext[Deps]) -> StreamableHttpTransport | None:
    if ctx.deps.logfire_token is None:
        return None
    url = LOGFIRE_EU_MCP_URL if ctx.deps.logfire_region == 'eu' else LOGFIRE_US_MCP_URL
    return StreamableHttpTransport(url, auth=ctx.deps.logfire_token)


capability = LogfireMCP(client=logfire_client)
```

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. To add more than one `LogfireMCP` to an agent, give each a distinct `id` and wrap them in [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/), since their tool names are the same.

## Provider settings

The default endpoint is Logfire's US region. Set `url=LOGFIRE_EU_MCP_URL` for EU data, or pass the MCP URL of a self-hosted Logfire. The API key's scopes decide which projects and actions are allowed.

The capability adds short guidance to the agent's instructions: the current UTC time, a reminder that timestamps in examples are not the current time, that queries cover a short time window unless widened, and that Logfire links should be created only when asked for. `include_instructions=False` turns this off, along with the server's own instructions.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, this can leave none. The credential is still what controls access.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

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

Handle the approval requests with the [deferred tools workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Pass `client` to use your own FastMCP client or transport, for example one with custom OAuth token storage. The client then owns the URL, authentication, and server settings, so set those on it rather than on the capability. `read_only` and `include_instructions` still apply.

A fixed `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections whose tool names overlap, give them distinct `id`s and add [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)

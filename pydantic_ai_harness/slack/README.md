# Slack

`Slack` gives an agent Slack's hosted MCP tools. It does not connect the agent to incoming Slack messages.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](../../docs/index.md#version-policy).

Install the capability with:

```bash
uv add "pydantic-ai-harness[slack]"
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.slack import Slack

agent = Agent('anthropic:claude-fable-5', capabilities=[Slack()])
result = agent.run_sync('Summarize the last 20 messages in #general')
print(result.output)
#> ...
```

`Slack` reads `SLACK_USER_TOKEN`, or you can pass `Slack(auth=...)`. The token must be a user token: in your app's settings under OAuth & Permissions, add user token scopes, reinstall the app, and copy the User OAuth Token, which starts with `xoxp-`. The Bot User OAuth Token on the same page (`xoxb-`) does not work; Slack's MCP server answers it, or any other invalid token, with a 401 at run time. Slack offers MCP to internal or directory-published apps, as described in the [Slack MCP server documentation](https://docs.slack.dev/ai/slack-mcp-server/).

For a token supplied dynamically, `auth` also accepts an `httpx.Auth` instance.

The tools run directly as the token's user, including the ones that post messages, add reactions, create channels, and edit canvases, so grant only the [scopes](https://docs.slack.dev/reference/scopes/) the agent needs. `Slack(read_only=True)` keeps only the tools Slack marks read-only.

A token is one person's identity, so an agent serving many people needs a token per run: pass a function that returns `Slack(auth=...)` for the run's context, as with any capability.

## Two workspaces

Give each `Slack` its own `id`, and wrap each in [`PrefixTools`](https://pydantic.dev/docs/ai/capabilities/prefix-tools/) so the two copies of Slack's tool catalog get different names. The prefix is also how the model tells the workspaces apart.

```python
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai_harness.slack import Slack

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        PrefixTools(wrapped=Slack(auth=os.environ['ACME_SLACK_TOKEN'], id='acme'), prefix='acme'),
        PrefixTools(wrapped=Slack(auth=os.environ['BETA_SLACK_TOKEN'], id='beta'), prefix='beta'),
    ],
)
```

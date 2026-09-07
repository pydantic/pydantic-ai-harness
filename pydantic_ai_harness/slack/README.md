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

`Slack` reads `SLACK_USER_TOKEN`, or you can pass `Slack(token=...)`. The tools come from Slack's hosted MCP server and act as the user the token belongs to. Slack's server accepts user tokens (`xoxp-`) only, not bot tokens, and offers MCP to internal or directory-published apps, as described in the [Slack MCP server documentation](https://docs.slack.dev/ai/slack-mcp-server/). To get a user token for an existing app, add user token scopes under OAuth & Permissions and reinstall it.

A token is one person's identity, so an agent serving many people needs a token per run: pass a function that returns `Slack(token=...)` for the run's context, as with any capability.

## Two workspaces

Give each `Slack` its own `id`, and wrap each in [`PrefixTools`](https://pydantic.dev/docs/ai/capabilities/overview/#prefixtools) so the two copies of Slack's tool catalog get different names. The prefix is also how the model tells the workspaces apart.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai_harness.slack import Slack

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        PrefixTools(wrapped=Slack(token=acme_token, id='acme'), prefix='acme'),
        PrefixTools(wrapped=Slack(token=beta_token, id='beta'), prefix='beta'),
    ],
)
```

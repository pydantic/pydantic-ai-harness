---
title: Slack
description: Give a Pydantic AI agent Slack tools.
---

# Slack

`Slack` gives an agent tools to read and write Slack. It works in any agent run and does not connect the agent to incoming Slack messages.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Slack tools

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

`Slack` reads `SLACK_USER_TOKEN` or `SLACK_BOT_TOKEN`, or you can pass `Slack(token=...)`. A user token (`xoxp-`) gives Slack's full hosted MCP tool catalog and acts as that user; Slack offers this to internal or directory-published apps with MCP enabled, as described in the [Slack MCP server documentation](https://docs.slack.dev/ai/slack-mcp-server/). A bot token (`xoxb-`) gives the built-in `send_message`, `add_reaction`, and `read_thread` tools. These need `chat:write`, `reactions:write`, and the history scope for the channel type being read, respectively; for example, `channels:history` applies to public channels, while other channel types use their corresponding history scope.

For per-user credentials, add an async capability factory that receives the run context:

```python
from pydantic_ai import RunContext

async def slack_for_user(ctx: RunContext[Deps]) -> Slack:
    return Slack(token=lookup_token(ctx.deps.user_id))

agent = Agent('anthropic:claude-fable-5', capabilities=[slack_for_user])
```

`current_slack_context()` returns the sender, channel, and thread when the run was started by `SlackApp`, otherwise `None`.

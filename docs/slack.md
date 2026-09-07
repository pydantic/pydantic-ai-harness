---
title: Slack
description: Give an agent Slack tools, or put an agent on Slack.
---

# Slack

Slack support is two separate pieces. `Slack` gives an agent tools to read and write Slack, and works in any agent run. `SlackApp` puts an agent on Slack so people can message it. Use either alone, or both together.

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

`Slack` reads `SLACK_USER_TOKEN` or `SLACK_BOT_TOKEN`, or you can pass `Slack(token=...)`. A user token (`xoxp-`) gives Slack's full hosted MCP tool catalog and acts as that user; Slack offers this to internal or directory-published apps with MCP enabled, as described in the [Slack MCP server documentation](https://docs.slack.dev/ai/slack-mcp-server/). A bot token (`xoxb-`) gives the built-in `send_message`, `add_reaction`, and `read_thread` tools. They need the `chat:write`, `reactions:write`, and `channels:history` scopes (private channels and DMs use their own history scope).

For per-user credentials, add an async capability factory that receives the run context:

```python
from pydantic_ai import RunContext

async def slack_for_user(ctx: RunContext[Deps]) -> Slack:
    return Slack(token=lookup_token(ctx.deps.user_id))

agent = Agent('anthropic:claude-fable-5', capabilities=[slack_for_user])
```

`current_slack_context()` returns the sender, channel, and thread when the run was started from Slack, otherwise `None`.

## Put an agent on Slack

`SlackApp` receives Slack messages, runs your agent, and posts the answer in the thread. It works with or without the `Slack` capability.

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness.slack import Slack, SlackApp

agent = Agent('anthropic:claude-fable-5', capabilities=[Slack()])
app = SlackApp(agent)
```

Run it with `uvicorn my_module:app`. Slack calls `/slack/events` on your server.

Set these before starting:

| Variable | What it is |
| --- | --- |
| `SLACK_BOT_TOKEN` | The bot token, `xoxb-...`, from **OAuth & Permissions**. |
| `SLACK_SIGNING_SECRET` | From **Basic Information**. Proves a request came from Slack. |
| `SLACK_APP_TOKEN` | Only for Socket Mode, `xapp-...`, from **App-Level Tokens**. |

Without a public URL, use Socket Mode instead of a server:

```python {test="skip"}
import anyio

anyio.run(app.serve)
```

### What it does

- Direct messages always get a reply. In a channel, mention the app to start a thread; later replies in that thread need no mention.
- Shows "Thinking..." while the agent runs, then streams the reply into the thread as markdown.
- Remembers each thread's conversation. The default store keeps 1,000 threads in memory for 24 hours; pass `history=` with your own `SlackHistory` implementation to persist it.
- Handles one thread's messages in order and different threads at the same time. Ignores Slack's duplicate deliveries.
- If the agent has a `Slack` capability, its tools act as the sender when the app was installed with user permissions, and as the bot otherwise.

Run one process per Slack app. History and duplicate detection are per process.

### Slack app settings

Create the app from this manifest at <https://api.slack.com/apps>. For Socket Mode, set `socket_mode_enabled` to `true` and drop `request_url`.

```json
{
  "display_information": {"name": "My agent"},
  "features": {
    "agent_view": {"agent_description": "A Pydantic AI agent"},
    "app_home": {"messages_tab_enabled": true, "messages_tab_read_only_enabled": false},
    "bot_user": {"display_name": "my-agent", "always_online": true}
  },
  "oauth_config": {
    "scopes": {
      "bot": ["app_mentions:read", "assistant:write", "channels:history", "chat:write", "groups:history", "im:history"]
    }
  },
  "settings": {
    "event_subscriptions": {
      "request_url": "https://your-domain/slack/events",
      "bot_events": ["app_mention", "message.channels", "message.groups", "message.im"]
    },
    "socket_mode_enabled": false
  }
}
```

### Agents with dependencies

Pass `deps_factory`. It receives a `SlackContext` with the workspace, channel, thread, sender, and any attached file names, and returns your deps:

```python {test="skip"}
from pydantic_ai_harness.slack import SlackContext


def make_deps(context: SlackContext) -> Deps:
    return Deps(user_id=context.user_id)


app = SlackApp(agent, deps_factory=make_deps)
```

### Installing in many workspaces

Pass Bolt's `AsyncOAuthSettings` as `oauth_settings=` instead of a bot token. `SlackApp` then serves `/slack/install` and `/slack/oauth_redirect` too, and uses each workspace's own tokens. Add `user_scopes` there if you want the `Slack` capability's tools to act as the sender. See [Bolt's OAuth guide](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/).

### Not included yet

Attached files are described to the agent by name but not downloaded. Tools that need human approval, suggested prompts, and the app home tab are not wired up. Agents with structured output are rejected at construction; `SlackApp` posts text.

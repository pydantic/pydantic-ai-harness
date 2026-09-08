---
title: Slack
description: Give a Pydantic AI agent Slack tools.
---

# Slack

`Slack` connects an agent to Slack's hosted MCP server, so it can search Slack, read conversations, and post as the person whose user token it holds. The default exposes Slack's write tools, so the token's scopes are what decide whether the agent can only look or can also act.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

Install the capability with:

```bash
uv add "pydantic-ai-harness[slack]"
```

## Setup

`Slack` reads `SLACK_USER_TOKEN`, or you can pass `Slack(auth=...)`. Slack's MCP server acts as a person, so it wants a user token, which starts with `xoxp-`, and it is open only to internal or directory-published apps. The [Slack MCP server documentation](https://docs.slack.dev/ai/slack-mcp-server/) covers both, and lists the scopes each tool needs. A YAML spec cannot carry the token, so an agent configured from one reads `SLACK_USER_TOKEN` from the environment.

```bash
export SLACK_USER_TOKEN='xoxp-...'
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.slack import Slack

agent = Agent('anthropic:claude-fable-5', capabilities=[Slack()])
result = agent.run_sync('Summarize the last 20 messages in #general')
print(result.output)
#> ...
```

## Things to know

- The token's scopes are the only boundary: the agent can do anything the token can, posting
  included. To put a person in front of the writes, use the
  [tool approval](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/#requiring-tool-approval).
- `Slack(read_only=True)` keeps only the tools Slack marks read-only, so anything Slack leaves
  unannotated is dropped.
- A token is one person's identity, so an agent serving many people needs a token per run: pass a
  function that returns `Slack(auth=...)` for the run's context.

## API reference

::: pydantic_ai_harness.slack.Slack

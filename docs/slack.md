---
title: Slack
description: Give an agent native Slack MCP tools and register it on a caller-owned Bolt app.
---

# Slack

Use `Slack` to give an agent the native Slack MCP tools for the invoking
user. Use `register_slack` to register that agent on a caller-owned asynchronous
Bolt app.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation and eligibility

```bash
uv add "pydantic-ai-harness[slack]"
```

Use an internal Slack app or an app published in Slack's directory, and enable
Slack's hosted MCP server for it. Slack owns the MCP catalog, schemas, result
formats, OAuth scopes, and pagination. Recheck the [hosted MCP guide](https://docs.slack.dev/ai/slack-mcp-server/)
before changing those assumptions.

> External assumptions, verified 2026-09-06: the native endpoint is
> `https://mcp.slack.com/mcp`; the installed Bolt APIs provide
> `AsyncOAuthSettings`, `AsyncSlackRequestHandler`, and
> `AsyncSocketModeHandler.start_async`/`close_async`. Recheck the endpoint and
> installed signatures against the [hosted MCP guide](https://docs.slack.dev/ai/slack-mcp-server/),
> [Bolt OAuth guide](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/),
> and [Bolt agent features](https://docs.slack.dev/tools/bolt-python/concepts/adding-agent-features/).

## Quick start

The caller configures Bolt OAuth and supplies a per-user token to each event.
`FileInstallationStore` is a native Slack SDK async-compatible store. Its
directory contains OAuth credentials, so create `SLACK_INSTALLATION_DIR` with
owner-only (`0700`) permissions and keep its files private.

```python
import os

from pydantic_ai import Agent
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore

from pydantic_ai_harness.slack import Slack, register_slack

agent = Agent('openai:gpt-5.6-sol', capabilities=[Slack()])
installation_store = FileInstallationStore(
    base_dir=os.environ['SLACK_INSTALLATION_DIR'],
    client_id=os.environ['SLACK_CLIENT_ID'],
)
oauth_settings = AsyncOAuthSettings(
    client_id=os.environ['SLACK_CLIENT_ID'],
    client_secret=os.environ['SLACK_CLIENT_SECRET'],
    scopes=[
        'app_mentions:read',
        'assistant:write',
        'channels:history',
        'chat:write',
        'groups:history',
        'im:history',
    ],
    user_scopes=os.environ['SLACK_USER_SCOPES'],
    redirect_uri=os.environ['SLACK_REDIRECT_URI'],
    installation_store=installation_store,
    installation_store_bot_only=False,
)
bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
register_slack(bolt, agent)
asgi_app = AsyncSlackRequestHandler(bolt, path='/slack/events')
```

Serve `asgi_app` directly with one worker. Bolt's OAuth routes, normally
`/slack/install` and `/slack/oauth_redirect`, must be served by the same HTTP
application so users can authorize the app. `SLACK_USER_SCOPES` is a
comma-separated value selected from Slack's current MCP documentation. Those
user scopes are the actual native read authorization; Harness does not add a
second disclosure policy.

For typed dependencies, replace the quickstart `register_slack(bolt, agent)`
call with a typed registration. The factory receives one `SlackContext` after
an event is accepted, and the agent's output type must be `str`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.slack import Slack, SlackContext, register_slack

@dataclass
class Deps:
    slack: SlackContext

def deps_factory(context: SlackContext) -> Deps:
    return Deps(slack=context)

typed_agent: Agent[Deps, str] = Agent('openai:gpt-5.6-sol', capabilities=[Slack()])
register_slack(bolt, typed_agent, deps_factory=deps_factory)
```

## Events, OAuth, and Socket Mode

The embedded HTTP manifest uses `app_mention`, `message.channels`,
`message.groups`, and `message.im`. The six bot scopes in the example match
those routes and agent messaging requirements. Remove a message event family
when it is not needed, and remove its corresponding history scope. User MCP
scopes remain provider-configured by Slack.

```json
{
  "display_information": {"name": "Pydantic AI Agent"},
  "features": {
    "agent_view": {"agent_description": "A Pydantic AI agent"},
    "app_home": {"home_tab_enabled": false, "messages_tab_enabled": true, "messages_tab_read_only_enabled": false},
    "bot_user": {"display_name": "pydantic-ai-agent", "always_online": true}
  },
  "oauth_config": {
    "redirect_urls": ["https://agent.example/slack/oauth_redirect"],
    "scopes": {
      "bot": [
        "app_mentions:read",
        "assistant:write",
        "channels:history",
        "chat:write",
        "groups:history",
        "im:history"
      ]
    }
  },
  "settings": {
    "event_subscriptions": {
      "request_url": "https://agent.example/slack/events",
      "bot_events": ["app_mention", "message.channels", "message.groups", "message.im"]
    },
    "is_mcp_enabled": true,
    "socket_mode_enabled": false
  }
}
```

For Socket Mode, construct the native handler inside the running event loop:

```python {test="skip"}
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

async def serve_socket_mode() -> None:
    handler = AsyncSocketModeHandler(bolt, 'xapp-your-app-level-token')
    try:
        await handler.start_async()
    finally:
        await handler.close_async()
```

Set `socket_mode_enabled` to `true` for that deployment, and continue serving
the OAuth HTTP routes. Socket Mode carries events; it does not replace the
OAuth install and redirect flow.

## Conversation behavior and context

Direct messages route without a mention. In public and private channels, an
`app_mention` engages that thread for the lifetime of this registration, so
later replies in the registered thread do not need another tag. Engagement is
an in-memory set owned by the registration closure. It is lost on restart;
there are no locks, queues, event
deduplication, exactly-once, distributed, or durable guarantees. Bolt
middleware and matchers remain the caller's audience and routing controls.

Each run receives a `SlackContext` with `team_id`, `channel_id`, `thread_ts`,
`message_ts`, `user_id`, optional `enterprise_id`, and event file metadata.
`current_slack_context()` reads it during the run. File metadata is not a file
transfer or an access grant. Native MCP reads use the invoking user's token and
current Slack permissions.

The adapter sends the event text and metadata to the agent, then delivers the
model output with Bolt's native `say` using literal text. It does not store
model-message history. Each event starts with fresh agent history. If prior
discussion is needed, the model should retrieve visible messages through native
MCP under the invoking user's token, using the current context to distinguish a
channel from a thread and following Slack's real pagination. A user's read
authorization does not authorize disclosing that data to every other thread
participant. Slack permissions, retention, and the model's final disclosure
remain the relevant limits.

The adapter's conversation ID is private and identity-qualified. It does not
turn the adapter into a history store or a Slack access-control boundary. The
core `RunContext.conversation_id` remains public.

## Unsupported behavior

This adapter does not provide:

- stored model-message history or history files;
- `SlackApp`, `ConversationStore`, `allowed_users`, or `install_url` APIs;
- a Harness transport wrapper, status/Stop handling, or streaming;
- mpim/group-DM routing;
- locks, queues, deduplication, exactly-once, distributed, or durable execution;
- a copied Slack MCP catalog, schema, pagination implementation, or user-scope policy.

## Public exports

The package exports exactly five names:

- `Slack` -- the native hosted Slack MCP capability;
- `SlackContext` -- metadata for the current event;
- `SlackFile` -- metadata for an event-attached file;
- `current_slack_context` -- the current run's context getter;
- `register_slack` -- the caller-owned Bolt registration function.

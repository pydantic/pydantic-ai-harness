---
title: Slack
description: Give an agent native Slack MCP tools and register it on a caller-owned Bolt app.
---

# Slack

`Slack` gives an agent Slack's hosted MCP tools with an explicit user OAuth
token. `register_slack` connects an ordinary string-output agent to a
caller-owned asynchronous Bolt app and supplies the invoking user's token for
each event.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation and eligibility

```bash
uv add "pydantic-ai-harness[slack]"
```

Use an internal Slack app or an app published in Slack's directory, and enable
Slack's hosted MCP server for it. Slack owns the MCP catalog, schemas, result
formats, OAuth scopes, and pagination. Recheck the [hosted MCP guide](https://docs.slack.dev/ai/slack-mcp-server/)
when changing those assumptions.

For a standalone script or test, pass the user's token to `Slack`:

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.slack import Slack

slack_token = os.environ['SLACK_USER_TOKEN']
agent = Agent('openai:gpt-5.6-sol', capabilities=[Slack(token=slack_token)])
result = await agent.run('Summarize the discussion in #support')
```

## Quick start

The caller configures Bolt OAuth and supplies a per-user token to each event.
`FileInstallationStore` stores OAuth credentials, so create
`SLACK_INSTALLATION_DIR` with owner-only (`0700`) permissions and keep its
files private.

```python
import os

from pydantic_ai import Agent
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore

from pydantic_ai_harness.slack import register_slack

agent = Agent('openai:gpt-5.6-sol')
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
)
bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
register_slack(bolt, agent)
asgi_app = AsyncSlackRequestHandler(bolt, path='/slack/events')
```

Registration adds the per-event `Slack(token=...)` capability and host-specific
instructions to each run. Do not configure `Slack` directly or through a
capability factory on this hosted agent. Registration rejects a concrete
`Slack` capability already present on the agent, so it cannot silently replace
a configured credential. A dynamic factory result cannot be inspected during
registration; core's run-level override applies to a factory result with the
same capability ID.

Serve `asgi_app` directly with one worker. Bolt's OAuth routes, normally
`/slack/install` and `/slack/oauth_redirect`, must be served by the same HTTP
application. Publish the full install URL, such as
`https://agent.example/slack/install`, in the Slack app description or App
Home so users who see the missing-identity reply know where to authorize.
Leave Bolt's `installation_store_bot_only` default unchanged so authorization
looks up the invoking user's OAuth installation alongside the bot installation.

`SLACK_USER_SCOPES` is comma-separated. A minimal read example for this
adapter is:

```bash
export SLACK_USER_SCOPES='channels:history,groups:history,im:history'
```

Add scopes for the MCP tools your hosted agent may use. Slack exposes its
provider tool catalog to the capability, and the user's OAuth scopes are the
authoritative limit on those tools. Slack's [OAuth scopes needed on user
token for different tools](https://docs.slack.dev/ai/slack-mcp-server/#oauth-scopes-needed-on-user-token-for-different-tools)
table maps read channels and threads to `channels:history`,
`groups:history`, `mpim:history`, and `im:history`, and lists the additional
search, write, file, user, canvas, reaction, and list scopes. Group DMs
(`mpim`) are not routed by this adapter, but a standalone MCP client can
request the corresponding scope.

For typed dependencies, the factory receives one `SlackContext` after an event
is accepted. It may be synchronous or asynchronous. The agent's output type
must be `str`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.slack import SlackContext, register_slack

@dataclass
class Deps:
    slack: SlackContext

def deps_factory(context: SlackContext) -> Deps:
    return Deps(slack=context)

typed_agent: Agent[Deps, str] = Agent('openai:gpt-5.6-sol')
register_slack(bolt, typed_agent, deps_factory=deps_factory)
```

An asynchronous factory can load user-specific dependencies before the run:

```python {test="skip"}
from dataclasses import dataclass

@dataclass
class AsyncDeps:
    slack: SlackContext
    user_name: str

async def async_deps_factory(context: SlackContext) -> AsyncDeps:
    user_name = await user_store.get_name(context.user_id)
    return AsyncDeps(slack=context, user_name=user_name)
```

For a non-Slack multi-user application, use Pydantic AI's capability factory
to look up a token from the run dependencies:

```python {test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.slack import Slack

@dataclass
class Deps:
    user_id: str

async def slack_for_user(ctx: RunContext[Deps]) -> Slack:
    token = await token_store.get_user_token(ctx.deps.user_id)
    return Slack(token=token)

agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[slack_for_user])
```

## Events, OAuth, and Socket Mode

The embedded HTTP manifest uses `app_mention`, `message.channels`,
`message.groups`, and `message.im`. The six bot scopes in the example match
those routes and agent messaging requirements. Remove a message event family
when it is not needed, and remove its corresponding history scope.

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

Direct messages route without a mention. An unthreaded DM receives a flat
reply. If the incoming DM is already threaded, the reply stays in that
thread. In public and private channels, an `app_mention` engages that thread.
Later human follow-ups in the engaged thread are accepted without another
mention, and accepted authorized follow-ups refresh the engagement's position
in a per-registration least-recently-used cache capped at 1,024 threads. The
least-recently-used thread is evicted when the cache is full; restart clears
the cache. Until eviction or restart, every valid human message in an engaged
thread can trigger a model run. There is no disengage control.

Slack Connect events from an external team are ignored. Ordinary channel
messages outside an engaged thread and group DMs are ignored, as are bot
messages and invalid events. An `app_mention` can engage a new channel thread. After an engaged thread is evicted or the
process restarts, a new `app_mention` is required.
There are no locks, queues, event deduplication, exactly-once, distributed,
or durable guarantees. Bolt middleware and matchers remain the caller's
audience and routing controls.

Each hosted run receives a `SlackContext` with `team_id`, `channel_id`,
`thread_ts`, `message_ts`, `user_id`, optional `enterprise_id`, and event
file metadata. `current_slack_context()` reads it during the run. File
metadata is not a file transfer or an access grant. Native MCP reads use the
invoking user's token and current Slack permissions.

The host puts event metadata and text in the model prompt. For an
unthreaded DM, the private conversation ID uses that message's timestamp, so
each DM message starts its own conversation-keyed state. Code should use
`current_slack_context()` or the `SlackContext` supplied by
`deps_factory`. The adapter delivers the model output with Bolt's native
`say` using literal text. It does not store model-message history. Each event
starts with fresh agent history. If prior discussion is needed, the model
should retrieve visible messages through native MCP under the invoking user's
token, using the current context to distinguish a channel from a thread and
following Slack's real pagination. A user's read authorization does not
authorize disclosing that data to every other thread participant. Slack
permissions, retention, and the model's final disclosure remain the relevant
limits.

The host's conversation ID is private and identity-qualified. Consequently,
conversation-keyed capabilities such as `Memory` or `ConversationSearch` see
one conversation per Slack participant in a thread. This keeps each user's
state isolated. It does not turn the adapter into a history store or a Slack
access-control boundary. The core `RunContext.conversation_id` remains public.

With Pydantic AI core 2.38.0, cancelling a run under anyio level cancellation
may leave its MCP client open. This is an unresolved core cleanup limitation
when a Slack run is cancelled during an MCP operation.

## Unsupported behavior

This adapter does not provide stored model-message history or history files, a
Harness transport wrapper, status/Stop handling, streaming, mpim/group-DM
routing, locks, queues, deduplication, exactly-once, distributed, or durable
execution. It does not copy Slack's MCP catalog, schemas, pagination, or
user-scope policy. `Slack` requires a non-empty token when constructed; token
revocation, missing installations, insufficient scopes, and network failures
remain run-time failures.

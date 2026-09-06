---
title: Slack
description: Give an agent native Slack MCP tools and serve it through a caller-owned Bolt app.
---

# Slack

Use `Slack` when an agent should answer questions about the Slack conversation
that invoked it, using Slack's hosted MCP server. Use `SlackApp` to connect that
agent to a caller-owned asynchronous Slack Bolt app.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
uv add "pydantic-ai-harness[slack]"
```

The Slack extra supplies the Bolt integration. Create a Slack app that is
eligible for Slack's hosted MCP server: use an internal app or an app published
in the Slack directory, and enable MCP for that app. Slack's [hosted MCP
guide](https://docs.slack.dev/ai/slack-mcp-server/) is authoritative for
eligibility, tool schemas, result shapes, OAuth scopes, and pagination.

> External assumptions, verified 2026-09-06: the native MCP endpoint is
> `https://mcp.slack.com/mcp`, and the installed Slack Bolt APIs provide
> `AsyncOAuthSettings`, `installation_store_bot_only`,
> `AsyncSocketModeHandler.start_async`, `close_async`, and
> `AsyncSlackRequestHandler`. Re-check these signatures and the provider
> contract against the [hosted MCP guide](https://docs.slack.dev/ai/slack-mcp-server/),
> [Bolt OAuth guide](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/),
> and the installed SDK before changing integration code.

## Quick start

```python
import os

from pydantic_ai import Agent
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore

from pydantic_ai_harness.slack import Slack, SlackApp

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
        'mpim:history',
    ],
    user_scopes=os.environ['SLACK_USER_SCOPES'],
    redirect_uri=os.environ['SLACK_REDIRECT_URI'],
    installation_store=installation_store,
    installation_store_bot_only=False,
)
bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
SlackApp(agent, app=bolt, allowed_users={'T01234567': {'U01234567'}})
```

Serve the resulting Bolt app through `AsyncSlackRequestHandler` as shown in
the [ASGI section](#asgi). Set `SLACK_INSTALLATION_DIR` to a caller-chosen
directory with owner-only permissions because it contains OAuth credentials.
Create it with mode `0700` and keep its files inaccessible to other users;
`FileInstallationStore` persists the bot and per-user tokens there.
`SLACK_USER_SCOPES` is a comma-separated list copied from Slack's current MCP
guide. The [runnable example](../../examples/slack_agent.py) uses the same
configuration and serves one ASGI worker.

`SlackApp` registers Bolt listeners on the `AsyncApp` supplied by the caller.
The caller then starts Socket Mode or serves the Bolt app through an ASGI
adapter. `allowed_users` is workspace-qualified: the keys are Slack workspace
IDs and each value is a set of user IDs. Use `{'T01234567': {'U01234567'}}`,
not a bare user ID. `allowed_users='all'` allows every user who can reach the
installed app, so use it only when that audience is intentional.

## Configure Bolt and OAuth

For a multi-user installation, the caller owns [Bolt OAuth](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/) and must configure it
to resolve a per-user OAuth token for each event. Use a user-keyed installation
store and set `installation_store_bot_only=False`; bot-only installations do not
provide the user token that Slack MCP needs. Keep the `AsyncApp` instance and
its installation store in the caller's application.

```python {test="skip"}
import os

from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore

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
        'mpim:history',
    ],
    user_scopes=os.environ['SLACK_USER_SCOPES'],
    redirect_uri=os.environ['SLACK_REDIRECT_URI'],
    installation_store=installation_store,
    installation_store_bot_only=False,
)
bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
SlackApp(
    agent,
    app=bolt,
    allowed_users={'T01234567': {'U01234567'}},
    install_url='https://agent.example/slack/install',
)
```

The exact user scopes are selected from Slack's current hosted MCP
documentation and are real authorization: Harness does not turn them into a
second policy or pretend to confine native MCP calls. Do not copy a catalog or
schema into the application. Slack owns the catalog, schemas, result formats,
and pagination behavior.

The install URL is optional. If configured, it is included when an allowed
event has no per-user OAuth token. If OAuth is configured without a user token,
the run fails before the model request and the user receives a connection
message. There is no fixed-token fallback. An unauthorized user is ignored by
the event handler.

## Socket Mode

Socket Mode is started on the caller-owned Bolt app. The native asynchronous
handler method is `start_async`; construct and close the handler inside the
running event loop.

```python {test="skip"}
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

async def serve_socket_mode() -> None:
    handler = AsyncSocketModeHandler(bolt, 'xapp-your-app-level-token')
    try:
        await handler.start_async()
    finally:
        await handler.close_async()
```

Create an app-level token with `connections:write`. Socket Mode still needs
OAuth HTTP routes for onboarding: authorization, installation storage, the
install page, and the OAuth redirect are separate from the Socket Mode
connection. Configure those routes in the caller's web service.

## ASGI

For HTTP events, serve the installed Bolt ASGI adapter directly around the same
caller-owned `AsyncApp`, with one worker:

```python {test="skip"}
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler

SlackApp(
    agent,
    app=bolt,
    allowed_users={'T01234567': {'U01234567'}},
    install_url='https://agent.example/slack/install',
)
asgi_app = AsyncSlackRequestHandler(bolt, path='/slack/events')
```

Serve `asgi_app` directly at `/slack/events` and expose Bolt's configured
`/slack/install` and `/slack/oauth_redirect` routes for OAuth onboarding. Do
not mount it below another path that changes the request path. Events and OAuth
onboarding are both required for a multi-user app. If you switch to Socket
Mode, set `socket_mode_enabled` to `true` and run the Socket Mode handler, but
continue serving the OAuth HTTP routes for onboarding.

## Slack event configuration

The app must be able to receive `app_mention`, supported `message` events, and
`agent_session_stopped`. Subscribe to the message event families used by your
deployment, such as `message.channels`, `message.groups`, `message.im`, and
`message.mpim`. The app needs the corresponding history scopes and `chat:write`
to post replies. `assistant:write` enables the native agent status and Stop
behavior. Do not add approval subscriptions or active-context subscriptions.
Removing a message event family also permits removing its corresponding bot
history scope from the manifest and OAuth `scopes` list. User MCP scopes remain
provider-configured by Slack.

This starting manifest keeps only the event and status behavior owned by
`SlackApp`; replace the IDs and URLs and choose message families for your app:

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
        "im:history",
        "mpim:history"
      ]
    }
  },
  "settings": {
    "event_subscriptions": {
      "request_url": "https://agent.example/slack/events",
      "bot_events": [
        "app_mention",
        "message.channels",
        "message.groups",
        "message.im",
        "message.mpim",
        "agent_session_stopped"
      ]
    },
    "is_mcp_enabled": true,
    "socket_mode_enabled": false
  }
}
```

The [Slack agent guide](https://docs.slack.dev/ai/developing-agents/), [Bolt agent features](https://docs.slack.dev/tools/bolt-python/concepts/adding-agent-features/), and [agent sessions](https://docs.slack.dev/ai/agent-sessions/) guides, together with the
[Slack app manifest reference](https://docs.slack.dev/reference/app-manifest/)
define the provider-owned fields. Slack may change them independently of a
Harness release.

## Conversation and context

A direct message invokes the agent without a mention. A channel or group-DM
conversation starts when the app is mentioned; replies in the engaged thread
continue the conversation. Unrelated top-level messages and threads are
ignored. Within one running process, bot messages, edits, deletions, and
duplicate retry deliveries do not start duplicate runs. This in-memory
deduplication is lost on restart. The host posts the model's final output, so the agent must not
send a duplicate ordinary answer through a Slack messaging tool. Slack status
and Stop behavior use native [agent features](https://docs.slack.dev/tools/bolt-python/concepts/adding-agent-features/)
and [agent sessions](https://docs.slack.dev/ai/agent-sessions/).

Each run has a `SlackContext` with the workspace (`team_id`), channel
(`channel_id`), thread (`thread_ts`), invoking message (`message_ts`), user
(`user_id`), optional enterprise (`enterprise_id`), and files attached to the
invoking message. `current_slack_context()` returns it while the run is active.
This context is interpretive metadata for resolving the user's request. It is
not a confinement or security boundary. OAuth authorization and Slack's native
MCP authorization remain authoritative.

For example, to answer “How many messages did Priya send in the current
conversation?” the agent should use the current context to identify the
workspace, channel, and thread, use native Slack discovery to resolve Priya,
then use the native read or search tools with their current schemas. It must
distinguish a channel-wide count from a thread-reply count, follow real
pagination until the result is complete, and say when the available result is
partial. The result format and pagination rules come from Slack MCP, not from a
Harness catalog or wrapper.

Files in the context are metadata from the invoking event. Native Slack MCP
schemas and the invoking user's OAuth authorization decide whether a file can
be read. Do not infer file authority from model-visible history.

## History, stores, and execution limits

Runs are serialized per Slack thread and the default store is in memory. Pass a
`ConversationStore` implementation, or use `FileConversationStore`, when a
single-process bot should retain thread history after restart:

```python {test="skip"}
from pydantic_ai_harness.slack import FileConversationStore

SlackApp(
    agent,
    app=bolt,
    allowed_users={'T01234567': {'U01234567'}},
    store=FileConversationStore('~/.slack-agent'),
)
```

`FileConversationStore` provides restart persistence for one process. It does
not coordinate workers or make a deployment distributed. Keep `SlackApp` to a
single process and one worker. Its in-memory locks, event deduplication,
engagement state, cancellation, and per-event OAuth identity are process-local.
The running process retains up to 30,000 event IDs per workspace for 10 minutes;
those entries are lost on process restart.

Important: an invoking user's Slack read authorization does not authorize
disclosure of that data to every participant in the thread. Shared thread
history is an information-flow boundary to consider: every user
who can participate in the same workspace, channel, and thread can cause the
stored model history for that thread to be replayed. Do not put data in shared
thread history that another participant must not see. A custom store can apply
the application's own retention and isolation policy.

`SlackApp` does not support durable execution capabilities. There are no
deferred Slack results. Native MCP tool errors are returned as retryable tool
results so the model can recover when possible. Missing OAuth fails before the
model request and produces the connection reply, including `install_url` when
configured. Model failures and terminal host failures produce the generic
error reply. If delivery succeeds but saving history fails, the host posts the
history-save warning. These failures do not cause a second ordinary model
reply.

## Dependencies

`deps` supplies one dependency value to every run. `deps_factory` can derive a
value from the typed Slack context; pass one or the other, not both.

```python {test="skip"}
SlackApp(
    agent,
    app=bolt,
    allowed_users={'T01234567': {'U01234567'}},
    deps_factory=lambda context: make_dependencies(context.team_id, context.user_id),
)
```

## Public exports

The Slack package has eight public exports:

- `Slack` adds the native hosted Slack MCP toolset to an agent.
- `SlackApp` registers the Slack event and stop handlers on a caller-owned Bolt app.
- `SlackContext` describes the current Slack run metadata.
- `SlackFile` describes a file attached to the invoking message.
- `current_slack_context` reads the current context during a run.
- `ConversationStore` is the protocol for thread-history storage.
- `InMemoryConversationStore` stores history in the current process.
- `FileConversationStore` stores history in private JSON files for restart persistence.

The native Slack MCP catalog is intentionally not re-exported or copied into
Harness. Read Slack's [hosted MCP guide](https://docs.slack.dev/ai/slack-mcp-server/)
for current tools, schemas, scopes, and pagination.

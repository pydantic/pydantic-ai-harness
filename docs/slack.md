---
title: Slack
description: Give an agent typed Slack workspace tools and serve it inside Slack with Bolt.
---

# Slack

`Slack` gives an agent Slack workspace tools, and `SlackApp` serves that agent
inside Slack. The capability uses Slack's hosted MCP server for model-selected
search, reads, and actions. The app adapter uses Bolt for events, OAuth, thread
routing, and response delivery.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
uv add "pydantic-ai-harness[slack]"
```

## Quick start

```python
from pydantic_ai import Agent

from pydantic_ai_harness.slack import Slack, SlackApp

agent = Agent('openai:gpt-5.6-sol', capabilities=[Slack()])
SlackApp(agent).run()
```

For this Socket Mode example, set `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, and
`SLACK_MCP_TOKEN`. Set `SLACK_ALLOWED_USER_IDS` to the user ID that owns that
MCP token. This fixed-identity setup is intended for a local or internal
single-user app; use per-user OAuth below before allowing multiple users.
Invite the app to a channel, then mention it once to start a thread. Direct
messages need no mention.

## Slack app setup

Create one internal or Marketplace Slack app for both Bolt and MCP. Enable
Socket Mode for `SlackApp.run()`, or configure HTTP request URLs for
`SlackApp.http_app()`.

For ingress and delivery, subscribe the bot to `app_mention`, `message.im`,
`message.channels`, `app_context_changed`, and `agent_session_stopped`. Add
`message.groups` for private-channel threads and `message.mpim` for group DMs.
Grant the matching bot scopes:

- `app_mentions:read`
- `chat:write`
- `im:write` for private approval prompts
- `channels:history`
- `groups:history` when subscribing to `message.groups`
- `im:history`
- `mpim:history` when subscribing to `message.mpim`

When using Slack's `agent_view`, `SlackContext.active_entities` preserves any
active-view entities attached to a message, including typed message coordinates.
Slack's agent surface requires the `assistant:write` bot scope. `SlackApp` uses
`agents.sessions.setStatus` for the visible working state and handles Slack's
native stop button.

Enable interactivity for approval buttons. In Socket Mode it needs no request
URL. For the Events API, point both event subscriptions and interactivity to the
public URL served at `/slack/events`. Public and private channels must invite
the app before it can receive their events.

Generate an app-level token with `connections:write` for Socket Mode. The Events
API uses the bot token and signing secret instead.

Enable the [Slack MCP server](https://docs.slack.dev/ai/slack-mcp-server/) for
the same app. `SlackTools.required_user_scopes` returns the user scopes for an
exact typed selection. The default `Slack()` needs `search:read.users`,
`files:read`, `channels:history`, `groups:history`, `im:history`, and
`mpim:history`.

The following manifest is the complete starting point for the default toolset.
Replace the name and request URLs. Remove event families your app will not use.

```json
{
  "display_information": {"name": "Pydantic AI Agent"},
  "features": {
    "agent_view": {"agent_description": "A Pydantic AI agent", "suggested_prompts": []},
    "app_home": {
      "home_tab_enabled": false,
      "messages_tab_enabled": true,
      "messages_tab_read_only_enabled": false
    },
    "bot_user": {"display_name": "Pydantic AI Agent", "always_online": true}
  },
  "oauth_config": {
    "redirect_urls": ["https://agent.example/slack/oauth_redirect"],
    "scopes": {
      "user": [
        "search:read.users",
        "files:read",
        "channels:history",
        "groups:history",
        "im:history",
        "mpim:history"
      ],
      "bot": [
        "app_mentions:read",
        "assistant:write",
        "channels:history",
        "chat:write",
        "groups:history",
        "im:history",
        "im:write",
        "mpim:history"
      ]
    }
  },
  "settings": {
    "event_subscriptions": {
      "request_url": "https://agent.example/slack/events",
      "bot_events": [
        "agent_session_stopped",
        "app_context_changed",
        "app_mention",
        "message.channels",
        "message.groups",
        "message.im",
        "message.mpim"
      ]
    },
    "interactivity": {"is_enabled": true, "request_url": "https://agent.example/slack/events"},
    "is_mcp_enabled": true,
    "socket_mode_enabled": true,
    "token_rotation_enabled": false
  }
}
```

`SLACK_MCP_TOKEN` is a fixed user identity. `SlackApp` accepts it only when
exactly one Slack user is allowed to invoke the agent. A distributed or
multi-user app should let Bolt perform OAuth, store installations by user, and
pass a caller-configured `AsyncApp` to `SlackApp`.

For a multi-user app, configure Bolt OAuth with a user-keyed installation store.
This example uses Slack SDK's SQLite store; use a shared database-backed store
when more than one process serves the app.

That store persists OAuth installations only. Approval waits, stop routing,
per-thread locks, engagement, and retry deduplication are process-local. Run one
`SlackApp` worker, or configure ingress affinity so every event for a workspace
and thread reaches the same worker. A shared `ConversationStore` does not provide
that coordination.

```python {test="skip"}
import os

from pydantic_ai import Agent
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.sqlite3 import SQLite3InstallationStore

from pydantic_ai_harness.slack import Slack, SlackAccess, SlackApp

slack = Slack()
agent = Agent('openai:gpt-5.6-sol', capabilities=[slack])
client_id = os.environ['SLACK_CLIENT_ID']
oauth_settings = AsyncOAuthSettings(
    client_id=client_id,
    client_secret=os.environ['SLACK_CLIENT_SECRET'],
    scopes=[
        'app_mentions:read',
        'assistant:write',
        'channels:history',
        'chat:write',
        'groups:history',
        'im:history',
        'im:write',
        'mpim:history',
    ],
    user_scopes=sorted(slack.tools.required_user_scopes),
    redirect_uri='https://agent.example/slack/oauth_redirect',
    installation_store=SQLite3InstallationStore(database='slack-installations.db', client_id=client_id),
)

bolt = AsyncApp(oauth_settings=oauth_settings)
slack_app = SlackApp(
    agent,
    app=bolt,
    access=SlackAccess.workspace(),
    install_url='https://agent.example/slack/install',
)
asgi_app = slack_app.http_app()
```

Every user must visit the install URL and authorize the requested user scopes;
Slack's MCP setup requires a separate user installation before it can query on
that person's behalf. Bolt then resolves both identities for an authorized
event: its Web API client delivers the response as the app, while
the OAuth user token creates a new Slack MCP session for the invoking user. Set
`install_url` or `SLACK_INSTALL_URL` so an unauthorized user receives the link
instead of a generic configuration error.

In OAuth mode, a missing per-event user token fails closed rather than falling
back to `SLACK_MCP_TOKEN`. Pass `mcp_token` explicitly only when the whole
installation is intentionally meant to use one fixed user identity.

Serve `asgi_app` directly so it receives `/slack/events`, `/slack/install`, and
`/slack/oauth_redirect`. When combining it with FastAPI or Starlette, mount it
at `/` after more specific application routes. Do not mount the handler at
`/slack/events`: ASGI mounting strips that prefix before Bolt performs its
exact path match.

## What each layer owns

| Layer | Responsibility |
| --- | --- |
| `SlackApp` and Bolt | Events, OAuth installations, request verification, conversation routing, and final-response delivery |
| `Slack` and Slack MCP | Agent-selected search, channel and thread reads, user lookup, and explicitly enabled Slack actions |
| Slack Web API client | Deterministic response delivery and approval UI that Slack MCP does not provide |

Composio and Pipedream can supply tools for other services used by the same
agent. They are not required for the Slack conversation loop or Slack tools.

## Conversation behavior

`SlackApp` follows normal Slack expectations:

- A DM invokes the agent without a mention.
- A group DM requires one mention to start; replies in its engaged thread do not.
- One `@agent` mention starts a channel thread.
- Later human replies in that engaged thread invoke the agent without another mention.
- Top-level channel messages and unrelated threads are ignored.
- Bot messages, edits, deletions, and Slack retry deliveries do not start duplicate runs.
- User-authored file shares and thread broadcasts continue an engaged thread. Attached file IDs and metadata are typed,
  and the default toolset can read only files attached to that invoking message.
- A follow-up that needs the same file should include the attachment again. Harness does not infer file authority from
  model-visible history.
- Runs are serialized per thread, and each thread has separate Pydantic AI message history.

The default store is process memory. Pass a `ConversationStore` implementation
for production persistence. In-memory event deduplication covers Slack's
standard five-minute retry schedule. High-availability deployments should also
deduplicate event IDs in their durable ingress queue.

## Typed tool selection

The default is `SlackTools.current_conversation()`: user lookup, attached-file
reads, and channel and thread reads. Runtime argument checks confine those reads
to the invoking conversation, its active-view context, and its attached files.
It handles questions like "How many messages did Aditya send in this channel?"
without giving the model workspace search or unrelated private history. Exact
aggregation over a channel too large for the model context should be provided as
a deterministic application tool. Write tools are absent.

Select exact additions with `SlackTool` values:

```python
from pydantic_ai_harness.slack import Slack, SlackTool, SlackTools

slack = Slack(
    tools=(
        SlackTools.read_only()
        | SlackTools.of(SlackTool.SEND_MESSAGE, SlackTool.ADD_REACTION)
    )
)
```

Call `.restrict_to_current_conversation()` on an exact selection when its
channel, thread, and file reads should keep the default runtime boundary. Agent
specs apply that boundary to exact selections by default; set
`read_scope: workspace` only when unrestricted reads are intentional.

Use `SlackTools.workspace_read()` to add public workspace search,
`SlackTools.read_only()` for every supported read surface including private
search, or `SlackTools.of(...)` for the smallest exact set. Use
`SlackTools.none()` when Slack only hosts the agent and no MCP tools are needed.
Conversation restrictions are sticky under `|`, so adding a duplicate tool cannot
widen its authority. Choose `workspace_read()`, `read_only()`, or an exact `of()`
selection directly when broader reads are intended.

Slack can publish a new MCP tool before Harness releases a matching enum value.
Advanced users can describe its exact name and OAuth scopes explicitly:

```python
from pydantic_ai_harness.slack import Slack, SlackCustomTool, SlackTools

canvas = SlackCustomTool('slack_create_canvas', user_scopes={'canvases:write'})
slack = Slack(tools=SlackTools.custom(canvas))
```

The name is checked against Slack's discovered catalog, its scopes contribute to
`required_user_scopes`, and `approval='writes'` requires approval because Harness
has not classified the custom tool as read-only. Prefer `SlackTool` values once
they are available.

Tool names are checked against Slack's discovered MCP catalog on connection. A
selected tool that Slack does not supply raises an error instead of disappearing
from the model's toolset.

The supported typed values are:

- `SEARCH_PUBLIC`
- `SEARCH_PUBLIC_AND_PRIVATE`
- `SEARCH_CHANNELS`
- `SEARCH_USERS`
- `READ_CHANNEL`
- `READ_THREAD`
- `READ_FILE`
- `READ_USER_PROFILE`
- `LIST_CHANNEL_MEMBERS`
- `SEND_MESSAGE`
- `SCHEDULE_MESSAGE`
- `ADD_REACTION`

Slack may add tools to its hosted server independently. Harness adds a typed
enum member after the tool and its behavior have been verified.

## Approval policy

Known write tools require Pydantic AI approval by default. `SlackApp` privately
DMs each allowed approver the pending call and its complete arguments with Slack
buttons, then resumes the run with the first valid decision. For channel and
group-DM runs, approval details stay out of the invoking conversation. A run
invoked in the approver's direct message necessarily uses that same private DM.

```python {test="skip"}
Slack(
    tools=SlackTools.of(SlackTool.SEND_MESSAGE),
    approval='writes',
    approver_ids=['U01REVIEWER'],
)
```

Use `approval='all'` to review every selected Slack call, or
`approval='none'` when authorization and confirmation are enforced elsewhere.
Set `approval_timeout_seconds` when the default ten-minute decision window does
not fit the deployment. A timeout denies the action.
Approval is confirmation, not authentication. Slack OAuth scopes and the app's
own authorization rules remain the access boundary.

## Identity and concurrency

The bot token and MCP user token have different jobs and are not interchangeable:

- The bot token lets Bolt receive events and deliver app responses.
- The user token lets Slack MCP act with the invoking user's Slack access.
- The app token opens Socket Mode and never authorizes Web API or MCP calls.
- The signing secret verifies HTTP requests and is not an API credential.

`Slack` creates a new `MCPToolset` once per agent run. It does not cache a shared
authenticated session, so overlapping runs cannot inherit another user's token.
The current channel, thread, message, workspace, enterprise, and user IDs are
available as `current_slack_context()` while the run executes. Tokens are
kept in private run context and are not fields on `SlackContext`.

## Slack MCP requirements

As verified on 2026-09-05, Slack allows hosted MCP access for internal apps and apps published
in the Slack Marketplace. Enable MCP for the Slack app and request the user
OAuth scopes needed by the selected tools. Search, private search, files,
history, profiles, messages, reactions, and other tool families use distinct
scopes.

Before changing the typed catalog, scopes, or manifest, re-check Slack's
[hosted MCP guide](https://docs.slack.dev/ai/slack-mcp-server/),
[agent guide](https://docs.slack.dev/ai/developing-agents/), and
[app manifest reference](https://docs.slack.dev/reference/app-manifest/). Slack
can change these provider-owned contracts independently of Harness releases.

If no per-event user token, explicit `mcp_token`, or `SLACK_MCP_TOKEN` is
available, the capability fails before the model request. `SlackApp` turns that
failure into an actionable connection message instead of letting the agent
claim that it cannot read Slack.

## Existing agent dependencies

`SlackApp` does not replace an agent's dependency type:

```python {test="skip"}
agent = Agent('openai:gpt-5.6-sol', deps_type=Warehouse, capabilities=[Slack()])

SlackApp(agent, deps=Warehouse(dsn=DSN)).run()
```

Use `deps_factory` when dependencies vary by Slack thread:

```python {test="skip"}
SlackApp(agent, deps_factory=lambda thread: Warehouse(dsn=dsn_for(thread.channel_id))).run()
```

## Durable execution

The `Slack` capability can be used by an agent that runs outside Slack with a
fixed MCP token and an unrestricted selection such as `SlackTools.workspace_read()`.
The default selection requires a `SlackApp`-bound conversation. The `SlackApp`
conversation loop is process-local: the invoking
OAuth identity, Bolt client, cancellation scope, and approval interaction are
not replayable on another worker. Keep that adapter outside durable work and
hand durable tasks to it through an explicit application boundary.

A custom host with a fixed token can bind trusted conversation coordinates with
`SlackContext.bind()` around `agent.run(...)`. This does not bind an OAuth token,
a delivery client, or approval routing.

## API reference

::: pydantic_ai_harness.slack.Slack

::: pydantic_ai_harness.slack.SlackApp

::: pydantic_ai_harness.slack.SlackTools

::: pydantic_ai_harness.slack.SlackCustomTool

::: pydantic_ai_harness.slack.SlackAccess

::: pydantic_ai_harness.slack.SlackContext

::: pydantic_ai_harness.slack.SlackFile

# Slack

`Slack` gives an agent Slack workspace tools, and `SlackApp` serves that agent
inside Slack. The capability uses Slack's hosted MCP server for model-selected
search, reads, and actions. The app adapter uses Bolt for events, OAuth, thread
routing, and response delivery.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

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

For ingress and delivery, subscribe the bot to `app_mention`, `message.im`, and
`message.channels`. Add `message.groups` for private-channel threads and
`message.mpim` for group DMs. Grant the matching bot scopes:

- `app_mentions:read`
- `chat:write`
- `channels:history`
- `groups:history` when subscribing to `message.groups`
- `im:history`
- `mpim:history` when subscribing to `message.mpim`

When using Slack's `agent_view`, also subscribe to `app_context_changed` so a
DM event can tell the agent what channel, thread, canvas, or list the user is
currently viewing. `SlackContext.active_entities` preserves Slack's relevance
order. Slack's agent surface also requires the `assistant:write` bot scope.

Enable interactivity for approval buttons. In Socket Mode it needs no request
URL. For the Events API, point both event subscriptions and interactivity to the
public URL served at `/slack/events`. Public and private channels must invite
the app before it can receive their events.

Generate an app-level token with `connections:write` for Socket Mode. The Events
API uses the bot token and signing secret instead.

Enable the [Slack MCP server](https://docs.slack.dev/ai/slack-mcp-server/) for
the same app. `SlackTools.workspace_read()` needs Slack's search, user-search,
and history scopes. `SlackTools.read_only()` adds files, profiles, and channel
membership, with their corresponding user scopes. An exact
`SlackTools.of(...)` selection can request only the scopes for those tools.

`SLACK_MCP_TOKEN` is a fixed user identity. `SlackApp` accepts it only when
exactly one Slack user is allowed to invoke the agent. A distributed or
multi-user app should let Bolt perform OAuth, store installations by user, and
pass a caller-configured `AsyncApp` to `SlackApp`.

After constructing `oauth_settings` with `AsyncOAuthSettings` and a user-keyed
installation store as shown in the
[Bolt for Python OAuth guide](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/),
the Harness integration is the following fragment. The `agent` is the Pydantic
AI agent constructed above.

```python {test="skip"}
from slack_bolt.app.async_app import AsyncApp
from pydantic_ai_harness.slack import SlackAccess

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
`context.user_token` creates a new Slack MCP session for the invoking user. Set
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
- One `@agent` mention starts a channel thread.
- Later human replies in that engaged thread invoke the agent without another mention.
- Top-level channel messages and unrelated threads are ignored.
- Bot messages, message subtypes, and Slack retry deliveries do not start duplicate runs.
- Runs are serialized per thread, and each thread has separate Pydantic AI message history.

The default store is process memory. Pass a `ConversationStore` implementation
for production persistence. In-memory event deduplication covers Slack's
standard five-minute retry schedule. High-availability deployments should also
deduplicate event IDs in their durable ingress queue.

## Typed tool selection

The default is `SlackTools.workspace_read()`: user and channel lookup, public
and private search, and channel and thread reads. It is enough for questions
like "How many messages did Aditya send in this channel?" without also asking
for file, profile, or channel-membership access. Write tools are absent.

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

Use `SlackTools.read_only()` for every supported read surface, or
`SlackTools.of(...)` for the smallest exact set.

Slack can publish a new MCP tool before Harness releases a matching enum value.
Advanced users can select that exact provider name with
`SlackTools.named('slack_create_canvas')`. The name is still checked against
Slack's discovered catalog, and `approval='writes'` requires approval for every
unclassified named tool. Prefer `SlackTool` values once they are available.

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

Known write tools require Pydantic AI approval by default. `SlackApp` renders
the pending call and its complete arguments with Slack buttons, then resumes the
run with the decision.

```python
Slack(
    tools=SlackTools.of(SlackTool.SEND_MESSAGE),
    approval='writes',
    approver_ids=['U01REVIEWER'],
)
```

Use `approval='all'` to review every selected Slack call, or
`approval='none'` when authorization and confirmation are enforced elsewhere.
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
redacted from its representation.

## Slack MCP requirements

Slack currently allows hosted MCP access for internal apps and apps published
in the Slack Marketplace. Enable MCP for the Slack app and request the user
OAuth scopes needed by the selected tools. Search, private search, files,
history, profiles, messages, reactions, and other tool families use distinct
scopes.

If no per-event user token, explicit `mcp_token`, or `SLACK_MCP_TOKEN` is
available, the capability fails before the model request. `SlackApp` turns that
failure into an actionable connection message instead of letting the agent
claim that it cannot read Slack.

## Existing agent dependencies

`SlackApp` does not replace an agent's dependency type:

```python
agent = Agent('openai:gpt-5.6-sol', deps_type=Warehouse, capabilities=[Slack()])

SlackApp(agent, deps=Warehouse(dsn=DSN)).run()
```

Use `deps_factory` when dependencies vary by Slack thread:

```python
SlackApp(agent, deps_factory=lambda thread: Warehouse(dsn=dsn_for(thread.channel_id))).run()
```

# Channels

Put a text-output Pydantic AI agent where your users already send messages.
Channels are useful for:

- a personal assistant in a private chat
- a support agent in team messages
- an internal agent that can use the same tools and capabilities as your app

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/channels/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](../../docs/index.md#version-policy).

These examples use Anthropic. Install the provider extra and set
`ANTHROPIC_API_KEY` before running them:

```bash
uv add "pydantic-ai-harness[anthropic]" starlette uvicorn
```

## Slack: DMs and app mentions

Use Slack to answer direct messages or respond when somebody mentions the agent
in a channel.

Create a Slack app with `chat:write`, `app_mentions:read`, and `im:history` bot
scopes. Subscribe it to `app_mention` and `message.im`. Set `SLACK_BOT_TOKEN`
and `SLACK_SIGNING_SECRET` from the app settings, keeping both outside source
control. Replace `U0123456789` below with your Slack member id from
**Profile > Copy member ID**, and `C0123456789` with a channel where the bot may
answer mentions.

```python
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from pydantic_ai import Agent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from pydantic_ai_harness.channels import ChannelHost, WebhookRequest
from pydantic_ai_harness.channels.slack import SlackChannel

agent = Agent('anthropic:claude-fable-5')
channel = SlackChannel(
    os.environ['SLACK_BOT_TOKEN'],
    os.environ['SLACK_SIGNING_SECRET'],
    allowed_channel_ids={'C0123456789'},
)
host = ChannelHost(agent, channel, allowed_senders={'U0123456789'})


async def receive_slack(request: Request) -> PlainTextResponse:
    result = await channel.handle_webhook(
        WebhookRequest(request.method, request.headers, request.query_params, await request.body())
    )
    return PlainTextResponse(result.body, status_code=result.status_code)


@asynccontextmanager
async def channel_lifespan(_app: Starlette) -> AsyncIterator[None]:
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(host.serve)
        yield
        tasks.cancel_scope.cancel()


app = Starlette(
    routes=[Route('/slack/events', receive_slack, methods=['POST'])],
    lifespan=channel_lifespan,
)
```

Save this as `app.py`, run `uvicorn app:app`, and point Slack's Events API URL
at `/slack/events`.
For GovSlack, pass `api_base_url='https://slack-gov.com/api'` to `SlackChannel`.

The task group owns the channel service and waits for cancellation and cleanup during shutdown.
The lifespan and route must use the same process and async event-loop thread.
Route each Slack app installation to exactly one live host process. Terminate TLS and limit
request body size in the ASGI server or reverse proxy.

## WhatsApp: messages to a business number

Use WhatsApp when customers or teammates should reach the agent through a
WhatsApp Business phone number.

Create a Meta app and WhatsApp Business phone number, then set a System User
access token, phone number id, app secret, and a private webhook verification
token. The access token needs `whatsapp_business_messaging` for the phone number.
Replace `15551234567` below with the sender's international phone number
without `+`, matching the webhook `from` value.

```python
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from pydantic_ai import Agent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from pydantic_ai_harness.channels import ChannelHost, WebhookRequest
from pydantic_ai_harness.channels.whatsapp import WhatsAppChannel

agent = Agent('anthropic:claude-fable-5')
whatsapp = WhatsAppChannel(
    os.environ['WHATSAPP_ACCESS_TOKEN'],
    os.environ['WHATSAPP_PHONE_NUMBER_ID'],
    os.environ['META_APP_SECRET'],
    os.environ['WHATSAPP_VERIFY_TOKEN'],
)
whatsapp_host = ChannelHost(agent, whatsapp, allowed_senders={'15551234567'})


async def receive_whatsapp(request: Request) -> PlainTextResponse:
    result = await whatsapp.handle_webhook(
        WebhookRequest(request.method, request.headers, request.query_params, await request.body())
    )
    return PlainTextResponse(result.body, status_code=result.status_code)


@asynccontextmanager
async def whatsapp_lifespan(_app: Starlette) -> AsyncIterator[None]:
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(whatsapp_host.serve)
        yield
        tasks.cancel_scope.cancel()


app = Starlette(
    routes=[Route('/whatsapp/webhook', receive_whatsapp, methods=['GET', 'POST'])],
    lifespan=whatsapp_lifespan,
)
```

Save this as `app.py`, run `uvicorn app:app`, and register `/whatsapp/webhook`
and the verification token in Meta. Subscribe the route to `messages`.

The task group owns the channel service and waits for cancellation and cleanup during shutdown.
The lifespan and route must use the same process and async event-loop thread.
Route each WhatsApp phone number to exactly one live host process. Terminate TLS and limit
request body size in the ASGI server or reverse proxy.
The default Graph API version is `v26.0`; pass `api_version` for another
supported version.

## How channels fit Pydantic AI

Each inbound message starts an ordinary `Agent.run()`. The agent keeps its
instructions, tools, capabilities, dependencies, and history processors.

`ChannelHost` runs outside the agent loop. It loads the conversation's Pydantic
AI message history, runs the agent, saves the updated history, and asks the
adapter to send the text response. A long-lived channel connection is therefore
an integration, not an `AbstractCapability`.

## Conversation behavior

- `allowed_senders` is required and cannot be empty. Messages from other sender
  ids are dropped before the history store or agent is called.
- Turns in one conversation run in arrival order. Different conversations may
  run concurrently.
- At most 100 turns may be running or waiting inside the host by default. Set
  `max_pending_turns` to tune this limit. Webhook adapters separately buffer up
  to `max_queued_messages` verified messages before the host accepts them.
- `/new` waits for earlier turns in the conversation, then deletes its stored
  Pydantic AI message history. It does not cancel an active turn.
- `InMemoryConversationStore` is the default. It keeps history only for the
  current process. Implement `ConversationStore` for durable storage.
- One `ChannelHost` serves one adapter. Use separate hosts and stores for
  multiple bot accounts or providers.
- Route each bot installation to exactly one live host process. The in-process lane and
  replace-style store API do not serialize turns across multiple workers.

The host accepts agents with text output. It does not convert structured or
deferred tool outputs into chat messages.

## Delivery and failure behavior

The host does not retry `send_text()`. A timeout can occur after a provider
accepted a message, so retrying at this layer can duplicate replies. Each
adapter owns retry decisions it can make from provider-specific responses.

When an agent run or conversation store operation fails, the host logs the
exception and sends the static `error_reply`. Exception text is not sent to the
chat. If sending a reply fails, the host logs the failure and continues serving.
History remains saved after a successful run even when delivery cannot be
confirmed.

For message-bearing requests, both webhook adapters authenticate and enqueue
before returning HTTP 200. They return HTTP 503 when not open or when their
bounded queue is full, so the provider can retry. Each queue and its 10,000-id
duplicate window are process-local. A restart can reprocess a redelivery or lose
an acknowledged message that was still queued.

`SlackChannel` retries one `chat.postMessage` call after an HTTP 429 with a
valid non-negative `Retry-After` of at most 60 seconds. Longer delays fail the
delivery instead of holding that conversation's turn slot. It does not retry
timeouts or other ambiguous failures because Slack may already have accepted
the message. If a later text chunk fails, earlier chunks remain visible.

WhatsApp Cloud API limits outbound free-form text to the 24-hour customer
service window. It surfaces error 131047 when that window is closed; it does not
substitute an approved template. It retries one response carrying an explicit
Cloud API throttling code after `retry_delay`, but does not retry network
timeouts. If a later text chunk fails, earlier chunks remain visible. Meta may
batch messages and retry failed webhook deliveries for up to seven days.

`serve()` runs until it is cancelled or the adapter ends. It owns every turn in
an AnyIO task group, so no turn task outlives it. Cancellation closes the
message iterator, cancels in-flight turns, and then closes the adapter.

## Security

Anyone allowed to message the agent can supply model input to an agent that may
have access to tools, credentials, files, and network services. Use a narrow
sender allowlist and apply normal Pydantic AI guardrails to the connected agent.

Slack signatures are checked over the exact request bytes with the app signing
secret. Requests more than five minutes from the local clock are rejected.
Replies disable link and media unfurling so Slack does not fetch URLs generated
by the agent.
Slack direct messages are enabled by default. App mentions are accepted only
from `allowed_channel_ids`. The adapter also drops events from other workspaces,
bot-authored messages, edits, messages with a subtype (including file shares),
and unaddressed channel messages.

WhatsApp POST signatures are checked over the exact request bytes with the Meta
app secret. The GET handshake uses the separate verification token. The adapter
accepts text for its configured phone number and drops delivery statuses,
non-text messages, and updates for other numbers.

## Adapters and stores

`ChannelAdapter` has three responsibilities: manage its connection, yield
normalized `InboundMessage` values, and send text once. Provider-specific
polling, webhooks, formatting, limits, and retry classification stay inside the
adapter.

The async-iterator interface supports both polling and webhooks. A webhook
adapter can authenticate a request, put an accepted message onto a bounded
queue, and yield that queue from `messages()` without changing `ChannelHost`.

`ConversationStore` exposes `load`, replace-style `save`, and `delete`. Store
failures fail the current turn. A store can add its own retry or fallback policy
without adding storage policy to the host. A shared durable store does not make
multiple live hosts safe without external per-conversation serialization.

## Not included

The Slack adapter does not include Socket Mode, multi-workspace OAuth routing,
files, reactions, message edits, or messages in channels that do not mention
the bot. Socket Mode requires a WebSocket client and can be added separately.

The WhatsApp adapter uses the official Cloud API. It does not include media,
message templates, delivery-status callbacks, embedded signup, or the unofficial
browser automation used by `whatsapp-web.js`.

The host does not define media, reactions, typing indicators, streaming edits,
tool approvals, or provider authentication. Adapters add only the provider
behavior they document.

## API reference

- [`ChannelHost`][pydantic_ai_harness.channels.ChannelHost]
- [`ChannelAdapter`][pydantic_ai_harness.channels.ChannelAdapter]
- [`ChannelError`][pydantic_ai_harness.channels.ChannelError]
- [`InboundMessage`][pydantic_ai_harness.channels.InboundMessage]
- [`ConversationStore`][pydantic_ai_harness.channels.ConversationStore]
- [`InMemoryConversationStore`][pydantic_ai_harness.channels.InMemoryConversationStore]
- [`WebhookRequest`][pydantic_ai_harness.channels.WebhookRequest]
- [`WebhookResponse`][pydantic_ai_harness.channels.WebhookResponse]
- [`SlackChannel`][pydantic_ai_harness.channels.slack.SlackChannel]
- [`WhatsAppChannel`][pydantic_ai_harness.channels.whatsapp.WhatsAppChannel]

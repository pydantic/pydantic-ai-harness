# Channels

Channels let a Pydantic AI agent answer Slack mentions and continue each Slack thread with its own message history.

## Install

Channels uses the base Harness package. This example uses an OpenAI model and FastAPI:

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[openai]" fastapi uvicorn
```

## Set up Slack

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps).
2. Under **OAuth & Permissions**, add the bot scopes `app_mentions:read` and `chat:write`, then install the app to the workspace. Copy the bot token that starts with `xoxb-`.
3. Under **Basic Information**, copy the signing secret.
4. Start the example below on a public HTTPS URL, enable **Event Subscriptions**, set the request URL to `https://your-host/slack/events`, and subscribe to the `app_mention` bot event.
5. Copy the workspace ID that starts with `T` from the Slack browser URL `app.slack.com/client/T.../`.

Set the four required environment variables:

```bash
export OPENAI_API_KEY='your-openai-api-key'
export SLACK_SIGNING_SECRET='your-slack-signing-secret'
export SLACK_BOT_TOKEN='xoxb-your-bot-token'
export SLACK_TEAM_ID='T-your-workspace-id'
```

Slack's installation screen handles OAuth. The package does not open a browser or implement an OAuth redirect. Browser interaction is required once to create, configure, and install the Slack app.

## Run

Save this as `app.py`:

```python {names="defined"}
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic_ai import Agent

from pydantic_ai_harness.channels import ChannelEvent, ChannelHost
from pydantic_ai_harness.channels.slack import (
    SlackChannel,
    SlackError,
    SlackSignatureError,
    SlackUrlVerification,
)

agent = Agent('openai:gpt-5.6-sol', output_type=str)
slack = SlackChannel(
    signing_secret=os.environ['SLACK_SIGNING_SECRET'],
    bot_token=os.environ['SLACK_BOT_TOKEN'],
    team_id=os.environ['SLACK_TEAM_ID'],
)
host = ChannelHost(agent, slack)
worker_streams = [anyio.create_memory_object_stream[ChannelEvent](5) for _ in range(20)]


def worker_for(event: ChannelEvent) -> int:
    return hash(event.conversation_id) % len(worker_streams)


async def consume_events(worker: int) -> None:
    async with worker_streams[worker][1] as receive_events:
        async for event in receive_events:
            try:
                await host.handle(event)
            except Exception:
                logging.exception('Slack event failed')


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    async with anyio.create_task_group() as group:
        for worker in range(len(worker_streams)):
            group.start_soon(consume_events, worker)
        yield
        group.cancel_scope.cancel()


app = FastAPI(lifespan=lifespan)


@app.post('/slack/events')
async def receive_slack_event(request: Request) -> Response:
    raw_body = await request.body()
    try:
        parsed = slack.parse_request(raw_body, request.headers)
    except SlackSignatureError:
        return Response(status_code=401)
    except SlackError:
        return Response(status_code=400)

    if isinstance(parsed, SlackUrlVerification):
        return PlainTextResponse(parsed.challenge)
    if parsed is not None:
        send_events = worker_streams[worker_for(parsed)][0]
        with anyio.move_on_after(2) as enqueue_scope:
            await send_events.send(parsed)
        if enqueue_scope.cancel_called:
            return Response(status_code=503)
    return Response(status_code=200)
```

Run it:

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Expose port 8000 through your HTTPS host, invite the bot to a channel, and mention it. The endpoint verifies Slack's signature before enqueueing the event, returns the URL-verification challenge when Slack configures the endpoint, and replies in the original thread.

## Things to ask

- `@bot Summarize this update in three bullets.`
- `@bot Rewrite this announcement for customers.`
- `@bot Turn these notes into a checklist.`
- Follow up in the same thread with `@bot Make it shorter` or `@bot Explain the second point`.

## Operational notes

- Pass the untouched request bytes to `parse_request()`. Reading and re-encoding JSON first breaks signature verification.
- Limit request-body size at the HTTPS server or proxy before FastAPI reads it. `parse_request()` verifies supplied bytes but does not own network admission.
- Slack expects a response within three seconds and retries failed delivery. The example reserves two seconds for bounded, sharded in-process queues and returns 503 when one is full. Use a durable queue for production.
- Claim `event_id` before calling `handle()` if duplicate agent turns are unacceptable. The host does not claim, acknowledge, or retry events.
- The default history store is process-local and loses history on restart. Pass a `ConversationStore` implementation for persistent history.
- Events are serialized per Slack workspace, channel, and thread inside one `ChannelHost`. Coordinate workers outside the package when several processes share a store.
- Signature verification authenticates Slack, not the sender. Check `sender_id` and `delivery_id` before enqueueing if only selected users or channels may invoke the agent.
- The host sends the reply before saving history. A save failure can occur after Slack receives the reply, so do not blindly retry an ambiguous whole handler call.
- Slack may truncate replies over 40,000 characters. The adapter raises `SlackAPIError` when Slack reports that warning instead of treating a partial reply as success.
- On the first HTTP 429, this adapter instance pauses its replies in the current process, waits for `Retry-After`, and retries the same generated reply once. A second 429 waits again before it propagates; timeouts and 5xx responses are not retried because their delivery outcome can be ambiguous.
- The caller owns FastAPI, the queue, OAuth, and any injected `httpx.AsyncClient`.
- While Harness is on 0.x releases, minor releases may change this API. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/channels/)

---
title: Channels
description: Connect text-output Pydantic AI agents to messaging services through ordinary agent runs.
---

# Channels

Put a text-output Pydantic AI agent where your users already send messages.
Channels are useful for:

- a personal assistant in a private chat
- a support agent in team messages
- an internal agent that can use the same tools and capabilities as your app

The agent-side integration is the same for every provider:

```python
from pydantic_ai import Agent

from pydantic_ai_harness.channels import ChannelAdapter, ChannelHost


async def serve(channel: ChannelAdapter) -> None:
    agent = Agent('anthropic:claude-fable-5', instructions='Be concise and helpful.')
    host = ChannelHost(agent, channel, allowed_senders={'provider-user-id'})
    await host.serve()
```

The provider adapter supplies real sender ids and delivers each reply. Replace
`provider-user-id` with an id from that provider.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/channels/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

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
- At most 100 accepted turns may be running or waiting by default. Set
  `max_pending_turns` to tune this backpressure limit.
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

`serve()` runs until it is cancelled or the adapter ends. It owns every turn in
an AnyIO task group, so no turn task outlives it. Cancellation closes the
message iterator, cancels in-flight turns, and then closes the adapter.

## Security

Anyone allowed to message the agent can supply model input to an agent that may
have access to tools, credentials, files, and network services. Use a narrow
sender allowlist and apply normal Pydantic AI guardrails to the connected agent.

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

The host does not define media, reactions, typing indicators, streaming edits,
tool approvals, or provider authentication. Adapters add only the provider
behavior they document.

## API reference

::: pydantic_ai_harness.channels.ChannelHost

::: pydantic_ai_harness.channels.ChannelAdapter

::: pydantic_ai_harness.channels.ChannelError

::: pydantic_ai_harness.channels.InboundMessage

::: pydantic_ai_harness.channels.ConversationStore

::: pydantic_ai_harness.channels.InMemoryConversationStore

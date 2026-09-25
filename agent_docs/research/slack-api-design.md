# Slack interface design research

Status: reference proposal. No runtime implementation is included. Researched 2026-09-06.

## Problem and constraints

An existing agent should answer people on Slack with its existing model,
instructions, tools, and capabilities. Slack tools must be optional. Receiving
a message, supplying conversation context, and returning an answer must work
without model-callable Slack tools.

Ordinary `agent.run()` remains a finite call returning its result to its caller.
No new agent lifecycle, background listener during construction or ordinary
runs, or required connection context for unrelated runs. The application can
own its connection and deployment explicitly. Configure the agent-to-Slack
relationship once. Do not require Bolt handlers, manual reply routing, or a
second registration step in the common case.

## Proposed common-case interface

Proposed `slack_bot.py`, using an existing agent and explicitly supplied credentials:

```python
from my_agent import agent
from pydantic_ai_harness.slack import SlackApp

app = SlackApp(
    agent,
    bot_token=bot_token,
    signing_secret=signing_secret,
    tools=False,
)
```

The returned object is directly serveable as a standard Python web application:

```sh
uvicorn slack_bot:app
```

Changing `tools=False` to `tools=True` enables the documented bot Web API tools
for Slack-origin runs. Receiving, conversation context, reply delivery, and
deployment remain identical. It does not expose every Slack API method or
grant permissions the bot lacks. Default to `False`; more granular selection
uses the same argument, for example `tools=['read_thread', 'read_channel']`.
The proposed initial catalog is `read_thread`, `read_channel`, `list_channels`,
and `post_message`. `True` selects that documented catalog. Do not silently
expand it with unrelated administrative actions. User-authorized search or
personal-account actions are not implied by this bot tool catalog.
`False` means this integration adds no Slack tools. It does not remove tools
or capabilities the agent's author already configured.

These map to the existing Slack Web API, with pagination and permissions
handled by the implementation:
[thread messages](https://docs.slack.dev/reference/methods/conversations.replies/),
[channel messages](https://docs.slack.dev/reference/methods/conversations.history/),
[channel listing](https://docs.slack.dev/reference/methods/conversations.list/),
[posting](https://docs.slack.dev/reference/methods/chat.postMessage/).

No separate `register`, `asgi`, capability attachment, or manual event handler
is required. This is an application interface composed around ordinary runs,
not a second agent execution loop. The agent is passed once. The caller starts
the server, just as for the existing web interface.

This constructor targets one installed workspace. The extension for multiple
installations is an explicit installation-store/authorization integration,
not a second token field on the same instance. That extension is not required
to chat with the bot in its installed workspace.

The basic public-channel mention and DM path needs `app_mentions:read`,
`chat:write`, `channels:history`, and `im:history`, plus the corresponding
message event subscriptions. Private channels require their applicable
additional scopes. `tools=False` does not mean the bot needs no read permission:
the integration still reads conversation context. Extra tools require their
own applicable scopes, such as channel-read permission for listing channels.

Source: [Slack app setup](https://docs.slack.dev/tools/bolt-python/creating-an-app/)
and the method references above. A tested manifest must accompany the feature.

An `agent.to_slack(...)` convenience would resemble `to_web`, but it would be
a new core method with an optional Harness dependency to design. It is not
needed for this proposal and should not be claimed as existing.

### Ordinary runs

```python
result = await agent.run('Summarize this report')
```

This returns to the Python caller. It does not post the result to Slack or
start receiving Slack events. Tools added by the Slack application are local
to its runs and do not mutate the original agent. Capabilities already on the
agent remain available as usual.

A user who wants Slack tools in programmatic runs can configure a standalone
Slack tools capability. That is a separate use case, not a second required
step for putting the agent on Slack. A programmatic post needs an explicit
destination; there is no implicit last-used channel.

| Operation | Reply destination | Added Slack tools | Receiver starts? |
| --- | --- | --- | --- |
| Ordinary `agent.run(...)` | Python caller | None from `SlackApp` | No |
| Slack message, `tools=False` | Originating conversation | None | Server already owns receipt |
| Slack message, `tools=True` | Originating conversation | Configured bot tools | Same hosting as above |
| Ordinary run with an explicit Slack tools capability | Python caller; tools may explicitly post | That capability's tools | No |

### Connection alternative

Slack Socket Mode avoids a public webhook and instead requires a running
process with an app-level token and bot token. If we offer it, it must use the
same message handling implementation and tools policy. It should have a
separately explicit construction path rather than guessing transport from
which token happens to be present. It does not change `agent.run()` semantics.

Source: [Slack Socket Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/).

Do not make the research recommendation depend on inventing a generic host or
CLI. The primary proposal uses an existing web server. A Socket Mode convenience
can use an explicit factory, with one binding and the same dispatch policy:

```python
async with SlackApp.socket_mode(
    agent, bot_token=bot_token, app_token=app_token, tools=False
) as connection:
    await connection.wait()
```

This proposed factory returns a connection context, not a web application.
The process runs an ordinary async entry point rather than Uvicorn. No prior
Slack capability attachment is required. Both transports should share message
handling rather than become separate agent implementations.

## What other frameworks do

### Mastra

The linked `channels.adapters.slack` example declares an adapter on the agent.
The Mastra application supplies the receiving server and generated route
`/api/agents/<id>/channels/slack/webhook`. Installation still requires a Slack
app, credentials, event subscriptions, and a reachable server. The first
non-DM mention fetches recent platform messages; later conversation context
uses memory. Serverless deployment needs explicit support for work after the
HTTP acknowledgement. This is a compact interface backed by an application
runtime, not an agent constructor opening a connection.

Sources: [Slack setup](https://mastra.ai/integrations/channels/slack),
[channels, history, and deployment](https://mastra.ai/docs/channels).

Current source binds `AgentChannels` to an agent and initializes it through
Mastra. Incoming messages use `sendMessage`: they can wake an idle run or be
delivered to an active run. Consequently, describing this as exactly one new
run for every event would be inaccurate. Replies use an output processor.
`getTools()` exposes optional reaction tools and explicitly documents that
they are not automatically injected. It is not a general Slack search/read
toolkit. Initialization accepts custom channel state; otherwise it requires
Mastra memory storage. The guide's suggestion to configure storage understates
that default-path requirement.

Source: [AgentChannels implementation](https://github.com/mastra-ai/mastra/blob/main/packages/core/src/channels/agent-channels.ts).

Assessment: the declaration is good within Mastra's existing server. The
important property is one binding with automatic context and reply delivery.
Copying the constructor option into Pydantic AI without its application owner
would omit the part that makes it work. We should not copy Mastra's active-run
message delivery into Harness as a new execution mechanism.

### Agno

AgentOS attaches `Slack(agent=agent)` as an interface. AgentOS creates and serves
the web application. The Slack interface routes users and threads to agents;
the application owns delivery and the agent remains independently runnable.

Source: [Agno Slack interface and complete startup example](https://docs.agno.com/agent-os/interfaces/slack/introduction).

### Vercel Chat SDK and LangGraph

Chat SDK supplies platform adapters, message handlers, and conversation state;
an AI agent is invoked by a handler. LangGraph's first-party messaging example
adds Slack routes to its deployed server and invokes graph runs through its
SDK. These are useful implementation precedents, but exposing their handler
wiring would not satisfy our common-case goal.

Sources: [Chat SDK guide](https://vercel.com/kb/guide/the-complete-guide-to-chat-sdk),
[LangGraph messaging integration](https://github.com/langchain-ai/langgraph-messaging-integrations).

## Existing Pydantic AI precedent

Pydantic AI already separates an agent from its interfaces. `agent.to_web()`
returns a web application; a server starts it. The same agent remains usable
through `run()`, the CLI, and other interfaces. Slack can follow this principle
without changing the agent graph or adding a lifecycle hook.

Sources: [interfaces](https://pydantic.dev/docs/ai/overview/interfaces/),
[web application example](https://pydantic.dev/docs/ai/guides/web/).

Installed core in this checkout also exposes per-run capabilities, instructions,
dependencies, and message history. `for_agent` can return a bound capability
copy. It is not a connection-start signal. Agent context entry manages tool
and model resources. A previous fake-listener probe established technical
possibility through toolset context entry; it did not establish a suitable
public contract for an inbound messaging interface.

Sources: [agent API](https://pydantic.dev/docs/ai/api/pydantic-ai/agent/),
[capability API](https://pydantic.dev/docs/ai/api/pydantic-ai/capabilities/),
[repository ownership rule](../core-boundary.md).

## Alternatives evaluated

| Interface | Useful property | Main cost | Decision |
| --- | --- | --- | --- |
| Start listening in `before_run` | Fits a capability hook | Requires a run before any incoming message can start one | Reject |
| Start listening in `async with agent` | Reuses an existing context | Changes resource-context behavior; ordinary runs provide no receiving intent | Reject as the public interface |
| `slack.session(agent)` after attaching `slack` | Explicit connection ownership | Configures the same relationship twice | Reject |
| Context-managed `Slack` capability bound through `for_agent` | One object, no repeated agent argument | Hosting ownership must survive capability merging, copies, wrappers, and reuse across agents | Feasible candidate, not preferred |
| Explicit connection plus `Slack(connection=...)` | Clear resource ownership | Adds a connection object and capability binding rules to the common case | Advanced alternative |
| One application adapter taking the existing agent | One binding; ordinary runs unchanged; uses existing server lifetime | Hosting is expressed outside `capabilities` | Recommended |

The recommendation intentionally chooses a clear interface over insisting
that all application entry points must be capabilities. The underlying Slack
tools can still be implemented as a capability. A conversation interface also
has to receive requests when no run exists and deliver answers independently
of model tool selection.

## Required behavior before implementation is considered complete

- Replies work with zero Slack tools. Conversation context is bounded and
  supplied by the integration, rather than relying on the model to fetch it.
- Bot identity handles the basic conversation and bot tools. Personal user
  authorization is an optional independent extension. Native Slack MCP uses
  user authorization; it must not be treated as interchangeable with bot APIs.
- Existing dependencies are retained through a typed per-message dependency
  factory. Slack identity is available to that factory and is not substituted
  for the user's dependency type.
- History is keyed by installation and conversation, with an explicit shared
  thread policy. Credential and dependency values are not saved in history.
  A conversation ID alone does not persist model history. The storage policy,
  retention, restart behavior, and concurrency contract must be documented.
  Prefer visible Slack conversation context as the baseline. Do not assume
  sharing hidden tool-result history between senders is safe just because the
  Slack token belongs to a bot: other existing agent tools may use per-user
  dependencies. Preserving private model history is a separate memory policy.
- Structured-output agents require a typed rendering function. Do not change
  their output type to string to make Slack hosting work.
- Existing tools and guards remain active. Deferred approvals need an actual
  Slack approval interaction or an explicit unsupported-case error; do not
  imply that every capability works unchanged merely because normal runs are
  used.
- Signature verification precedes acceptance. HTTP acknowledgement precedes
  slow model work. Delivery deduplication and bounded per-conversation work are
  defined. A successful acknowledgement is not a completed reply.
- Startup errors are visible. Shutdown stops intake, drains within a stated
  bound, then cancels and joins remaining tasks before closing clients.
- The initial web deployment uses a persistent process. Multi-worker or
  serverless support requires suitable shared state and background execution;
  do not infer it from being ASGI-compatible.

Sources for identity choices: [Slack tokens](https://docs.slack.dev/authentication/tokens/),
[Slack MCP authorization](https://docs.slack.dev/ai/slack-mcp-server/).

## Scope

This PR records the proposed interface and its tradeoffs for review. It adds no
runtime code, dependency, public export, or released feature documentation.
The examples illustrate the proposed contract and are not runnable against
the current package. Live Slack behavior and implementation compatibility
remain to be validated in an implementation PR.

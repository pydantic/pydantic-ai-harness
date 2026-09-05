# Slack integration architecture: Slack, Composio, and Pipedream

Date: 2026-09-05

## Decision

This section is an architectural recommendation derived from the verified facts in the sections below.

For the Slack capability in Pydantic AI Harness, use Slack's own platform at both Slack-specific seams:

1. Use Bolt for Python as the adapter framework for inbound events, OAuth installations, request verification, agent
   surfaces, streaming, status, stop events, Block Kit interactions, and thread routing. Harness still owns the
   application policy for deduplication, concurrency, and which events should invoke the agent.
2. Use Slack's hosted MCP server for agent-initiated Slack search, reads, and actions when the app and workspace are
   eligible to use it.
3. Use direct Slack Web API calls only for delivery mechanics and gaps that Slack MCP does not cover. Do not recreate a general Slack tool catalog over `WebClient`.

Composio and Pipedream should not be dependencies of the Slack capability and should not host the Slack-facing conversation loop. They are integration catalogs for the other applications an agent may use while it happens to be hosted in Slack.

If Pydantic wants a first integration-catalog capability, Composio is the stronger candidate for an agent-native
integration. This is a recommendation, not a measured benchmark. Composio's documented sessions explicitly scope a
user, connected accounts, authentication, and tools; they can expose an exact fixed tool list and filter by behavior
tags. Pipedream should initially be supported through Pydantic AI's standard MCP client and a focused recipe. A
dedicated Pipedream capability would add little until it hides a concrete lifecycle or safety problem that generic MCP
cannot.

This is a split architecture, not a choice of one vendor for everything:

```text
Slack user
    |
    v
Slack Bolt adapter -----------------------> Slack UI and delivery
    |
    v
Pydantic AI agent
    |                     |
    |                     +---------------> Composio or Pipedream
    |                                       Gmail, GitHub, Linear, etc.
    v
Slack MCP server
Slack search, history, users, files, actions
```

## Proposed product acceptance test

The conversation supplied for this review is the useful acceptance test:

> A user asks the agent in a channel, "How many messages did Aditya send in this channel?"

The proposed successful flow is:

1. A channel mention starts the conversation.
2. The Slack adapter binds a typed context containing at least workspace, channel, thread, message, and invoking-user identifiers.
3. The Slack capability creates a fresh Slack MCP toolset for this run using the invoking user's token.
4. The agent resolves "Aditya" to a Slack user, scopes the search to the current channel, retrieves all relevant pages,
   counts the messages, and answers with source links or enough context to verify the result. Complete pagination is a
   Harness behavior requirement, not a guarantee supplied by the model or MCP transport.
5. Follow-up messages in the same channel thread invoke the agent without another mention.
6. Direct messages and the Slack agent Messages surface never require a mention.
7. Unrelated top-level channel traffic is ignored.

Slack's own Pydantic AI starter demonstrates the important routing behavior: DMs are accepted, a top-level channel
message is handled by `app_mention`, and an unmentioned channel-thread reply is handled only when stored history proves
that the bot is already participating in the thread. It also passes `context.user_token` into a newly constructed
`MCPToolset` for each agent invocation. [Slack message listener](https://github.com/slack-samples/bolt-python-starter-agent/blob/main/pydantic-ai/listeners/events/message.py)
[Slack Pydantic AI runner](https://github.com/slack-samples/bolt-python-starter-agent/blob/main/pydantic-ai/agent/agent.py)

## The seams are different

### Slack application seam

This seam receives Slack events and presents agent activity back to Slack users. Bolt is the provider-owned implementation for it.

Slack's current agent guidance covers the response loop from input through reasoning, tools, and streaming. It defines
the agent Messages experience, `message.im`, `app_context_changed`, loading status, stop events, session titles,
streaming, and interactive controls. Its Python examples use Bolt, and the guide points Python users to Bolt's streaming
utility. [Developing an agent](https://docs.slack.dev/ai/developing-agents/)

As of this research date, `agent_view` is Slack's default forward path and `assistant_view` is the legacy experience.
Slack says `assistant_view` will be deprecated in February 2027, so the capability should not design its public API
around legacy `assistant.threads.*` names. [Slack agent-view migration announcement](https://docs.slack.dev/changelog/2026/08/20/agent-updates/)

Bolt provides foundational Slack machinery that should not be reproduced in Harness: request verification,
authorization middleware, OAuth routes and state verification, installation storage integration, event listeners,
Socket Mode, and framework adapters. Multi-workspace apps can give Bolt an installation store and OAuth settings; Bolt
sets up the install and redirect routes and looks up the appropriate installation while processing a request. Event
retry deduplication and application-level idempotency remain Harness responsibilities.
[Bolt OAuth](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/)

Composio and Pipedream both document trigger products. That is useful for background automation. The conclusion that
they should not replace Bolt for this capability is an architectural recommendation based on the Slack-specific
routing, context, streaming, status, stop, Block Kit, and installation behavior that the Slack sources define.

Pipedream has an additional Slack-specific limitation: custom OAuth clients are supported for Slack actions and code steps, but not for Slack triggers. That makes it unsuitable as the required ingress for a branded Slack app. [Pipedream OAuth client limitations](https://pipedream.com/docs/apps/oauth-clients)

### Agent tools seam

This seam lets the model search and act in Slack. Slack now supplies a hosted MCP server at `https://mcp.slack.com/mcp` using Streamable HTTP. Its documented tools cover message and file search, channel and thread reads, user and channel search, message sending and drafting, channel creation, reactions, canvases, files, and lists. [Slack MCP overview](https://docs.slack.dev/ai/slack-mcp-server/)

The following are reasons to prefer it to hand-maintaining general Slack tools over the Python SDK:

- Slack owns the MCP tool schemas and can evolve them with the product. This is also a compatibility risk for any
  exhaustive enum maintained in Harness.
- Slack defines the OAuth scopes required by each tool family.
- Actions run as the authenticated Slack user, preserving Slack's access model.
- Slack administrators can approve and audit the MCP client app.
- Pydantic AI documents an MCP client and toolset composition, including filtering and approval wrappers.

The direct Python Slack SDK remains important, but mostly inside the Bolt adapter. It covers the Web API and supplies
the primitives for posting the agent's response, streaming, status, reactions, and UI. Its Python interface is only
partially static: Web API methods accept `**kwargs`, return a mapping-like `SlackResponse`, and Bolt listener payloads
are commonly `dict`. Harness therefore still needs validation and nominal types at its own seam rather than exposing
raw SDK payloads as its public interface.
[Python Slack SDK WebClient source](https://github.com/slackapi/python-slack-sdk/blob/main/slack_sdk/web/client.py)
[Python Slack SDK `SlackResponse` source](https://github.com/slackapi/python-slack-sdk/blob/main/slack_sdk/web/slack_response.py)

## Slack MCP constraints that affect the design

Slack MCP is not a universal zero-configuration backend. The following are verified Slack platform constraints:

- The MCP client must be backed by a registered Slack app with a fixed app ID, and Slack directs client applications to
  hardcode that ID.
- Only internal apps and apps published in the Slack Marketplace may currently use Slack MCP. Unlisted distributed apps are prohibited.
- It uses confidential OAuth with the app's client ID and secret and issues user tokens. A bot token is still needed for the in-Slack experience.
- Search and history access require explicit user-token scopes. Public search, private search, DMs, files, users, and conversation history are separate permissions.
- The MCP server has the same method-level rate limits as the corresponding Slack APIs.

These constraints are documented by Slack. [Slack MCP identity and authentication](https://docs.slack.dev/ai/slack-mcp-server/#app-identity)

Recommended consequence: the first release should have one agent-tool path, Slack MCP, rather than a second general
read catalog over the Web API. Bolt and its `WebClient` still host and render the conversation when Slack MCP is not
configured, but workspace search and other agent-facing Slack tools should then be reported as unavailable. Add a
direct-Web-API agent tool only for a concrete unsupported gap that Harness is prepared to own, type, test, scope, and
maintain.

## Identity is the most important correctness constraint

A Slack-hosted agent has at least two relevant identities:

- The bot installation identity receives events and delivers the agent response.
- The invoking user's identity determines which Slack information the agent may search and which actions it may take on the user's behalf.

Those credentials are not interchangeable. A public constructor should not accept an ambiguous `token: str` and decide at runtime what it means.

Pydantic AI's MCP documentation identifies a concurrency hazard directly: one shared `MCPToolset` maintains one MCP session and therefore one identity. If overlapping runs derive authentication from task-local state on a shared instance, calls can use the identity of the run that opened the session first. The documented fix is a fresh MCP toolset per run, built from explicit run dependencies or passed to that run. [Pydantic AI per-user MCP authentication](https://pydantic.dev/docs/ai/mcp/client/#per-user-authentication)

Slack's own Pydantic AI sample follows that shape by constructing the Slack MCP toolset inside `run_agent` with the current Bolt context's user token. [Slack Pydantic AI sample](https://github.com/slack-samples/bolt-python-starter-agent/blob/main/pydantic-ai/agent/agent.py)

The following are implementation recommendations derived from that constraint:

- A Slack MCP toolset configured with per-user authentication is created once per agent run, never cached on the long-lived capability.
- Bot, app-level, signing-secret, and user credentials have distinct types and fields.
- There is no precedence ladder such as client, explicit token, bot token from another object, then environment variable.
- A missing user authorization produces a typed reconnect/authentication outcome that the Slack adapter can render as
  a sign-in button. Slack documents the OAuth endpoints, but this outcome and rendering contract are Harness design
  choices.
- The Slack invocation context is serializable so durable execution does not depend on a process-local `ContextVar`.

## Typed public interface

The SDK implementation may be dynamic, but the Harness interface should be narrow and typed.

A direction worth prototyping is shown below. All items in this subsection are recommendations unless explicitly tied
to a source.

```python
slack = Slack(
    tools=SlackTools.workspace_read(),
)

agent = Agent(model, capabilities=[slack])
SlackApp(agent).run()
```

Advanced tool selection should compose typed values rather than string names:

```python
slack = Slack(
    tools=(
        SlackTools.workspace_read()
        | SlackTools.of(SlackTool.SEND_MESSAGE, SlackTool.ADD_REACTION)
    ),
    approval='writes',
)
```

The exact names need a prototype, but the important properties are:

- `SlackTool` is a versioned enum for a curated, compatibility-tested subset of the Slack MCP catalog, not an attempted
  static mirror of every provider tool and not `list[str]`. An advanced escape hatch may be needed because Slack owns
  and can change the discovered catalog.
- Tool presets are immutable typed values. The default is the smallest useful read-only set needed for questions about the current channel and workspace.
- Enabling a write tool automatically requires approval unless the caller explicitly supplies a different approval policy.
- Requested tools missing from the discovered server catalog fail during startup or first connection. They are not dropped quietly.
- Unknown spec keys and invalid combinations fail validation.
- Runtime-only objects and secrets do not serialize into an agent spec.
- Slack IDs and timestamps use nominal or constrained types so a workspace ID cannot be passed where a channel ID belongs.
- The invocation exposes a typed `SlackContext` with workspace, enterprise, channel, thread, message, invoking user, and ordered app-context entities.
- The convenience path reads conventional environment variables, while advanced constructors accept explicit typed credentials or a caller-owned Bolt app. These should be separate constructors or discriminated configuration models rather than a collection of optional arguments.

Pydantic AI documents `FilteredToolset`, `ApprovalRequiredToolset`, deferred loading, prefixes, metadata, and per-run
dynamic toolsets. The recommendation is for the Slack capability to compose those primitives rather than implement
parallel filtering and approval semantics. [Pydantic AI toolset composition](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/#toolset-composition)

## Conversation behavior

The following is the proposed default behavior:

- Direct messages and the agent Messages surface invoke the agent without a mention.
- A top-level channel conversation starts with one `@agent` mention.
- Every later human message in that same thread invokes the agent without another mention.
- A top-level channel message without a mention is ignored.
- Bot messages, edits, deletions, and retry deliveries do not create duplicate runs.
- Turns in one thread are serialized or rejected with visible busy state so histories cannot interleave.

Slack's Bolt agent guide explicitly shows the channel-thread rule: only handle an unmentioned thread reply if an
existing stored session proves the agent is engaged. [Bolt agent features](https://docs.slack.dev/tools/bolt-python/concepts/adding-agent-features/)

For the agent Messages surface, Slack supplies `app_context_changed` and also includes current `app_context` on message events when configured. This is how phrases such as "this channel" can resolve even when the user is speaking from the agent panel rather than inside the channel itself. [Slack context management](https://docs.slack.dev/ai/agent-context-management/)

## Approvals and trust

The final response posted by the Slack adapter is delivery, not an agent-selected side effect. It should not require approval. An agent-selected action such as sending a message to another channel, inviting a user, changing a canvas, or deleting content is different.

The following safe default is a Harness recommendation:

- Read tools enabled.
- Write tools absent unless selected.
- Selected write tools wrapped in Pydantic AI's approval toolset.
- Approval requests rendered by the Slack adapter as native Block Kit controls and resumable deferred tool calls.
- Destructive and administrative tools excluded from broad presets.

Slack's agent design guidance says that real-world actions should have explicit confirmation, should preview the proposed action, and should not prompt so frequently that users learn to approve without reading. Slack's governance guidance also recommends approval gates for creates, sends, and deletes and visible status/control surfaces. [Slack agent design](https://docs.slack.dev/concepts/agent-design/) [Slack governance](https://docs.slack.dev/ai/agent-governance/)

The approval policy therefore needs more precision than an `approvals: bool`. It should classify tools or tool calls and let callers replace the policy. Slack also now has approval message metadata that can surface approve/deny actions in Slack's Activity feed, which may be a useful adapter target after the basic Block Kit flow is correct. [Slack approval metadata](https://docs.slack.dev/messaging/message-metadata/#approval-schema)

## Composio assessment

The facts in this subsection describe Composio's published product behavior. The strengths, weaknesses, and final
recommendation are this review's assessment.

Composio is designed around an agent session. A session binds the application's user ID, the allowed toolkits and tools,
authentication configuration, selected connected accounts, and execution state. By default it supplies meta-tools for
discovery and connection management. It can instead expose a fixed direct tool set.
[Composio sessions](https://docs.composio.dev/docs/how-composio-works)
[Configuring Composio sessions](https://docs.composio.dev/docs/configuring-sessions)

Strengths:

- Agent-native per-user session and connected-account model.
- Managed OAuth, token storage, refresh, custom OAuth apps, and reconnect flows.
- Exact toolkit and tool inclusion/exclusion.
- Filtering by behavior tags including read-only, destructive, idempotent, and open-world hints.
- A direct-tools preset for a small stable tool surface, plus dynamic discovery for broad agents.
- Triggers for background events from connected apps.
- A large catalog and separate Slack and Slackbot token models. Composio documents Slack as acting as an individual
  user and Slackbot as acting as the bot. Its Slackbot catalog page also describes workspace reads on behalf of the
  installing user while writes remain bot-authored. [Composio Slack and Slackbot token models](https://docs.composio.dev/kb/guide/toolkits-slackbot)
- Hosted MCP endpoints for each configured session, so Pydantic AI can connect without a provider-specific tool format.

The fixed-tool and behavior-tag controls are particularly relevant to Pydantic AI's safety model. This is an inference
from the documented filters, not evidence that the annotations have been independently audited.
[Configuring Composio sessions](https://docs.composio.dev/docs/configuring-sessions)

Weaknesses and constraints:

- It adds a third-party control plane, credential store, execution hop, account model, logs, quotas, and billing relationship.
- The default session exposes every toolkit for discovery unless the developer restricts it. A Pydantic wrapper would need a narrower default.
- Composio's current official provider list does not include Pydantic AI. Pydantic AI can use a session's documented
  MCP URL and headers; a direct provider would require a custom Composio provider.
  [Composio providers](https://docs.composio.dev/docs/providers)
  [Composio sessions over MCP](https://docs.composio.dev/docs/sessions-via-mcp)
- Composio documents that SDK execution modifiers do not run through its hosted MCP route. Local logging, schema modification, and gating must therefore happen in Pydantic AI's toolset wrappers, or a native provider path must be developed. [Composio MCP trade-offs](https://docs.composio.dev/docs/sessions-via-mcp#trade-offs)
- Tool catalogs and identifiers are vendor-defined and versioned. A universal static enum in Pydantic AI would be too large and change too often.
- The platform becomes part of the data and credential path. Composio documents encryption and isolation, but standard plans do not provide an unconditional end-to-end zero-retention guarantee; current retention and contractual requirements must be evaluated for the deployment. [Composio data handling](https://docs.composio.dev/kb/guide/platform-compliance-data-handling)

Recommendation for Composio:

- Do not use its Slack or Slackbot toolkit as the foundation of `pydantic_ai_harness.slack`.
- Explore a separate `Composio` capability for cross-application tools.
- Prefer a partner-owned `composio-pydantic-ai` provider for direct execution if Composio wants its schema modifiers and lifecycle hooks to apply.
- Otherwise build a small Harness capability around a per-run Composio session MCP endpoint, exact toolkit restrictions, no sandbox by default, and Pydantic-owned approval wrappers.
- In a Slack-hosted agent, map the stable application user ID deliberately. Do not assume the Slack user ID alone is the application's global user identity.

## Pipedream assessment

The facts in this subsection describe Pipedream's published product behavior. The strengths, weaknesses, and final
recommendation are this review's assessment.

Pipedream Connect provides managed authentication, actions, triggers, a proxy, hosted MCP, and SDKs. Its developer MCP
endpoint uses developer credentials plus project, environment, external-user, and app headers. It advertises more than
10,000 tools across more than 3,000 APIs. [Pipedream Connect](https://pipedream.com/docs/connect)
[Pipedream developer MCP](https://pipedream.com/docs/connect/mcp/developers)

Strengths:

- Very broad API and action catalog.
- Managed end-user connections and OAuth clients.
- Hosted MCP with both Streamable HTTP and SSE support.
- Strong event-trigger and workflow story for background automation.
- Source-available action and trigger components.
- MCP tool annotations for read-only, destructive, and open-world behavior are documented for registry components;
  current Pipedream changelog material says those annotations are included in its hosted MCP catalog. The annotations
  are provider assertions and still need compatibility and accuracy checks before driving automatic approvals.
  [Pipedream component annotations](https://pipedream.com/docs/components/contributing/guidelines)
  [Pipedream MCP annotation release](https://pipedream.com/docs/changelog)
- Official Python, TypeScript, and Java server SDKs for the Connect API. The source does not support the stronger claim
  that the Python SDK makes the dynamic MCP tool catalog statically typed.
  [Pipedream SDKs](https://pipedream.com/docs/connect/api-reference/sdks)
- Connect states that API/MCP request and response bodies are not persisted, while credentials are encrypted at rest. [Pipedream security](https://pipedream.com/docs/privacy-and-security)

Weaknesses and constraints:

- Its primary abstraction is an integration and workflow platform, not a Slack agent interface.
- The developer MCP connection needs a Pipedream project and developer OAuth credentials in addition to an external-user mapping.
- Older Pipedream MCP tool modes included a stateful full-config mode that required clients to reload tools. Pipedream's
  page now says that, as of 2026-04-15, tool modes are no longer necessary on the current `/v3` endpoint and the full
  schema is returned in a tools-only shape. Compatibility with dynamic properties and Pydantic AI still needs a real
  integration test, but the previous claim that `/v3` lacks the current mode is false.
  [Pipedream MCP tool modes](https://pipedream.com/docs/connect/mcp/tool-modes)
- Slack custom OAuth clients do not work with Pipedream Slack triggers, even though they work for actions and code steps.
- Production Connect usage adds external-user and compute-credit billing. [Pipedream pricing](https://pipedream.com/docs/pricing)
- A Pipedream-specific capability risks being a shallow wrapper over `MCPToolset` plus headers unless it owns token refresh, exact tool selection, annotation mapping, and per-user lifecycle.

Recommendation for Pipedream:

- Do not use it as Slack ingress or as the Slack tool backend for this PR.
- Publish and test a Pydantic AI MCP recipe for Pipedream Connect.
- Consider a dedicated capability only after proving a deep interface around developer-token refresh, external-user binding, tool-mode compatibility, and safe annotation-to-approval mapping.
- Prefer it when the main product requirement is event-driven workflows or extremely broad app coverage, not when the requirement is the best Slack-native experience.

## Comparative result

| Criterion | Slack official stack | Composio | Pipedream |
| --- | --- | --- | --- |
| Slack-native conversation UX | Recommended. Bolt exposes the current Slack agent features | Generic triggers do not provide the same Slack-specific interface | Generic triggers do not provide the same Slack-specific interface; custom Slack OAuth cannot drive triggers |
| Slack tool fidelity | Provider-owned Slack MCP schemas; complete Web API primitives for deterministic app mechanics | Broad Slack and Slackbot catalogs, but vendor-normalized | Broad action catalog, but vendor-normalized |
| Invoking-user identity | Direct fit through Slack OAuth and Bolt context | Requires Slack user to application user mapping and connected account | Requires Slack user to external user mapping and connected account |
| Typed Pydantic interface | Harness must add typed seam over dynamic SDK/MCP | Session config is structured, but catalog is dynamic; no current Pydantic provider | Management SDK is typed, but MCP/action catalog is dynamic |
| Approvals | Pydantic toolset approvals plus Slack-native UI | Pydantic wrappers work over MCP; Composio SDK modifiers do not | Pydantic wrappers can use MCP annotations after verification |
| Cross-app breadth | Slack only | Broad, with agent-session-oriented controls | Broad, with workflow and event-oriented controls |
| Additional control plane | No integration intermediary | Yes | Yes |
| Best role in this product | Slack host and Slack tools | Optional other-app capability | Generic MCP/workflow option |

## Implications for PR 808

The current PR should be reshaped rather than incrementally expanded.

Keep or redesign:

- A thin `SlackApp` adapter built on caller-owned or convenience-configured Bolt.
- Typed Slack invocation context.
- Per-thread conversation/history port, with the adapter applying the one-mention rule.
- Slack rendering of Pydantic deferred approvals.
- Typed access policy and exact tool selection.

Replace:

- Custom general Slack tools such as model-driven `post_message`, `post_plan`, and upload behavior with Slack MCP tools or deterministic adapter rendering.
- A manually mirrored partial Slack client protocol as the public advanced interface.
- Ambiguous token/client precedence and mutation between the bot host and capability.
- Boolean configuration for approvals and interactions.
- A capability that caches a user-authenticated MCP client.

Defer:

- A universal integration catalog in the Slack package.
- Supporting both Composio and Pipedream behind one abstraction. They have materially different session, tool-discovery, trigger, and billing semantics, so there is not yet a real shared seam.
- Full administrative Slack tool coverage through a Harness-maintained direct-Web-API backend.

## Implemented PR boundary

The reshaped PR implements the first coherent product slice:

- `Slack()` adds a per-run, user-authenticated Slack MCP toolset with a curated typed read default.
- `SlackTools.of(...)` selects compatibility-tested enum values. `SlackTools.named(...)` is the explicit forward-
  compatibility seam for newly released provider tools; unclassified tools require approval under the default policy.
- `SlackApp` uses Bolt for Socket Mode or Events API ingress, request verification, OAuth installation lookup,
  channel-thread engagement, direct messages, retries, per-thread serialization, and deterministic final delivery.
- Fixed MCP tokens are documented only for single-user installations. Multi-user applications use a Bolt installation
  store, require each user to complete Slack's install flow, and surface that install URL when a per-event user token is
  absent.
- Direct Web API usage is limited to final-response delivery and approval Block Kit. The earlier general model-facing
  Web API toolset was removed.

Live partial-response streaming, loading status, and user-triggered stop handling are not part of this slice. They need
an agent-surface compatibility test across DMs, channel threads, `agent_view`, and Slack plan tiers before becoming
default behavior. The current adapter returns a final answer in the correct thread and does not claim those lifecycle
features in its user documentation.

## Recommended delivery sequence

1. Define the behavior contract and typed public interface in tests before preserving any current implementation.
2. Implement Slack routing through Bolt: DMs, agent Messages, initial channel mention, unmentioned replies in an engaged thread, retries, self-event filtering, stop events, and per-thread serialization.
3. Add typed `SlackContext` and serializable conversation identity.
4. Add per-run Slack MCP toolsets using the invoking user's OAuth token and a curated read-only default.
5. Add opt-in write tools with Pydantic deferred approvals rendered in Slack.
6. Add a narrow direct Web API agent tool only when a concrete Slack MCP gap blocks an important supported workflow.
   Do not introduce a general fallback catalog merely because Slack MCP is not configured.
7. Separately prototype a Composio capability for Gmail, GitHub, Linear, and other user-connected apps.
8. Publish a Pipedream MCP recipe and collect concrete gaps before adding a Pipedream-specific capability.

## Unresolved uncertainties

- Slack's overview documents tool families and scopes but does not state a compatibility or versioning contract for
  exact MCP tool names and schemas. Discover the live catalog in an integration test before freezing a curated enum.
- A Bolt listener receives `context.user_token` only when the installation and user authorization flow produced one.
  The adapter now exposes the install URL on this failure, but installation-store lookup and multi-workspace behavior
  still need an end-to-end OAuth test.
- Some current Slack agent features require a paid workspace or Developer Program sandbox. The capability needs a
  documented minimum Slack plan and SDK version matrix.
- Slack's message-metadata page currently labels the approval schema inconsistently: prose refers to
  `slack_approval`, while its JSON examples use `event_type: "approval"`. Treat Activity-feed approval metadata as a
  later integration target until the accepted wire value is verified against the API.
- The implementation uses a non-cached `DynamicToolset` with `per_run_step=False`, producing one MCP toolset and one
  Slack user identity per agent run. A live concurrent-user test is still needed against Slack's hosted endpoint.
- Pipedream says `/v3` no longer needs its older stateful tool modes, but dynamic-property behavior with Pydantic AI has
  not been exercised here.
- Composio and Pipedream publish behavior annotations. This research did not audit the accuracy of those annotations
  tool by tool, so they should not become the sole approval policy without runtime safeguards.

## Source list

- [Slack MCP server overview](https://docs.slack.dev/ai/slack-mcp-server/)
- [Developing an app with Slack MCP](https://docs.slack.dev/ai/slack-mcp-server/developing/)
- [Slack agent development](https://docs.slack.dev/ai/developing-agents/)
- [Slack agent context management](https://docs.slack.dev/ai/agent-context-management/)
- [Slack governance and trust](https://docs.slack.dev/ai/agent-governance/)
- [Slack agent design](https://docs.slack.dev/concepts/agent-design/)
- [Slack agent sessions](https://docs.slack.dev/ai/agent-sessions/)
- [Slack approval message metadata](https://docs.slack.dev/messaging/message-metadata/#approval-schema)
- [Slack agent-view migration announcement](https://docs.slack.dev/changelog/2026/08/20/agent-updates/)
- [Bolt for Python agent features](https://docs.slack.dev/tools/bolt-python/concepts/adding-agent-features/)
- [Bolt for Python OAuth](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/)
- [Slack's Pydantic AI starter agent](https://github.com/slack-samples/bolt-python-starter-agent)
- [Python Slack SDK WebClient](https://github.com/slackapi/python-slack-sdk/blob/main/slack_sdk/web/client.py)
- [Python Slack SDK SlackResponse](https://github.com/slackapi/python-slack-sdk/blob/main/slack_sdk/web/slack_response.py)
- [Pydantic AI MCP client](https://pydantic.dev/docs/ai/mcp/client/)
- [Pydantic AI toolset composition](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
- [Composio sessions](https://docs.composio.dev/docs/how-composio-works)
- [Composio session configuration](https://docs.composio.dev/docs/configuring-sessions)
- [Composio sessions through MCP](https://docs.composio.dev/docs/sessions-via-mcp)
- [Composio Slackbot toolkit](https://docs.composio.dev/toolkits/slackbot)
- [Composio Slack and Slackbot token models](https://docs.composio.dev/kb/guide/toolkits-slackbot)
- [Composio data handling](https://docs.composio.dev/kb/guide/platform-compliance-data-handling)
- [Composio triggers](https://docs.composio.dev/docs/triggers)
- [Pipedream Connect](https://pipedream.com/docs/connect)
- [Pipedream developer MCP](https://pipedream.com/docs/connect/mcp/developers)
- [Pipedream MCP tool modes](https://pipedream.com/docs/connect/mcp/tool-modes)
- [Pipedream OAuth client limitations](https://pipedream.com/docs/apps/oauth-clients)
- [Pipedream privacy and security](https://pipedream.com/docs/privacy-and-security)
- [Pipedream SDKs](https://pipedream.com/docs/connect/api-reference/sdks)
- [Pipedream component annotations](https://pipedream.com/docs/components/contributing/guidelines)
- [Pipedream pricing](https://pipedream.com/docs/pricing)

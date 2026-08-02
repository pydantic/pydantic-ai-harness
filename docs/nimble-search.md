---
title: Nimble Search
description: Give a Pydantic AI agent web research tools backed by the Nimble API -- search, page extract, opt-in site map and resumable crawl, plus a separate NimbleAgent capability for Web Search Agents.
---

# Nimble Search

`NimbleSearch` gives an agent web research tools backed by the
[Nimble](https://www.nimbleway.com/) API: web search with content or
descriptions, full-page markdown extract, and opt-in site mapping and
resumable crawl jobs. The separate `NimbleAgent` capability runs Nimble
[Web Search Agents](https://www.nimbleway.com/) (Agent API V2) with
resumable start/status/result tools.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/nimble/)

## The problem

Research agents need more than titles and snippets, but dumping full pages into
context for every search hit is wasteful. Wiring search, extract, and crawl
together - with attribution headers, result budgets, and prompts that keep long
jobs resumable - is boilerplate every integration reinvents.

`NimbleSearch` bundles that plumbing into a single
[capability](/ai/core-concepts/capabilities/): the research tools, per-tool
output budgets, and short research guidance in the system prompt. Hosted Web
Search Agents are a separate product surface, so they ship as `NimbleAgent`
(same split as `ExaSearch` / `ExaAgent`).

## Usage

Install the `nimble` extra and set the `NIMBLE_API_KEY` environment variable
(create a key at <https://online.nimbleway.com/account-settings/api-keys>):

```bash
uv add "pydantic-ai-harness[nimble]"
```

Then pass `NimbleSearch` to an `Agent` via the `capabilities` parameter:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.nimble import NimbleSearch

agent = Agent('openai:gpt-5.2', capabilities=[NimbleSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

Compose search with Web Search Agents when you need both. Pass the same
`client=` to both capabilities so they share one HTTP session (otherwise each
builds its own factory client):

```python
from nimble_python import AsyncNimble
from pydantic_ai import Agent
from pydantic_ai_harness.nimble import NimbleAgent, NimbleSearch

client = AsyncNimble()  # or AsyncNimble(api_key=...)
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[NimbleSearch(client=client), NimbleAgent(client=client)],
)
```

## Tools

`NimbleSearch` contributes these tools to the agent:

| Tool | Purpose |
|---|---|
| `web_search` | Search the web (default depth `lite`) and return titles, URLs, and content/descriptions. |
| `get_page` | Extract one URL as markdown. |
| `map_site` | Discover links on a website. Opt-in via `include_map=True`. |
| `crawl_start` / `crawl_status` | Start a crawl and poll progress across turns. Opt-in via `include_crawl=True`. |
| Agent API tools | Provided by the separate `NimbleAgent` capability (see below). |

`get_page` markdown is capped at `max_text_chars` characters for the **entire**
tool return (URL prefix + body + truncation marker), keeping the head. When
truncated, the output ends with
`[... page text truncated at N characters]`.

A rate limit or a transient API or network failure surfaces to the model as a
[`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries) rather than a
hard error. Empty search results return a normal tool message
(`No results found for '...'.`). Empty page extract retries via `ModelRetry`.
Authentication failures (401/403) are configuration errors and propagate.

Factory-built `AsyncNimble` clients are retained for the duration of each agent
run via `wrap_run` and closed when the last concurrent run ends, including on
failure or cancellation.

## Opt-in map and crawl

```python
from pydantic_ai_harness.nimble import NimbleSearch

NimbleSearch(
    include_map=True,
    include_crawl=True,
)
```

Crawl tools are **non-blocking**: start a job, then call `crawl_status` on later
turns. Do not expect a single tool call to wait until a multi-minute job finishes.

## Web Search Agents (`NimbleAgent`)

`NimbleAgent` exposes Nimble Web Search Agents (Agent API V2) as separate
model-callable tools. Runs are resumable across turns via start / status /
result - unlike Exa's deferred single-tool polling.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.nimble import NimbleAgent

agent = Agent('openai:gpt-5.2', capabilities=[NimbleAgent()])
```

| Tool | Purpose |
|---|---|
| `agents_list` / `agent_templates_list` | Discover agents and templates. |
| `agent_run_start` / `agent_run_status` / `agent_run_result` | Agent API V2 lifecycle. |

### Modes (pick from host architecture)

| Mode | When | How the model starts a run |
|---|---|---|
| **1** (default for Pydantic AI) | Stateless host - keep `wsa_...` / `task_run_...` in message history | `agent_run_start(..., agent_name='...')` |
| **2** | App already persists a `wsa_...` | `agent_run_start(..., agent_id='wsa_...')` |
| **3** | One-off anonymous create | `agent_run_start(...)` with neither id nor name |

Mode 1 create-or-reuse: the same `agent_name` reuses the account agent. A failed
first Mode 1 run does **not** brick the name - retry with the same name.

```python
# Mode 1 - create-or-reuse (recommended default)
# The model calls agent_run_start with agent_name + input (+ optional use_case/skill).

# Mode 2 - explicit agent id
# agent_run_start(input=..., agent_id='wsa_...')

# Mode 3 - anonymous one-shot (still returns web_search_agent_id)
# agent_run_start(input=...)
```

### `use_case` (locked at create)

| Value | Output |
|---|---|
| `research` | Free-form cited answer (`output.type: "text"`) |
| `enrichment` | Fill `input_data` against a schema (`output.type: "json"`) |
| `dataset_building` | Structured table from scratch (`output.type: "json"`) |

Set `use_case` once when the agent is created (Mode 1 first call, Mode 3, or a
prior create). Against an existing agent: omit it or pass the **same** value -
a different value returns **422**. It is never a silent per-run override.

### Effort tiers

`effort`: `low` | `medium` | `high` | `x-high` | `max`.

Deeper tiers often take several minutes. Never block-poll inside one tool call -
use `agent_run_status` / `agent_run_result` across turns.

### Overrides vs persist

On an **existing** agent, `sources`, `output_schema`, and `skill` are one-time
run overrides (they do not mutate the stored agent). On Mode 1 **first** call
(new `agent_name`), create-time fields are stored on the new agent.

`input_data` is enrichment payload only (run-level; never stored). It is
distinct from `output_schema` (data vs shape).

### `sources` shape

```python
sources = {
    'allow': [{'title': 'Official filings', 'domains': ['sec.gov'], 'order': 0}],
    'block': [{'title': 'Junk', 'domains': ['example.com'], 'order': 0}],
    'avoid': 'free-text domains or source types to avoid',
    'prioritize': 'free-text domains or source types to prefer',
}
```

### Trust / citations

`agent_run_result` returns the full JSON payload (text or JSON output plus
trust). Citation URLs from `output.trust.sources` are also copied into
`ToolReturn.metadata['sources']` for application-side rendering without parsing
model text.

### Events and follow-ups (intentional gaps)

The HTTP API supports `enable_events=True` plus
`GET /v2/agents/{id}/runs/{run_id}/events` (SSE), and a
`previous_interaction_id` field for follow-up threads. `agent_run_start` accepts
`enable_events` for forward compatibility, but this capability does **not**
expose an events subscription tool or `previous_interaction_id` - use
start/status/result across turns.

```python
from pydantic_ai_harness.nimble import NimbleAgent

NimbleAgent(
    guidance=None,  # None = default instructions; '' = none
    client=None,    # NimbleClient - None builds AsyncNimble from NIMBLE_API_KEY
)
```

### Structured citations

Search hits and completed agent results carry URLs in
`ToolReturn.metadata['sources']` (a list of `{url, title}` dicts). Use that
metadata in your application instead of scraping model text for links.

### Agent specs

Both capabilities support `from_spec` / AgentSpec loading. The `client` field is
not serializable, so spec-loaded instances always build from `NIMBLE_API_KEY`:

```yaml
model: openai:gpt-5.2
capabilities:
  - NimbleSearch:
      num_results: 5
      search_depth: lite
  - NimbleAgent: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.nimble import NimbleAgent, NimbleSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[NimbleSearch, NimbleAgent])
```

### Multiple instances

To expose more than one configured searcher on the same agent, wrap with
[`PrefixTools`](/ai/core-concepts/capabilities/#prefixtools) so tool names do not
collide (`web_search` also overlaps with core `WebSearch` if both are present).

## Configuration

```python
from pydantic_ai_harness.nimble import NimbleSearch

NimbleSearch(
    num_results=5,
    max_text_chars=10_000,
    search_depth='lite',
    time_range=None,
    include_domains=[],
    exclude_domains=[],
    include_map=False,
    include_crawl=False,
    guidance=None,
    client=None,
)
```

Factory-built clients send `X-Client-Source: pydantic-ai` for attribution.
Pass `client=` to reuse an existing `AsyncNimble` instance or a test double.

## API reference

::: pydantic_ai_harness.nimble.NimbleSearch

::: pydantic_ai_harness.nimble.NimbleAgent

::: pydantic_ai_harness.nimble.NimbleSource

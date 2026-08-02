# Nimble Search

Give an agent web research tools backed by the [Nimble](https://www.nimbleway.com/) API:
search, page extract (markdown), and opt-in site map / resumable crawl. Web Search
Agents (Agent API V2) ship as the separate `NimbleAgent` capability.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/nimble/)

## Install

```bash
uv add "pydantic-ai-harness[nimble]"
```

Set `NIMBLE_API_KEY` (create a key at
<https://online.nimbleway.com/account-settings/api-keys>), or pass a configured client (see
configuration below).

## Usage

```python
from nimble_python import AsyncNimble
from pydantic_ai import Agent
from pydantic_ai_harness.nimble import NimbleAgent, NimbleSearch

client = AsyncNimble()
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[NimbleSearch(client=client), NimbleAgent(client=client)],
)

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

## Tools

| Tool | Purpose |
|---|---|
| `web_search` | Search the web (default depth `lite`) and return titles, URLs, and content/descriptions. |
| `get_page` | Extract a URL as markdown. |
| `map_site` | Discover links on a site. Opt-in via `include_map=True`. |
| `crawl_start` / `crawl_status` | Start and poll a crawl across turns (no long-polling in one call). Opt-in via `include_crawl=True`. |
| `agents_list` / `agent_templates_list` / `agent_run_start` / `agent_run_status` / `agent_run_result` | Web Search Agents lifecycle via `NimbleAgent`. |

```python
from pydantic_ai_harness.nimble import NimbleSearch

NimbleSearch(
    include_map=True,
    include_crawl=True,
)
```

## Web Search Agents

Default bootstrap is **Mode 1** (`agent_name` create-or-reuse) because a typical
Pydantic AI agent is a stateless host - keep returned `wsa_...` /
`task_run_...` ids in message history. Pass `agent_id` for Mode 2; omit both for
Mode 3 anonymous create.

`agent_run_start` exposes `use_case` (`research` | `enrichment` |
`dataset_building`, locked at create), `skill`, `effort`
(`low`...`max`, plan for **several minutes**), and run overrides (`sources`,
`output_schema`, `input_data`). Prefer start / status / result across turns -
do not block-poll inside one tool call.

See the [Harness docs](https://pydantic.dev/docs/ai/harness/nimble-search/) for
mode examples, override vs persist rules, and the `sources` shape.

## Configuration

```python
from pydantic_ai_harness.nimble import NimbleAgent, NimbleSearch

NimbleSearch(
    num_results=5,              # web_search result count (1-100)
    max_text_chars=10_000,      # get_page markdown budget (entire return)
    search_depth='lite',        # lite | fast | deep
    time_range=None,            # hour | day | week | month | year
    include_domains=[],         # mutually exclusive with exclude_domains
    exclude_domains=[],
    include_map=False,
    include_crawl=False,
    guidance=None,              # None = default instructions; '' = none
    client=None,                # NimbleClient - None builds AsyncNimble from NIMBLE_API_KEY
)

NimbleAgent(
    guidance=None,
    client=None,
)
```

Factory-built clients send `X-Client-Source: pydantic-ai` for attribution and
are closed when the last concurrent run ends (including failed or cancelled
runs). Pass the same `client=` to `NimbleSearch` and `NimbleAgent` when using
both so they share one HTTP session.

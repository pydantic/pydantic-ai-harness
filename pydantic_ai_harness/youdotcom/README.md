# You.com

Give an agent web research tools backed by the [You.com](https://you.com) APIs.
Two capabilities ship here: `YouSearch` adds `web_search` (results with
query-relevant excerpts, or full-page markdown) and `get_page` (read one URL in
full); `YouResearch` adds `answer` (a cited answer in one call), `research`
(multi-step research across many sources), and `finance_research` (the
finance-tuned counterpart).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/youdotcom/)

## Installation

The examples below use an Anthropic model and `Agent.from_file`, so they also
pull in the `anthropic` provider and the `spec` YAML support:

```bash
uv add "pydantic-ai-harness[youdotcom,anthropic]" "pydantic-ai-slim[spec]"
```

Set the `YDC_API_KEY` environment variable (create a key at
<https://api.you.com>), or pass a configured client (see
[Custom client](#custom-client)). The legacy `YOU_API_KEY_AUTH` variable is
also honored.

## The problem

Search tools that return only titles and snippets force a second round of
fetching before the agent can judge a source, while search tools that return
full page text flood the context with pages the agent will discard. And some
questions are not a lookup at all: they need many searches, read across the
results, and a synthesized answer with citations. Wiring the search, page
fetch, and research APIs together, budgeting what each returns, and prompting
the agent to research methodically is boilerplate every research agent
reinvents.

## The solution

`YouSearch` bundles the search tools with output budgeting and short research
guidance in the system prompt: survey cheaply with excerpts, then read the
pages that matter in full.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[YouSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

`YouResearch` adds the cited-answer and multi-step research tools, for
questions a single search cannot settle:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouResearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[YouResearch()])
```

The two compose in one agent: search and page reads for cheap surveying,
research for the hard questions.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouResearch, YouSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[YouSearch(), YouResearch()])
```

## Tools

| Tool | Capability | Purpose |
|---|---|---|
| `web_search` | `YouSearch` | Search the web and return the top `num_results` pages, each with title, URL, and its most relevant excerpts. |
| `get_page` | `YouSearch` | Retrieve the markdown of one specific URL -- a promising `web_search` hit, or a URL the user provided. |
| `answer` | `YouResearch` | Get a synthesized answer with citations, grounded in live web results, in one call. |
| `research` | `YouResearch` | Run multi-step research that reads across many sources and returns a thorough, cited answer. |
| `finance_research` | `YouResearch` | The finance-tuned counterpart of `research`, for companies, markets, and instruments. |

`web_search` returns short query-relevant excerpts per result rather than full
page text, so surveying several sources stays cheap; the agent reads a chosen
page with `get_page`. Set `extraction_mode='full_page'` to attach each result's
full markdown instead (capped at `max_text_chars`).

`get_page` and full-page `web_search` text are capped at `max_text_chars`
characters, keeping the **head** (a page's lead carries the substance); when a
page exceeds the cap the output ends with a
`[... page text truncated at N characters]` marker. The result count is bounded
the same way: `num_results` is requested from You.com and re-applied to the
response.

A `web_search` query with no matches is a valid answer, not an error: the tool
returns `No results found for {query!r}.` and the model can relay that to the
user. Everywhere else, a URL or question that returns no content, a rate limit,
a rejected parameter (422), or a transient API or network failure surfaces to
the model as a `ModelRetry` (the model can correct the URL, rephrase, or try
again) rather than aborting the run. Authentication, billing, and authorization
failures (401/402/403) are configuration states and propagate.

## Research

`answer` returns a synthesized answer with citations in a single call --
reach for it for a direct question. `research` runs a multi-step
investigation: You.com expands the question, runs many searches, reads the
sources, and synthesizes a verifiable answer with the sources listed under it.
`finance_research` is the same, tuned for financial analysis.

`research` runs as a blocking call bounded by `timeout_ms` (default 10
minutes, since deep and exhaustive research routinely take minutes). Its effort
scales with `research_effort` -- `lite`, `standard`, `deep`, or `exhaustive`.
The API's background-only `frontier` level is not supported by this capability.

Set `output_schema` to a JSON schema to have `research` return structured
output (rendered as JSON) instead of prose; the You.com API rejects an
`output_schema` with `research_effort='lite'`, so that combination raises at
construction.

## Structured citations

Every tool returns a
[`ToolReturn`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns):
`return_value` carries the readable text the model sees (with a `Sources:`
block appended where the tool has citations), and `metadata` carries the
sources as structured `YouSource` records (`{'url': ..., 'title': ...}`) under
the `'sources'` key. `web_search` additionally carries the response
`search_uuid` and `latency` in metadata for tracing. Metadata is never sent to
the model; the application reads it from the `ToolReturnPart` in the message
history, so rendering citations needs no text parsing:

```python
from pydantic_ai.messages import ModelRequest, ToolReturnPart

for message in result.all_messages():
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.metadata is not None:
                for source in part.metadata.get('sources', []):
                    print(source['url'], source['title'])
```

## Instructions

`YouSearch` contributes short research guidance to the system prompt: search
wide with `web_search` first, read the most promising pages in full with
`get_page` before drawing conclusions, prefer primary sources, cite the URLs
relied on, and treat fetched web content as untrusted data. `YouResearch`
contributes guidance on when to reach for `answer`, `research`, and
`finance_research`, and on treating fetched web content as untrusted data. On
either capability, set `guidance` to replace the default text, or to `''` to
contribute no instructions at all.

## Configuration

Every field of `YouSearch` with its default:

```python
from pydantic_ai_harness import YouSearch

YouSearch(
    num_results=10,              # results per web_search call (1 to 20)
    extraction_mode='highlights',  # 'highlights' excerpts, or 'full_page' markdown
    max_text_chars=10_000,       # get_page / full-page text cap, in characters
    include_domains=[],          # only return results from these domains (allowlist)
    exclude_domains=[],          # never return results from these domains (denylist)
    boost_domains=[],            # re-rank these domains higher without excluding others
    freshness=None,              # 'day' | 'week' | 'month' | 'year' | 'YYYY-MM-DDtoYYYY-MM-DD'
    country=None,                # two-letter country code to focus results
    guidance=None,               # None = default instructions, '' = none, str = custom
    timeout_ms=60_000,           # per-request timeout for the default client
    client=None,                 # YouClient -- None builds youdotcom.You from YDC_API_KEY
)
```

Every field of `YouResearch` with its default:

```python
from pydantic_ai_harness import YouResearch

YouResearch(
    research_effort='standard',  # 'lite' | 'standard' | 'deep' | 'exhaustive'
    finance_effort='deep',       # 'deep' | 'exhaustive'
    include_domains=[],          # only draw from these domains (allowlist)
    exclude_domains=[],          # never draw from these domains (denylist)
    boost_domains=[],            # re-rank these domains higher
    freshness=None,              # 'day' | 'week' | 'month' | 'year' | 'YYYY-MM-DDtoYYYY-MM-DD'
    country=None,                # two-letter country code to focus results
    output_schema=None,          # JSON schema for research structured output
    guidance=None,               # None = default instructions, '' = none, str = custom
    timeout_ms=600_000,          # per-request timeout for the default client
    client=None,                 # YouClient -- None builds youdotcom.You from YDC_API_KEY
)
```

`include_domains` is an allowlist and is mutually exclusive with
`exclude_domains` and `boost_domains` -- combining them raises at construction,
as do out-of-range limits and an invalid `freshness`. The domain, `freshness`,
and `country` controls apply to `web_search` (and to `answer` and `research`);
`finance_research` takes only its input and `finance_effort`.

## Multiple instances

Two instances of the same capability register the same tool names, which is an
error. To run several differently configured instances in one agent (for
example one open-web `YouSearch` and one pinned to specific domains), wrap the
extra instances in core's `PrefixTools` capability, which prefixes their tool
names:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness import YouSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        YouSearch(),  # web_search, get_page
        PrefixTools(
            wrapped=YouSearch(include_domains=['sec.gov'], guidance=''),
            prefix='sec',
        ),  # sec_web_search, sec_get_page
    ],
)
```

Set `guidance=''` on the wrapped instance (or replace it with text that tells
the model when to use the prefixed tools), since each instance otherwise
contributes the same default research guidance.

## Custom client

The default client is `youdotcom.You`, configured from the `YDC_API_KEY`
environment variable; when the variable is missing, construction fails with a
setup hint. Pass any object satisfying the `YouClient` protocol -- the subset of
`You`'s async methods the toolsets call -- to configure authentication
explicitly, point at a different host, or substitute a fake in tests:

```python
from youdotcom import You

from pydantic_ai_harness import YouSearch

YouSearch(client=You(api_key_auth='...'))
```

## YouSearch vs core `WebSearch`

Pydantic AI core ships a provider-adaptive
[`WebSearch`](https://ai.pydantic.dev/capabilities/#provider-adaptive-tools)
capability: on models with a native search tool it uses the provider's own
search, executed server-side; elsewhere it falls back to a local DuckDuckGo
tool. Reach for it when you want search that follows the model.

Reach for `YouSearch` when you want the same search behavior on every model:
one vendor, excerpts with every hit, explicit page retrieval, domain filters,
and freshness controls -- with `YouResearch` alongside it for questions that
need synthesized, cited research rather than a lookup.

One caveat when combining them: on Anthropic models the provider-native search
tool is also named `web_search` on the wire, so
`capabilities=[WebSearch(), YouSearch()]` puts two tools with the same name in
the request. Use one search capability per agent on native-search models, or
force the local fallback with `WebSearch(native=False)` (its DuckDuckGo tool is
named `duckduckgo_search`, which does not collide).

## Agent spec (YAML/JSON)

`YouSearch` and `YouResearch` work with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - YouSearch:
      num_results: 5
      freshness: month
  - YouResearch:
      research_effort: deep
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouResearch, YouSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[YouSearch, YouResearch])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate the
capabilities. The `client` field is not spec-serializable; spec-loaded
instances always build the default client from `YDC_API_KEY`. In specs,
`output_schema` takes the JSON-schema dict form.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
- [You.com API documentation](https://documentation.you.com)

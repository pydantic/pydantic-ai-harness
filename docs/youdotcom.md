---
title: You.com
description: Give a Pydantic AI agent web search, content extraction, and research via You.com APIs, with configurable parameter locking.
---

# You.com

`Youdotcom` exposes four tools backed by You.com APIs:

- `you_search`: Web and news search with configurable filters.
- `you_contents`: Extract clean HTML or Markdown from known URLs.
- `you_research`: Deep research with cited, synthesized answers.
- `you_finance_research`: Finance-focused research with cited answers.

Parameters set at construction are locked: they are removed from each tool's
schema, so the LLM neither sees nor can override them. Parameters left unset are
exposed to the LLM, giving it dynamic control over search behavior. `offset`,
`max_age`, and `output_schema` are never exposed to the LLM.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/youdotcom/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

Agents need live information from the web: searching for current events, reading
page contents, and getting cited answers to complex questions. Wiring each of
these into agent tools by hand is repetitive -- parameter validation, response
parsing, and deciding which parameters the LLM should control versus which the
developer should lock.

`Youdotcom` centralizes that wiring so you configure the boundary once and reuse
it across agents.

## Usage

Add `Youdotcom` to your agent's `capabilities` with your You.com API key. The
agent can call any of the four tools.

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.youdotcom import Youdotcom

agent = Agent(
    'openai:gpt-5.1',
    capabilities=[Youdotcom(api_key=os.environ['YOU_API_KEY'], count=5, freshness='day')],
    system_prompt='Use you_search to find live information, you_research for complex questions.',
)

result = agent.run_sync('What happened in the world today, and why?')
print(result.output)
```

You.com is a paid service with free credits to explore. Create an account at
<https://you.com/platform> to get an API key.

## Tools

### `you_search`

Web and news search via the
[Search API](https://docs.you.com/api-reference/search/v1-search). Returns
unified results from web and news sources. A response without a `results` object
returns an empty list.

### `you_contents`

Extract clean HTML or Markdown from known URLs via the
[Contents API](https://docs.you.com/api-reference/contents). Pass a list of URLs
and get back full page content, ready for LLM consumption. Useful for competitive
intelligence, knowledge base ingestion, or reading specific pages the agent
already knows about.

### `you_research`

Deep research via the
[Research API](https://docs.you.com/api-reference/research/v1-research). Runs
multiple searches, reads through sources, and synthesizes a thorough,
well-cited answer with inline citations. Use it when a question is too complex
for a simple lookup -- comparative analyses, multi-factor evaluations, or
questions that span multiple domains.

This capability uses the synchronous Research API. It supports `lite`,
`standard`, `deep`, and `exhaustive`; the asynchronous `frontier` mode is not
exposed.

### `you_finance_research`

Finance-focused research via the
[Finance Research API](https://docs.you.com/api-reference/finance-research/v1-finance_research).
Uses a finance-optimized index to research earnings, filings, market data, and
financial news. Use it for company fundamentals, market trends, competitive
analysis, or earnings summaries. It returns a `YouTextResearchResult`;
structured output is available only from `you_research`.

## Parameters

### Search parameters

| Parameter | Description | LLM Control |
|---|---|---|
| `count` | Maximum results per section (web/news). Range 1-100, default 10. | Only if not configured |
| `offset` | Pagination offset (0-9). | Never (human-only) |
| `freshness` | Time filter: `'day'`, `'week'`, `'month'`, `'year'`, or `'YYYY-MM-DDtoYYYY-MM-DD'`. | Only if not configured |
| `country` | Geographic focus: ISO 3166-1 alpha-2 code. | Only if not configured |
| `language` | Language of results: BCP 47 code. | Only if not configured |
| `safesearch` | Content moderation: `'off'`, `'moderate'`, `'strict'`. | Only if not configured |
| `livecrawl` | Full page content retrieval: `'web'`, `'news'`, `'all'`. | Only if not configured |
| `livecrawl_formats` | Livecrawl format(s): `'html'`, `'markdown'`, or both. | Only if not configured |
| `include_domains` | Domain allowlist (max 500). Cannot be combined with `exclude_domains` or `boost_domains`. Setting any domain filter sends the search as a POST request. | Only if not configured |
| `exclude_domains` | Domain blocklist (max 500). May be combined with `boost_domains`. | Only if not configured |
| `boost_domains` | Domains to boost in ranking without filtering others (max 500). | Only if not configured |
| `search_crawl_timeout` | Per-URL livecrawl timeout in seconds (1-60). Separate from Contents `crawl_timeout`. | Only if not configured |

### Contents parameters

| Parameter | Description | LLM Control |
|---|---|---|
| `contents_formats` | Formats to return: `'html'`, `'markdown'`, `'metadata'`. | Only if not configured |
| `crawl_timeout` | Per-URL timeout in seconds (1-60). Default: 10. | Only if not configured |
| `max_age` | Max age of cached content in seconds. | Never (human-only) |

### Research parameters

| Parameter | Description | LLM Control |
|---|---|---|
| `research_effort` | Depth: `'lite'`, `'standard'`, `'deep'`, `'exhaustive'`. Default: `'standard'`. | Only if not configured |
| `research_include_domains` | Domain allowlist for sources (max 500). Cannot be combined with `research_exclude_domains` or `research_boost_domains`. | Only if not configured |
| `research_exclude_domains` | Domain blocklist for sources (max 500). | Only if not configured |
| `research_boost_domains` | Domains to boost in source ranking without filtering others (max 500). | Only if not configured |
| `research_freshness` | Source recency filter: `'day'`, `'week'`, `'month'`, `'year'`, or `'YYYY-MM-DDtoYYYY-MM-DD'`. | Only if not configured |
| `research_country` | ISO 3166-1 alpha-2 country code to geographically focus sources. | Only if not configured |
| `output_schema` | JSON Schema for structured output. When set, the result is a `YouObjectResearchResult` (`content` is a JSON object, `content_type` is `'object'`); otherwise a `YouTextResearchResult`. Not valid with `research_effort='lite'`. | Never (human-only) |

### Finance research parameters

| Parameter | Description | LLM Control |
|---|---|---|
| `finance_research_effort` | Depth: `'deep'`, `'exhaustive'`. Default: `'deep'`. | Only if not configured |

### Parameter control behavior

- **Configured parameters**: When you set a parameter at construction time, that
  value is locked and the LLM cannot override it.
- **Unconfigured parameters**: When you don't set a parameter, the LLM can
  dynamically choose appropriate values based on the user's query.
- **Human-only parameters**: `offset`, `max_age`, and `output_schema` are never exposed to the LLM.

`count` specifies results **per section** (web and news), so `count=5` may
return up to 10 total results (5 web + 5 news).

## Configuration

```python
from pydantic_ai_harness.youdotcom import Youdotcom

Youdotcom(
    api_key='...',              # required -- You.com API key (excluded from repr)
    http_client=None,           # optional httpx.AsyncClient for connection pooling
    timeout=None,               # request timeout override; defaults: 300s research, 60s search/contents
    # Search
    count=5,                    # results per section
    offset=None,                # pagination offset (never exposed to LLM)
    freshness='day',            # 'day', 'week', 'month', 'year', or date range
    country='US',               # ISO 3166-1 alpha-2
    language='EN',              # BCP 47
    safesearch='moderate',      # 'off', 'moderate', 'strict'
    livecrawl=None,             # 'web', 'news', 'all'
    livecrawl_formats=None,     # ['html'], ['markdown'], or ['html', 'markdown']
    include_domains=None,       # ['nytimes.com', 'bbc.com'] -- allowlist
    exclude_domains=None,       # ['spam.com'] -- blocklist
    boost_domains=None,         # ['good.com'] -- ranking boost
    search_crawl_timeout=None,  # per-URL livecrawl timeout (1-60 seconds)
    # Contents
    contents_formats=None,      # ['html', 'markdown', 'metadata']
    crawl_timeout=None,         # per-URL timeout (1-60 seconds)
    max_age=None,               # max cache age in seconds (never exposed to LLM)
    # Research
    research_effort=None,       # 'lite', 'standard', 'deep', 'exhaustive'
    research_include_domains=None,  # ['arxiv.org'] -- source allowlist
    research_exclude_domains=None,  # ['spam.com'] -- source blocklist
    research_boost_domains=None,    # ['good.com'] -- source ranking boost
    research_freshness=None,    # 'day', 'week', 'month', 'year', or date range
    research_country=None,      # ISO 3166-1 alpha-2 for source geographic focus
    output_schema=None,         # JSON Schema for structured output (never exposed to LLM; not valid with research_effort='lite')
    # Finance research
    finance_research_effort=None,  # 'deep', 'exhaustive'
)
```

## Agent spec (YAML/JSON)

`Youdotcom` works with Pydantic AI's
[agent spec](/ai/core-concepts/agent-spec/). Loading agents from files needs the
`spec` extra (`pip install "pydantic-ai-slim[spec]"`), which this harness already
pulls in.

Spec values are used verbatim -- the loader does not expand `${...}` or
environment variables. A placeholder like `${YOU_API_KEY}` would be sent to
You.com unchanged, so inject the real key in code rather than the file:

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.youdotcom import Youdotcom

agent = Agent(
    'openai:gpt-5.1',
    capabilities=[Youdotcom(api_key=os.environ['YOU_API_KEY'], count=5, freshness='day')],
)
```

If you do load `Youdotcom` from a file, `api_key` is required and stored
literally, so treat the spec file itself as a secret:

```yaml
# agent.yaml -- contains a secret; api_key is stored verbatim
model: openai:gpt-5.1
capabilities:
  - Youdotcom:
      api_key: your-you-com-api-key
      count: 5
      freshness: day
      country: US
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.youdotcom import Youdotcom

agent = Agent.from_file('agent.yaml', custom_capability_types=[Youdotcom])
```

## Error handling

The toolset retries one `429 Too Many Requests` response when You.com's
`Retry-After` delay is at most 60 seconds. Longer or repeated rate limits, and
configuration errors (`401`, `402`, `403`, or `404`), propagate as
`httpx.HTTPStatusError`. Other HTTP and transport errors become `ModelRetry` so
the agent can recover.

## Further reading

- [You.com API docs](https://docs.you.com/)
- [Search API reference](https://docs.you.com/api-reference/search/v1-search)
- [Contents API reference](https://docs.you.com/api-reference/contents)
- [Research API reference](https://docs.you.com/api-reference/research/v1-research)
- [Finance Research API reference](https://docs.you.com/api-reference/finance-research/v1-finance_research)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Toolsets](/ai/tools-toolsets/toolsets/)

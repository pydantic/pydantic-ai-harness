"""You.com capability for Pydantic AI agents."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

import httpx
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.youdotcom._toolset import (
    ContentsFormat,
    Country,
    CrawlTimeoutSeconds,
    Domains,
    FinanceResearchEffort,
    Freshness,
    Language,
    LiveCrawl,
    LiveCrawlFormats,
    ResearchEffort,
    SafeSearch,
    SearchCount,
    SearchOffset,
    YoudotcomToolset,
)

__all__ = ('Youdotcom',)


@dataclass
class Youdotcom(AbstractCapability[AgentDepsT]):
    """Web search, content extraction, and research via the [You.com API](https://docs.you.com/).

    Exposes four tools backed by You.com:

    - `you_search`: Web and news search with configurable filters.
    - `you_contents`: Extract clean HTML or Markdown from known URLs.
    - `you_research`: Deep research with cited, synthesized answers.
    - `you_finance_research`: Finance-focused research with cited answers.

    Parameters set at construction are locked -- the LLM cannot override them.
    Parameters left as `None` are exposed to the LLM, giving it dynamic control.
    `offset`, `max_age`, and `output_schema` are never exposed to the LLM; they
    are always human-controlled.

    ```python
    import os

    from pydantic_ai import Agent
    from pydantic_ai_harness.youdotcom import Youdotcom

    agent = Agent(
        'openai:gpt-5.1',
        capabilities=[Youdotcom(api_key=os.environ['YOU_API_KEY'], count=5, freshness='day')],
        system_prompt='Use you_search to find live information from the web.',
    )

    result = agent.run_sync('What happened in the world today?')
    print(result.output)
    ```

    You.com is a paid service with free credits to explore. Create an account at
    <https://you.com/platform> to get an API key.
    """

    api_key: str = field(repr=False)
    """You.com API key. Get one at <https://you.com/platform/api-keys>. Excluded from `repr` to avoid leaking the secret."""

    _: KW_ONLY

    http_client: httpx.AsyncClient | None = None
    """Optional shared `httpx.AsyncClient` for connection pooling. If `None`, a new client is created per request."""

    timeout: float | None = None
    """Request timeout in seconds applied to all four tools. If `None`, research/finance use 300s and search/contents use 60s."""

    # Search
    count: SearchCount | None = None
    """Maximum results per section (web/news). Range 1-100. API default is 10."""

    offset: SearchOffset | None = None
    """Pagination offset (0-9). Never exposed to the LLM."""

    freshness: Freshness | None = None
    """Result freshness: 'day', 'week', 'month', 'year', or 'YYYY-MM-DDtoYYYY-MM-DD'."""

    country: Country | None = None
    """ISO 3166-1 alpha-2 country code for geographic focus."""

    language: Language | None = None
    """BCP 47 language code for results."""

    safesearch: SafeSearch | None = None
    """Content moderation: 'off', 'moderate', or 'strict'. API default is 'moderate'."""

    livecrawl: LiveCrawl | None = None
    """Sections to livecrawl for full page content: 'web', 'news', or 'all'."""

    livecrawl_formats: LiveCrawlFormats | None = None
    """Format(s) for livecrawled content: one or both of 'html' and 'markdown'."""

    include_domains: Domains | None = None
    """Domain allowlist for search results (max 500 domains)."""

    exclude_domains: Domains | None = None
    """Domain blocklist for search results (max 500 domains)."""

    boost_domains: Domains | None = None
    """Domains to boost in search ranking without filtering others (max 500 domains)."""

    search_crawl_timeout: CrawlTimeoutSeconds | None = None
    """Per-URL livecrawl timeout for `you_search` in seconds (1-60). Separate from Contents `crawl_timeout`."""

    # Contents
    contents_formats: list[ContentsFormat] | None = None
    """Formats for `you_contents`: 'html', 'markdown', and/or 'metadata'."""

    crawl_timeout: CrawlTimeoutSeconds | None = None
    """Per-URL timeout for `you_contents` in seconds (1-60). API default is 10."""

    max_age: int | None = None
    """Max age of cached content for `you_contents` in seconds. Never exposed to the LLM."""

    # Research
    research_effort: ResearchEffort | None = None
    """Synchronous depth for `you_research`: 'lite', 'standard', 'deep', or 'exhaustive'. API default is 'standard'."""

    research_include_domains: Domains | None = None
    """Domain allowlist for research sources (max 500 domains)."""

    research_exclude_domains: Domains | None = None
    """Domain blocklist for research sources (max 500 domains)."""

    research_boost_domains: Domains | None = None
    """Domains to boost in research source ranking without filtering others (max 500 domains)."""

    research_freshness: Freshness | None = None
    """Recency filter for research sources: 'day', 'week', 'month', 'year', or 'YYYY-MM-DDtoYYYY-MM-DD'."""

    research_country: Country | None = None
    """ISO 3166-1 alpha-2 country code to geographically focus research sources."""

    output_schema: dict[str, object] | None = None
    """JSON Schema for structured research output. When set, `content` is a JSON object and `content_type` is 'object'. Never exposed to the LLM."""

    # Finance research
    finance_research_effort: FinanceResearchEffort | None = None
    """Depth for `you_finance_research`: 'deep' or 'exhaustive'. API default is 'deep'."""

    def __post_init__(self) -> None:
        """Validate configuration when the public capability is constructed."""
        self.get_toolset()

    def get_toolset(self) -> YoudotcomToolset[AgentDepsT]:
        """Build and return the You.com toolset."""
        return YoudotcomToolset[AgentDepsT](
            api_key=self.api_key,
            http_client=self.http_client,
            timeout=self.timeout,
            count=self.count,
            offset=self.offset,
            freshness=self.freshness,
            country=self.country,
            language=self.language,
            safesearch=self.safesearch,
            livecrawl=self.livecrawl,
            livecrawl_formats=self.livecrawl_formats,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            boost_domains=self.boost_domains,
            search_crawl_timeout=self.search_crawl_timeout,
            contents_formats=self.contents_formats,
            crawl_timeout=self.crawl_timeout,
            max_age=self.max_age,
            research_effort=self.research_effort,
            research_include_domains=self.research_include_domains,
            research_exclude_domains=self.research_exclude_domains,
            research_boost_domains=self.research_boost_domains,
            research_freshness=self.research_freshness,
            research_country=self.research_country,
            output_schema=self.output_schema,
            finance_research_effort=self.finance_research_effort,
        )

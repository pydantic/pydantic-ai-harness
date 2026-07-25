"""You.com toolset providing web search, content extraction, and research via the You.com API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Annotated, Final, Literal

import anyio
import httpx
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    WithJsonSchema,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import NotRequired, TypedDict

__all__ = (
    'ContentsFormat',
    'Country',
    'CrawlTimeoutSeconds',
    'Domains',
    'FinanceResearchEffort',
    'Freshness',
    'Language',
    'LiveCrawl',
    'LiveCrawlFormats',
    'ResearchEffort',
    'SafeSearch',
    'SearchCount',
    'SearchOffset',
    'YouContentsMetadata',
    'YouContentsResult',
    'YouLivecrawlContents',
    'YouObjectResearchResult',
    'YouResearchResult',
    'YouResearchSource',
    'YouSearchResult',
    'YouTextResearchResult',
    'YoudotcomToolset',
)

_YOU_SEARCH_URL: Final[str] = 'https://ydc-index.io/v1/search'
_YOU_CONTENTS_URL: Final[str] = 'https://ydc-index.io/v1/contents'
_YOU_RESEARCH_URL: Final[str] = 'https://api.you.com/v1/research'
_YOU_FINANCE_RESEARCH_URL: Final[str] = 'https://api.you.com/v1/finance_research'

_DEFAULT_TIMEOUT: Final[float] = 60.0
"""Request timeout (seconds) for search and contents calls."""

_RESEARCH_TIMEOUT: Final[float] = 300.0
"""Request timeout (seconds) for research and finance research calls; exhaustive runs are slow."""

Country = Literal[
    'AR',
    'AU',
    'AT',
    'BE',
    'BR',
    'CA',
    'CL',
    'DK',
    'FI',
    'FR',
    'DE',
    'HK',
    'IN',
    'ID',
    'IT',
    'JP',
    'KR',
    'MY',
    'MX',
    'NL',
    'NZ',
    'NO',
    'CN',
    'PL',
    'PT',
    'PH',
    'RU',
    'SA',
    'ZA',
    'ES',
    'SE',
    'CH',
    'TW',
    'TR',
    'GB',
    'US',
]
"""ISO 3166-1 alpha-2 country codes for geographic focus of search results."""

Language = Literal[
    'AR',
    'EU',
    'BN',
    'BG',
    'CA',
    'ZH-HANS',
    'ZH-HANT',
    'HR',
    'CS',
    'DA',
    'NL',
    'EN',
    'EN-GB',
    'ET',
    'FI',
    'FR',
    'GL',
    'DE',
    'EL',
    'GU',
    'HE',
    'HI',
    'HU',
    'IS',
    'IT',
    'JA',
    'KN',
    'KO',
    'LV',
    'LT',
    'MS',
    'ML',
    'MR',
    'NB',
    'PL',
    'PT-BR',
    'PT-PT',
    'PA',
    'RO',
    'RU',
    'SR',
    'SK',
    'SL',
    'ES',
    'SV',
    'TA',
    'TE',
    'TH',
    'TR',
    'UK',
    'VI',
]
"""BCP 47 language codes for search results."""

_DateRange = Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$')]
Freshness = Literal['day', 'week', 'month', 'year'] | _DateRange
"""Result freshness filter: a named interval or a date range ``YYYY-MM-DDtoYYYY-MM-DD``."""

SafeSearch = Literal['off', 'moderate', 'strict']
"""Content moderation level for search results."""

LiveCrawl = Literal['web', 'news', 'all']
"""Which sections to livecrawl for full page content."""

LiveCrawlFormats = Annotated[list[Literal['html', 'markdown']], Field(max_length=2)]
"""Format(s) for livecrawled content. Pass one or both of 'html', 'markdown'."""

ContentsFormat = Literal['html', 'markdown', 'metadata']
"""Format for content extraction via the Contents API."""

ResearchEffort = Literal['lite', 'standard', 'deep', 'exhaustive']
"""Synchronous research depth level. The asynchronous `frontier` mode is not exposed."""

FinanceResearchEffort = Literal['deep', 'exhaustive']
"""Research depth level for the Finance Research API."""

SearchCount = Annotated[int, Field(ge=1, le=100)]
"""Maximum number of search results per section (1-100)."""

SearchOffset = Annotated[int, Field(ge=0, le=9)]
"""Pagination offset for search results (0-9)."""

CrawlTimeoutSeconds = Annotated[int, Field(ge=1, le=60)]
"""Per-URL crawl timeout in seconds (1-60)."""

_AnyUrlAdapter: TypeAdapter[AnyUrl] = TypeAdapter(AnyUrl)


def _validate_uri(value: str) -> str:
    """Validate a URI while preserving the string value sent to You.com."""
    _AnyUrlAdapter.validate_python(value)
    return value


ContentsUrl = Annotated[
    str,
    AfterValidator(_validate_uri),
    WithJsonSchema({'type': 'string', 'format': 'uri', 'minLength': 1}),
]
ContentsUrls = list[ContentsUrl]
"""URIs to fetch content from."""

ResearchInput = Annotated[str, Field(max_length=40000)]
"""The research question (maximum 40,000 characters)."""

Domains = Annotated[list[str], Field(max_length=500)]
"""Domain list for search or source control filtering (maximum 500 domains)."""


class YouLivecrawlContents(TypedDict, total=False):
    """Contents of a page when livecrawl is enabled in search."""

    html: str
    """The HTML content of the page."""

    markdown: str
    """The Markdown content of the page."""


class YouSearchResult(TypedDict, total=False):
    """A single You.com search result.

    Fields depend on the API response and livecrawl settings.
    """

    title: str
    """The title of the search result."""

    url: str
    """The URL of the search result."""

    description: NotRequired[str]
    """A description or snippet of the search result."""

    snippets: NotRequired[list[str]]
    """Text snippets from the search result, providing a preview of the content."""

    thumbnail_url: NotRequired[str]
    """URL of the thumbnail image for the search result."""

    page_age: NotRequired[datetime]
    """The age or publication date of the search result (ISO 8601 format)."""

    favicon_url: NotRequired[str]
    """The URL of the favicon of the search result's domain."""

    contents: NotRequired[YouLivecrawlContents]
    """Contents of the page if livecrawl was enabled."""

    authors: NotRequired[list[str]]
    """An array of authors of the search result."""


class YouContentsMetadata(TypedDict, total=False):
    """Metadata about a web page from the Contents API."""

    site_name: str
    """The OpenGraph site name of the web page."""

    favicon_url: str
    """The URL of the favicon of the web page's domain."""


class YouContentsResult(TypedDict, total=False):
    """A single result from the Contents API.

    Fields depend on the requested formats and what the API could extract.
    """

    url: str
    """The webpage URL whose content was fetched."""

    title: str
    """The title of the web page."""

    html: NotRequired[str]
    """The HTML content of the web page, if requested and available."""

    markdown: NotRequired[str]
    """The Markdown content of the web page, if requested and available."""

    metadata: NotRequired[YouContentsMetadata]
    """Metadata about the web page, if requested."""


class YouResearchSource(TypedDict):
    """A source cited in a research result."""

    url: str
    """The URL of the source webpage."""

    title: NotRequired[str]
    """The title of the source webpage."""

    snippets: NotRequired[list[str]]
    """Relevant excerpts from the source page used in generating the answer."""


class _ResearchSourceControl(TypedDict, total=False):
    """Internal request-body shape controlling which web sources research visits.

    Beta feature. `include_domains` cannot be combined with `exclude_domains` or
    `boost_domains`. Each list supports up to 500 domains.
    """

    include_domains: list[str]
    exclude_domains: list[str]
    boost_domains: list[str]
    freshness: str
    country: str


class YouTextResearchResult(TypedDict):
    """A free-form research answer (the default when no `output_schema` is set)."""

    content: str
    """The comprehensive response with inline citations, in Markdown."""

    content_type: Literal['text']
    """Always 'text' for a Markdown answer."""

    sources: list[YouResearchSource]
    """Web sources used to generate the answer."""


class YouObjectResearchResult(TypedDict):
    """A structured research answer, returned when `output_schema` is configured."""

    content: dict[str, object]
    """The structured answer as a JSON object matching the configured `output_schema`."""

    content_type: Literal['object']
    """Always 'object' for structured JSON output."""

    sources: list[YouResearchSource]
    """Web sources used to generate the answer."""


YouResearchResult = YouTextResearchResult | YouObjectResearchResult
"""A result from the Research API, discriminated on `content_type`."""


# ---------------------------------------------------------------------------
# Internal parsing models -- search
# ---------------------------------------------------------------------------


class _RawLivecrawlContents(BaseModel):
    """Raw livecrawl contents field from the search API response."""

    html: str | None = None
    markdown: str | None = None

    def to_contents(self) -> YouLivecrawlContents | None:
        """Build a `YouLivecrawlContents` if either field is present."""
        if not self.html and not self.markdown:
            return None
        contents: YouLivecrawlContents = {}
        if self.html:
            contents['html'] = self.html
        if self.markdown:
            contents['markdown'] = self.markdown
        return contents


class _RawSearchResult(BaseModel):
    """Shared fields present in both web and news search results."""

    title: str | None = None
    url: str | None = None
    description: str | None = None
    thumbnail_url: str | None = None
    page_age: datetime | None = None
    contents: _RawLivecrawlContents | None = None

    def to_result(self) -> YouSearchResult:
        """Convert to a public `YouSearchResult` TypedDict."""
        result: YouSearchResult = {}
        if self.title is not None:
            result['title'] = self.title
        if self.url is not None:
            result['url'] = self.url
        if self.description:
            result['description'] = self.description
        if self.thumbnail_url:
            result['thumbnail_url'] = self.thumbnail_url
        if self.page_age is not None:
            result['page_age'] = self.page_age
        if self.contents is not None:
            contents = self.contents.to_contents()
            if contents is not None:
                result['contents'] = contents
        return result


class _RawWebResult(_RawSearchResult):
    """Web result with additional fields not present in news results."""

    snippets: list[str] | None = None
    favicon_url: str | None = None
    authors: list[str] | None = None

    def to_result(self) -> YouSearchResult:
        """Convert to a public `YouSearchResult`, adding web-only fields."""
        result = super().to_result()
        if self.snippets:
            result['snippets'] = self.snippets
        if self.favicon_url:
            result['favicon_url'] = self.favicon_url
        if self.authors:
            result['authors'] = self.authors
        return result


class _RawSearchResults(BaseModel):
    """The `results` object from the You.com search API response."""

    web: list[_RawWebResult] | None = None
    news: list[_RawSearchResult] | None = None


class _RawSearchResponse(BaseModel):
    """Top-level You.com search API response."""

    results: _RawSearchResults | None = None


# ---------------------------------------------------------------------------
# Internal parsing models -- contents
# ---------------------------------------------------------------------------


class _RawContentsMetadata(BaseModel):
    """Metadata field from the Contents API response."""

    site_name: str | None = None
    favicon_url: str | None = None

    def to_metadata(self) -> YouContentsMetadata | None:
        """Build a `YouContentsMetadata` if any field is present."""
        if not self.site_name and not self.favicon_url:
            return None
        metadata: YouContentsMetadata = {}
        if self.site_name:
            metadata['site_name'] = self.site_name
        if self.favicon_url:
            metadata['favicon_url'] = self.favicon_url
        return metadata


class _RawContentsItem(BaseModel):
    """A single item from the Contents API response array."""

    url: str | None = None
    title: str | None = None
    html: str | None = None
    markdown: str | None = None
    metadata: _RawContentsMetadata | None = None

    def to_result(self) -> YouContentsResult:
        """Convert to a public `YouContentsResult` TypedDict."""
        result: YouContentsResult = {}
        if self.url is not None:
            result['url'] = self.url
        if self.title is not None:
            result['title'] = self.title
        if self.html:
            result['html'] = self.html
        if self.markdown:
            result['markdown'] = self.markdown
        if self.metadata is not None:
            metadata = self.metadata.to_metadata()
            if metadata is not None:
                result['metadata'] = metadata
        return result


_ContentsResponseAdapter: TypeAdapter[list[_RawContentsItem]] = TypeAdapter(list[_RawContentsItem])


# ---------------------------------------------------------------------------
# Internal parsing models -- research / finance research
# ---------------------------------------------------------------------------


class _RawResearchSource(BaseModel):
    """A source cited in a research response."""

    url: str
    title: str | None = None
    snippets: list[str] | None = None

    def to_source(self) -> YouResearchSource:
        """Convert to a public `YouResearchSource` TypedDict."""
        source: YouResearchSource = {'url': self.url}
        if self.title:
            source['title'] = self.title
        if self.snippets:
            source['snippets'] = self.snippets
        return source


class _RawTextResearchOutput(BaseModel):
    """The `output` object for a free-form (Markdown) research answer."""

    content: str
    content_type: Literal['text']
    sources: list[_RawResearchSource]


class _RawObjectResearchOutput(BaseModel):
    """The `output` object for a structured (JSON) research answer."""

    content: dict[str, object]
    content_type: Literal['object']
    sources: list[_RawResearchSource]


class _RawResearchResponse(BaseModel):
    """Top-level research or finance research API response."""

    output: _RawTextResearchOutput | _RawObjectResearchOutput = Field(discriminator='content_type')


class _RawFinanceResearchResponse(BaseModel):
    """Top-level finance research API response."""

    output: _RawTextResearchOutput


class _ConfigValidator(BaseModel):
    """Validates configured (construction-time) values against You.com's documented limits.

    Tool-argument aliases only constrain values the LLM supplies; constructor
    values are validated here so out-of-range configuration fails at build time.
    """

    model_config = ConfigDict(strict=True)

    api_key: str
    timeout: Annotated[float, Field(gt=0)] | None = None
    count: SearchCount | None = None
    offset: SearchOffset | None = None
    freshness: Freshness | None = None
    country: Country | None = None
    language: Language | None = None
    safesearch: SafeSearch | None = None
    livecrawl: LiveCrawl | None = None
    livecrawl_formats: LiveCrawlFormats | None = None
    include_domains: Domains | None = None
    exclude_domains: Domains | None = None
    boost_domains: Domains | None = None
    search_crawl_timeout: CrawlTimeoutSeconds | None = None
    contents_formats: list[ContentsFormat] | None = None
    crawl_timeout: CrawlTimeoutSeconds | None = None
    max_age: Annotated[int, Field(ge=0)] | None = None
    research_effort: ResearchEffort | None = None
    research_include_domains: Domains | None = None
    research_exclude_domains: Domains | None = None
    research_boost_domains: Domains | None = None
    research_freshness: Freshness | None = None
    research_country: Country | None = None
    output_schema: dict[str, object] | None = None
    finance_research_effort: FinanceResearchEffort | None = None


class YoudotcomToolset(FunctionToolset[AgentDepsT]):
    """Toolset exposing You.com web search, content extraction, and research tools.

    Provides four tools backed by the You.com API:

    - `you_search`: Web and news search with configurable filters.
    - `you_contents`: Extract clean HTML or Markdown from known URLs.
    - `you_research`: Deep research with cited, synthesized answers.
    - `you_finance_research`: Finance-focused research with cited answers.

    Configured parameters (set at construction time) are locked: they are removed
    from each tool's schema so the LLM neither sees nor can override them.
    Unconfigured parameters are exposed to the LLM with sensible defaults.
    `offset`, `max_age`, and `output_schema` are never exposed to the LLM -- they
    are always human-controlled.

    Research and finance research use a 300s request timeout by default (exhaustive
    runs are slow); search and contents use 60s. Pass `timeout` to override all four.
    """

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float | None = None,
        # Search params
        count: SearchCount | None = None,
        offset: SearchOffset | None = None,
        freshness: Freshness | None = None,
        country: Country | None = None,
        language: Language | None = None,
        safesearch: SafeSearch | None = None,
        livecrawl: LiveCrawl | None = None,
        livecrawl_formats: LiveCrawlFormats | None = None,
        include_domains: Domains | None = None,
        exclude_domains: Domains | None = None,
        boost_domains: Domains | None = None,
        search_crawl_timeout: CrawlTimeoutSeconds | None = None,
        # Contents params
        contents_formats: list[ContentsFormat] | None = None,
        crawl_timeout: CrawlTimeoutSeconds | None = None,
        max_age: int | None = None,
        # Research params
        research_effort: ResearchEffort | None = None,
        research_include_domains: Domains | None = None,
        research_exclude_domains: Domains | None = None,
        research_boost_domains: Domains | None = None,
        research_freshness: Freshness | None = None,
        research_country: Country | None = None,
        output_schema: dict[str, object] | None = None,
        # Finance research params
        finance_research_effort: FinanceResearchEffort | None = None,
    ) -> None:
        super().__init__()
        # Validate configured values against You.com's documented limits. Tool-argument
        # aliases only constrain LLM-supplied values, so constructor values are checked here.
        _ConfigValidator(
            api_key=api_key,
            timeout=timeout,
            count=count,
            offset=offset,
            freshness=freshness,
            country=country,
            language=language,
            safesearch=safesearch,
            livecrawl=livecrawl,
            livecrawl_formats=livecrawl_formats,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            boost_domains=boost_domains,
            search_crawl_timeout=search_crawl_timeout,
            contents_formats=contents_formats,
            crawl_timeout=crawl_timeout,
            max_age=max_age,
            research_effort=research_effort,
            research_include_domains=research_include_domains,
            research_exclude_domains=research_exclude_domains,
            research_boost_domains=research_boost_domains,
            research_freshness=research_freshness,
            research_country=research_country,
            output_schema=output_schema,
            finance_research_effort=finance_research_effort,
        )

        self._api_key = api_key
        self._http_client = http_client
        self._timeout = timeout
        # Search
        self._count = count
        self._offset = offset
        self._freshness = freshness
        self._country = country
        self._language = language
        self._safesearch = safesearch
        self._livecrawl = livecrawl
        self._livecrawl_formats = livecrawl_formats
        self._include_domains = include_domains
        self._exclude_domains = exclude_domains
        self._boost_domains = boost_domains
        self._search_crawl_timeout = search_crawl_timeout
        # Contents
        self._contents_formats = contents_formats
        self._crawl_timeout = crawl_timeout
        self._max_age = max_age
        # Research
        self._research_effort = research_effort
        self._research_include_domains = research_include_domains
        self._research_exclude_domains = research_exclude_domains
        self._research_boost_domains = research_boost_domains
        self._research_freshness = research_freshness
        self._research_country = research_country
        self._output_schema = output_schema
        # Finance research
        self._finance_research_effort = finance_research_effort

        # Fail fast on locked-value combinations the API rejects with 422.
        self._check_domain_combo(include_domains, exclude_domains, boost_domains, ValueError)
        self._check_domain_combo(research_include_domains, research_exclude_domains, research_boost_domains, ValueError)
        if output_schema is not None and research_effort == 'lite':
            raise ValueError("output_schema is not supported with research_effort='lite'.")

        # Configured parameters are locked and removed from each tool's schema so the
        # LLM neither sees nor can override them.
        self._locked_by_tool: dict[str, frozenset[str]] = {
            'you_search': self._locked_search_params(),
            'you_contents': self._locked_contents_params(),
            'you_research': self._locked_research_params(),
            'you_finance_research': self._locked_finance_params(),
        }
        self.add_function(self.search, name='you_search', prepare=self._strip_locked_params)
        self.add_function(self.extract_contents, name='you_contents', prepare=self._strip_locked_params)
        self.add_function(self.research, name='you_research', prepare=self._strip_locked_params)
        self.add_function(self.finance_research, name='you_finance_research', prepare=self._strip_locked_params)

    # ------------------------------------------------------------------
    # Locked-parameter schema stripping
    # ------------------------------------------------------------------

    def _locked_search_params(self) -> frozenset[str]:
        """Tool-argument names of `you_search` that are locked by configuration."""
        configured: dict[str, object | None] = {
            'count': self._count,
            'freshness': self._freshness,
            'country': self._country,
            'language': self._language,
            'safesearch': self._safesearch,
            'livecrawl': self._livecrawl,
            'livecrawl_formats': self._livecrawl_formats,
            'include_domains': self._include_domains,
            'exclude_domains': self._exclude_domains,
            'boost_domains': self._boost_domains,
            'crawl_timeout': self._search_crawl_timeout,
        }
        locked = {name for name, value in configured.items() if value is not None}
        if self._include_domains is not None:
            locked.update(('exclude_domains', 'boost_domains'))
        elif self._exclude_domains is not None or self._boost_domains is not None:
            locked.add('include_domains')
        return frozenset(locked)

    def _locked_contents_params(self) -> frozenset[str]:
        """Tool-argument names of `you_contents` that are locked by configuration."""
        configured: dict[str, object | None] = {
            'formats': self._contents_formats,
            'crawl_timeout': self._crawl_timeout,
        }
        return frozenset(name for name, value in configured.items() if value is not None)

    def _locked_research_params(self) -> frozenset[str]:
        """Tool-argument names of `you_research` that are locked by configuration."""
        configured: dict[str, object | None] = {
            'research_effort': self._research_effort,
            'include_domains': self._research_include_domains,
            'exclude_domains': self._research_exclude_domains,
            'boost_domains': self._research_boost_domains,
            'freshness': self._research_freshness,
            'country': self._research_country,
        }
        locked = {name for name, value in configured.items() if value is not None}
        if self._research_include_domains is not None:
            locked.update(('exclude_domains', 'boost_domains'))
        elif self._research_exclude_domains is not None or self._research_boost_domains is not None:
            locked.add('include_domains')
        return frozenset(locked)

    def _locked_finance_params(self) -> frozenset[str]:
        """Tool-argument names of `you_finance_research` that are locked by configuration."""
        if self._finance_research_effort is not None:
            return frozenset({'research_effort'})
        return frozenset()

    def _strip_locked_params(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
        """`prepare` hook that removes construction-locked parameters from a tool's schema."""
        locked = self._locked_by_tool.get(tool_def.name, frozenset())
        if not locked:
            return tool_def
        original = tool_def.parameters_json_schema
        properties: dict[str, object] = original.get('properties', {})
        schema: dict[str, object] = dict(original)
        schema['properties'] = {key: value for key, value in properties.items() if key not in locked}
        return replace(tool_def, parameters_json_schema=schema)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        count: SearchCount | None = None,
        freshness: Freshness | None = None,
        country: Country | None = None,
        language: Language | None = None,
        safesearch: SafeSearch | None = None,
        livecrawl: LiveCrawl | None = None,
        livecrawl_formats: LiveCrawlFormats | None = None,
        include_domains: Domains | None = None,
        exclude_domains: Domains | None = None,
        boost_domains: Domains | None = None,
        crawl_timeout: CrawlTimeoutSeconds | None = None,
    ) -> list[YouSearchResult]:
        """Search the web and news using the You.com Search API.

        Args:
            query: The search query to execute.
            count: Maximum number of results per section (1-100). Only used if not
                configured at tool creation.
            freshness: Result freshness: 'day', 'week', 'month', 'year', or
                'YYYY-MM-DDtoYYYY-MM-DD'. Only used if not configured at tool creation.
            country: ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB'). Only used if
                not configured at tool creation.
            language: BCP 47 language code (e.g. 'EN', 'FR'). Only used if not
                configured at tool creation.
            safesearch: Content moderation: 'off', 'moderate', or 'strict'. Only used
                if not configured at tool creation.
            livecrawl: Sections to livecrawl: 'web', 'news', or 'all'. Only used if
                not configured at tool creation.
            livecrawl_formats: Format(s) for livecrawled content: one or both of
                'html' and 'markdown'. Only used if not configured at tool creation.
            include_domains: Restrict results to these domains (max 500). Cannot be
                combined with exclude_domains or boost_domains. Only used if not
                configured at tool creation.
            exclude_domains: Block results from these domains (max 500). Only used if
                not configured at tool creation.
            boost_domains: Boost these domains in ranking without filtering others
                (max 500). Only used if not configured at tool creation.
            crawl_timeout: Per-URL livecrawl timeout in seconds (1-60). Only used if
                not configured at tool creation.
        """
        domains = self._resolve_search_domains(
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            boost_domains=boost_domains,
        )
        params = self._build_search_params(
            query=query,
            count=count,
            freshness=freshness,
            country=country,
            language=language,
            safesearch=safesearch,
            livecrawl=livecrawl,
            livecrawl_formats=livecrawl_formats,
            crawl_timeout=crawl_timeout,
        )
        timeout = self._timeout_for(_DEFAULT_TIMEOUT)
        # Domain filters must be sent as JSON arrays via POST; the GET endpoint has no
        # unambiguous encoding for them.
        if domains:
            response = await self._post(_YOU_SEARCH_URL, {**params, **domains}, timeout=timeout)
        else:
            response = await self._get(_YOU_SEARCH_URL, params, timeout=timeout)
        return self._parse_search_results(_RawSearchResponse.model_validate(response.json()))

    async def extract_contents(
        self,
        urls: ContentsUrls,
        *,
        formats: list[ContentsFormat] | None = None,
        crawl_timeout: CrawlTimeoutSeconds | None = None,
    ) -> list[YouContentsResult]:
        """Extract clean HTML or Markdown content from web pages.

        Pass a list of URLs to fetch full page content for each. The API crawls
        them in parallel and returns clean, LLM-ready content.

        Args:
            urls: The URIs to fetch content from.
            formats: Content formats to return: 'markdown', 'html', 'metadata'.
                Only used if not configured at tool creation.
            crawl_timeout: Per-URL timeout in seconds (1-60). Only used if not
                configured at tool creation.
        """
        body = self._build_contents_body(urls=urls, formats=formats, crawl_timeout=crawl_timeout)
        response = await self._post(_YOU_CONTENTS_URL, body, timeout=self._timeout_for(_DEFAULT_TIMEOUT))
        items = _ContentsResponseAdapter.validate_python(response.json())
        return [item.to_result() for item in items]

    async def research(
        self,
        input: ResearchInput,
        *,
        research_effort: ResearchEffort | None = None,
        include_domains: Domains | None = None,
        exclude_domains: Domains | None = None,
        boost_domains: Domains | None = None,
        freshness: Freshness | None = None,
        country: Country | None = None,
    ) -> YouResearchResult:
        """Research a complex question and return a cited, synthesized answer.

        The Research API runs multiple searches, reads through sources, and
        synthesizes everything into a thorough, well-cited answer. Use it when a
        question is too complex for a simple lookup.

        Args:
            input: The research question (max 40,000 characters).
            research_effort: Depth of research: 'lite', 'standard', 'deep', or
                'exhaustive'. Only used if not configured at tool creation.
            include_domains: Restrict sources to these domains (max 500). Cannot be
                combined with exclude_domains or boost_domains. Only used if not
                configured at tool creation.
            exclude_domains: Block sources from these domains (max 500). Only used if
                not configured at tool creation.
            boost_domains: Boost these domains in source ranking without filtering
                others (max 500). Only used if not configured at tool creation.
            freshness: Filter sources by recency: 'day', 'week', 'month', 'year', or
                'YYYY-MM-DDtoYYYY-MM-DD'. Only used if not configured at tool creation.
            country: ISO 3166-1 alpha-2 country code to geographically focus sources.
                Only used if not configured at tool creation.
        """
        body = self._build_research_body(
            input=input,
            research_effort=research_effort,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            boost_domains=boost_domains,
            freshness=freshness,
            country=country,
        )
        response = await self._post(_YOU_RESEARCH_URL, body, timeout=self._timeout_for(_RESEARCH_TIMEOUT))
        return self._parse_research_result(_RawResearchResponse.model_validate(response.json()))

    async def finance_research(
        self,
        input: ResearchInput,
        *,
        research_effort: FinanceResearchEffort | None = None,
    ) -> YouTextResearchResult:
        """Research a financial question and return a cited, synthesized answer.

        The Finance Research API uses a finance-optimized index to research
        earnings, filings, market data, and financial news. Use it for company
        fundamentals, market trends, competitive analysis, or earnings summaries.

        Args:
            input: The financial research question (max 40,000 characters).
            research_effort: Depth of research: 'deep' or 'exhaustive'. Only used
                if not configured at tool creation.
        """
        body = self._build_finance_research_body(input=input, research_effort=research_effort)
        response = await self._post(_YOU_FINANCE_RESEARCH_URL, body, timeout=self._timeout_for(_RESEARCH_TIMEOUT))
        output = _RawFinanceResearchResponse.model_validate(response.json()).output
        return {
            'content': output.content,
            'content_type': 'text',
            'sources': [source.to_source() for source in output.sources],
        }

    # ------------------------------------------------------------------
    # Parameter building
    # ------------------------------------------------------------------

    def _build_search_params(
        self,
        *,
        query: str,
        count: int | None,
        freshness: Freshness | None,
        country: Country | None,
        language: Language | None,
        safesearch: SafeSearch | None,
        livecrawl: LiveCrawl | None,
        livecrawl_formats: LiveCrawlFormats | None,
        crawl_timeout: CrawlTimeoutSeconds | None,
    ) -> dict[str, str | int | Sequence[str]]:
        """Merge configured search defaults with LLM-provided values (excluding domains).

        Configured values (set at construction) always win. `offset` is always
        included if set, regardless of LLM input. `livecrawl_formats` is passed as a
        list so httpx repeats the query parameter for each entry (GET) or serializes
        it as a JSON array (POST). Domain filters are resolved separately by
        `_resolve_search_domains` because they force the POST path.
        """
        params: dict[str, str | int | Sequence[str]] = {'query': query}

        effective_count = self._count if self._count is not None else count
        if effective_count is not None:
            params['count'] = effective_count
        if self._offset is not None:
            params['offset'] = self._offset

        effective_values: tuple[tuple[str, object | None], ...] = (
            ('freshness', self._freshness if self._freshness is not None else freshness),
            ('country', self._country if self._country is not None else country),
            ('language', self._language if self._language is not None else language),
            ('safesearch', self._safesearch if self._safesearch is not None else safesearch),
            ('livecrawl', self._livecrawl if self._livecrawl is not None else livecrawl),
        )
        for key, value in effective_values:
            normalized = self._normalize_param(value)
            if normalized is not None:
                params[key] = normalized

        # livecrawl_formats is a list -- pass directly so httpx repeats the param.
        effective_formats = self._livecrawl_formats if self._livecrawl_formats is not None else livecrawl_formats
        if effective_formats is not None:
            params['livecrawl_formats'] = effective_formats

        effective_timeout = self._search_crawl_timeout if self._search_crawl_timeout is not None else crawl_timeout
        if effective_timeout is not None:
            params['crawl_timeout'] = effective_timeout

        return params

    def _resolve_search_domains(
        self,
        *,
        include_domains: Domains | None,
        exclude_domains: Domains | None,
        boost_domains: Domains | None,
    ) -> dict[str, Sequence[str]]:
        """Resolve effective search domain filters, rejecting combinations the API 422s on."""
        effective_include = self._include_domains if self._include_domains is not None else include_domains
        effective_exclude = self._exclude_domains if self._exclude_domains is not None else exclude_domains
        effective_boost = self._boost_domains if self._boost_domains is not None else boost_domains
        self._check_domain_combo(effective_include, effective_exclude, effective_boost, ModelRetry)

        domains: dict[str, Sequence[str]] = {}
        if effective_include is not None:
            domains['include_domains'] = effective_include
        if effective_exclude is not None:
            domains['exclude_domains'] = effective_exclude
        if effective_boost is not None:
            domains['boost_domains'] = effective_boost
        return domains

    def _build_contents_body(
        self,
        *,
        urls: list[str],
        formats: list[ContentsFormat] | None,
        crawl_timeout: int | None,
    ) -> dict[str, object]:
        """Build the JSON body for a Contents API request.

        `max_age` is configured-only (never from the LLM). `formats` and
        `crawl_timeout` use configured values when set, otherwise LLM values.
        """
        body: dict[str, object] = {'urls': urls}

        effective_formats = self._contents_formats if self._contents_formats is not None else formats
        if effective_formats is not None:
            body['formats'] = effective_formats

        effective_timeout = self._crawl_timeout if self._crawl_timeout is not None else crawl_timeout
        if effective_timeout is not None:
            body['crawl_timeout'] = effective_timeout

        if self._max_age is not None:
            body['max_age'] = self._max_age

        return body

    def _build_research_body(
        self,
        *,
        input: str,
        research_effort: ResearchEffort | None,
        include_domains: Domains | None,
        exclude_domains: Domains | None,
        boost_domains: Domains | None,
        freshness: Freshness | None,
        country: Country | None,
    ) -> dict[str, object]:
        """Build the JSON body for a Research API request.

        Configured values always win over LLM-provided values. A `source_control`
        object is included only when at least one source-control field has a
        non-None effective value. `output_schema` is human-only and never comes
        from the LLM.
        """
        body: dict[str, object] = {'input': input}
        effective_effort = self._research_effort if self._research_effort is not None else research_effort
        if effective_effort is not None:
            body['research_effort'] = effective_effort

        if self._output_schema is not None and effective_effort == 'lite':
            raise ModelRetry(
                "output_schema is not supported with research_effort='lite'; use 'standard', 'deep', or 'exhaustive'."
            )

        effective_include = (
            self._research_include_domains if self._research_include_domains is not None else include_domains
        )
        effective_exclude = (
            self._research_exclude_domains if self._research_exclude_domains is not None else exclude_domains
        )
        effective_boost = self._research_boost_domains if self._research_boost_domains is not None else boost_domains
        self._check_domain_combo(effective_include, effective_exclude, effective_boost, ModelRetry)

        source_control: _ResearchSourceControl = {}
        if effective_include is not None:
            source_control['include_domains'] = effective_include
        if effective_exclude is not None:
            source_control['exclude_domains'] = effective_exclude
        if effective_boost is not None:
            source_control['boost_domains'] = effective_boost
        effective_freshness = self._research_freshness if self._research_freshness is not None else freshness
        if effective_freshness is not None:
            source_control['freshness'] = effective_freshness
        effective_country = self._research_country if self._research_country is not None else country
        if effective_country is not None:
            source_control['country'] = effective_country
        if source_control:
            body['source_control'] = source_control

        if self._output_schema is not None:
            body['output_schema'] = self._output_schema

        return body

    def _build_finance_research_body(
        self,
        *,
        input: str,
        research_effort: FinanceResearchEffort | None,
    ) -> dict[str, object]:
        """Build the JSON body for a Finance Research API request."""
        body: dict[str, object] = {'input': input}
        effective_effort = (
            self._finance_research_effort if self._finance_research_effort is not None else research_effort
        )
        if effective_effort is not None:
            body['research_effort'] = effective_effort
        return body

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _timeout_for(self, default: float) -> float:
        """Return the configured timeout override, or *default* when unset."""
        return self._timeout if self._timeout is not None else default

    @staticmethod
    def _check_domain_combo(
        include_domains: Sequence[str] | None,
        exclude_domains: Sequence[str] | None,
        boost_domains: Sequence[str] | None,
        error_cls: type[Exception],
    ) -> None:
        """Reject `include_domains` combined with `exclude_domains` or `boost_domains`.

        You.com returns 422 for that combination; `exclude_domains` and
        `boost_domains` may be combined with each other.
        """
        if include_domains is not None and (exclude_domains is not None or boost_domains is not None):
            raise error_cls(
                'include_domains cannot be combined with exclude_domains or boost_domains; '
                'use include_domains alone, or combine exclude_domains and boost_domains.'
            )

    async def _get(self, url: str, params: dict[str, str | int | Sequence[str]], *, timeout: float) -> httpx.Response:
        """Execute a GET request with the API key header."""
        return await self._request('GET', url, params=params, json_body=None, timeout=timeout)

    async def _post(self, url: str, json_body: dict[str, object], *, timeout: float) -> httpx.Response:
        """Execute a POST request with the API key header."""
        return await self._request('POST', url, params=None, json_body=json_body, timeout=timeout)

    async def _request(
        self,
        method: Literal['GET', 'POST'],
        url: str,
        *,
        params: dict[str, str | int | Sequence[str]] | None,
        json_body: dict[str, object] | None,
        timeout: float,
    ) -> httpx.Response:
        """Execute a request, retrying one rate-limited response with provider guidance."""
        if self._http_client is not None:
            return await self._request_with_client(
                self._http_client,
                method,
                url,
                params=params,
                json_body=json_body,
                timeout=timeout,
            )
        async with httpx.AsyncClient() as client:
            return await self._request_with_client(
                client,
                method,
                url,
                params=params,
                json_body=json_body,
                timeout=timeout,
            )

    async def _request_with_client(
        self,
        client: httpx.AsyncClient,
        method: Literal['GET', 'POST'],
        url: str,
        *,
        params: dict[str, str | int | Sequence[str]] | None,
        json_body: dict[str, object] | None,
        timeout: float,
    ) -> httpx.Response:
        headers = {'X-API-Key': self._api_key}
        for attempt in range(2):
            try:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=timeout,
                )
            except httpx.RequestError as error:
                raise ModelRetry(f'You.com request failed: {error}') from error

            if response.status_code == 429 and attempt == 0:
                delay = self._retry_delay(response, attempt)
                if delay is not None:
                    await anyio.sleep(delay)
                    continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.status_code in (401, 402, 403, 404, 429):
                    raise
                raise ModelRetry(f'You.com request failed: {error}') from error
            return response
        raise AssertionError('rate-limit retry loop exhausted')  # pragma: no cover

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float | None:
        """Return a short `Retry-After` delay or fallback backoff.

        A long provider delay returns `None`, leaving the 429 to propagate instead
        of contacting the provider before that delay expires.
        """
        retry_after = response.headers.get('Retry-After')
        if retry_after is not None:
            try:
                delay = max(float(retry_after), 0.0)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    return None
                delay = max((retry_at - datetime.now(retry_at.tzinfo)).total_seconds(), 0.0)
            return delay if delay <= 60.0 else None
        return min(float(2**attempt), 60.0)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_search_results(self, response: _RawSearchResponse) -> list[YouSearchResult]:
        """Convert the parsed search response into a flat list of search results."""
        results: list[YouSearchResult] = []
        if response.results is None:
            return results
        for item in response.results.web or []:
            results.append(item.to_result())
        for item in response.results.news or []:
            results.append(item.to_result())
        return results

    def _parse_research_result(self, response: _RawResearchResponse) -> YouResearchResult:
        """Convert the parsed research response into a public `YouResearchResult`."""
        output = response.output
        sources = [s.to_source() for s in output.sources]
        if isinstance(output, _RawObjectResearchOutput):
            return {'content': output.content, 'content_type': 'object', 'sources': sources}
        return {'content': output.content, 'content_type': 'text', 'sources': sources}

    @staticmethod
    def _normalize_param(value: object | None) -> str | None:
        """Convert a parameter value to its string form for the API query string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

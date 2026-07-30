"""Nimble toolset -- web search, page extract, and opt-in map/crawl tools."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Concatenate, Literal, ParamSpec, Protocol, TypedDict, TypeVar, cast

import httpx
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

try:
    from nimble_python import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AsyncNimble,
        AuthenticationError,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'nimble_python is required for Nimble capabilities. Install it with: pip install "pydantic-ai-harness[nimble]"'
    ) from _import_error

NIMBLE_MAX_NUM_RESULTS = 100
"""Upper bound for developer-controlled `num_results` (aligned with ExaSearch)."""

NIMBLE_MAX_PAGE_TEXT_CHARS = 50_000
"""Upper bound for `get_page` markdown truncation."""

_CLIENT_SOURCE = 'pydantic-ai'
"""Stable attribution slug sent as `X-Client-Source` on every Nimble request."""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')

SearchDepth = Literal['lite', 'fast', 'deep']
TimeRange = Literal['hour', 'day', 'week', 'month', 'year']
AgentEffort = Literal['low', 'medium', 'high', 'x-high', 'max']


class NimbleSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


def _source_list(sources: Mapping[str, str | None]) -> list[NimbleSource]:
    """Convert a `url -> title` mapping into the metadata `sources` list."""
    return [{'url': url, 'title': title} for url, title in sources.items()]


class NimbleClient(Protocol):
    """The subset of `AsyncNimble` that Nimble capabilities call.

    Pass a real `AsyncNimble` or a test double via `NimbleSearch.client` /
    `NimbleAgent.client`.
    """

    async def search(self, **kwargs: Any) -> Any:
        """Run a Nimble web search."""
        ...  # pragma: no cover

    @property
    def extract(self) -> Any:
        """Extract resource namespace."""
        ...  # pragma: no cover

    async def map(self, **kwargs: Any) -> Any:
        """Discover links on a site."""
        ...  # pragma: no cover

    @property
    def crawl(self) -> Any:
        """Crawl resource namespace."""
        ...  # pragma: no cover

    @property
    def agents(self) -> Any:
        """Web Search Agents / Agent API V2 namespace."""
        ...  # pragma: no cover


GetNimbleClient = Callable[[], NimbleClient]


def _default_client() -> NimbleClient:  # pyright: ignore[reportUnusedFunction]
    """Build an `AsyncNimble` client from `NIMBLE_API_KEY`.

    Used by `NimbleSearch` / `NimbleAgent` when no explicit `client` is passed.
    """
    api_key = os.getenv('NIMBLE_API_KEY')
    if not api_key:
        raise UserError(
            'Nimble capabilities need a Nimble API key: set the NIMBLE_API_KEY environment '
            'variable, or pass a configured client, e.g. '
            'NimbleSearch(client=AsyncNimble(api_key=...)).'
        )
    # AsyncNimble satisfies the runtime surface; keyword-only SDK signatures are
    # stricter than the protocol's `**kwargs` shape used for test doubles.
    return cast(NimbleClient, AsyncNimble(api_key=api_key, client_source=_CLIENT_SOURCE))


async def _aclose_client(client: NimbleClient) -> None:  # pyright: ignore[reportUnusedFunction]
    """Close an owned AsyncNimble-like client if it exposes `close`."""
    close = getattr(client, 'close', None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


@dataclass
class _OwnedClientLifecycle:  # pyright: ignore[reportUnusedClass]
    """Reference-counted factory client shared safely across concurrent agent runs.

    `before_run` retains; `after_run` releases and closes only when the last run ends.
    Explicit `client=` on the capability bypasses this entirely.
    """

    explicit_client: NimbleClient | None
    _owned_client: NimbleClient | None = field(default=None, init=False, repr=False)
    _active_runs: int = field(default=0, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def resolve(self) -> NimbleClient:
        """Return the explicit client, or lazily create a factory-owned one."""
        if self.explicit_client is not None:
            return self.explicit_client
        if self._owned_client is None:
            self._owned_client = _default_client()
        return self._owned_client

    async def retain_for_run(self) -> None:
        """Increment the active-run count, creating the owned client if needed."""
        if self.explicit_client is not None:
            return
        async with self._lock:
            if self._owned_client is None:
                self._owned_client = _default_client()
            self._active_runs += 1

    async def release_after_run(self) -> None:
        """Decrement the active-run count; close the owned client when it hits zero."""
        if self.explicit_client is not None:
            return
        async with self._lock:
            if self._active_runs > 0:
                self._active_runs -= 1
            if self._active_runs == 0 and self._owned_client is not None:
                await _aclose_client(self._owned_client)
                self._owned_client = None


def _page_return_text(url: str, markdown: str, max_text_chars: int) -> str:
    """Format `get_page` output so the entire return value fits `max_text_chars`."""
    prefix = f'URL: {url}\n\n'
    if max_text_chars <= len(prefix):
        return (prefix + markdown)[:max_text_chars]
    available = max_text_chars - len(prefix)
    if len(markdown) <= available:
        return prefix + markdown
    marker = f'\n[... page text truncated at {max_text_chars} characters]'
    if available <= len(marker):
        return (prefix + markdown)[:max_text_chars]
    return prefix + markdown[: available - len(marker)] + marker


def _recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
    """Convert transient Nimble API failures into `ModelRetry`.

    Auth/permission failures (401/403) propagate as configuration errors.
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await fn(self, *args, **kwargs)
        except (AuthenticationError, PermissionDeniedError):
            raise
        except RateLimitError as error:
            raise ModelRetry(f'Nimble rate limit: {error}') from error
        except (APITimeoutError, APIConnectionError, httpx.HTTPError) as error:
            raise ModelRetry(f'Nimble request failed: {error}') from error
        except APIStatusError as error:
            if error.status_code in {401, 403}:
                raise
            raise ModelRetry(f'Nimble request failed: {error}') from error

    return wrapper


def _format_search_result(title: str, url: str, content: str) -> str:
    """Render one search hit as labelled metadata lines plus body."""
    lines = [f'Title: {title or "(untitled)"}', f'URL: {url}']
    if content:
        lines.extend(['', content])
    return '\n'.join(lines)


def _json_dump(payload: Any) -> str:
    """Serialize a JSON-compatible payload for the model."""
    return json.dumps(payload, indent=2, default=str)


class NimbleSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent Nimble-backed web research tools.

    Defaults: `web_search` and `get_page` (extract markdown). Opt-in: `map_site`
    and resumable crawl start/status. For Web Search Agents, use
    [`NimbleAgentToolset`][pydantic_ai_harness.nimble.NimbleAgentToolset].
    """

    def __init__(
        self,
        *,
        get_client: GetNimbleClient,
        num_results: int,
        max_text_chars: int,
        search_depth: SearchDepth,
        time_range: TimeRange | None,
        include_domains: Sequence[str],
        exclude_domains: Sequence[str],
        include_map: bool,
        include_crawl: bool,
    ) -> None:
        super().__init__()
        self._get_client = get_client
        self._num_results = num_results
        self._max_text_chars = max_text_chars
        self._search_depth = search_depth
        self._time_range = time_range
        self._include_domains = list(include_domains) if include_domains else None
        self._exclude_domains = list(exclude_domains) if exclude_domains else None

        self.add_function(self.web_search, name='web_search')
        self.add_function(self.get_page, name='get_page')
        if include_map:
            self.add_function(self.map_site, name='map_site')
        if include_crawl:
            self.add_function(self.crawl_start, name='crawl_start')
            self.add_function(self.crawl_status, name='crawl_status')

    @property
    def _client(self) -> NimbleClient:
        """Resolve the live client (may be replaced after a prior run closes)."""
        return self._get_client()

    @_recoverable
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web with Nimble and return matching pages.

        Args:
            query: The search query. Natural-language questions and keyword queries both work.

        Returns:
            Matching pages with title, URL, and content/description.
        """
        search_kwargs: dict[str, Any] = {
            'query': query,
            'search_depth': self._search_depth,
            'max_results': self._num_results,
        }
        if self._time_range is not None:
            search_kwargs['time_range'] = self._time_range
        if self._include_domains is not None:
            search_kwargs['include_domains'] = self._include_domains
        if self._exclude_domains is not None:
            search_kwargs['exclude_domains'] = self._exclude_domains

        response = await self._client.search(**search_kwargs)
        results = list(response.results)[: self._num_results]
        if not results:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})

        sources = _source_list({result.url: result.title for result in results})
        sections = [
            _format_search_result(
                result.title or '',
                result.url,
                result.content or result.description or '',
            )
            for result in results
        ]
        plural = 's' if len(sections) != 1 else ''
        joined = '\n\n---\n\n'.join(sections)
        return ToolReturn(
            f'Found {len(sections)} result{plural} for {query!r}:\n\n{joined}',
            metadata={'sources': sources},
        )

    @_recoverable
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Extract page content as markdown from a URL.

        Use it to read a promising URL from `web_search` results in full, or a
        URL the user provided.

        Args:
            url: The URL of the page to read.

        Returns:
            The page's URL and markdown content.
        """
        response = await self._client.extract.run(url=url, formats=['markdown'])
        markdown = ''
        if response.data and response.data.markdown:
            markdown = response.data.markdown
        if not markdown:
            raise ModelRetry(f'No content could be retrieved for {url!r}. Check the URL or try another page.')
        return ToolReturn(
            _page_return_text(url, markdown, self._max_text_chars),
            metadata={'sources': _source_list({url: None})},
        )

    @_recoverable
    async def map_site(
        self,
        url: str,
        limit: int | None = None,
        domain_filter: Literal['domain', 'subdomain', 'all'] | None = None,
        sitemap: Literal['skip', 'include', 'only'] | None = None,
    ) -> ToolReturn[str]:
        """Discover links on a website.

        Args:
            url: The website URL to map.
            limit: Maximum number of links to return.
            domain_filter: Scope of domains to include (`domain`, `subdomain`, or `all`).
            sitemap: Sitemap handling strategy (`skip`, `include`, or `only`).

        Returns:
            Discovered links with optional titles and descriptions.
        """
        map_kwargs: dict[str, Any] = {'url': url}
        if limit is not None:
            map_kwargs['limit'] = limit
        if domain_filter is not None:
            map_kwargs['domain_filter'] = domain_filter
        if sitemap is not None:
            map_kwargs['sitemap'] = sitemap
        response = await self._client.map(**map_kwargs)
        links = list(response.links)
        if not links:
            return ToolReturn(f'No links found for {url!r}.', metadata={'sources': []})
        sources = _source_list({link.url: link.title for link in links})
        lines = [f'Found {len(links)} link(s) for {url!r}:', '']
        for link in links:
            title = link.title or '(untitled)'
            desc = f' — {link.description}' if link.description else ''
            lines.append(f'- {title}: {link.url}{desc}')
        return ToolReturn('\n'.join(lines), metadata={'sources': sources})

    @_recoverable
    async def crawl_start(
        self,
        url: str,
        limit: int | None = None,
        max_discovery_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        sitemap: Literal['skip', 'include', 'only'] | None = None,
        name: str | None = None,
    ) -> ToolReturn[str]:
        """Start a Nimble crawl job and return its id immediately (does not wait for completion).

        Args:
            url: The URL to start crawling from.
            limit: Maximum number of pages to crawl.
            max_discovery_depth: Maximum link-following depth from the start URL.
            include_paths: URL path patterns to include.
            exclude_paths: URL path patterns to exclude.
            sitemap: Sitemap handling strategy.
            name: Optional name for the crawl job.

        Returns:
            Crawl job snapshot including `crawl_id` for later `crawl_status` calls.
        """
        crawl_kwargs: dict[str, Any] = {'url': url}
        if limit is not None:
            crawl_kwargs['limit'] = limit
        if max_discovery_depth is not None:
            crawl_kwargs['max_discovery_depth'] = max_discovery_depth
        if include_paths is not None:
            crawl_kwargs['include_paths'] = include_paths
        if exclude_paths is not None:
            crawl_kwargs['exclude_paths'] = exclude_paths
        if sitemap is not None:
            crawl_kwargs['sitemap'] = sitemap
        if name is not None:
            crawl_kwargs['name'] = name
        response = await self._client.crawl.run(**crawl_kwargs)
        payload = {
            'crawl_id': response.crawl_id,
            'status': response.status,
            'url': response.url,
            'completed': response.completed,
            'failed': response.failed,
            'pending': response.pending,
            'total': response.total,
        }
        return ToolReturn(
            f'Started crawl {response.crawl_id} ({response.status}). Use crawl_status to poll.\n\n{_json_dump(payload)}',
            metadata={'crawl_id': response.crawl_id, 'sources': _source_list({response.url: None})},
        )

    @_recoverable
    async def crawl_status(self, crawl_id: str) -> ToolReturn[str]:
        """Get the status of a Nimble crawl job.

        Args:
            crawl_id: The crawl job id returned by `crawl_start`.

        Returns:
            The latest crawl job snapshot.
        """
        response = await self._client.crawl.status(crawl_id)
        payload = {
            'crawl_id': response.crawl_id,
            'status': response.status,
            'url': response.url,
            'completed': response.completed,
            'failed': response.failed,
            'pending': response.pending,
            'total': response.total,
        }
        return ToolReturn(
            f'Crawl {response.crawl_id}: {response.status}\n\n{_json_dump(payload)}',
            metadata={'crawl_id': response.crawl_id, 'sources': _source_list({response.url: None})},
        )

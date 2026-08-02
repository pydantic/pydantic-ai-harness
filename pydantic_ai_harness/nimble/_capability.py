"""Nimble search capability that gives an agent web research tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.nimble._toolset import (
    NIMBLE_MAX_NUM_RESULTS,
    NIMBLE_MAX_PAGE_TEXT_CHARS,
    NimbleClient,
    NimbleSearchToolset,
    SearchDepth,
    TimeRange,
    _OwnedClientLifecycle,  # pyright: ignore[reportPrivateUsage]
    _VALID_SEARCH_DEPTHS,  # pyright: ignore[reportPrivateUsage]
    _VALID_TIME_RANGES,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    'You have web research tools backed by the Nimble search API. Start broad: use `web_search` '
    'to survey several sources on a topic, then use `get_page` to read the most promising '
    'URLs in full before drawing conclusions. Prefer primary sources, and cite the URLs of '
    'the pages you relied on in your answer.'
)

_MAP_INSTRUCTIONS_SUFFIX = (
    ' Use `map_site` to discover URLs on a website before targeted extraction or crawling.'
)

_CRAWL_INSTRUCTIONS_SUFFIX = (
    ' For multi-page jobs, call `crawl_start` then `crawl_status` across turns - do not wait '
    'inside a single tool call for a crawl to finish.'
)


@dataclass
class NimbleSearch(AbstractCapability[AgentDepsT]):
    """Web research for agents, backed by the [Nimble](https://www.nimbleway.com/) API.

    Adds two tools by default: `web_search`, which returns search results with
    content/description, and `get_page`, which extracts a URL as markdown.
    Opt in to site mapping and resumable crawl jobs via `include_map` and
    `include_crawl`. For Nimble Web Search Agents (Agent API V2), use the
    separate [`NimbleAgent`][pydantic_ai_harness.nimble.NimbleAgent] capability.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.nimble import NimbleSearch

    agent = Agent('openai:gpt-5.2', capabilities=[NimbleSearch()])
    ```

    Authentication comes from the `NIMBLE_API_KEY` environment variable by
    default; pass `client` to configure it explicitly. When composing with
    [`NimbleAgent`][pydantic_ai_harness.nimble.NimbleAgent], pass the same
    `client=` to both capabilities to share one HTTP session.
    """

    num_results: int = 5
    """Number of results `web_search` returns per query (1 to 100)."""

    max_text_chars: int = 10_000
    """Maximum characters of markdown `get_page` returns (1 to 50,000)."""

    search_depth: SearchDepth = 'lite'
    """Nimble search depth for every `web_search` call (`lite`, `fast`, or `deep`)."""

    time_range: TimeRange | None = None
    """Optional time filter applied to every `web_search` call."""

    include_domains: Sequence[str] = field(default_factory=list[str])
    """If non-empty, search results only come from these domains (allowlist).

    Mutually exclusive with `exclude_domains`.
    """

    exclude_domains: Sequence[str] = field(default_factory=list[str])
    """Search results never come from these domains (denylist).

    Mutually exclusive with `include_domains`.
    """

    include_map: bool = False
    """Also expose the `map_site` tool. Off by default."""

    include_crawl: bool = False
    """Also expose resumable `crawl_start` / `crawl_status` tools. Off by default."""

    guidance: str | None = None
    """Custom research guidance for the system prompt.

    Leave as `None` for the default guidance (which adapts to opt-in tools), or
    set `''` to contribute no instructions at all.
    """

    client: NimbleClient | None = None
    """Nimble client to use; when `None`, an `AsyncNimble` is built from `NIMBLE_API_KEY`.

    Any object satisfying the `NimbleClient` protocol works: use it to pass an API
    key explicitly or substitute a fake in tests. Factory-built clients send
    `X-Client-Source: pydantic-ai` and are closed when the last concurrent run ends
    (including failed or cancelled runs).
    """

    _client_lifecycle: _OwnedClientLifecycle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration bounds and domain filter exclusivity."""
        if not 1 <= self.num_results <= NIMBLE_MAX_NUM_RESULTS:
            raise ValueError(f'num_results must be between 1 and {NIMBLE_MAX_NUM_RESULTS}, got {self.num_results}')
        if not 1 <= self.max_text_chars <= NIMBLE_MAX_PAGE_TEXT_CHARS:
            raise ValueError(
                f'max_text_chars must be between 1 and {NIMBLE_MAX_PAGE_TEXT_CHARS}, got {self.max_text_chars}'
            )
        if self.search_depth not in _VALID_SEARCH_DEPTHS:
            raise ValueError(f'search_depth must be lite, fast, or deep, got {self.search_depth!r}')
        if self.time_range is not None and self.time_range not in _VALID_TIME_RANGES:
            raise ValueError(
                f'time_range must be one of {sorted(_VALID_TIME_RANGES)}, got {self.time_range!r}'
            )
        if isinstance(self.include_domains, str) or isinstance(self.exclude_domains, str):
            raise ValueError('include_domains and exclude_domains must be a sequence of strings, not a single str')
        if self.include_domains and self.exclude_domains:
            raise ValueError('Specify include_domains or exclude_domains, not both.')
        self._client_lifecycle = _OwnedClientLifecycle(explicit_client=self.client)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static research guidance adapted to which opt-in tools are enabled."""
        if self.guidance is not None:
            return self.guidance or None
        instructions = _INSTRUCTIONS
        if self.include_map:
            instructions += _MAP_INSTRUCTIONS_SUFFIX
        if self.include_crawl:
            instructions += _CRAWL_INSTRUCTIONS_SUFFIX
        return instructions

    def get_toolset(self) -> NimbleSearchToolset[AgentDepsT]:
        """Build the toolset providing Nimble research tools."""
        self._client_lifecycle.ensure_configured()
        return NimbleSearchToolset[AgentDepsT](
            get_client=self._client_lifecycle.resolve,
            num_results=self.num_results,
            max_text_chars=self.max_text_chars,
            search_depth=self.search_depth,
            time_range=self.time_range,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            include_map=self.include_map,
            include_crawl=self.include_crawl,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Retain a factory client for the run and always release it afterward."""
        await self._client_lifecycle.retain_for_run()
        try:
            return await handler()
        finally:
            await self._client_lifecycle.release_after_run()

    @classmethod
    def from_spec(
        cls,
        *,
        num_results: int = 5,
        max_text_chars: int = 10_000,
        search_depth: Literal['lite', 'fast', 'deep'] = 'lite',
        time_range: TimeRange | None = None,
        include_domains: Sequence[str] = (),
        exclude_domains: Sequence[str] = (),
        include_map: bool = False,
        include_crawl: bool = False,
        guidance: str | None = None,
    ) -> NimbleSearch[AgentDepsT]:
        """Construct the capability from serializable spec options.

        The `client` field is not spec-serializable, so spec-loaded instances
        always build the default `AsyncNimble` from `NIMBLE_API_KEY`.
        """
        return cls(
            num_results=num_results,
            max_text_chars=max_text_chars,
            search_depth=search_depth,
            time_range=time_range,
            include_domains=list(include_domains),
            exclude_domains=list(exclude_domains),
            include_map=include_map,
            include_crawl=include_crawl,
            guidance=guidance,
        )

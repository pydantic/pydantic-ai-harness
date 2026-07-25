"""Tests for the Youdotcom capability and YoudotcomToolset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.youdotcom import Youdotcom, YoudotcomToolset

T = TypeVar('T')


def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    """Build a `RunContext` for invoking toolsets directly in tests."""
    return RunContext[T](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
        pending_messages=[],
    )


class _CapturedRequest:
    """Attributes of an outgoing httpx request captured for assertions."""

    def __init__(self) -> None:
        self.method: str = ''
        self.params: dict[str, str] = {}
        self.param_items: list[tuple[str, str]] = []
        self.body: dict[str, object] | None = None


def _search_capture() -> tuple[httpx.AsyncClient, _CapturedRequest]:
    """Return a client whose mock transport records the outgoing search request."""
    cap = _CapturedRequest()

    def handler(request: httpx.Request) -> httpx.Response:
        cap.method = request.method
        cap.param_items = list(request.url.params.multi_items())
        cap.params = dict(cap.param_items)
        if request.content:
            cap.body = json.loads(request.content)
        return httpx.Response(200, json=_make_empty_search_payload())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), cap


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _make_web_payload() -> dict[str, object]:
    """Build a minimal search API response with one web result."""
    return {
        'results': {
            'web': [
                {
                    'title': 'Example Page',
                    'url': 'https://example.com',
                    'description': 'An example page.',
                    'snippets': ['snippet one', 'snippet two'],
                    'thumbnail_url': 'https://example.com/thumb.png',
                    'favicon_url': 'https://example.com/favicon.ico',
                    'authors': ['Jane Doe'],
                    'page_age': '2025-01-15T10:30:00Z',
                }
            ],
            'news': [],
        }
    }


def _make_news_payload() -> dict[str, object]:
    """Build a minimal search API response with one news result."""
    return {
        'results': {
            'web': [],
            'news': [
                {
                    'title': 'Breaking News',
                    'url': 'https://news.example.com/story',
                    'description': 'Something happened.',
                    'page_age': '2025-06-01T12:00:00Z',
                }
            ],
        }
    }


def _make_livecrawl_payload() -> dict[str, object]:
    """Build a search API response with livecrawled content."""
    return {
        'results': {
            'web': [
                {
                    'title': 'Live Page',
                    'url': 'https://live.example.com',
                    'contents': {
                        'html': '<p>Hello</p>',
                        'markdown': 'Hello',
                    },
                }
            ],
            'news': [],
        }
    }


def _make_empty_search_payload() -> dict[str, object]:
    """Build an empty search API response."""
    return {'results': {'web': [], 'news': []}}


def _make_malformed_search_payload() -> dict[str, object]:
    """Build a search payload missing the results key entirely."""
    return {'unrelated': 'data'}


def _make_contents_payload() -> list[dict[str, object]]:
    """Build a Contents API response with one result."""
    return [
        {
            'url': 'https://example.com/page',
            'title': 'Example Page',
            'markdown': '# Example\n\nHello world.',
            'html': '<h1>Example</h1><p>Hello world.</p>',
            'metadata': {
                'site_name': 'Example',
                'favicon_url': 'https://example.com/favicon.ico',
            },
        }
    ]


def _make_contents_minimal_payload() -> list[dict[str, object]]:
    """Build a Contents API response with only required fields."""
    return [{'url': 'https://example.com', 'title': 'Minimal'}]


def _make_contents_partial_payload() -> list[dict[str, object]]:
    """Build a Contents API response where one URL failed (null content)."""
    return [
        {'url': 'https://ok.com', 'title': 'OK', 'markdown': 'content'},
        {'url': 'https://fail.com', 'title': 'Fail', 'html': None, 'markdown': None},
    ]


def _make_research_payload() -> dict[str, object]:
    """Build a Research API response."""
    return {
        'output': {
            'content': '## Answer\n\nSomething happened [[1, 2]].',
            'content_type': 'text',
            'sources': [
                {
                    'url': 'https://source1.com',
                    'title': 'Source 1',
                    'snippets': ['relevant excerpt'],
                },
                {
                    'url': 'https://source2.com',
                    'title': 'Source 2',
                },
            ],
        }
    }


def _make_research_minimal_payload() -> dict[str, object]:
    """Build a Research API response with minimal fields."""
    return {
        'output': {
            'content': 'Short answer.',
            'content_type': 'text',
            'sources': [{'url': 'https://src.com'}],
        }
    }


def _make_research_empty_payload() -> dict[str, object]:
    """Build a Research API response with no sources."""
    return {'output': {'content': 'No sources needed.', 'content_type': 'text', 'sources': []}}


def _make_finance_research_payload() -> dict[str, object]:
    """Build a Finance Research API response."""
    return {
        'output': {
            'content': 'Revenue grew 114% [[1]].',
            'content_type': 'text',
            'sources': [
                {
                    'url': 'https://sec.gov/filing',
                    'title': '10-K Filing',
                    'snippets': ['Total revenue $130.5B'],
                }
            ],
        }
    }


def _make_malformed_research_payload() -> dict[str, object]:
    """Build a research payload missing the output key entirely."""
    return {'error': 'something went wrong'}


def _make_research_structured_payload() -> dict[str, object]:
    """Build a Research API response with structured (object) output."""
    return {
        'output': {
            'content': {'answer': 'The sky is blue.', 'confidence': 0.95},
            'content_type': 'object',
            'sources': [{'url': 'https://source.com', 'title': 'Source'}],
        }
    }


# ---------------------------------------------------------------------------
# Search: result field integration tests
# ---------------------------------------------------------------------------


class TestSearchResultFields:
    """Exercise search result field mapping through the public search() tool."""

    @staticmethod
    def _toolset_with_payload(payload: dict[str, object]) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
        """Create a toolset and its mock client backed by *payload*."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        client = httpx.AsyncClient(transport=transport)
        return YoudotcomToolset(api_key='test', http_client=client), client

    async def test_web_result_all_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_web_payload())
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            r = results[0]
            assert r.get('title') == 'Example Page'
            assert r.get('url') == 'https://example.com'
            assert r.get('description') == 'An example page.'
            assert r.get('snippets') == ['snippet one', 'snippet two']
            assert r.get('thumbnail_url') == 'https://example.com/thumb.png'
            assert r.get('favicon_url') == 'https://example.com/favicon.ico'
            assert r.get('authors') == ['Jane Doe']
            assert r.get('page_age') == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
        finally:
            await client.aclose()

    async def test_web_result_minimal_fields(self) -> None:
        payload: dict[str, object] = {'results': {'web': [{'title': 'T', 'url': 'https://x.com'}], 'news': []}}
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            assert results[0].get('title') == 'T'
            assert results[0].get('url') == 'https://x.com'
            assert 'description' not in results[0]
            assert 'snippets' not in results[0]
            assert 'thumbnail_url' not in results[0]
            assert 'favicon_url' not in results[0]
            assert 'authors' not in results[0]
            assert 'page_age' not in results[0]
            assert 'contents' not in results[0]
        finally:
            await client.aclose()

    async def test_web_result_empty_description_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': 'T', 'url': 'https://x.com', 'description': ''}], 'news': []}
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert 'description' not in results[0]
        finally:
            await client.aclose()

    async def test_web_result_empty_snippets_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': 'T', 'url': 'https://x.com', 'snippets': []}], 'news': []}
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert 'snippets' not in results[0]
        finally:
            await client.aclose()

    async def test_news_result_no_web_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_news_payload())
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            assert results[0].get('title') == 'Breaking News'
            assert 'snippets' not in results[0]
            assert 'favicon_url' not in results[0]
            assert 'authors' not in results[0]
        finally:
            await client.aclose()

    async def test_livecrawl_both_formats(self) -> None:
        toolset, client = self._toolset_with_payload(_make_livecrawl_payload())
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            assert results[0].get('contents') == {'html': '<p>Hello</p>', 'markdown': 'Hello'}
        finally:
            await client.aclose()

    async def test_livecrawl_html_only(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'html': '<p>Hi</p>'}}],
                'news': [],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert results[0].get('contents') == {'html': '<p>Hi</p>'}
            assert 'markdown' not in results[0].get('contents', {})
        finally:
            await client.aclose()

    async def test_livecrawl_markdown_only(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'markdown': 'Hi'}}],
                'news': [],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert results[0].get('contents') == {'markdown': 'Hi'}
            assert 'html' not in results[0].get('contents', {})
        finally:
            await client.aclose()

    async def test_livecrawl_empty_strings_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'html': '', 'markdown': ''}}],
                'news': [],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert 'contents' not in results[0]
        finally:
            await client.aclose()

    async def test_both_web_and_news(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'W', 'url': 'https://w.com'}],
                'news': [{'title': 'N', 'url': 'https://n.com'}],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert len(results) == 2
            assert results[0].get('title') == 'W'
            assert results[1].get('title') == 'N'
        finally:
            await client.aclose()

    async def test_empty_results(self) -> None:
        toolset, client = self._toolset_with_payload(_make_empty_search_payload())
        try:
            results = await toolset.search('q')
            assert results == []
        finally:
            await client.aclose()

    async def test_malformed_payload_raises(self) -> None:
        """Missing results key raises ValidationError instead of returning empty list."""
        toolset, client = self._toolset_with_payload(_make_malformed_search_payload())
        try:
            with pytest.raises(ValidationError):
                await toolset.search('q')
        finally:
            await client.aclose()

    async def test_partial_item_is_preserved(self) -> None:
        """Provider-valid search items may omit every optional field."""
        toolset, client = self._toolset_with_payload({'results': {'web': [{}]}})
        try:
            assert await toolset.search('q') == [{}]
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Contents: result field integration tests
# ---------------------------------------------------------------------------


class TestContentsResultFields:
    """Exercise contents result field mapping through the public extract_contents() tool."""

    @staticmethod
    def _toolset_with_payload(payload: list[dict[str, object]]) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
        """Create a toolset and its mock client backed by *payload*."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        client = httpx.AsyncClient(transport=transport)
        return YoudotcomToolset(api_key='test', http_client=client), client

    async def test_all_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_contents_payload())
        try:
            results = await toolset.extract_contents(['https://example.com/page'])
            assert len(results) == 1
            r = results[0]
            assert r.get('url') == 'https://example.com/page'
            assert r.get('title') == 'Example Page'
            assert r.get('html') == '<h1>Example</h1><p>Hello world.</p>'
            assert r.get('markdown') == '# Example\n\nHello world.'
            assert r.get('metadata') == {
                'site_name': 'Example',
                'favicon_url': 'https://example.com/favicon.ico',
            }
        finally:
            await client.aclose()

    async def test_minimal_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_contents_minimal_payload())
        try:
            results = await toolset.extract_contents(['https://example.com'])
            assert len(results) == 1
            assert results[0].get('url') == 'https://example.com'
            assert results[0].get('title') == 'Minimal'
            assert 'html' not in results[0]
            assert 'markdown' not in results[0]
            assert 'metadata' not in results[0]
        finally:
            await client.aclose()

    async def test_partial_item_is_preserved(self) -> None:
        """Provider-valid contents items may omit every optional field."""
        toolset, client = self._toolset_with_payload([{}])
        try:
            assert await toolset.extract_contents(['https://example.com']) == [{}]
        finally:
            await client.aclose()

    async def test_empty_html_not_included(self) -> None:
        payload: list[dict[str, object]] = [{'url': 'https://x.com', 'title': 'X', 'html': ''}]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert 'html' not in results[0]
        finally:
            await client.aclose()

    async def test_empty_markdown_not_included(self) -> None:
        payload: list[dict[str, object]] = [{'url': 'https://x.com', 'title': 'X', 'markdown': ''}]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert 'markdown' not in results[0]
        finally:
            await client.aclose()

    async def test_metadata_only_site_name(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'site_name': 'Example'}}
        ]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert results[0].get('metadata') == {'site_name': 'Example'}
            assert 'favicon_url' not in results[0].get('metadata', {})
        finally:
            await client.aclose()

    async def test_metadata_only_favicon(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'favicon_url': 'https://x.com/fav.ico'}}
        ]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert results[0].get('metadata') == {'favicon_url': 'https://x.com/fav.ico'}
            assert 'site_name' not in results[0].get('metadata', {})
        finally:
            await client.aclose()

    async def test_empty_metadata_not_included(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'site_name': '', 'favicon_url': ''}}
        ]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert 'metadata' not in results[0]
        finally:
            await client.aclose()

    async def test_empty_results(self) -> None:
        toolset, client = self._toolset_with_payload([])
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert results == []
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Research: result field integration tests
# ---------------------------------------------------------------------------


class TestResearchResultFields:
    """Exercise research result field mapping through the public research() tool."""

    @staticmethod
    def _toolset_with_payload(payload: dict[str, object]) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
        """Create a toolset and its mock client backed by *payload*."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        client = httpx.AsyncClient(transport=transport)
        return YoudotcomToolset(api_key='test', http_client=client), client

    async def test_full_response(self) -> None:
        toolset, client = self._toolset_with_payload(_make_research_payload())
        try:
            result = await toolset.research('What happened?')
            assert result['content'] == '## Answer\n\nSomething happened [[1, 2]].'
            assert result['content_type'] == 'text'
            assert len(result['sources']) == 2
            assert result['sources'][0]['url'] == 'https://source1.com'
            assert result['sources'][0].get('title') == 'Source 1'
            assert result['sources'][0].get('snippets') == ['relevant excerpt']
            assert result['sources'][1]['url'] == 'https://source2.com'
            assert result['sources'][1].get('title') == 'Source 2'
            assert 'snippets' not in result['sources'][1]
        finally:
            await client.aclose()

    async def test_source_empty_title_not_included(self) -> None:
        payload: dict[str, object] = {
            'output': {
                'content': 'A',
                'content_type': 'text',
                'sources': [{'url': 'https://x.com', 'title': ''}],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            result = await toolset.research('q')
            assert 'title' not in result['sources'][0]
        finally:
            await client.aclose()

    async def test_source_empty_snippets_not_included(self) -> None:
        payload: dict[str, object] = {
            'output': {
                'content': 'A',
                'content_type': 'text',
                'sources': [{'url': 'https://x.com', 'snippets': []}],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            result = await toolset.research('q')
            assert 'snippets' not in result['sources'][0]
        finally:
            await client.aclose()

    async def test_minimal_response(self) -> None:
        toolset, client = self._toolset_with_payload(_make_research_minimal_payload())
        try:
            result = await toolset.research('q')
            assert result['content'] == 'Short answer.'
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://src.com'
            assert 'title' not in result['sources'][0]
            assert 'snippets' not in result['sources'][0]
        finally:
            await client.aclose()

    async def test_empty_sources(self) -> None:
        toolset, client = self._toolset_with_payload(_make_research_empty_payload())
        try:
            result = await toolset.research('q')
            assert result['sources'] == []
        finally:
            await client.aclose()

    async def test_structured_output(self) -> None:
        """Structured output returns content as a dict with content_type 'object'."""
        toolset, client = self._toolset_with_payload(_make_research_structured_payload())
        try:
            result = await toolset.research('q')
            assert result['content_type'] == 'object'
            assert result['content'] == {'answer': 'The sky is blue.', 'confidence': 0.95}
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://source.com'
        finally:
            await client.aclose()

    async def test_malformed_payload_raises(self) -> None:
        """Missing output key raises ValidationError."""
        toolset, client = self._toolset_with_payload(_make_malformed_research_payload())
        try:
            with pytest.raises(ValidationError):
                await toolset.research('q')
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Search: parameter building tests
# ---------------------------------------------------------------------------


class TestSearchRequest:
    """Search parameter locking and GET/POST selection, exercised through `search()`."""

    async def test_query_only_uses_get(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('hello')
            assert cap.method == 'GET'
            assert cap.params['query'] == 'hello'
            assert cap.body is None
        finally:
            await client.aclose()

    async def test_configured_count_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, count=5)
        try:
            await toolset.search('q', count=8)
            assert cap.params['count'] == '5'
        finally:
            await client.aclose()

    async def test_llm_count_when_not_configured(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', count=8)
            assert cap.params['count'] == '8'
        finally:
            await client.aclose()

    async def test_offset_included_when_configured(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, offset=5)
        try:
            await toolset.search('q')
            assert cap.params['offset'] == '5'
        finally:
            await client.aclose()

    async def test_offset_absent_by_default(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q')
            assert 'offset' not in cap.params
        finally:
            await client.aclose()

    async def test_configured_freshness_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, freshness='day')
        try:
            await toolset.search('q', freshness='week')
            assert cap.params['freshness'] == 'day'
        finally:
            await client.aclose()

    async def test_llm_freshness_when_not_configured(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', freshness='week')
            assert cap.params['freshness'] == 'week'
        finally:
            await client.aclose()

    async def test_llm_freshness_accepts_date_range(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', freshness='2024-01-01to2024-01-31')
            assert cap.params['freshness'] == '2024-01-01to2024-01-31'
        finally:
            await client.aclose()

    async def test_configured_country_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, country='US')
        try:
            await toolset.search('q', country='GB')
            assert cap.params['country'] == 'US'
        finally:
            await client.aclose()

    async def test_llm_country_when_not_configured(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', country='GB')
            assert cap.params['country'] == 'GB'
        finally:
            await client.aclose()

    async def test_configured_language_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, language='JA')
        try:
            await toolset.search('q', language='EN')
            assert cap.params['language'] == 'JA'
        finally:
            await client.aclose()

    async def test_configured_safesearch_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, safesearch='strict')
        try:
            await toolset.search('q', safesearch='off')
            assert cap.params['safesearch'] == 'strict'
        finally:
            await client.aclose()

    async def test_configured_livecrawl_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, livecrawl='all')
        try:
            await toolset.search('q', livecrawl='web')
            assert cap.params['livecrawl'] == 'all'
        finally:
            await client.aclose()

    async def test_livecrawl_formats_sent_as_repeated_params(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', livecrawl_formats=['html', 'markdown'])
            formats = [v for k, v in cap.param_items if k == 'livecrawl_formats']
            assert formats == ['html', 'markdown']
        finally:
            await client.aclose()

    async def test_configured_crawl_timeout_locks(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, search_crawl_timeout=30)
        try:
            await toolset.search('q', crawl_timeout=10)
            assert cap.params['crawl_timeout'] == '30'
        finally:
            await client.aclose()

    async def test_llm_crawl_timeout_when_not_configured(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', crawl_timeout=10)
            assert cap.params['crawl_timeout'] == '10'
        finally:
            await client.aclose()

    async def test_no_crawl_timeout_when_absent(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q')
            assert 'crawl_timeout' not in cap.params
        finally:
            await client.aclose()

    async def test_domain_filter_uses_post_with_json_arrays(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', include_domains=['nytimes.com', 'bbc.com'])
            assert cap.method == 'POST'
            assert cap.body is not None
            assert cap.body['query'] == 'q'
            assert cap.body['include_domains'] == ['nytimes.com', 'bbc.com']
        finally:
            await client.aclose()

    async def test_configured_domains_lock_and_use_post(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client, include_domains=['arxiv.org'])
        try:
            await toolset.search('q', include_domains=['bbc.com'])
            assert cap.method == 'POST'
            assert cap.body is not None
            assert cap.body['include_domains'] == ['arxiv.org']
        finally:
            await client.aclose()

    async def test_exclude_and_boost_combine(self) -> None:
        client, cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', exclude_domains=['spam.com'], boost_domains=['good.com'])
            assert cap.method == 'POST'
            assert cap.body is not None
            assert cap.body['exclude_domains'] == ['spam.com']
            assert cap.body['boost_domains'] == ['good.com']
        finally:
            await client.aclose()

    async def test_include_with_exclude_rejected(self) -> None:
        client, _cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            with pytest.raises(ModelRetry, match='include_domains cannot be combined'):
                await toolset.search('q', include_domains=['a.com'], exclude_domains=['b.com'])
        finally:
            await client.aclose()

    async def test_include_with_boost_rejected(self) -> None:
        client, _cap = _search_capture()
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            with pytest.raises(ModelRetry, match='include_domains cannot be combined'):
                await toolset.search('q', include_domains=['a.com'], boost_domains=['c.com'])
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Contents: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildContentsBody:
    def test_urls_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert body['urls'] == ['https://x.com']
        assert len(body) == 1

    def test_configured_formats_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', contents_formats=['markdown'])
        body = toolset._build_contents_body(urls=['https://x.com'], formats=['html', 'metadata'], crawl_timeout=None)
        assert body['formats'] == ['markdown']

    def test_llm_formats_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=['html'], crawl_timeout=None)
        assert body['formats'] == ['html']

    def test_configured_crawl_timeout_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', crawl_timeout=30)
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=10)
        assert body['crawl_timeout'] == 30

    def test_llm_crawl_timeout_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=15)
        assert body['crawl_timeout'] == 15

    def test_max_age_always_included_when_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test', max_age=86400)
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert body['max_age'] == 86400

    def test_max_age_never_from_llm(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert 'max_age' not in body

    def test_all_params_configured(self) -> None:
        toolset = YoudotcomToolset(
            api_key='test',
            contents_formats=['html', 'markdown'],
            crawl_timeout=20,
            max_age=3600,
        )
        body = toolset._build_contents_body(urls=['https://x.com'], formats=['metadata'], crawl_timeout=5)
        assert body == {
            'urls': ['https://x.com'],
            'formats': ['html', 'markdown'],
            'crawl_timeout': 20,
            'max_age': 3600,
        }


# ---------------------------------------------------------------------------
# Contents: integration tests
# ---------------------------------------------------------------------------


class TestContentsIntegration:
    async def test_contents_returns_results(self) -> None:
        """End-to-end contents extraction with a mock transport."""
        payload = _make_contents_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'POST'
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            results = await toolset.extract_contents(['https://example.com/page'])
            assert len(results) == 1
            assert results[0].get('url') == 'https://example.com/page'
            assert results[0].get('title') == 'Example Page'
            assert results[0].get('markdown') == '# Example\n\nHello world.'
        finally:
            await client.aclose()

    async def test_contents_with_configured_formats(self) -> None:
        """Configured formats are sent, not LLM-provided ones."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_contents_minimal_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, contents_formats=['markdown'])
        try:
            await toolset.extract_contents(['https://x.com'], formats=['html'])
            assert captured_body['formats'] == ['markdown']
        finally:
            await client.aclose()

    async def test_contents_partial_failure(self) -> None:
        """URLs that fail to crawl return with no content fields."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_contents_partial_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            results = await toolset.extract_contents(['https://ok.com', 'https://fail.com'])
            assert len(results) == 2
            assert results[0].get('markdown') == 'content'
            assert 'markdown' not in results[1]
            assert 'html' not in results[1]
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Research: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildResearchBody:
    def test_input_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='question',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['input'] == 'question'
        assert len(body) == 1

    def test_configured_effort_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_effort='deep')
        body = toolset._build_research_body(
            input='q',
            research_effort='lite',
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['research_effort'] == 'deep'

    def test_llm_effort_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort='exhaustive',
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['research_effort'] == 'exhaustive'

    def test_source_control_not_included_when_none(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert 'source_control' not in body

    def test_configured_research_include_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_include_domains=['arxiv.org'])
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=['bbc.com'],
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'include_domains': ['arxiv.org']}

    def test_llm_research_include_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=['arxiv.org', 'nature.com'],
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'include_domains': ['arxiv.org', 'nature.com']}

    def test_configured_research_exclude_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_exclude_domains=['spam.com'])
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=['other.com'],
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'exclude_domains': ['spam.com']}

    def test_llm_research_exclude_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=['spam.com'],
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'exclude_domains': ['spam.com']}

    def test_configured_research_boost_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_boost_domains=['good.com'])
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=['other.com'],
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'boost_domains': ['good.com']}

    def test_llm_research_boost_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=['good.com'],
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'boost_domains': ['good.com']}

    def test_configured_research_freshness_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_freshness='day')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness='week',
            country=None,
        )
        assert body['source_control'] == {'freshness': 'day'}

    def test_llm_research_freshness_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness='month',
            country=None,
        )
        assert body['source_control'] == {'freshness': 'month'}

    def test_configured_research_country_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_country='US')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country='GB',
        )
        assert body['source_control'] == {'country': 'US'}

    def test_llm_research_country_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country='GB',
        )
        assert body['source_control'] == {'country': 'GB'}

    def test_source_control_only_includes_set_fields(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness='day',
            country='US',
        )
        assert body['source_control'] == {'freshness': 'day', 'country': 'US'}

    def test_output_schema_included_when_configured(self) -> None:
        schema: dict[str, object] = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
        toolset = YoudotcomToolset(api_key='test', output_schema=schema)
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['output_schema'] == schema

    def test_output_schema_not_included_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert 'output_schema' not in body


# ---------------------------------------------------------------------------
# Research: integration tests
# ---------------------------------------------------------------------------


class TestResearchIntegration:
    async def test_research_returns_result(self) -> None:
        """End-to-end research with a mock transport."""
        payload = _make_research_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'POST'
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            result = await toolset.research('What happened?')
            assert result['content'] == '## Answer\n\nSomething happened [[1, 2]].'
            assert result['content_type'] == 'text'
            assert len(result['sources']) == 2
        finally:
            await client.aclose()

    async def test_research_with_configured_effort(self) -> None:
        """Configured research_effort is sent, not LLM-provided."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, research_effort='deep')
        try:
            await toolset.research('q', research_effort='lite')
            assert captured_body['research_effort'] == 'deep'
        finally:
            await client.aclose()

    async def test_research_with_configured_source_control(self) -> None:
        """Configured source_control fields are sent in the request body."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test',
            http_client=client,
            research_include_domains=['arxiv.org'],
            research_freshness='day',
            research_country='US',
        )
        try:
            await toolset.research(
                'q',
                include_domains=['bbc.com'],
                freshness='week',
                country='GB',
            )
            assert captured_body['source_control'] == {
                'include_domains': ['arxiv.org'],
                'freshness': 'day',
                'country': 'US',
            }
        finally:
            await client.aclose()

    async def test_research_source_control_from_llm(self) -> None:
        """LLM-provided source_control fields are sent when not configured."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.research(
                'q',
                exclude_domains=['spam.com'],
                boost_domains=['good.com'],
                freshness='day',
                country='US',
            )
            assert captured_body['source_control'] == {
                'exclude_domains': ['spam.com'],
                'boost_domains': ['good.com'],
                'freshness': 'day',
                'country': 'US',
            }
        finally:
            await client.aclose()

    async def test_research_with_configured_output_schema(self) -> None:
        """Configured output_schema is included in the request body."""
        captured_body: dict[str, object] = {}
        schema: dict[str, object] = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_structured_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, output_schema=schema)
        try:
            await toolset.research('q')
            assert captured_body['output_schema'] == schema
        finally:
            await client.aclose()

    async def test_research_structured_output(self) -> None:
        """Research with structured output returns content as a dict with content_type 'object'."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_research_structured_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            result = await toolset.research('q')
            assert result['content_type'] == 'object'
            assert result['content'] == {'answer': 'The sky is blue.', 'confidence': 0.95}
            assert len(result['sources']) == 1
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Finance research: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildFinanceResearchBody:
    def test_input_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_finance_research_body(input='question', research_effort=None)
        assert body['input'] == 'question'
        assert len(body) == 1

    def test_configured_effort_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', finance_research_effort='exhaustive')
        body = toolset._build_finance_research_body(input='q', research_effort='deep')
        assert body['research_effort'] == 'exhaustive'

    def test_llm_effort_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_finance_research_body(input='q', research_effort='exhaustive')
        assert body['research_effort'] == 'exhaustive'


# ---------------------------------------------------------------------------
# Finance research: integration tests
# ---------------------------------------------------------------------------


class TestFinanceResearchIntegration:
    async def test_finance_research_returns_result(self) -> None:
        """End-to-end finance research with a mock transport."""
        payload = _make_finance_research_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'POST'
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            result = await toolset.finance_research('NVDA revenue growth')
            assert result['content'] == 'Revenue grew 114% [[1]].'
            assert result['content_type'] == 'text'
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://sec.gov/filing'
        finally:
            await client.aclose()

    async def test_finance_research_with_configured_effort(self) -> None:
        """Configured finance_research_effort is sent, not LLM-provided."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, finance_research_effort='exhaustive')
        try:
            await toolset.finance_research('q', research_effort='deep')
            assert captured_body['research_effort'] == 'exhaustive'
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# HTTP helper tests
# ---------------------------------------------------------------------------


class TestNormalizeParam:
    def test_none(self) -> None:
        assert YoudotcomToolset._normalize_param(None) is None

    def test_str(self) -> None:
        assert YoudotcomToolset._normalize_param('hello') == 'hello'

    def test_int(self) -> None:
        assert YoudotcomToolset._normalize_param(42) == '42'


class TestHttpGet:
    async def test_with_custom_client(self) -> None:
        """_get uses the provided http_client."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_web_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test-key', http_client=client)
        try:
            response = await toolset._get('https://api.you.com/v1/search', {'query': 'test'}, timeout=60.0)
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
        finally:
            await client.aclose()

    async def test_raises_on_http_error(self) -> None:
        """_get raises for non-2xx responses."""
        transport = httpx.MockTransport(lambda req: httpx.Response(403, json={'error': 'forbidden'}))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='bad-key', http_client=client)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await toolset._get('https://api.you.com/v1/search', {'query': 'test'}, timeout=60.0)
        finally:
            await client.aclose()

    async def test_recoverable_http_error_becomes_model_retry(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(500, json={'error': 'server'}))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            with pytest.raises(ModelRetry, match='You.com request failed'):
                await toolset._get('https://api.you.com/v1/search', {'query': 'test'}, timeout=60.0)
        finally:
            await client.aclose()

    async def test_network_error_becomes_model_retry(self) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('offline', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            with pytest.raises(ModelRetry, match='You.com request failed'):
                await toolset._get('https://api.you.com/v1/search', {'query': 'test'}, timeout=60.0)
        finally:
            await client.aclose()

    async def test_without_custom_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_get creates a new httpx.AsyncClient when no http_client is provided."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_web_payload()))
        real_async_client = httpx.AsyncClient

        class _MockAsyncClient(real_async_client):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(transport=transport)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, 'AsyncClient', _MockAsyncClient)
        toolset = YoudotcomToolset(api_key='test')
        response = await toolset._get('https://api.you.com/v1/search', {'query': 'test'}, timeout=60.0)
        assert response.status_code == 200


class TestHttpPost:
    async def test_with_custom_client(self) -> None:
        """_post uses the provided http_client."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_research_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test-key', http_client=client)
        try:
            response = await toolset._post('https://api.you.com/v1/research', {'input': 'test'}, timeout=60.0)
            assert response.status_code == 200
        finally:
            await client.aclose()

    async def test_recoverable_http_error_becomes_model_retry(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(422, json={'error': 'invalid'}))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='bad-key', http_client=client)
        try:
            with pytest.raises(ModelRetry, match='You.com request failed'):
                await toolset._post('https://api.you.com/v1/research', {'input': 'test'}, timeout=60.0)
        finally:
            await client.aclose()

    async def test_auth_error_propagates(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(401, json={'error': 'unauthorized'}))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='bad-key', http_client=client)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await toolset._post('https://api.you.com/v1/research', {'input': 'test'}, timeout=60.0)
        finally:
            await client.aclose()

    async def test_network_error_becomes_model_retry(self) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('offline', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            with pytest.raises(ModelRetry, match='You.com request failed'):
                await toolset._post('https://api.you.com/v1/research', {'input': 'test'}, timeout=60.0)
        finally:
            await client.aclose()

    async def test_without_custom_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_post creates a new httpx.AsyncClient when no http_client is provided."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_research_empty_payload()))
        real_async_client = httpx.AsyncClient

        class _MockAsyncClient(real_async_client):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(transport=transport)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, 'AsyncClient', _MockAsyncClient)
        toolset = YoudotcomToolset(api_key='test')
        response = await toolset._post('https://api.you.com/v1/research', {'input': 'test'}, timeout=60.0)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Search: integration tests
# ---------------------------------------------------------------------------


class TestSearchIntegration:
    async def test_search_returns_results(self) -> None:
        """End-to-end search through the tool method with a mock transport."""
        payload = _make_web_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            results = await toolset.search('example query')
            assert len(results) == 1
            assert results[0].get('title') == 'Example Page'
        finally:
            await client.aclose()

    async def test_search_with_configured_params(self) -> None:
        """Configured params are sent in the request, not LLM-provided ones."""
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.multi_items():
                captured_params[key] = value
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test',
            http_client=client,
            count=5,
            freshness='day',
            country='US',
        )
        try:
            await toolset.search('q', count=100, freshness='year', country='GB')
            assert captured_params['count'] == '5'
            assert captured_params['freshness'] == 'day'
            assert captured_params['country'] == 'US'
        finally:
            await client.aclose()

    async def test_search_with_livecrawl_formats_list(self) -> None:
        """livecrawl_formats list is sent as repeated query params."""
        captured_formats: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_formats.extend(v for k, v in request.url.params.multi_items() if k == 'livecrawl_formats')
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', livecrawl_formats=['html', 'markdown'])
            assert captured_formats == ['html', 'markdown']
        finally:
            await client.aclose()

    async def test_search_with_configured_crawl_timeout(self) -> None:
        """Configured search crawl_timeout is sent, not LLM-provided."""
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.multi_items():
                captured_params[key] = value
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, search_crawl_timeout=30)
        try:
            await toolset.search('q', crawl_timeout=10)
            assert captured_params['crawl_timeout'] == '30'
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Capability tests
# ---------------------------------------------------------------------------


class TestCapability:
    def test_get_toolset_returns_youdotcom_toolset(self) -> None:
        cap = Youdotcom(api_key='test')
        toolset = cap.get_toolset()
        assert isinstance(toolset, YoudotcomToolset)

    async def test_capability_passes_search_params(self) -> None:
        client, cap_req = _search_capture()
        cap = Youdotcom(
            api_key='k',
            http_client=client,
            count=3,
            offset=6,
            freshness='week',
            country='GB',
            language='EN',
            safesearch='moderate',
            livecrawl='web',
            livecrawl_formats=['html'],
            exclude_domains=['spam.com'],
            boost_domains=['good.com'],
            search_crawl_timeout=30,
        )
        toolset = cap.get_toolset()
        try:
            await toolset.search('q')
            # Domain filters force a POST, so every configured field lands in the JSON body.
            assert cap_req.method == 'POST'
            assert cap_req.body is not None
            assert cap_req.body['count'] == 3
            assert cap_req.body['offset'] == 6
            assert cap_req.body['freshness'] == 'week'
            assert cap_req.body['country'] == 'GB'
            assert cap_req.body['language'] == 'EN'
            assert cap_req.body['safesearch'] == 'moderate'
            assert cap_req.body['livecrawl'] == 'web'
            assert cap_req.body['livecrawl_formats'] == ['html']
            assert cap_req.body['exclude_domains'] == ['spam.com']
            assert cap_req.body['boost_domains'] == ['good.com']
            assert cap_req.body['crawl_timeout'] == 30
        finally:
            await client.aclose()

    def test_capability_passes_contents_params(self) -> None:
        cap = Youdotcom(
            api_key='k',
            contents_formats=['markdown', 'metadata'],
            crawl_timeout=15,
            max_age=3600,
        )
        toolset = cap.get_toolset()
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert body['formats'] == ['markdown', 'metadata']
        assert body['crawl_timeout'] == 15
        assert body['max_age'] == 3600

    async def test_capability_passes_research_params(self) -> None:
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cap = Youdotcom(
            api_key='k',
            http_client=client,
            research_effort='deep',
            research_exclude_domains=['spam.com'],
            research_boost_domains=['good.com'],
            research_freshness='week',
            research_country='US',
            output_schema={'type': 'object', 'properties': {}},
        )
        toolset = cap.get_toolset()
        try:
            await toolset.research('q')
            assert captured_body['research_effort'] == 'deep'
            assert captured_body['source_control'] == {
                'exclude_domains': ['spam.com'],
                'boost_domains': ['good.com'],
                'freshness': 'week',
                'country': 'US',
            }
            assert captured_body['output_schema'] == {'type': 'object', 'properties': {}}
        finally:
            await client.aclose()

    def test_capability_passes_finance_research_params(self) -> None:
        cap = Youdotcom(api_key='k', finance_research_effort='exhaustive')
        toolset = cap.get_toolset()
        body = toolset._build_finance_research_body(input='q', research_effort=None)
        assert body['research_effort'] == 'exhaustive'

    def test_capability_registers_all_four_tools(self) -> None:
        cap = Youdotcom(api_key='test')
        toolset = cap.get_toolset()
        assert 'you_search' in toolset.tools
        assert 'you_contents' in toolset.tools
        assert 'you_research' in toolset.tools
        assert 'you_finance_research' in toolset.tools

    def test_capability_with_agent(self) -> None:
        """The capability registers tools with an Agent."""
        cap = Youdotcom(api_key='test')
        agent: Agent[None, str] = Agent(
            TestModel(custom_output_text='done', call_tools=[]),
            capabilities=[cap],
            name='test-agent',
        )
        toolset = cap.get_toolset()
        assert 'you_search' in toolset.tools
        result = agent.run_sync('Search for test')
        assert result.output == 'done'

    def test_capability_loads_from_agent_file(self, tmp_path: Path) -> None:
        spec_path = tmp_path / 'agent.yaml'
        spec_path.write_text('capabilities:\n  - Youdotcom:\n      api_key: test\n')
        agent = Agent.from_file(
            spec_path,
            custom_capability_types=[Youdotcom],
            model=TestModel(custom_output_text='done', call_tools=[]),
        )
        assert agent.run_sync('test').output == 'done'


# ---------------------------------------------------------------------------
# Locked-parameter schema stripping
# ---------------------------------------------------------------------------


class TestLockedSchema:
    """Construction-locked parameters are removed from each tool's JSON schema."""

    async def test_locked_search_params_removed_from_schema(self) -> None:
        toolset = YoudotcomToolset(api_key='test', count=5, include_domains=['a.com'])
        tools = await toolset.get_tools(build_run_context(None))
        props: dict[str, object] = tools['you_search'].tool_def.parameters_json_schema.get('properties', {})
        assert 'count' not in props
        assert 'include_domains' not in props
        assert 'exclude_domains' not in props
        assert 'boost_domains' not in props
        assert 'query' in props

    async def test_unlocked_search_params_present_in_schema(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        tools = await toolset.get_tools(build_run_context(None))
        props: dict[str, object] = tools['you_search'].tool_def.parameters_json_schema.get('properties', {})
        assert 'count' in props
        assert 'include_domains' in props

    async def test_configured_research_exclude_hides_incompatible_include(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_exclude_domains=['a.com'])
        tools = await toolset.get_tools(build_run_context(None))
        props: dict[str, object] = tools['you_research'].tool_def.parameters_json_schema.get('properties', {})
        assert 'include_domains' not in props
        assert 'exclude_domains' not in props
        assert 'boost_domains' in props

    async def test_configured_contents_formats_are_removed(self) -> None:
        toolset = YoudotcomToolset(api_key='test', contents_formats=['markdown'])
        tools = await toolset.get_tools(build_run_context(None))
        props: dict[str, object] = tools['you_contents'].tool_def.parameters_json_schema.get('properties', {})
        assert 'formats' not in props

    async def test_contents_urls_have_uri_schema(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        tools = await toolset.get_tools(build_run_context(None))
        tool = tools['you_contents']
        props: dict[str, object] = tool.tool_def.parameters_json_schema.get('properties', {})
        urls_schema = TypeAdapter(dict[str, object]).validate_python(props['urls'])
        assert urls_schema['items'] == {'type': 'string', 'format': 'uri', 'minLength': 1}
        tool.args_validator.validate_python({'urls': ['https://example.com']})
        with pytest.raises(ValidationError):
            tool.args_validator.validate_python({'urls': ['not-a-url']})

    async def test_locked_field_removed_from_required(self) -> None:
        toolset = YoudotcomToolset(api_key='test', finance_research_effort='deep')
        tools = await toolset.get_tools(build_run_context(None))
        schema = tools['you_finance_research'].tool_def.parameters_json_schema
        props: dict[str, object] = schema.get('properties', {})
        required: list[object] = schema.get('required', [])
        assert 'research_effort' not in props
        assert 'research_effort' not in required


# ---------------------------------------------------------------------------
# Constructor validation of configured values
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Configured values are validated at construction, not only at tool-call time."""

    def test_count_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            YoudotcomToolset(api_key='test', count=0)

    def test_capability_validates_immediately(self) -> None:
        with pytest.raises(ValidationError):
            Youdotcom(api_key='test', count=0)

    def test_offset_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            YoudotcomToolset(api_key='test', offset=10)

    def test_crawl_timeout_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            YoudotcomToolset(api_key='test', search_crawl_timeout=61)

    def test_non_positive_request_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            YoudotcomToolset(api_key='test', timeout=0)

    def test_freshness_bad_date_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            YoudotcomToolset(api_key='test', freshness='2024/01/01')

    def test_configured_include_with_exclude_rejected(self) -> None:
        with pytest.raises(ValueError, match='include_domains cannot be combined'):
            YoudotcomToolset(api_key='test', include_domains=['a.com'], exclude_domains=['b.com'])

    def test_configured_research_include_with_boost_rejected(self) -> None:
        with pytest.raises(ValueError, match='include_domains cannot be combined'):
            YoudotcomToolset(api_key='test', research_include_domains=['a.com'], research_boost_domains=['c.com'])

    def test_output_schema_with_lite_effort_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported with research_effort='lite'"):
            YoudotcomToolset(api_key='test', output_schema={'type': 'object'}, research_effort='lite')


# ---------------------------------------------------------------------------
# Runtime guards for LLM-supplied values
# ---------------------------------------------------------------------------


class TestResearchGuards:
    """Runtime guards raise ModelRetry for LLM-supplied invalid combinations."""

    async def test_research_include_with_exclude_rejected(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        with pytest.raises(ModelRetry, match='include_domains cannot be combined'):
            await toolset.research('q', include_domains=['a.com'], exclude_domains=['b.com'])

    async def test_research_output_schema_with_llm_lite_rejected(self) -> None:
        toolset = YoudotcomToolset(api_key='test', output_schema={'type': 'object'})
        with pytest.raises(ModelRetry, match="not supported with research_effort='lite'"):
            await toolset.research('q', research_effort='lite')


class TestMalformedResearchResponse:
    """A research response missing required fields surfaces as a validation error."""

    async def test_missing_sources_raises(self) -> None:
        payload: dict[str, object] = {'output': {'content': 'answer', 'content_type': 'text'}}
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            with pytest.raises(ValidationError):
                await toolset.research('q')
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Timeouts and secret handling
# ---------------------------------------------------------------------------


class TestTimeouts:
    """Research/finance use a long default timeout; search/contents use a shorter one."""

    @staticmethod
    def _timeout_capture(payload: dict[str, object]) -> tuple[httpx.AsyncClient, dict[str, object]]:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            ext = request.extensions.get('timeout')
            seen['read'] = ext['read'] if isinstance(ext, dict) else None
            return httpx.Response(200, json=payload)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen

    async def test_research_uses_long_default_timeout(self) -> None:
        client, seen = self._timeout_capture(_make_research_empty_payload())
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.research('q')
            assert seen['read'] == 300.0
        finally:
            await client.aclose()

    async def test_search_uses_short_default_timeout(self) -> None:
        client, seen = self._timeout_capture(_make_empty_search_payload())
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q')
            assert seen['read'] == 60.0
        finally:
            await client.aclose()

    async def test_configured_timeout_overrides_default(self) -> None:
        client, seen = self._timeout_capture(_make_research_empty_payload())
        toolset = YoudotcomToolset(api_key='test', http_client=client, timeout=5.0)
        try:
            await toolset.research('q')
            assert seen['read'] == 5.0
        finally:
            await client.aclose()


class TestSecretHandling:
    """The API key is not exposed through object reprs."""

    def test_capability_api_key_excluded_from_repr(self) -> None:
        cap = Youdotcom(api_key='super-secret-key')
        assert 'super-secret-key' not in repr(cap)

    def test_toolset_api_key_not_in_repr(self) -> None:
        toolset = YoudotcomToolset(api_key='super-secret-key')
        assert 'super-secret-key' not in repr(toolset)

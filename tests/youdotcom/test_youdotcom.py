"""Tests for the Youdotcom capability and YoudotcomToolset."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import anyio
import httpx
import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_core import to_json

from pydantic_ai_harness.youdotcom import Youdotcom, YoudotcomHTTPClient, YoudotcomToolset

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


def _toolset_with_payload(payload: object) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
    """Create a toolset and mock client backed by a provider response."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    return YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client)), client


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


def _make_missing_results_payload() -> dict[str, object]:
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
        },
        'warnings': ['One source could not be accessed.'],
    }


def _make_research_minimal_payload() -> dict[str, object]:
    """Build a Research API response with minimal fields."""
    return {
        'output': {
            'content': 'Short answer.',
            'content_type': 'text',
            'sources': [{'url': 'https://src.com'}],
        },
        'warnings': [],
    }


def _make_research_empty_payload() -> dict[str, object]:
    """Build a Research API response with no sources."""
    return {'output': {'content': 'No sources needed.', 'content_type': 'text', 'sources': []}, 'warnings': []}


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
        },
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
        },
        'warnings': [],
    }


# ---------------------------------------------------------------------------
# Search: result field integration tests
# ---------------------------------------------------------------------------


class TestSearchResultFields:
    """Exercise search result field mapping through the public search() tool."""

    async def test_web_result_all_fields(self) -> None:
        toolset, client = _toolset_with_payload(_make_web_payload())
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert len(results) == 1
            r = results[0]
            assert r.get('title') == 'Example Page'
            assert r.get('url') == 'https://example.com'
            assert r.get('description') == 'An example page.'
            assert r.get('snippets') == ['snippet one', 'snippet two']
            assert r.get('thumbnail_url') == 'https://example.com/thumb.png'
            assert r.get('favicon_url') == 'https://example.com/favicon.ico'
            assert r['kind'] == 'web'
            assert r.get('page_age') == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)

    async def test_web_result_minimal_fields(self) -> None:
        payload: dict[str, object] = {'results': {'web': [{'title': 'T', 'url': 'https://x.com'}], 'news': []}}
        toolset, client = _toolset_with_payload(payload)
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].get('title') == 'T'
            assert results[0].get('url') == 'https://x.com'
            assert 'description' not in results[0]
            assert 'snippets' not in results[0]
            assert 'thumbnail_url' not in results[0]
            assert 'favicon_url' not in results[0]
            assert results[0]['kind'] == 'web'
            assert 'page_age' not in results[0]
            assert 'contents' not in results[0]

    async def test_news_result_no_web_fields(self) -> None:
        toolset, client = _toolset_with_payload(_make_news_payload())
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].get('title') == 'Breaking News'
            assert 'snippets' not in results[0]
            assert 'favicon_url' not in results[0]
            assert results[0]['kind'] == 'news'

    async def test_livecrawl_both_formats(self) -> None:
        toolset, client = _toolset_with_payload(_make_livecrawl_payload())
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].get('contents') == {'html': '<p>Hello</p>', 'markdown': 'Hello'}

    @pytest.mark.parametrize(
        ('contents', 'expected'),
        [({'html': '<p>Hi</p>'}, {'html': '<p>Hi</p>'}), ({'markdown': 'Hi'}, {'markdown': 'Hi'})],
    )
    async def test_livecrawl_single_format(self, contents: dict[str, str], expected: dict[str, str]) -> None:
        payload: dict[str, object] = {'results': {'web': [{'contents': contents}]}}
        toolset, client = _toolset_with_payload(payload)
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert results[0].get('contents') == expected

    async def test_livecrawl_empty_strings_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'html': '', 'markdown': ''}}],
                'news': [],
            }
        }
        toolset, client = _toolset_with_payload(payload)
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert 'contents' not in results[0]

    async def test_both_web_and_news(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'W', 'url': 'https://w.com'}],
                'news': [{'title': 'N', 'url': 'https://n.com'}],
            }
        }
        toolset, client = _toolset_with_payload(payload)
        async with client:
            results = await toolset.search('q')
            assert isinstance(results, list)
            assert len(results) == 2
            assert results[0].get('title') == 'W'
            assert results[1].get('title') == 'N'

    async def test_missing_results_returns_empty(self) -> None:
        """The provider may omit the optional results object."""
        toolset, client = _toolset_with_payload(_make_missing_results_payload())
        async with client:
            assert await toolset.search('q') == []

    async def test_partial_item_is_preserved(self) -> None:
        """Provider-valid search items may omit every optional field."""
        payload: dict[str, object] = {'results': {'web': [{}]}}
        toolset, client = _toolset_with_payload(payload)
        async with client:
            assert await toolset.search('q') == [{'kind': 'web'}]


# ---------------------------------------------------------------------------
# Contents: result field integration tests
# ---------------------------------------------------------------------------


class TestContentsResultFields:
    """Exercise contents result field mapping through the public extract_contents() tool."""

    async def test_all_fields(self) -> None:
        toolset, client = _toolset_with_payload(_make_contents_payload())
        async with client:
            results = await toolset.extract_contents(['https://example.com/page'])
            assert isinstance(results, list)
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

    async def test_minimal_fields(self) -> None:
        toolset, client = _toolset_with_payload(_make_contents_minimal_payload())
        async with client:
            results = await toolset.extract_contents(['https://example.com'])
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].get('url') == 'https://example.com'
            assert results[0].get('title') == 'Minimal'
            assert 'html' not in results[0]
            assert 'markdown' not in results[0]
            assert 'metadata' not in results[0]

    @pytest.mark.parametrize(
        ('metadata', 'expected'),
        [
            ({'site_name': 'Example'}, {'site_name': 'Example'}),
            ({'favicon_url': 'https://x.com/favicon.ico'}, {'favicon_url': 'https://x.com/favicon.ico'}),
        ],
    )
    async def test_partial_metadata(self, metadata: dict[str, str], expected: dict[str, str]) -> None:
        toolset, client = _toolset_with_payload([{'metadata': metadata}])
        async with client:
            results = await toolset.extract_contents(['https://x.com'])
            assert isinstance(results, list)
            assert results[0].get('metadata') == expected

    async def test_partial_item_is_preserved(self) -> None:
        """Provider-valid contents items may omit every optional field."""
        payload: list[dict[str, object]] = [{}]
        toolset, client = _toolset_with_payload(payload)
        async with client:
            assert await toolset.extract_contents(['https://example.com']) == [{}]

    async def test_empty_metadata_not_included(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'site_name': '', 'favicon_url': ''}}
        ]
        toolset, client = _toolset_with_payload(payload)
        async with client:
            results = await toolset.extract_contents(['https://x.com'])
            assert isinstance(results, list)
            assert 'metadata' not in results[0]


# ---------------------------------------------------------------------------
# Research: result field integration tests
# ---------------------------------------------------------------------------


class TestResearchResultFields:
    """Exercise research result field mapping through the public research() tool."""

    async def test_full_response(self) -> None:
        toolset, client = _toolset_with_payload(_make_research_payload())
        async with client:
            result = await toolset.research('What happened?')
            assert result['content_type'] == 'text'
            assert result['content'] == '## Answer\n\nSomething happened [[1, 2]].'
            assert len(result['sources']) == 2
            assert result['sources'][0]['url'] == 'https://source1.com'
            assert result['sources'][0].get('title') == 'Source 1'
            assert result['sources'][0].get('snippets') == ['relevant excerpt']
            assert result['sources'][1]['url'] == 'https://source2.com'
            assert result['sources'][1].get('title') == 'Source 2'
            assert 'snippets' not in result['sources'][1]
            assert result['warnings'] == ['One source could not be accessed.']

    async def test_minimal_response(self) -> None:
        toolset, client = _toolset_with_payload(_make_research_minimal_payload())
        async with client:
            result = await toolset.research('q')
            assert result['content_type'] == 'text'
            assert result['content'] == 'Short answer.'
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://src.com'
            assert 'title' not in result['sources'][0]
            assert 'snippets' not in result['sources'][0]

    async def test_structured_output(self) -> None:
        """Structured output returns content as a dict with content_type 'object'."""
        toolset, client = _toolset_with_payload(_make_research_structured_payload())
        async with client:
            result = await toolset.research('q')
            assert result['content_type'] == 'object'
            assert result['content'] == {'answer': 'The sky is blue.', 'confidence': 0.95}
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://source.com'

    async def test_malformed_payload_raises(self) -> None:
        """Missing output key raises ValidationError."""
        toolset, client = _toolset_with_payload(_make_malformed_research_payload())
        async with client:
            with pytest.raises(ValidationError):
                await toolset.research('q')


# ---------------------------------------------------------------------------
# Search: parameter building tests
# ---------------------------------------------------------------------------


class TestSearchRequest:
    async def test_search_without_options_uses_get(self) -> None:
        client, request = _search_capture()
        async with client:
            assert await YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client)).search('q') == []
        assert request.method == 'GET'
        assert request.params == {'query': 'q'}
        assert request.body is None

    async def test_search_sends_unconfigured_options(self) -> None:
        client, request = _search_capture()
        async with client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            await toolset.search(
                'q',
                count=5,
                freshness='2026-01-01to2026-01-31',
                country='US',
                language='EN',
                safesearch='strict',
                livecrawl='all',
                livecrawl_formats=['html', 'markdown'],
                crawl_timeout=10,
            )
        assert request.params == {
            'query': 'q',
            'count': '5',
            'freshness': '2026-01-01to2026-01-31',
            'country': 'US',
            'language': 'EN',
            'safesearch': 'strict',
            'livecrawl': 'all',
            'livecrawl_formats': 'markdown',
            'crawl_timeout': '10',
        }
        assert [item for item in request.param_items if item[0] == 'livecrawl_formats'] == [
            ('livecrawl_formats', 'html'),
            ('livecrawl_formats', 'markdown'),
        ]

    @pytest.mark.parametrize(
        ('include_domains', 'exclude_domains', 'boost_domains'),
        [
            (['docs.example.com'], None, None),
            (None, ['spam.example.com'], ['trusted.example.com']),
        ],
    )
    async def test_domain_filters_use_post(
        self,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        boost_domains: list[str] | None,
    ) -> None:
        client, request = _search_capture()
        async with client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            await toolset.search(
                'q',
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                boost_domains=boost_domains,
            )
        assert request.method == 'POST'
        assert request.body == {
            key: value
            for key, value in {
                'query': 'q',
                'include_domains': include_domains,
                'exclude_domains': exclude_domains,
                'boost_domains': boost_domains,
            }.items()
            if value is not None
        }

    @pytest.mark.parametrize(
        ('exclude_domains', 'boost_domains'),
        [(['blocked.example.com'], None), (None, ['boosted.example.com'])],
    )
    async def test_include_domains_rejects_other_domain_filters(
        self,
        exclude_domains: list[str] | None,
        boost_domains: list[str] | None,
    ) -> None:
        client, _ = _search_capture()
        async with client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            with pytest.raises(ModelRetry, match='include_domains cannot be combined'):
                await toolset.search(
                    'q',
                    include_domains=['allowed.example.com'],
                    exclude_domains=exclude_domains,
                    boost_domains=boost_domains,
                )


# ---------------------------------------------------------------------------
# Contents: integration tests
# ---------------------------------------------------------------------------


class TestContentsIntegration:
    async def test_contents_with_configured_formats(self) -> None:
        """Configured formats are sent, not LLM-provided ones."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_contents_minimal_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client), contents_formats=['markdown'])
        async with client:
            await toolset.extract_contents(['https://x.com'], formats=['html'])
            assert captured_body['formats'] == ['markdown']


# ---------------------------------------------------------------------------
# Research: integration tests
# ---------------------------------------------------------------------------


class TestResearchIntegration:
    async def test_research_with_configured_effort(self) -> None:
        """Configured research_effort is sent, not LLM-provided."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client), research_effort='deep')
        async with client:
            await toolset.research('q', research_effort='lite')
            assert captured_body['research_effort'] == 'deep'

    async def test_research_with_configured_source_control(self) -> None:
        """Configured source_control fields are sent in the request body."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test',
            client=YoudotcomHTTPClient(client),
            research_include_domains=['arxiv.org'],
            research_freshness='day',
            research_country='US',
        )
        async with client:
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

    async def test_research_source_control_from_llm(self) -> None:
        """LLM-provided source_control fields are sent when not configured."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
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

    async def test_research_with_configured_output_schema(self) -> None:
        """Configured output_schema is included in the request body."""
        captured_body: dict[str, object] = {}
        schema: dict[str, object] = {
            'type': 'object',
            'properties': {'answer': {'type': 'string'}},
            'required': ['answer'],
            'additionalProperties': False,
        }

        def handler(request: httpx.Request) -> httpx.Response:

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_structured_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client), output_schema=schema)
        async with client:
            await toolset.research('q')
            assert captured_body['output_schema'] == schema


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
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            result = await toolset.finance_research('NVDA revenue growth')
            assert result['content_type'] == 'text'
            assert result['content'] == 'Revenue grew 114% [[1]].'
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://sec.gov/filing'

    async def test_finance_research_with_configured_effort(self) -> None:
        """Configured finance_research_effort is sent, not LLM-provided."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test', client=YoudotcomHTTPClient(client), finance_research_effort='exhaustive'
        )
        async with client:
            await toolset.finance_research('q', research_effort='deep')
            assert captured_body['research_effort'] == 'exhaustive'

    async def test_finance_research_rejects_structured_output(self) -> None:
        """Finance research only documents text output."""
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_make_research_structured_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            with pytest.raises(ValidationError):
                await toolset.finance_research('q')

    @pytest.mark.parametrize(
        'payload',
        [
            {'unrelated': 'data'},
            {'output': {'content': 'answer', 'content_type': 'text'}},
        ],
    )
    async def test_finance_research_rejects_malformed_response(self, payload: dict[str, object]) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            with pytest.raises(ValidationError):
                await toolset.finance_research('q')


# ---------------------------------------------------------------------------
# HTTP behavior
# ---------------------------------------------------------------------------


class TestHttpBehavior:
    @pytest.mark.parametrize('status_code', [401, 402, 403, 404, 429])
    async def test_configuration_and_rate_limit_errors_propagate(self, status_code: int) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(status_code))
        async with httpx.AsyncClient(transport=transport) as client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            with pytest.raises(httpx.HTTPStatusError):
                await toolset.search('test')

    @pytest.mark.parametrize('failure', ['http', 'network'])
    async def test_recoverable_http_and_network_errors_become_model_retry(self, failure: str) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if failure == 'network':
                raise httpx.ConnectError('offline', request=request)
            return httpx.Response(500)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            with pytest.raises(ModelRetry, match='You.com request failed'):
                await toolset.search('test')

    async def test_locked_output_schema_makes_unprocessable_response_non_retryable(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(422))
        async with httpx.AsyncClient(transport=transport) as client:
            toolset = YoudotcomToolset(
                api_key='test',
                client=YoudotcomHTTPClient(client),
                output_schema={
                    'type': 'object',
                    'properties': {},
                    'required': [],
                    'additionalProperties': False,
                },
            )
            with pytest.raises(httpx.HTTPStatusError):
                await toolset.research('test')

    @pytest.mark.parametrize(
        'retry_after',
        ['100', 'Wed, 25 Jul 2099 12:00:00 GMT', 'Wed, 25 Jul 999999999999 12:00:00 GMT', 'invalid'],
    )
    async def test_long_or_invalid_retry_after_propagates(self, retry_after: str) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(429, headers={'Retry-After': retry_after}))
        async with httpx.AsyncClient(transport=transport) as client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            with pytest.raises(httpx.HTTPStatusError):
                await toolset.search('test')

    @pytest.mark.parametrize('second_status', [200, 429])
    async def test_short_rate_limit_retries_once(self, second_status: int) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            status = 429 if attempts == 1 else second_status
            payload = _make_empty_search_payload() if status == 200 else None
            return httpx.Response(status, headers={'Retry-After': '0'}, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            if second_status == 429:
                with pytest.raises(httpx.HTTPStatusError):
                    await toolset.search('test')
            else:
                assert await toolset.search('test') == []
            assert attempts == 2

    async def test_missing_retry_after_uses_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        delays: list[float] = []
        attempts = 0

        async def capture_sleep(delay: float) -> None:
            delays.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429)
            return httpx.Response(200, json=_make_empty_search_payload())

        monkeypatch.setattr(anyio, 'sleep', capture_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
            assert await toolset.search('test') == []
        assert delays == [1.0]

    async def test_default_client_is_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_make_empty_search_payload()))
        real_async_client = httpx.AsyncClient

        class MockAsyncClient(real_async_client):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(transport=transport)

        monkeypatch.setattr(httpx, 'AsyncClient', MockAsyncClient)
        assert await YoudotcomToolset(api_key='test').search('test') == []


# ---------------------------------------------------------------------------
# Capability tests
# ---------------------------------------------------------------------------


class TestCapability:
    def test_get_toolset_returns_youdotcom_toolset(self) -> None:
        cap = Youdotcom(api_key='test')
        toolset = cap.get_toolset()
        assert isinstance(toolset, YoudotcomToolset)
        assert set(toolset.tools) == {'you_search', 'you_contents', 'you_research', 'you_finance_research'}

    async def test_capability_passes_search_params(self) -> None:
        client, cap_req = _search_capture()
        cap = Youdotcom(
            api_key='k',
            client=YoudotcomHTTPClient(client),
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
        async with client:
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

    async def test_capability_passes_contents_and_finance_params(self) -> None:
        bodies: dict[str, dict[str, object]] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            bodies[request.url.path] = json.loads(request.content)
            payload: object = [] if request.url.path.endswith('/contents') else _make_research_empty_payload()
            return httpx.Response(200, json=payload)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cap = Youdotcom(
            api_key='k',
            client=YoudotcomHTTPClient(client),
            contents_formats=['markdown', 'metadata'],
            crawl_timeout=15,
            max_age=3600,
            finance_research_effort='exhaustive',
        )
        toolset = cap.get_toolset()
        async with client:
            await toolset.extract_contents(['https://x.com'])
            await toolset.finance_research('q')
        assert bodies['/v1/contents'] == {
            'urls': ['https://x.com'],
            'formats': ['markdown', 'metadata'],
            'crawl_timeout': 15,
            'max_age': 3600,
        }
        assert bodies['/v1/finance_research'] == {'input': 'q', 'research_effort': 'exhaustive'}

    async def test_capability_passes_research_params(self) -> None:
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cap = Youdotcom(
            api_key='k',
            client=YoudotcomHTTPClient(client),
            research_effort='deep',
            research_exclude_domains=['spam.com'],
            research_boost_domains=['good.com'],
            research_freshness='week',
            research_country='US',
            output_schema={
                'type': 'object',
                'properties': {},
                'required': [],
                'additionalProperties': False,
            },
        )
        toolset = cap.get_toolset()
        async with client:
            await toolset.research('q')
            assert captured_body['research_effort'] == 'deep'
            assert captured_body['source_control'] == {
                'exclude_domains': ['spam.com'],
                'boost_domains': ['good.com'],
                'freshness': 'week',
                'country': 'US',
            }
            assert captured_body['output_schema'] == {
                'type': 'object',
                'properties': {},
                'required': [],
                'additionalProperties': False,
            }

    @pytest.mark.parametrize('anyio_backend', ['asyncio'])
    async def test_capability_with_agent(self, anyio_backend: str) -> None:
        """The public capability path registers and dispatches a You.com tool."""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_make_empty_search_payload()))
        )
        agent = Agent(
            TestModel(call_tools=['you_search']),
            capabilities=[Youdotcom(api_key='test', client=YoudotcomHTTPClient(client))],
            name='test-agent',
        )
        async with client:
            result = await agent.run('Search for test')
        returns = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'you_search'
        ]
        assert returns and returns[0].content == []

    def test_capability_loads_from_agent_file(self, tmp_path: Path) -> None:
        spec_path = tmp_path / 'agent.yaml'
        spec_path.write_text('capabilities:\n  - Youdotcom:\n      api_key: test\n')
        agent = Agent.from_file(
            spec_path,
            custom_capability_types=[Youdotcom],
            model=TestModel(custom_output_text='done', call_tools=[]),
        )
        assert agent.run_sync('test').output == 'done'

    @pytest.mark.parametrize(
        'config',
        [
            "api_key: test\n      timeout: '5'",
            'api_key: 123',
            'api_key: test\n      output_schema: []',
        ],
    )
    def test_capability_file_rejects_coerced_config(self, tmp_path: Path, config: str) -> None:
        spec_path = tmp_path / 'agent.yaml'
        spec_path.write_text(f'capabilities:\n  - Youdotcom:\n      {config}\n')
        with pytest.raises(ValueError, match="Failed to instantiate capability 'Youdotcom'"):
            Agent.from_file(
                spec_path,
                custom_capability_types=[Youdotcom],
                model=TestModel(custom_output_text='done', call_tools=[]),
            )


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
        livecrawl_formats = TypeAdapter(dict[str, object]).validate_python(props['livecrawl_formats'])
        variants = TypeAdapter(list[dict[str, object]]).validate_python(livecrawl_formats['anyOf'])
        assert variants[0]['minItems'] == 1
        assert variants[0]['maxItems'] == 2
        tool = tools['you_search']
        with pytest.raises(ValidationError):
            tool.args_validator.validate_python({'query': 'q', 'livecrawl_formats': []})

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

    @pytest.mark.parametrize(
        'build',
        [
            lambda: YoudotcomToolset(api_key='test', count=0),
            lambda: YoudotcomToolset(api_key='test', offset=10),
            lambda: YoudotcomToolset(api_key='test', search_crawl_timeout=61),
            lambda: YoudotcomToolset(api_key='test', timeout=0),
            lambda: YoudotcomToolset(api_key='test', max_output_bytes=511),
            lambda: YoudotcomToolset(api_key='test', max_output_lines=7),
            lambda: YoudotcomToolset(api_key='test', freshness='2024/01/01'),
            lambda: YoudotcomToolset(api_key='test', livecrawl_formats=[]),
            lambda: YoudotcomToolset(api_key='test', output_schema={'type': 'string'}),
            lambda: YoudotcomToolset(
                api_key='test',
                output_schema={
                    'type': 'object',
                    'properties': {'answer': {'type': 'string'}},
                    'required': [],
                    'additionalProperties': False,
                },
            ),
            lambda: YoudotcomToolset(
                api_key='test',
                output_schema={
                    'type': 'object',
                    'properties': {},
                    'required': [],
                    'additionalProperties': True,
                },
            ),
        ],
    )
    def test_invalid_ranges_are_rejected(self, build: Callable[[], object]) -> None:
        with pytest.raises(ValidationError):
            build()

    def test_capability_validates_immediately(self) -> None:
        with pytest.raises(ValidationError):
            Youdotcom(api_key='test', count=0)

    @pytest.mark.parametrize(
        'build',
        [
            lambda: YoudotcomToolset(api_key='test', include_domains=['a.com'], exclude_domains=['b.com']),
            lambda: YoudotcomToolset(
                api_key='test', research_include_domains=['a.com'], research_boost_domains=['c.com']
            ),
            lambda: YoudotcomToolset(
                api_key='test',
                output_schema={
                    'type': 'object',
                    'properties': {},
                    'required': [],
                    'additionalProperties': False,
                },
                research_effort='lite',
            ),
        ],
    )
    def test_invalid_combinations_are_rejected(self, build: Callable[[], object]) -> None:
        with pytest.raises(ValueError):
            build()


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
        toolset = YoudotcomToolset(
            api_key='test',
            output_schema={
                'type': 'object',
                'properties': {},
                'required': [],
                'additionalProperties': False,
            },
        )
        with pytest.raises(ModelRetry, match="not supported with research_effort='lite'"):
            await toolset.research('q', research_effort='lite')


class TestMalformedResearchResponse:
    """A research response missing required fields surfaces as a validation error."""

    async def test_missing_sources_raises(self) -> None:
        payload: dict[str, object] = {
            'output': {'content': 'answer', 'content_type': 'text'},
            'warnings': [],
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            with pytest.raises(ValidationError):
                await toolset.research('q')

    async def test_missing_warnings_raises(self) -> None:
        payload: dict[str, object] = {
            'output': {'content': 'answer', 'content_type': 'text', 'sources': []},
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            with pytest.raises(ValidationError):
                await toolset.research('q')


# ---------------------------------------------------------------------------
# Output limits
# ---------------------------------------------------------------------------


class TestOutputLimits:
    """Complete provider responses are bounded before they reach model context."""

    async def test_byte_limit_returns_bounded_metadata(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': 'large', 'description': 'x' * 1000}], 'news': []},
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client), max_output_bytes=512)
        async with client:
            result = await toolset.search('q')
        assert not isinstance(result, list)
        assert result['content_type'] == 'output_limit'
        assert result['output_bytes'] > 512
        assert result['max_output_bytes'] == 512
        assert len(to_json(result, indent=2)) <= 512
        assert len(to_json(result, indent=2).splitlines()) <= result['max_output_lines']

    async def test_line_limit_returns_bounded_metadata(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': f'result-{index}'} for index in range(20)], 'news': []},
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(
            api_key='test',
            client=YoudotcomHTTPClient(client),
            max_output_bytes=10_000,
            max_output_lines=8,
        )
        async with client:
            result = await toolset.search('q')
        assert not isinstance(result, list)
        assert result['content_type'] == 'output_limit'
        assert result['output_lines'] > 8
        assert result['max_output_lines'] == 8
        assert len(to_json(result, indent=2)) <= result['max_output_bytes']
        assert len(to_json(result, indent=2).splitlines()) <= 8

    async def test_literal_escape_text_does_not_count_as_serialized_lines(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': 'escape', 'description': r'x\ny'}], 'news': []},
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
        toolset = YoudotcomToolset(
            api_key='test',
            client=YoudotcomHTTPClient(client),
            max_output_bytes=10_000,
            max_output_lines=20,
        )
        async with client:
            result = await toolset.search('q')
        assert isinstance(result, list)


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
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            await toolset.research('q')
            assert seen['read'] == 300.0

    async def test_search_uses_short_default_timeout(self) -> None:
        client, seen = self._timeout_capture(_make_empty_search_payload())
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client))
        async with client:
            await toolset.search('q')
            assert seen['read'] == 60.0

    async def test_configured_timeout_overrides_default(self) -> None:
        client, seen = self._timeout_capture(_make_research_empty_payload())
        toolset = YoudotcomToolset(api_key='test', client=YoudotcomHTTPClient(client), timeout=5.0)
        async with client:
            await toolset.research('q')
            assert seen['read'] == 5.0


class TestSecretHandling:
    """The API key is not exposed through object reprs."""

    def test_capability_api_key_excluded_from_repr(self) -> None:
        cap = Youdotcom(api_key='super-secret-key')
        assert 'super-secret-key' not in repr(cap)

    def test_toolset_api_key_not_in_repr(self) -> None:
        toolset = YoudotcomToolset(api_key='super-secret-key')
        assert 'super-secret-key' not in repr(toolset)

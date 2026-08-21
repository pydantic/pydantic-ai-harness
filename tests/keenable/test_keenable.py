"""Tests for the KeenableSearch capability and KeenableSearchToolset."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.keenable import (
    HttpKeenableClient,
    KeenableSearch,
    KeenableSearchToolset,
    KeenableSource,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    # Pydantic AI's capability lifecycle uses `asyncio.create_task`, so the
    # agent-run test cannot execute under trio.
    return 'asyncio'


# A realistic Keenable result: the API populates `snippet` with the page text
# and leaves `description` empty on essentially every result.
def _result(url: str, title: str | None = 'A title', snippet: str = 'Page text') -> dict[str, Any]:
    return {'url': url, 'title': title, 'description': '', 'snippet': snippet}


class FakeClient:
    """Stands in for the HTTP client; records calls and returns canned payloads."""

    def __init__(
        self,
        *,
        results: list[dict[str, Any]] | None = None,
        page: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.page = page if page is not None else {}
        self.error = error
        self.queries: list[str] = []
        self.urls: list[str] = []

    async def search(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results

    async def fetch(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.page


def _toolset(client: FakeClient, **kwargs: Any) -> KeenableSearchToolset[None]:
    capability = KeenableSearch[None](client=client, **kwargs)
    return capability.get_toolset()


def _text(returned: ToolReturn[str]) -> str:
    # `ToolReturn.return_value` is typed as the full multi-modal content union,
    # so narrow it once here rather than at every assertion.
    value = returned.return_value
    assert isinstance(value, str)
    return value


async def test_agent_run_exposes_both_tools():
    # TestModel calls every tool the capability registers, so both paths must
    # return successfully for the run to finish.
    client = FakeClient(results=[_result('https://example.com/one')], page={'content': '# A page'})
    agent = Agent(TestModel(), capabilities=[KeenableSearch(client=client)])

    result = await agent.run('research something')

    assert result.output is not None
    assert client.queries and client.urls


async def test_web_search_reads_the_snippet_field():
    client = FakeClient(results=[_result('https://example.com/one', 'One', 'First page text')])

    returned = await _toolset(client).web_search('cats')

    assert isinstance(returned, ToolReturn)
    assert 'First page text' in _text(returned)
    assert returned.metadata == {'sources': [KeenableSource(url='https://example.com/one', title='One')]}


async def test_web_search_falls_back_to_description():
    client = FakeClient(results=[{'url': 'https://example.com/one', 'title': 'One', 'description': 'A description'}])

    returned = await _toolset(client).web_search('cats')

    assert 'A description' in _text(returned)


async def test_web_search_collapses_whitespace_and_caps_the_excerpt():
    # Snippets are raw page text, newlines included, and far longer than the
    # excerpt a search result should carry.
    client = FakeClient(results=[_result('https://example.com/one', 'One', 'line one\n\nline two' + ' pad' * 500)])

    returned = await _toolset(client, max_snippet_chars=40).web_search('cats')

    excerpt = _text(returned).splitlines()[-1].strip()
    assert len(excerpt) == 40
    assert excerpt.startswith('line one line two')


async def test_web_search_trims_to_num_results():
    client = FakeClient(results=[_result(f'https://example.com/{i}') for i in range(10)])

    returned = await _toolset(client, num_results=3).web_search('cats')

    assert len(returned.metadata['sources']) == 3


async def test_web_search_reports_no_results():
    client = FakeClient(results=[])

    returned = await _toolset(client).web_search('cats')

    assert _text(returned) == "No results found for 'cats'."
    assert returned.metadata == {'sources': []}


async def test_web_search_skips_results_without_a_url():
    client = FakeClient(results=[{'title': 'No URL', 'snippet': 'text'}, _result('https://example.com/one')])

    returned = await _toolset(client).web_search('cats')

    assert [source['url'] for source in returned.metadata['sources']] == ['https://example.com/one']


async def test_web_search_drops_url_less_results_before_applying_the_limit():
    # A URL-less entry ahead of good ones must not cost a result slot.
    client = FakeClient(results=[{'title': 'No URL'}, *(_result(f'https://example.com/{i}') for i in range(3))])

    returned = await _toolset(client, num_results=2).web_search('cats')

    assert [source['url'] for source in returned.metadata['sources']] == [
        'https://example.com/0',
        'https://example.com/1',
    ]


async def test_web_search_reports_no_results_when_every_result_lacks_a_url():
    client = FakeClient(results=[{'title': 'No URL', 'snippet': 'text'}])

    returned = await _toolset(client).web_search('cats')

    assert _text(returned) == "No results found for 'cats'."


async def test_web_search_omits_an_empty_excerpt():
    client = FakeClient(results=[{'url': 'https://example.com/one', 'title': 'One'}])

    returned = await _toolset(client).web_search('cats')

    assert _text(returned) == '1. One\n   https://example.com/one'


async def test_web_search_falls_back_to_the_url_when_a_result_has_no_title():
    client = FakeClient(results=[_result('https://example.com/one', None)])

    returned = await _toolset(client).web_search('cats')

    assert _text(returned).startswith('1. https://example.com/one')
    assert returned.metadata['sources'][0]['title'] is None


async def test_get_page_returns_markdown():
    client = FakeClient(page={'content': '# A page', 'url': 'https://example.com/one', 'title': 'One'})

    returned = await _toolset(client).get_page('https://example.com/one')

    assert _text(returned) == '# A page'
    assert returned.metadata == {'sources': [KeenableSource(url='https://example.com/one', title='One')]}


async def test_get_page_truncates_and_marks_long_pages():
    client = FakeClient(page={'content': 'x' * 100})

    returned = await _toolset(client, max_page_chars=23).get_page('https://example.com/one')

    # The marker is part of the budget, not extra: 10 characters of page text
    # plus the 13-character marker is exactly the 23 asked for.
    assert _text(returned) == f'{"x" * 10}\n\n[truncated]'
    assert len(_text(returned)) == 23
    assert returned.metadata['sources'][0] == KeenableSource(url='https://example.com/one', title=None)


async def test_get_page_drops_the_marker_when_the_budget_cannot_hold_it():
    client = FakeClient(page={'content': 'x' * 100})

    returned = await _toolset(client, max_page_chars=5).get_page('https://example.com/one')

    assert _text(returned) == 'x' * 5


async def test_get_page_retries_on_empty_content():
    client = FakeClient(page={'content': ''})

    with pytest.raises(ModelRetry, match='No readable content'):
        await _toolset(client).get_page('https://example.com/one')


@pytest.mark.parametrize('status', [429, 500])
async def test_transient_http_errors_become_retries(status: int):
    request = httpx.Request('POST', 'https://api.keenable.ai/v1/search/public')
    error = httpx.HTTPStatusError('boom', request=request, response=httpx.Response(status, request=request))
    client = FakeClient(error=error)

    with pytest.raises(ModelRetry, match='Keenable request failed'):
        await _toolset(client).web_search('cats')


@pytest.mark.parametrize('status', [401, 403])
async def test_auth_errors_propagate(status: int):
    request = httpx.Request('POST', 'https://api.keenable.ai/v1/search')
    error = httpx.HTTPStatusError('nope', request=request, response=httpx.Response(status, request=request))
    client = FakeClient(error=error)

    with pytest.raises(httpx.HTTPStatusError):
        await _toolset(client).web_search('cats')


async def test_network_errors_become_retries():
    client = FakeClient(error=httpx.ConnectError('unreachable'))

    with pytest.raises(ModelRetry, match='Keenable request failed'):
        await _toolset(client).get_page('https://example.com/one')


def test_default_instructions_mention_both_tools():
    instructions = KeenableSearch[None]().get_instructions()

    assert isinstance(instructions, str)
    assert '`web_search`' in instructions
    assert '`get_page`' in instructions
    assert 'untrusted' in instructions


def test_guidance_overrides_and_disables_instructions():
    assert KeenableSearch[None](guidance='Just search.').get_instructions() == 'Just search.'
    assert KeenableSearch[None](guidance='').get_instructions() is None


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'num_results': 0}, 'num_results must be at least 1'),
        ({'max_snippet_chars': 0}, 'max_snippet_chars must be at least 1'),
        ({'max_page_chars': 0}, 'max_page_chars must be at least 1'),
    ],
)
def test_budgets_are_validated(kwargs: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        KeenableSearch[None](**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'num_results': -1}, 'num_results must be at least 1'),
        ({'max_snippet_chars': 0}, 'max_snippet_chars must be at least 1'),
        ({'max_page_chars': -10}, 'max_page_chars must be at least 1'),
    ],
)
def test_the_toolset_validates_budgets_when_built_directly(kwargs: dict[str, Any], message: str):
    # The toolset is exported, so it cannot rely on `KeenableSearch` having
    # validated first: a negative budget slices from the end and would return
    # the wrong text rather than fail.
    budgets: dict[str, Any] = {'num_results': 5, 'max_snippet_chars': 500, 'max_page_chars': 10_000}

    with pytest.raises(ValueError, match=message):
        KeenableSearchToolset[None](client=FakeClient(), **{**budgets, **kwargs})


def test_from_spec_builds_the_default_client():
    capability = KeenableSearch[None].from_spec(num_results=2, max_snippet_chars=10, max_page_chars=20)

    assert capability.client is None
    assert capability.num_results == 2
    assert isinstance(capability.get_toolset(), KeenableSearchToolset)


def test_default_client_is_keyless(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
    monkeypatch.delenv('KEENABLE_API_URL', raising=False)

    client = HttpKeenableClient()

    assert 'X-API-Key' not in client._headers()  # pyright: ignore[reportPrivateUsage]
    assert client._path('/v1/search', '/v1/search/public') == '/v1/search/public'  # pyright: ignore[reportPrivateUsage]


def test_api_key_selects_the_keyed_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('KEENABLE_API_KEY', 'keen_live_x')

    client = HttpKeenableClient()

    assert client._headers()['X-API-Key'] == 'keen_live_x'  # pyright: ignore[reportPrivateUsage]
    assert client._path('/v1/search', '/v1/search/public') == '/v1/search'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('base_url', ['http://localhost:8080', 'https://keenable.internal'])
def test_base_url_accepts_https_and_loopback_http(base_url: str):
    assert HttpKeenableClient(api_key='', base_url=base_url) is not None


@pytest.mark.parametrize(
    'base_url',
    [
        'http://example.com',
        'https://',
        'not-a-url',
        # The endpoint path is appended to the base URL, so a query or fragment
        # would land in front of it and the request would miss the endpoint.
        'https://api.keenable.ai?token=x',
        'https://api.keenable.ai#frag',
    ],
)
def test_base_url_rejects_plaintext_hostless_and_query_bearing_urls(base_url: str):
    with pytest.raises(UserError, match='must be an https:// URL'):
        HttpKeenableClient(api_key='', base_url=base_url)


async def test_http_client_parses_search_and_fetch(monkeypatch: pytest.MonkeyPatch):
    # Exercises the real request/response plumbing without a network, so the
    # endpoint choice, attribution header, and payload shaping are pinned.
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['url'] = str(request.url)
        seen['headers'] = dict(request.headers)
        if request.url.path.startswith('/v1/search'):
            return httpx.Response(200, json={'results': [_result('https://example.com/one'), 'not-a-dict']})
        return httpx.Response(200, json={'content': '# A page'})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original(*args, **{**kwargs, 'transport': transport})

    monkeypatch.setattr(httpx, 'AsyncClient', patched)
    monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
    client = HttpKeenableClient()

    results = await client.search('cats')
    assert results == [_result('https://example.com/one')]
    assert seen['url'] == 'https://api.keenable.ai/v1/search/public'
    assert seen['headers']['x-keenable-title'] == 'Pydantic AI Harness'

    page = await client.fetch('https://example.com/one')
    assert page == {'content': '# A page'}
    assert seen['url'] == 'https://api.keenable.ai/v1/fetch/public?url=https%3A%2F%2Fexample.com%2Fone'


@pytest.mark.parametrize(
    ('payload', 'expected'),
    [({'results': 'nope'}, []), ([], []), ({}, [])],
)
async def test_http_client_tolerates_unexpected_payloads(
    monkeypatch: pytest.MonkeyPatch, payload: Any, expected: list[dict[str, Any]]
):
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    original = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original(*args, **{**kwargs, 'transport': transport})

    monkeypatch.setattr(httpx, 'AsyncClient', patched)
    monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
    client = HttpKeenableClient()

    assert await client.search('cats') == expected
    assert await client.fetch('https://example.com/one') == (payload if isinstance(payload, dict) else {})

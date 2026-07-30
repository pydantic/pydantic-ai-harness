"""Tests for NimbleSearch / NimbleAgent capabilities and toolsets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock

import httpx
import pytest
from nimble_python import AuthenticationError, RateLimitError
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.nimble import NimbleAgent, NimbleSearch, NimbleSearchToolset


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _text(output: ToolReturn[str]) -> str:
    """The model-facing text of a tool result."""
    body = output.return_value
    assert isinstance(body, str)
    return body


def _search_result(
    url: str = 'https://example.dev/page',
    *,
    title: str | None = 'Example page',
    content: str | None = 'body',
    description: str | None = 'desc',
) -> SimpleNamespace:
    return SimpleNamespace(url=url, title=title, content=content, description=description)


@dataclass
class _FakeNimbleClient:
    """In-memory Nimble client double."""

    search_results: list[SimpleNamespace] = field(default_factory=list[SimpleNamespace])
    extract_markdown: str | None = '# Hello'
    map_links: list[SimpleNamespace] = field(default_factory=list[SimpleNamespace])
    crawl_response: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            crawl_id='crawl_1',
            status='queued',
            url='https://example.com',
            completed=0,
            failed=0,
            pending=1,
            total=1,
        )
    )
    agent_items: list[Any] = field(default_factory=list[Any])
    template_items: list[Any] = field(default_factory=list[Any])
    run_payload: Any = field(
        default_factory=lambda: SimpleNamespace(
            model_dump=lambda mode='json': {
                'id': 'task_run_1',
                'status': 'queued',
                'web_search_agent_id': 'wsa_1',
            }
        )
    )
    result_payload: Any = field(
        default_factory=lambda: SimpleNamespace(
            model_dump=lambda mode='json': {
                'run': {'id': 'task_run_1', 'status': 'completed', 'web_search_agent_id': 'wsa_1'},
                'output': {
                    'type': 'text',
                    'content': 'done',
                    'trust': {
                        'confidence': 'high',
                        'sources': [{'url': 'https://example.com', 'title': 'Example'}],
                    },
                },
            }
        )
    )
    error: Exception | None = None
    search_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    extract_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    mode1_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    mode2_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    closed: bool = False

    def __post_init__(self) -> None:
        self.extract = SimpleNamespace(run=self._extract_run)
        self.crawl = SimpleNamespace(run=self._crawl_run, status=self._crawl_status)
        self.agents = SimpleNamespace(
            list=self._agents_list,
            templates=SimpleNamespace(list=self._templates_list),
            run=self._agents_run,
            runs=SimpleNamespace(
                create=self._runs_create,
                get=self._runs_get,
                result=self._runs_result,
            ),
        )

    async def close(self) -> None:
        self.closed = True

    async def search(self, **kwargs: Any) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        self.search_calls.append(kwargs)
        return SimpleNamespace(results=list(self.search_results))

    async def _extract_run(self, **kwargs: Any) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        self.extract_calls.append(kwargs)
        data = SimpleNamespace(markdown=self.extract_markdown) if self.extract_markdown is not None else None
        return SimpleNamespace(data=data)

    async def map(self, **kwargs: Any) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(links=list(self.map_links))

    async def _crawl_run(self, **kwargs: Any) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return self.crawl_response

    async def _crawl_status(self, crawl_id: str) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            crawl_id=crawl_id,
            status='running',
            url=self.crawl_response.url,
            completed=0,
            failed=0,
            pending=1,
            total=1,
        )

    async def _agents_list(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(items=list(self.agent_items))

    async def _templates_list(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(items=list(self.template_items))

    async def _agents_run(self, **kwargs: Any) -> Any:
        self.mode1_calls.append(kwargs)
        return self.run_payload

    async def _runs_create(self, agent_id: str, **kwargs: Any) -> Any:
        self.mode2_calls.append({'agent_id': agent_id, **kwargs})
        return self.run_payload

    async def _runs_get(self, run_id: str, *, agent_id: str) -> Any:
        return SimpleNamespace(
            model_dump=lambda mode='json': {
                'id': run_id,
                'agent_id': agent_id,
                'web_search_agent_id': agent_id,
                'status': 'running',
            }
        )

    async def _runs_result(self, run_id: str, *, agent_id: str) -> Any:
        return self.result_payload


def _toolset(
    client: _FakeNimbleClient,
    *,
    num_results: int = 5,
    max_text_chars: int = 10_000,
    search_depth: Literal['lite', 'fast', 'deep'] = 'lite',
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = None,
    include_domains: Sequence[str] = (),
    exclude_domains: Sequence[str] = (),
    include_map: bool = False,
    include_crawl: bool = False,
) -> NimbleSearchToolset[None]:
    return NimbleSearch[None](
        num_results=num_results,
        max_text_chars=max_text_chars,
        search_depth=search_depth,
        time_range=time_range,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_map=include_map,
        include_crawl=include_crawl,
        client=client,
    ).get_toolset()


class TestWebSearch:
    async def test_formats_results_and_sources(self) -> None:
        client = _FakeNimbleClient(
            search_results=[
                _search_result('https://a.dev', title='A', content='alpha', description=''),
                _search_result('https://b.dev', title=None, content='', description='fallback'),
            ]
        )
        output = await _toolset(client).web_search('rust web frameworks')
        assert "Found 2 results for 'rust web frameworks'" in _text(output)
        assert 'Title: A' in _text(output)
        assert 'alpha' in _text(output)
        assert 'fallback' in _text(output)
        assert output.metadata == {
            'sources': [{'url': 'https://a.dev', 'title': 'A'}, {'url': 'https://b.dev', 'title': None}]
        }

    async def test_requests_depth_results_and_domains(self) -> None:
        client = _FakeNimbleClient(search_results=[_search_result()])
        toolset = _toolset(
            client,
            num_results=3,
            search_depth='fast',
            time_range='week',
            include_domains=['a.dev'],
        )
        await toolset.web_search('q')
        assert client.search_calls == [
            {
                'query': 'q',
                'search_depth': 'fast',
                'max_results': 3,
                'time_range': 'week',
                'include_domains': ['a.dev'],
            }
        ]

    async def test_empty_results(self) -> None:
        client = _FakeNimbleClient(search_results=[])
        output = await _toolset(client).web_search('nothing')
        assert _text(output) == "No results found for 'nothing'."


class TestGetPage:
    async def test_returns_markdown(self) -> None:
        client = _FakeNimbleClient(extract_markdown='# Hello')
        output = await _toolset(client).get_page('https://example.com')
        assert _text(output).startswith('URL: https://example.com')
        assert '# Hello' in _text(output)
        assert client.extract_calls[0] == {'url': 'https://example.com', 'formats': ['markdown']}

    async def test_total_return_respects_max_text_chars(self) -> None:
        client = _FakeNimbleClient(extract_markdown='x' * 500)
        output = await _toolset(client, max_text_chars=80).get_page('https://example.com')
        assert len(_text(output)) <= 80
        assert 'truncated' in _text(output)

    async def test_missing_markdown_retries(self) -> None:
        client = _FakeNimbleClient(extract_markdown=None)
        with pytest.raises(ModelRetry, match='No content'):
            await _toolset(client).get_page('https://example.com')


class TestOptInTools:
    async def test_map_and_crawl(self) -> None:
        link = SimpleNamespace(url='https://example.com/a', title='A', description='d')
        client = _FakeNimbleClient(map_links=[link])
        toolset = _toolset(client, include_map=True, include_crawl=True)

        mapped = await toolset.map_site('https://example.com', limit=5, domain_filter='domain')
        assert 'https://example.com/a' in _text(mapped)

        started = await toolset.crawl_start('https://example.com', limit=2, name='docs')
        assert 'crawl_1' in _text(started)
        status = await toolset.crawl_status('crawl_1')
        assert 'running' in _text(status)

    def test_default_tool_names(self) -> None:
        toolset = _toolset(_FakeNimbleClient())
        assert sorted(toolset.tools) == ['get_page', 'web_search']

    def test_map_crawl_opt_in_tool_names(self) -> None:
        toolset = _toolset(_FakeNimbleClient(), include_map=True, include_crawl=True)
        assert sorted(toolset.tools) == [
            'crawl_start',
            'crawl_status',
            'get_page',
            'map_site',
            'web_search',
        ]


class TestNimbleAgent:
    async def test_mode1_start_with_agent_name_and_overrides(self) -> None:
        client = _FakeNimbleClient()
        toolset = NimbleAgent[None](client=client).get_toolset()
        started = await toolset.agent_run_start(
            'Find AI news',
            agent_name='pydantic_harness_research',
            use_case='research',
            skill='Prefer primary sources',
            effort='medium',
            sources={'prioritize': 'docs.python.org'},
        )
        assert 'task_run_1' in _text(started)
        assert 'wsa_1' in _text(started)
        assert started.metadata is not None
        assert started.metadata['web_search_agent_id'] == 'wsa_1'
        assert started.metadata['mode'] == 'mode1'
        assert client.mode1_calls
        call = client.mode1_calls[0]
        assert call['input'] == 'Find AI news'
        assert call['effort'] == 'medium'
        assert call['sources'] == {'prioritize': 'docs.python.org'}
        assert call['extra_body'] == {
            'use_case': 'research',
            'skill': 'Prefer primary sources',
            'agent_name': 'pydantic_harness_research',
        }

    async def test_mode2_and_mode3_start(self) -> None:
        client = _FakeNimbleClient()
        toolset = NimbleAgent[None](client=client).get_toolset()

        mode2 = await toolset.agent_run_start('q', agent_id='wsa_existing', effort='low')
        assert mode2.metadata is not None
        assert mode2.metadata['mode'] == 'mode2'
        assert client.mode2_calls[0]['agent_id'] == 'wsa_existing'

        mode3 = await toolset.agent_run_start('anonymous question')
        assert mode3.metadata is not None
        assert mode3.metadata['mode'] == 'mode3'
        assert client.mode1_calls[-1].get('extra_body') in (None, {})

    async def test_enrichment_overrides(self) -> None:
        client = _FakeNimbleClient()
        toolset = NimbleAgent[None](client=client).get_toolset()
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        await toolset.agent_run_start(
            'Enrich this company',
            agent_name='enrich_co',
            use_case='enrichment',
            output_schema=schema,
            input_data=[{'name': 'Acme'}],
            enable_events=True,
        )
        call = client.mode1_calls[-1]
        assert call['output_schema'] == schema
        assert call['input_data'] == [{'name': 'Acme'}]
        assert call['enable_events'] is True
        assert call['extra_body']['use_case'] == 'enrichment'

    async def test_status_result_and_trust_sources(self) -> None:
        client = _FakeNimbleClient()
        toolset = NimbleAgent[None](client=client).get_toolset()
        assert 'running' in _text(await toolset.agent_run_status('wsa_1', 'task_run_1'))
        result = await toolset.agent_run_result('wsa_1', 'task_run_1')
        assert 'done' in _text(result)
        assert result.metadata is not None
        assert result.metadata['sources'] == [{'url': 'https://example.com', 'title': 'Example'}]

    async def test_list_tools(self) -> None:
        agent_item = SimpleNamespace(model_dump=lambda mode='json': {'id': 'agent_1'})
        template_item = SimpleNamespace(model_dump=lambda mode='json': {'template_name': 'research'})
        client = _FakeNimbleClient(agent_items=[agent_item], template_items=[template_item])
        toolset = NimbleAgent[None](client=client).get_toolset()
        assert sorted(toolset.tools) == [
            'agent_run_result',
            'agent_run_start',
            'agent_run_status',
            'agent_templates_list',
            'agents_list',
        ]
        assert 'agent_1' in _text(await toolset.agents_list())
        assert 'research' in _text(await toolset.agent_templates_list())

    def test_from_spec_and_instructions(self) -> None:
        cap = NimbleAgent.from_spec()
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'agent_name' in instructions
        assert 'use_case' in instructions
        assert '3–15 minutes' in instructions or '3-15 minutes' in instructions
        assert NimbleAgent(guidance='').get_instructions() is None

    async def test_after_run_closes_owned_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeNimbleClient()

        def fake_async_nimble(**kwargs: Any) -> _FakeNimbleClient:
            assert kwargs.get('client_source') == 'pydantic-ai'
            return fake

        monkeypatch.setenv('NIMBLE_API_KEY', 'test-key')
        monkeypatch.setattr('pydantic_ai_harness.nimble._toolset.AsyncNimble', fake_async_nimble)
        cap = NimbleAgent[None]()
        await cap.before_run(AsyncMock(spec=RunContext))
        cap.get_toolset()
        result = await cap.after_run(AsyncMock(spec=RunContext), result=AsyncMock(spec=AgentRunResult))
        assert fake.closed is True
        assert result is not None

    async def test_concurrent_runs_do_not_close_shared_client_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeNimbleClient()

        def fake_async_nimble(**kwargs: Any) -> _FakeNimbleClient:
            return fake

        monkeypatch.setenv('NIMBLE_API_KEY', 'test-key')
        monkeypatch.setattr('pydantic_ai_harness.nimble._toolset.AsyncNimble', fake_async_nimble)
        cap = NimbleSearch[None]()
        ctx = AsyncMock(spec=RunContext)
        await cap.before_run(ctx)
        await cap.before_run(ctx)
        await cap.after_run(ctx, result=AsyncMock(spec=AgentRunResult))
        assert fake.closed is False
        await cap.after_run(ctx, result=AsyncMock(spec=AgentRunResult))
        assert fake.closed is True

    def test_missing_api_key_raises_user_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('NIMBLE_API_KEY', raising=False)
        with pytest.raises(UserError, match='NIMBLE_API_KEY'):
            NimbleAgent().get_toolset()


def _status_error(cls: type[AuthenticationError] | type[RateLimitError], status: int, message: str) -> Exception:
    response = httpx.Response(status, request=httpx.Request('POST', 'https://sdk.nimbleway.com'))
    return cls(message, response=response, body={'error': message})


class TestErrorsAndConfig:
    async def test_rate_limit_becomes_model_retry(self) -> None:
        client = _FakeNimbleClient(error=_status_error(RateLimitError, 429, 'slow down'))
        with pytest.raises(ModelRetry, match='rate limit'):
            await _toolset(client).web_search('q')

    async def test_auth_error_propagates(self) -> None:
        client = _FakeNimbleClient(error=_status_error(AuthenticationError, 401, 'bad key'))
        with pytest.raises(AuthenticationError):
            await _toolset(client).web_search('q')

    async def test_httpx_error_becomes_model_retry(self) -> None:
        client = _FakeNimbleClient(error=httpx.ConnectError('offline'))
        with pytest.raises(ModelRetry, match='request failed'):
            await _toolset(client).web_search('q')

    def test_missing_api_key_raises_user_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('NIMBLE_API_KEY', raising=False)
        with pytest.raises(UserError, match='NIMBLE_API_KEY'):
            NimbleSearch().get_toolset()

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match='num_results'):
            NimbleSearch(num_results=0)
        with pytest.raises(ValueError, match='max_text_chars'):
            NimbleSearch(max_text_chars=0)
        with pytest.raises(ValueError, match='include_domains or exclude_domains'):
            NimbleSearch(include_domains=['a.com'], exclude_domains=['b.com'])

    def test_from_spec_and_instructions(self) -> None:
        cap = NimbleSearch.from_spec(include_map=True, include_crawl=True)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'map_site' in instructions
        assert 'crawl_start' in instructions
        assert 'agent_run_start' not in instructions
        assert NimbleSearch(guidance='').get_instructions() is None
        assert NimbleSearch(guidance='custom').get_instructions() == 'custom'

    def test_factory_client_sets_attribution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('NIMBLE_API_KEY', 'test-key')
        created: dict[str, Any] = {}

        def fake_async_nimble(*, api_key: str, client_source: str) -> AsyncMock:
            created['api_key'] = api_key
            created['client_source'] = client_source
            return AsyncMock()

        monkeypatch.setattr('pydantic_ai_harness.nimble._toolset.AsyncNimble', fake_async_nimble)
        NimbleSearch().get_toolset()
        assert created == {'api_key': 'test-key', 'client_source': 'pydantic-ai'}

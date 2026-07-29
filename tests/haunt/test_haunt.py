"""Tests for the HauntExtract capability and HauntExtractToolset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.haunt import HauntExtract, HauntExtractToolset, HttpxHauntClient


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _text(output: ToolReturn[str]) -> str:
    """The model-facing text of a tool result."""
    body = output.return_value
    assert isinstance(body, str)
    return body


def _success(data: object) -> dict[str, object]:
    """A successful Haunt response body."""
    return {'success': True, 'data': data}


def _failure(error_code: str | None, message: str | None = None) -> dict[str, object]:
    """An unsuccessful Haunt response body."""
    body: dict[str, object] = {'success': False}
    if error_code is not None:
        body['error_code'] = error_code
    if message is not None:
        body['message'] = message
    return body


@dataclass
class _FakeHauntClient:
    """In-memory `HauntClient` double: canned responses, recorded call arguments."""

    response: dict[str, object] = field(default_factory=lambda: _success({'markdown': 'alpha'}))
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    async def extract(self, url: str, prompt: str, *, response_format: str | None = None) -> Mapping[str, object]:
        if self.error is not None:
            raise self.error
        self.calls.append({'url': url, 'prompt': prompt, 'response_format': response_format})
        return self.response


def _toolset(client: _FakeHauntClient, *, max_text_chars: int = 50_000) -> HauntExtractToolset[None]:
    return HauntExtract[None](max_text_chars=max_text_chars, client=client).get_toolset()


class TestReadPage:
    async def test_returns_markdown_and_requests_markdown_format(self) -> None:
        client = _FakeHauntClient(response=_success({'markdown': '# Title\n\nBody text.'}))
        output = await _toolset(client).read_page('https://a.dev/page')
        assert _text(output) == '# Title\n\nBody text.'
        assert output.metadata == {'error_code': None}
        assert client.calls == [
            {
                'url': 'https://a.dev/page',
                'prompt': 'Return the readable page content.',
                'response_format': 'markdown',
            }
        ]

    async def test_accepts_bare_string_data(self) -> None:
        client = _FakeHauntClient(response=_success('plain markdown'))
        output = await _toolset(client).read_page('https://a.dev')
        assert _text(output) == 'plain markdown'

    async def test_text_at_cap_is_not_truncated(self) -> None:
        client = _FakeHauntClient(response=_success({'markdown': 'x' * 10}))
        output = await _toolset(client, max_text_chars=10).read_page('https://a.dev')
        assert _text(output) == 'x' * 10

    async def test_text_over_cap_is_truncated_keeping_the_head(self) -> None:
        client = _FakeHauntClient(response=_success({'markdown': 'x' * 11}))
        output = await _toolset(client, max_text_chars=10).read_page('https://a.dev')
        assert _text(output) == f'{"x" * 10}\n[... content truncated at 10 characters]'

    async def test_empty_content_asks_for_retry(self) -> None:
        client = _FakeHauntClient(response=_success({'markdown': ''}))
        with pytest.raises(ModelRetry, match='no content'):
            await _toolset(client).read_page('https://a.dev')

    async def test_unrecognized_data_shape_asks_for_retry(self) -> None:
        client = _FakeHauntClient(response=_success({'html': '<p>not markdown</p>'}))
        with pytest.raises(ModelRetry, match='no content'):
            await _toolset(client).read_page('https://a.dev')


class TestExtractData:
    async def test_returns_data_as_json(self) -> None:
        client = _FakeHauntClient(response=_success({'price': '£4.99', 'in_stock': True}))
        output = await _toolset(client).extract_data('https://a.dev/p', 'the price and stock status')
        parsed = TypeAdapter(dict[str, object]).validate_json(_text(output))
        assert parsed == {'price': '£4.99', 'in_stock': True}
        assert output.metadata == {'error_code': None}
        assert client.calls == [
            {'url': 'https://a.dev/p', 'prompt': 'the price and stock status', 'response_format': None}
        ]

    async def test_string_data_returned_verbatim(self) -> None:
        client = _FakeHauntClient(response=_success('already text'))
        output = await _toolset(client).extract_data('https://a.dev', 'p')
        assert _text(output) == 'already text'

    async def test_missing_data_asks_for_retry(self) -> None:
        client = _FakeHauntClient(response={'success': True})
        with pytest.raises(ModelRetry, match='no data'):
            await _toolset(client).extract_data('https://a.dev', 'p')


class TestHonestFailures:
    @pytest.mark.parametrize('error_code', ['access_denied', 'login_required', 'captcha_required', 'not_found'])
    async def test_page_failures_return_text_not_exceptions(self, error_code: str) -> None:
        client = _FakeHauntClient(response=_failure(error_code, 'The site said no.'))
        output = await _toolset(client).read_page('https://a.dev')
        assert _text(output) == (
            f'Extraction failed for https://a.dev: {error_code}. The site said no.'
            ' The page itself is unavailable; retrying the same URL will not help.'
        )
        assert output.metadata == {'error_code': error_code}

    async def test_extract_data_reports_failures_the_same_way(self) -> None:
        client = _FakeHauntClient(response=_failure('login_required', 'Login wall.'))
        output = await _toolset(client).extract_data('https://a.dev', 'p')
        assert output.metadata == {'error_code': 'login_required'}
        assert 'login_required' in _text(output)

    async def test_unknown_failure_codes_carry_no_finality_hint(self) -> None:
        client = _FakeHauntClient(response=_failure('upstream_fetch_failed', 'Upstream hiccup.'))
        output = await _toolset(client).read_page('https://a.dev')
        assert _text(output) == 'Extraction failed for https://a.dev: upstream_fetch_failed. Upstream hiccup.'
        assert output.metadata == {'error_code': 'upstream_fetch_failed'}

    async def test_failure_without_code_or_message_still_reads_honestly(self) -> None:
        client = _FakeHauntClient(response=_failure(None))
        output = await _toolset(client).read_page('https://a.dev')
        assert _text(output) == (
            'Extraction failed for https://a.dev: extraction_failed. The page could not be extracted.'
        )
        assert output.metadata == {'error_code': 'extraction_failed'}

    async def test_error_field_used_when_message_absent(self) -> None:
        client = _FakeHauntClient(response={'success': False, 'error_code': 'not_found', 'error': 'No such page.'})
        output = await _toolset(client).read_page('https://a.dev')
        assert 'No such page.' in _text(output)


class TestHttpxHauntClient:
    async def test_posts_authenticated_extract_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == '/v1/extract'
            assert request.headers['X-API-Key'] == 'haunt_test'
            body = TypeAdapter(dict[str, object]).validate_json(request.content)
            assert body == {
                'url': 'https://a.dev',
                'prompt': 'Read it.',
                'response_format': 'markdown',
            }
            return httpx.Response(200, json={'success': True, 'data': {'markdown': 'body'}})

        transport = httpx.MockTransport(handler)
        async with HttpxHauntClient('haunt_test', transport=transport) as client:
            payload = await client.extract(
                'https://a.dev',
                'Read it.',
                response_format='markdown',
            )

        assert payload['success'] is True

    async def test_does_not_follow_redirects(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                302,
                headers={'Location': 'https://redirect.example/collect'},
                json={'message': 'Moved'},
            )

        transport = httpx.MockTransport(handler)
        async with HttpxHauntClient('haunt_test', transport=transport) as client:
            with pytest.raises(ModelRetry, match='HTTP status 302'):
                await client.extract('https://a.dev', 'Read it.')

        assert len(requests) == 1
        assert requests[0].url.host == 'hauntapi.com'

    @pytest.mark.parametrize('status_code', [401, 403])
    async def test_auth_failures_are_user_errors(self, status_code: int) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(status_code, request=request))
        async with HttpxHauntClient('bad-key', transport=transport) as client:
            with pytest.raises(UserError, match=f'HTTP status: {status_code}'):
                await client.extract('https://a.dev', 'Read it.')

    async def test_rate_limit_is_model_retry(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                429,
                json={'message': 'Monthly quota exceeded'},
                request=request,
            )
        )
        async with HttpxHauntClient('haunt_test', transport=transport) as client:
            with pytest.raises(ModelRetry, match='Monthly quota exceeded'):
                await client.extract('https://a.dev', 'Read it.')

    async def test_other_non_success_status_is_model_retry(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                503,
                json={'error': 'Temporarily unavailable'},
                request=request,
            )
        )
        async with HttpxHauntClient('haunt_test', transport=transport) as client:
            with pytest.raises(ModelRetry, match='HTTP status 503: Temporarily unavailable'):
                await client.extract('https://a.dev', 'Read it.')

    @pytest.mark.parametrize('content', [b'not json', b'[]'])
    async def test_invalid_response_body_is_model_retry(self, content: bytes) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=content,
                request=request,
            )
        )
        async with HttpxHauntClient('haunt_test', transport=transport) as client:
            with pytest.raises(ModelRetry, match='invalid JSON'):
                await client.extract('https://a.dev', 'Read it.')

    async def test_missing_success_field_is_model_retry(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={'data': {'price': 12}},
                request=request,
            )
        )
        async with HttpxHauntClient('haunt_test', transport=transport) as client:
            with pytest.raises(ModelRetry, match='missing boolean success field'):
                await client.extract('https://a.dev', 'Read it.')


class TestTransientFailures:
    async def test_network_errors_become_model_retry(self) -> None:
        client = _FakeHauntClient(error=httpx.ConnectError('boom'))
        with pytest.raises(ModelRetry, match='Haunt request failed'):
            await _toolset(client).read_page('https://a.dev')

    async def test_timeouts_become_model_retry(self) -> None:
        client = _FakeHauntClient(error=httpx.ReadTimeout('slow'))
        with pytest.raises(ModelRetry, match='Haunt request failed'):
            await _toolset(client).extract_data('https://a.dev', 'p')

    async def test_user_error_propagates(self) -> None:
        client = _FakeHauntClient(error=UserError('bad key'))
        with pytest.raises(UserError, match='bad key'):
            await _toolset(client).read_page('https://a.dev')


class TestHauntExtract:
    def test_default_client_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('HAUNT_API_KEY', raising=False)
        with pytest.raises(UserError, match='HAUNT_API_KEY'):
            HauntExtract[None]().get_toolset()

    async def test_default_client_built_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('HAUNT_API_KEY', 'test-key')
        toolset = HauntExtract[None]().get_toolset()
        assert isinstance(toolset, HauntExtractToolset)
        async with toolset:
            pass

    async def test_default_client_is_recreated_for_a_second_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('HAUNT_API_KEY', 'test-key')
        toolset = HauntExtract[None]().get_toolset()
        async with toolset:
            first_client = toolset._client  # pyright: ignore[reportPrivateUsage]
        async with toolset:
            second_client = toolset._client  # pyright: ignore[reportPrivateUsage]
        assert first_client is not second_client

    def test_max_text_chars_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_text_chars must be at least 1, got 0'):
            HauntExtract[None](max_text_chars=0)

    def test_max_text_chars_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_text_chars must be an integer, got True'):
            HauntExtract[None](max_text_chars=True)

    def test_max_text_chars_float_rejected_at_runtime(self) -> None:
        capability = HauntExtract[None].__new__(HauntExtract)
        object.__setattr__(capability, 'max_text_chars', 1.5)
        with pytest.raises(ValueError, match='max_text_chars must be an integer, got 1.5'):
            capability.__post_init__()

    def test_instructions_reference_the_tools_and_honesty(self) -> None:
        instructions = HauntExtract[None]().get_instructions()
        assert isinstance(instructions, str)
        assert 'read_page' in instructions
        assert 'extract_data' in instructions
        assert 'access_denied' in instructions

    def test_custom_guidance_replaces_default(self) -> None:
        capability = HauntExtract[None](guidance='Extract with the Haunt tools.')
        assert capability.get_instructions() == 'Extract with the Haunt tools.'

    def test_empty_guidance_disables_instructions(self) -> None:
        assert HauntExtract[None](guidance='').get_instructions() is None

    async def test_agent_run_uses_tools_and_instructions(self) -> None:
        client = _FakeHauntClient(response=_success({'markdown': 'page body'}))
        agent = Agent(TestModel(), capabilities=[HauntExtract(client=client)])

        result = await agent.run('Read something.')

        messages = result.all_messages()
        first = messages[0]
        assert isinstance(first, ModelRequest)
        assert first.instructions is not None
        assert 'read_page' in first.instructions

        calls = {
            part.tool_name
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        returns = {
            part.tool_name: part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        assert calls == {'read_page', 'extract_data'}
        assert returns['read_page'] == 'page body'


class TestAgentSpec:
    def test_spec_schema_includes_haunt_extract(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([HauntExtract])
        assert 'HauntExtract' in json.dumps(schema)

    def test_from_spec_builds_capability(self) -> None:
        capability = HauntExtract[None].from_spec(max_text_chars=20_000, guidance='Go.')
        assert capability.max_text_chars == 20_000
        assert capability.guidance == 'Go.'
        assert capability.client is None

    def test_agent_loads_from_spec_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('HAUNT_API_KEY', 'test-key')
        spec = tmp_path / 'agent.yaml'
        spec.write_text('model: test\ncapabilities:\n  - HauntExtract:\n      max_text_chars: 20000\n')
        agent = Agent.from_file(spec, custom_capability_types=[HauntExtract])
        assert isinstance(agent, Agent)

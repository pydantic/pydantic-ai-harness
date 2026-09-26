"""Tests for the `Xberg` capability: configuration, instructions, and behavior through an `Agent`."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.xberg import XBERG_DEFAULT_URL, XBERG_TOOL_NAMES, Xberg, XbergToolset

from .._tool_calls import call_tool
from .conftest import ClientFactory, document, extraction

pytestmark = pytest.mark.anyio


class TestConfiguration:
    def test_defaults_to_a_local_server_and_every_tool(self) -> None:
        capability = Xberg[None]()
        assert capability.url == XBERG_DEFAULT_URL
        assert capability.tools == XBERG_TOOL_NAMES

    def test_builds_the_toolset(self) -> None:
        toolset = Xberg[None]().get_toolset()
        assert isinstance(toolset, XbergToolset)
        assert toolset.id == 'xberg'

    def test_a_given_id_names_the_toolset(self) -> None:
        assert Xberg[None](id='xberg-eu').get_toolset().id == 'xberg-eu'

    def test_tools_are_validated_where_they_are_written(self) -> None:
        with pytest.raises(UserError, match='Unknown Xberg tool'):
            Xberg[None](tools=['extract_everything'])

    def test_headers_are_validated_where_they_are_written(self) -> None:
        with pytest.raises(UserError, match='`headers` cannot set content-type'):
            Xberg[None](headers={'content-type': 'text/plain'})

    def test_limits_are_validated_where_they_are_written(self) -> None:
        with pytest.raises(UserError, match='`timeout` must be a positive number up to 2\\*\\*63 - 1, not nan'):
            Xberg[None](timeout=math.nan)

    async def test_a_client_is_validated_where_it_is_written(self) -> None:
        async with httpx.AsyncClient(headers={'Transfer-Encoding': 'chunked'}) as client:
            with pytest.raises(UserError, match='`http_client` cannot set transfer-encoding'):
                Xberg[None](http_client=client)

    def test_the_repr_names_the_server_but_no_credential(self) -> None:
        capability = Xberg[None](
            url='http://user:secret@xberg.test:8000/gw?api_key=hunter2',
            headers={'Authorization': 'Bearer t0k3n'},
            redact=['hunter3'],
        )
        text = repr(capability)
        assert text.startswith('Xberg(') and "url='http://xberg.test:8000/gw'" in text
        assert not any(secret in text for secret in ('secret', 'hunter2', 't0k3n', 'headers', 'hunter3', 'redact'))

    def test_a_spec_can_list_text_to_redact(self) -> None:
        assert Xberg[None].from_spec(redact=['proxy-user:proxy-secret']).redact == ['proxy-user:proxy-secret']


class TestInstructions:
    def test_instructions_name_the_tools_and_the_size_cap(self) -> None:
        instructions = Xberg[None]().get_instructions()
        assert instructions is not None
        assert '`extract` with one path' in instructions
        assert 'truncated' in instructions

    def test_instructions_follow_the_chosen_tools(self) -> None:
        formats_only = Xberg[None](tools=['list_formats']).get_instructions()
        assert formats_only is not None
        assert '`list_formats`' in formats_only
        assert '`extract`' not in formats_only
        assert 'truncated' not in formats_only

        batch_only = Xberg[None](tools=['extract_batch']).get_instructions()
        assert batch_only is not None
        assert '`extract_batch`' in batch_only
        assert '`extract`' not in batch_only
        assert 'truncated' in batch_only

    def test_instructions_can_be_switched_off(self) -> None:
        assert Xberg[None](include_instructions=False).get_instructions() is None

    def test_description_is_set_for_deferred_loading(self) -> None:
        assert Xberg[None]().description == (
            'Read documents: PDFs, Office files, images, audio, archives, and source files.'
        )


class TestExecution:
    async def test_calls_in_one_response_run_one_at_a_time(self, tmp_path: Path) -> None:
        (tmp_path / 'report.pdf').write_bytes(b'%PDF-1.7')
        in_flight = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await anyio.sleep(0.05)
            in_flight -= 1
            if request.url.path == '/detect':
                return httpx.Response(200, json={'mime_type': 'application/pdf', 'filename': 'report.pdf'})
            return extraction(document('hello'))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('extract', {'path': 'report.pdf'}),
                        ToolCallPart('detect_mime_type', {'path': 'report.pdf'}),
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            capability = Xberg[None](http_client=client, root=tmp_path)
            agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[capability])
            result = await agent.run('read it')

        assert result.output == 'done'
        assert peak == 1


class TestSpecs:
    async def test_a_spec_sets_the_serializable_options(self) -> None:
        spec: dict[str, object] = {
            'capabilities': [{'Xberg': {'url': 'http://xberg.internal:8000', 'tools': ['extract']}}]
        }
        model = TestModel(call_tools=[])
        agent = Agent.from_spec(spec, custom_capability_types=[Xberg], model=model)
        await agent.run('go')

        parameters = model.last_model_request_parameters
        assert parameters is not None
        assert [tool.name for tool in parameters.function_tools] == ['extract']

    @pytest.mark.parametrize(
        ('option', 'complaint'),
        [
            pytest.param(
                {'max_output_bytes': 500.0},
                '`max_output_bytes` must be a positive integer up to 2\\*\\*63 - 1, not 500.0',
                id='float-cap',
            ),
            pytest.param(
                {'max_batch_inputs': 2**63},
                '`max_batch_inputs` must be a positive integer up to 2\\*\\*63 - 1, not 9223372036854775808',
                id='huge-cap',
            ),
            pytest.param(
                {'max_response_values': '1000'},
                "`max_response_values` must be a positive integer up to 2\\*\\*63 - 1, not '1000'",
                id='string-value-cap',
            ),
            pytest.param(
                {'timeout': '5'},
                "`timeout` must be a positive number up to 2\\*\\*63 - 1, not '5'",
                id='string-timeout',
            ),
            pytest.param(
                {'timeout': 10**400},
                '`timeout` must be a positive number up to 2\\*\\*63 - 1, not 1000',
                id='huge-timeout',
            ),
            pytest.param({'url': 5}, '`url` must be text, not 5', id='numeric-url'),
            pytest.param(
                {'url': 'xberg.internal:8000'}, '`url` must start with http:// or https://', id='schemeless-url'
            ),
            pytest.param({'url': 'https://'}, '`url` must name a host', id='hostless-url'),
            pytest.param(
                {'url': 'http://127.0.0.1:99999'}, '`url` port must be between 1 and 65535', id='port-out-of-range'
            ),
            pytest.param({'root': {}}, '`root` must be a path, not \\{\\}', id='mapping-root'),
            pytest.param(
                {'redact': ['top\udcffsecret']},
                '`redact` entries must be text UTF-8 can encode',
                id='surrogate-redact',
            ),
            pytest.param(
                {'headers': []}, '`headers` must map header names to text values, not list', id='list-headers'
            ),
            pytest.param(
                {'headers': False}, '`headers` must map header names to text values, not bool', id='false-headers'
            ),
            pytest.param({'config': {1: 'x'}}, '`config` keys must be text, not 1', id='numeric-config-key'),
            pytest.param(
                {'config': {'chunking': [{2: 'y'}]}}, '`config` keys must be text, not 2', id='nested-config-key'
            ),
            pytest.param(
                {'headers': {'X-Key': 12345}}, '`headers` must map header names to text values', id='numeric-header'
            ),
            pytest.param({'headers': {123: 'v'}}, '`headers` must map header names to text values', id='numeric-name'),
            pytest.param(
                {'config': 'bad'}, "`config` must be a mapping of extraction options, not 'bad'", id='string-config'
            ),
            pytest.param(
                {'config': [1]}, '`config` must be a mapping of extraction options, not \\[1\\]', id='list-config'
            ),
            pytest.param(
                {'config': {'cutoff': date(2026, 9, 22)}},
                '`config` must hold JSON values only: Object of type date is not JSON serializable',
                id='date-in-config',
            ),
            pytest.param(
                {'config': {'ratio': math.nan}},
                '`config` must hold JSON values only: Out of range float values are not JSON compliant',
                id='nan-in-config',
            ),
            pytest.param(
                {'tools': {'extract': False}},
                "`tools` must be a list of Xberg tool names, not \\{'extract': False\\}",
                id='mapping-tools',
            ),
            pytest.param(
                {'tools': ['extract', 1]}, '`tools` must be a list of Xberg tool names, not \\[', id='number-in-tools'
            ),
            pytest.param({'redact': 'hunter2'}, '`redact` must be a list of strings.$', id='string-redact'),
            pytest.param({'redact': [1]}, '`redact` must be a list of strings.$', id='number-in-redact'),
            pytest.param(
                {'output_format': {'x': 1}},
                "`output_format` must be one of plain, markdown, djot, html, json, doctags, not \\{'x': 1\\}",
                id='mapping-output-format',
            ),
            pytest.param(
                {'output_format': 'xml'},
                "`output_format` must be one of plain, markdown, djot, html, json, doctags, not 'xml'",
                id='unknown-output-format',
            ),
            pytest.param(
                {'include_instructions': 'false'},
                "`include_instructions` must be true or false, not 'false'",
                id='string-flag',
            ),
            pytest.param(
                {'defer_loading': 'true'}, "`defer_loading` must be true or false, not 'true'", id='string-defer'
            ),
            pytest.param({'description': 1}, '`description` must be text, not 1', id='numeric-description'),
            pytest.param({'id': 5}, '`id` must be text, not 5', id='numeric-id'),
        ],
    )
    def test_a_spec_cannot_set_an_option_of_the_wrong_type(self, option: dict[str, object], complaint: str) -> None:
        spec: dict[str, object] = {'capabilities': [{'Xberg': option}]}
        with pytest.raises(ValueError, match=complaint):
            Agent.from_spec(spec, custom_capability_types=[Xberg], model=TestModel())

    def test_a_spec_cannot_supply_a_client(self) -> None:
        spec: dict[str, object] = {'capabilities': [{'Xberg': {'http_client': {}}}]}
        with pytest.raises(ValueError, match='http_client'):
            Agent.from_spec(spec, custom_capability_types=[Xberg], model=TestModel())


class TestThroughAgent:
    async def test_the_model_extracts_a_document(self, api_client: ClientFactory, tmp_path: Path) -> None:
        (tmp_path / 'report.pdf').write_bytes(b'%PDF-1.7')
        client = api_client(extraction(document('Revenue rose 4%')))

        returned = await call_tool([Xberg(http_client=client, root=tmp_path)], 'extract', {'path': 'report.pdf'})

        assert 'Revenue rose 4%' in returned

    async def test_the_response_limit_reaches_the_toolset(self, api_client: ClientFactory, tmp_path: Path) -> None:
        (tmp_path / 'report.pdf').write_bytes(b'%PDF-1.7')
        client = api_client(extraction(document('x' * 1_000)))
        capability = Xberg(http_client=client, root=tmp_path, max_response_bytes=100)

        returned = await call_tool([capability], 'extract', {'path': 'report.pdf'})

        assert 'over the 100-byte response limit' in returned

    async def test_the_value_limit_reaches_the_toolset(self, api_client: ClientFactory, tmp_path: Path) -> None:
        (tmp_path / 'report.pdf').write_bytes(b'%PDF-1.7')
        metadata: dict[str, JsonValue] = {f'key{n}': n for n in range(100)}
        client = api_client(extraction(document('hello', metadata=metadata)))
        capability = Xberg(http_client=client, root=tmp_path, max_response_values=50)

        returned = await call_tool([capability], 'extract', {'path': 'report.pdf'})

        assert 'more than 50 JSON values, over the response value limit' in returned

    async def test_the_configured_tools_are_the_ones_offered(self) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Xberg(tools=['extract', 'detect_mime_type'])]).run('go')

        parameters = model.last_model_request_parameters
        assert parameters is not None
        assert sorted(tool.name for tool in parameters.function_tools) == ['detect_mime_type', 'extract']

    async def test_instructions_reach_the_model(self) -> None:
        result = await Agent(TestModel(call_tools=[]), capabilities=[Xberg()]).run('go')

        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert request.instructions is not None
        assert 'Xberg extracts text, tables, and metadata' in request.instructions

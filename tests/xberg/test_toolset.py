"""Tests for `XbergToolset`: the wire it speaks, the limits it applies, and how it fails."""

from __future__ import annotations

import contextlib
import math
import os
import socket
import sys
import threading
from collections.abc import AsyncIterator, Callable, Generator
from pathlib import Path
from types import MappingProxyType

import anyio
import anyio.from_thread
import anyio.to_thread
import httpx
import pytest
from anyio.abc import SocketAttribute, SocketStream
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.xberg import XbergDocument, XbergToolset

from .conftest import ClientFactory, config_of, document, extraction, form_fields, uploaded_filenames

pytestmark = pytest.mark.anyio


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """A file the extraction tools are allowed to read."""
    path = tmp_path / 'report.pdf'
    path.write_bytes(b'%PDF-1.7 report')
    return path


def closed_port() -> int:
    """A port nothing is listening on, for the unreachable-server path."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


class TestToolRegistration:
    def test_default_tools(self) -> None:
        toolset = XbergToolset[None]()
        assert sorted(toolset.tools) == ['detect_mime_type', 'extract', 'extract_batch', 'list_formats']

    def test_tools_can_be_narrowed(self) -> None:
        assert list(XbergToolset[None](tools=['extract']).tools) == ['extract']
        assert list(XbergToolset[None](tools=['list_formats']).tools) == ['list_formats']

    def test_id_reaches_the_toolset(self) -> None:
        assert XbergToolset[None](id='xberg-eu').id == 'xberg-eu'

    def test_rejects_an_empty_tool_list(self) -> None:
        with pytest.raises(UserError, match='must name at least one Xberg tool'):
            XbergToolset[None](tools=[])

    def test_rejects_headers_the_toolset_owns(self) -> None:
        with pytest.raises(UserError, match='`headers` cannot set Content-Length, Transfer-Encoding: the toolset sets'):
            XbergToolset[None](headers={'Content-Length': '1', 'Transfer-Encoding': 'chunked', 'X-Ok': 'yes'})

    def test_rejects_an_unknown_tool(self) -> None:
        with pytest.raises(UserError, match='Unknown Xberg tool\\(s\\) extract_all'):
            XbergToolset[None](tools=['extract', 'extract_all'])

    def test_rejects_a_url_httpx_cannot_send_to(self) -> None:
        with pytest.raises(UserError, match='`url` is not a URL httpx can send to: Invalid non-printable ASCII'):
            XbergToolset[None](url='http://127.0.0.1:8000\n')

    def test_rejects_a_url_that_is_not_http(self) -> None:
        with pytest.raises(UserError, match='`url` must start with http:// or https://'):
            XbergToolset[None](url='ftp://xberg.internal/')

    def test_rejects_a_url_without_a_host(self) -> None:
        with pytest.raises(UserError, match='`url` must name a host'):
            XbergToolset[None](url='http://')

    @pytest.mark.parametrize('port', [99999, -1, 0, 65536])
    def test_rejects_a_url_with_a_port_out_of_range(self, port: int) -> None:
        with pytest.raises(UserError, match='`url` port must be between 1 and 65535'):
            XbergToolset[None](url=f'http://127.0.0.1:{port}')

    def test_rejects_a_url_utf8_cannot_encode(self) -> None:
        with pytest.raises(UserError, match='`url` is not a URL httpx can send to: '):
            XbergToolset[None](url='http://xberg.internal/\udcff')

    def test_rejects_a_redact_entry_utf8_cannot_encode(self) -> None:
        with pytest.raises(UserError, match='`redact` entries must be text UTF-8 can encode.$') as info:
            XbergToolset[None](redact=['top\udcffsecret'])

        assert 'secret' not in str(info.value)

    @pytest.mark.parametrize(
        ('name', 'build'),
        [
            pytest.param('timeout', lambda: XbergToolset[None](timeout=math.nan), id='nan-timeout'),
            pytest.param('timeout', lambda: XbergToolset[None](timeout=math.inf), id='infinite-timeout'),
            pytest.param('timeout', lambda: XbergToolset[None](timeout=0), id='zero-timeout'),
            pytest.param('timeout', lambda: XbergToolset[None](timeout=True), id='boolean-timeout'),
            pytest.param('max_batch_inputs', lambda: XbergToolset[None](max_batch_inputs=True), id='boolean-batch-cap'),
            pytest.param('max_output_bytes', lambda: XbergToolset[None](max_output_bytes=-1), id='negative-output-cap'),
            pytest.param('max_batch_inputs', lambda: XbergToolset[None](max_batch_inputs=0), id='zero-batch-cap'),
            pytest.param('max_upload_bytes', lambda: XbergToolset[None](max_upload_bytes=0), id='zero-upload-cap'),
            pytest.param(
                'max_response_bytes', lambda: XbergToolset[None](max_response_bytes=0), id='zero-response-cap'
            ),
            pytest.param('max_response_values', lambda: XbergToolset[None](max_response_values=0), id='zero-value-cap'),
        ],
    )
    def test_rejects_a_limit_that_is_not_positive(self, name: str, build: Callable[[], XbergToolset[None]]) -> None:
        with pytest.raises(UserError, match=f'`{name}` must be a positive'):
            build()

    async def test_every_tool_runs_alone(self, run_context: RunContext[None]) -> None:
        tools = await XbergToolset[None]().get_tools(run_context)

        assert all(tool.tool_def.sequential for tool in tools.values())

    def test_rejects_tools_that_are_not_a_list(self) -> None:
        with pytest.raises(UserError, match="`tools` must be a list of Xberg tool names, not 'extract'"):
            XbergToolset[None](tools='extract')

    def test_rejects_redact_that_is_not_a_list(self) -> None:
        with pytest.raises(UserError, match='`redact` must be a list of strings.$') as info:
            XbergToolset[None](redact='hunter2')

        assert 'hunter2' not in str(info.value)

    async def test_accepts_any_mapping_as_config(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded: list[httpx.Request],
    ) -> None:
        config = MappingProxyType({'force_ocr': True})
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path, config=config)

        await toolset.extract(run_context, 'report.pdf')

        assert config_of(recorded[0]) == {'force_ocr': True}

    async def test_keeps_its_own_copy_of_the_config(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded: list[httpx.Request],
    ) -> None:
        ocr: dict[str, JsonValue] = {'language': 'eng'}
        config: dict[str, JsonValue] = {'ocr': ocr}
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path, config=config)
        ocr['language'] = 'deu'
        config['later'] = 'added'

        await toolset.extract(run_context, 'report.pdf')

        assert config_of(recorded[0]) == {'ocr': {'language': 'eng'}}

    def test_rejects_a_config_that_refers_to_itself(self) -> None:
        config: dict[str, JsonValue] = {}
        config['self'] = config
        with pytest.raises(UserError, match='`config` must hold JSON values only: Circular reference detected'):
            XbergToolset[None](config=config)

    async def test_accepts_a_config_nested_as_deeply_as_json_writes(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded: list[httpx.Request],
    ) -> None:
        nested: list[JsonValue] = []
        for _ in range(500):
            nested = [nested]
        config: dict[str, JsonValue] = {'deep': nested}
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path, config=config)

        await toolset.extract(run_context, 'report.pdf')

        sent = config_of(recorded[0])
        assert isinstance(sent, dict) and sent['deep'] == nested

    def test_rejects_a_config_nested_too_deeply_for_json(self) -> None:
        nested: list[JsonValue] = []
        for _ in range(400_000):
            nested = [nested]
        raised: list[Exception] = []

        def build() -> None:
            try:
                XbergToolset[None](config={'deep': nested})
            except Exception as e:
                raised.append(e)

        # `json` stops at a count of levels through 3.13 and at the C stack from 3.14, which a main
        # thread may have plenty of; a thread with its own stack makes the refusal the same everywhere.
        threading.stack_size(8 * 1024 * 1024)
        try:
            worker = threading.Thread(target=build)
            worker.start()
            worker.join()
        finally:
            threading.stack_size(0)

        assert [type(e) for e in raised] == [UserError]
        assert '`config` is nested too deeply to write as JSON' in str(raised[0])

    def test_rejects_a_config_value_that_is_not_json(self) -> None:
        with pytest.raises(UserError, match='`config` must hold JSON values only: Out of range float'):
            XbergToolset[None](config={'ratio': math.nan})

    @pytest.mark.parametrize(
        ('headers', 'complaint'),
        [
            pytest.param({'X-Token': 'secret\n'}, "the value of 'X-Token' is not printable ASCII", id='newline'),
            pytest.param({'X-Token': ' secret'}, "the value of 'X-Token' is not printable ASCII", id='leading-blank'),
            pytest.param({'X-Token': 's\u00e9cret'}, "the value of 'X-Token' is not printable ASCII", id='non-ascii'),
            pytest.param({'X Token': 'secret'}, "'X Token' is not a valid header name", id='blank-in-name'),
        ],
    )
    def test_rejects_a_header_the_transport_would_refuse(self, headers: dict[str, str], complaint: str) -> None:
        with pytest.raises(UserError, match='`headers` cannot be sent: ' + complaint) as info:
            XbergToolset[None](headers=headers)

        assert 'secret' not in str(info.value)

    async def test_rejects_a_client_header_the_transport_would_refuse(self) -> None:
        async with httpx.AsyncClient(headers={'X-Token': 'secret\n'}) as client:
            complaint = "`http_client` sets a header that cannot be sent: the value of 'X-Token'"
            with pytest.raises(UserError, match=complaint):
                XbergToolset[None](http_client=client)

    async def test_rejects_a_client_that_frames_requests_itself(self) -> None:
        async with httpx.AsyncClient(headers={'Content-Length': '1', 'Content-Type': 'text/plain'}) as client:
            with pytest.raises(UserError, match='`http_client` cannot set content-length, content-type: the toolset'):
                XbergToolset[None](http_client=client)

    async def test_ctx_is_not_part_of_the_tool_schema(self, run_context: RunContext[None]) -> None:
        tools = await XbergToolset[None]().get_tools(run_context)
        assert 'ctx' not in tools['extract'].tool_def.parameters_json_schema['properties']
        assert tools['extract'].tool_def.parameters_json_schema['required'] == ['path']


class TestExtract:
    async def test_uploads_the_file_and_projects_the_answer(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        response = extraction(
            document(
                'The findings',
                metadata={'page_count': 3},
                tables=[{'markdown': '| a |'}, {'markdown': ''}],
                detected_languages=['eng'],
            )
        )
        toolset = XbergToolset[None](http_client=api_client(response), root=tmp_path)
        result = await toolset.extract(run_context, 'report.pdf')

        assert result == XbergDocument(
            source='report.pdf',
            mime_type='application/pdf',
            content='The findings',
            metadata={'page_count': 3},
            tables=['| a |'],
            detected_languages=['eng'],
        )
        assert str(recorded[0].url) == 'http://127.0.0.1:8000/extract'
        assert uploaded_filenames(recorded[0]) == ['report.pdf']
        assert pdf.read_bytes() in recorded[0].content
        assert form_fields(recorded[0])['output_format'] == 'markdown'
        assert config_of(recorded[0]) is None

    async def test_output_format_and_ocr_arguments_reach_the_server(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())),
            root=tmp_path,
            config={'ocr': {'backend': 'paddleocr'}, 'chunking': {'max_characters': 10}},
        )
        await toolset.extract(run_context, 'report.pdf', output_format='plain', force_ocr=True, ocr_language='deu')

        assert form_fields(recorded[0])['output_format'] == 'plain'
        assert config_of(recorded[0]) == {
            'ocr': {'backend': 'paddleocr', 'language': 'deu'},
            'chunking': {'max_characters': 10},
            'force_ocr': True,
        }

    async def test_configured_output_format_is_the_default(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())), root=tmp_path, output_format='html'
        )
        await toolset.extract(run_context, 'report.pdf')

        assert form_fields(recorded[0])['output_format'] == 'html'

    async def test_ocr_language_without_configured_ocr_settings(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())),
            root=tmp_path,
            config={'ocr': 'not-a-mapping'},
        )
        await toolset.extract(run_context, 'report.pdf', ocr_language='eng')

        assert config_of(recorded[0]) == {'ocr': {'language': 'eng'}}

    async def test_reports_a_failed_input(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(
                    errors=[{'index': 0, 'error_type': 'OcrError', 'source': 'report.pdf', 'message': 'no text layer'}]
                )
            ),
            root=tmp_path,
        )
        with pytest.raises(ModelRetry, match=r"could not extract 'report.pdf' \(OcrError\): no text layer"):
            await toolset.extract(run_context, 'report.pdf')

    async def test_rejects_an_answer_with_no_outcome(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction()), root=tmp_path)
        with pytest.raises(ModelRetry, match='0 results and 0 errors that cannot be paired with the 1 uploaded file'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_rejects_a_result_without_its_identifying_fields(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction({})), root=tmp_path)
        with pytest.raises(ModelRetry, match='returned an unexpected response'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_an_explicit_false_overrides_configured_forced_ocr(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())), root=tmp_path, config={'force_ocr': True}
        )
        await toolset.extract(run_context, 'report.pdf')
        await toolset.extract(run_context, 'report.pdf', force_ocr=False)

        assert config_of(recorded[0]) == {'force_ocr': True}
        assert config_of(recorded[1]) == {'force_ocr': False}

    async def test_reports_an_unexpected_body(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, json={'results': 'nope'})), root=tmp_path
        )
        with pytest.raises(ModelRetry, match='returned an unexpected response: 1 validation error for'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_summarizes_many_validation_errors(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction({}, {}, {}, {}, {}, {})), root=tmp_path)
        with pytest.raises(ModelRetry, match='returned an unexpected response: 12 validation errors$'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_rejects_a_table_without_markdown(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document(tables=[{}]))), root=tmp_path)
        with pytest.raises(ModelRetry, match='returned an unexpected response'):
            await toolset.extract(run_context, 'report.pdf')


class TestExtractBatch:
    async def test_pairs_documents_and_errors_with_their_paths(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path
    ) -> None:
        for name in ('a.pdf', 'b.pdf', 'c.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1.7')

        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(
                    document('first'),
                    document('third'),
                    errors=[{'index': 1, 'source': 'b.pdf', 'error_type': 'ParsingError', 'message': 'corrupt'}],
                )
            ),
            root=tmp_path,
        )
        result = await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf', 'c.pdf'])

        assert [(d.source, d.content) for d in result.documents] == [('a.pdf', 'first'), ('c.pdf', 'third')]
        assert [(e.source, e.error_type, e.message) for e in result.errors] == [('b.pdf', 'ParsingError', 'corrupt')]

    async def test_uploads_every_file_in_order(
        self, run_context: RunContext[None], api_client: ClientFactory, recorded: list[httpx.Request], tmp_path: Path
    ) -> None:
        for name in ('a.pdf', 'b.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1.7')

        toolset = XbergToolset[None](http_client=api_client(extraction(document(), document())), root=tmp_path)
        await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf'])

        assert uploaded_filenames(recorded[0]) == ['a.pdf', 'b.pdf']

    async def test_rejects_an_answer_that_does_not_cover_every_input(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path
    ) -> None:
        for name in ('a.pdf', 'b.pdf', 'c.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1.7')
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(
                    document('first'),
                    errors=[{'index': 1, 'source': 'b.pdf', 'error_type': 'ParsingError', 'message': 'corrupt'}],
                )
            ),
            root=tmp_path,
        )
        with pytest.raises(ModelRetry, match='1 results and 1 errors that cannot be paired with the 3 uploaded files'):
            await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf', 'c.pdf'])

    async def test_error_source_is_the_indexed_path_not_the_uploaded_basename(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path
    ) -> None:
        for folder in ('a', 'b'):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / 'report.pdf').write_bytes(b'%PDF-1.7')
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(
                    document('first'),
                    errors=[{'index': 1, 'error_type': 'ParsingError', 'source': 'report.pdf', 'message': 'corrupt'}],
                )
            ),
            root=tmp_path,
        )
        result = await toolset.extract_batch(run_context, ['a/report.pdf', 'b/report.pdf'])

        assert [d.source for d in result.documents] == ['a/report.pdf']
        assert [e.source for e in result.errors] == ['b/report.pdf']

    @pytest.mark.parametrize(
        'error',
        [
            pytest.param({'error_type': 'ParsingError', 'message': 'bad'}, id='no-index'),
            pytest.param({'index': 0, 'source': 'report.pdf', 'message': 'bad'}, id='no-error-type'),
            pytest.param({'index': 0, 'source': 'report.pdf', 'error_type': 'ParsingError'}, id='no-message'),
            pytest.param({'index': 0, 'error_type': 'ParsingError', 'message': 'bad'}, id='no-source'),
        ],
    )
    async def test_rejects_an_error_missing_a_required_field(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        error: dict[str, JsonValue],
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(errors=[error])), root=tmp_path)
        with pytest.raises(ModelRetry, match='returned an unexpected response'):
            await toolset.extract(run_context, 'report.pdf')

    @pytest.mark.parametrize(
        'errors',
        [
            pytest.param(
                [{'index': -1, 'source': 'a.pdf', 'error_type': 'ParsingError', 'message': 'bad'}], id='negative'
            ),
            pytest.param(
                [{'index': 7, 'source': 'a.pdf', 'error_type': 'ParsingError', 'message': 'bad'}], id='beyond'
            ),
            pytest.param(
                [{'index': 0, 'source': 'a.pdf', 'error_type': 'ParsingError', 'message': 'bad'}] * 2, id='repeated'
            ),
        ],
    )
    async def test_rejects_errors_that_do_not_name_distinct_inputs(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        errors: list[dict[str, JsonValue]],
    ) -> None:
        for name in ('a.pdf', 'b.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1.7')
        results = [document('first')] * (2 - len(errors))
        toolset = XbergToolset[None](http_client=api_client(extraction(*results, errors=errors)), root=tmp_path)
        with pytest.raises(ModelRetry, match='cannot be paired with the 2 uploaded files'):
            await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf'])

    @pytest.mark.parametrize(
        'index', [pytest.param(True, id='boolean'), pytest.param(1.0, id='float'), pytest.param('1', id='string')]
    )
    async def test_rejects_an_error_index_that_is_not_an_integer(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        index: JsonValue,
    ) -> None:
        for name in ('a.pdf', 'b.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1.7')
        answer = extraction(
            document('first'),
            errors=[{'index': index, 'source': 'b.pdf', 'error_type': 'ParsingError', 'message': 'bad'}],
        )
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match='unexpected response: 1 validation error'):
            await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf'])

    async def test_rejects_an_error_whose_source_and_index_disagree(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path
    ) -> None:
        for name in ('a.pdf', 'b.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1.7')
        answer = extraction(
            document('first'), errors=[{'index': 1, 'source': 'a.pdf', 'error_type': 'ParsingError', 'message': 'bad'}]
        )
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match="the error for input 1 names 'a.pdf', not the uploaded 'b.pdf'"):
            await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf'])

    async def test_pairs_an_error_by_the_name_as_multipart_carried_it(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path
    ) -> None:
        (tmp_path / 'a"b.pdf').write_bytes(b'%PDF-1.7')
        error: dict[str, JsonValue] = {
            'index': 0,
            'source': 'a%22b.pdf',
            'error_type': 'ParsingError',
            'message': 'bad',
        }
        toolset = XbergToolset[None](http_client=api_client(extraction(errors=[error])), root=tmp_path)
        with pytest.raises(ModelRetry, match="Xberg could not extract 'a\"b.pdf' \\(ParsingError\\): bad"):
            await toolset.extract(run_context, 'a"b.pdf')

    async def test_reads_with_the_upload_limit_at_its_ceiling(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        answer = api_client(extraction(document()))
        toolset = XbergToolset[None](http_client=answer, root=tmp_path, max_upload_bytes=2**63 - 1)
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.content == 'hello'

    @pytest.mark.skipif(not Path('/proc/self/cmdline').exists(), reason='needs a file that reports no size')
    async def test_reads_a_file_that_reports_no_size(
        self, run_context: RunContext[None], api_client: ClientFactory, recorded: list[httpx.Request]
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root='/proc/self')
        await toolset.extract(run_context, 'cmdline')

        assert Path('/proc/self/cmdline').read_bytes() in recorded[0].content

    @pytest.mark.skipif(not Path('/proc/self/cmdline').exists(), reason='needs a file that reports no size')
    async def test_refuses_a_file_that_reads_past_the_limit(self, run_context: RunContext[None]) -> None:
        size = len(Path('/proc/self/cmdline').read_bytes())
        toolset = XbergToolset[None](root='/proc/self', max_upload_bytes=size - 1)
        with pytest.raises(ModelRetry, match=f"'cmdline' reads past the {size - 1}-byte upload limit\\.$"):
            await toolset.extract(run_context, 'cmdline')

    @pytest.mark.skipif(not Path('/proc/self/cmdline').exists(), reason='needs a file that reports no size')
    async def test_refuses_a_batch_that_reads_past_the_limit(self, run_context: RunContext[None]) -> None:
        size = len(Path('/proc/self/cmdline').read_bytes())
        toolset = XbergToolset[None](root='/proc/self', max_upload_bytes=2 * size - 1)
        with pytest.raises(ModelRetry, match='reads past the .*-byte upload limit\\. Split the batch\\.$'):
            await toolset.extract_batch(run_context, ['cmdline', 'cmdline'])

    async def test_clips_the_path_of_a_failed_batch_input(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        path = './' * 50_000 + 'report.pdf'
        error: dict[str, JsonValue] = {
            'index': 0,
            'source': 'report.pdf',
            'error_type': 'ParsingError',
            'message': 'bad',
        }
        toolset = XbergToolset[None](http_client=api_client(extraction(errors=[error])), root=tmp_path)
        result = await toolset.extract_batch(run_context, [path])

        assert result.errors[0].source == path[:500] + '...'

    async def test_returns_content_as_extracted_even_when_it_holds_a_credential(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        answer = extraction(document('the key is top-secret'))
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path, headers={'X-Key': 'top-secret'})
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.content == 'the key is top-secret'

    async def test_error_text_is_clipped(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(
                    errors=[{'index': 0, 'source': 'report.pdf', 'error_type': 'E' * 600, 'message': 'm' * 5_000}]
                )
            ),
            root=tmp_path,
        )
        result = await toolset.extract_batch(run_context, ['report.pdf'])

        assert result.errors[0].error_type == 'E' * 500 + '...'
        assert result.errors[0].message == 'm' * 500 + '...'

    async def test_rejects_an_empty_batch(self, run_context: RunContext[None]) -> None:
        toolset = XbergToolset[None]()
        with pytest.raises(ModelRetry, match='Name at least one path'):
            await toolset.extract_batch(run_context, [])

    async def test_rejects_a_batch_wider_than_the_limit(self, run_context: RunContext[None], tmp_path: Path) -> None:
        toolset = XbergToolset[None](root=tmp_path, max_batch_inputs=2)
        with pytest.raises(ModelRetry, match='3 paths is more than this server accepts in one batch \\(2\\)'):
            await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf', 'c.pdf'])


class TestSizeLimits:
    async def test_content_is_truncated_to_fit(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('x' * 5_000))), root=tmp_path, max_output_bytes=400
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.truncated
        assert result.omitted == []
        assert result.content.endswith('...[truncated]')
        assert len(result.model_dump_json().encode()) <= 400

    async def test_multibyte_content_stays_valid(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('é' * 2_000))), root=tmp_path, max_output_bytes=300
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert '�' not in result.content
        assert len(result.model_dump_json().encode()) <= 300

    async def test_multibyte_content_keeps_what_fits(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('\U0001f600' * 10_000))), root=tmp_path, max_output_bytes=20_000
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.truncated
        assert result.content.count('\U0001f600') >= 4_900
        assert len(result.model_dump_json().encode()) <= 20_000

    async def test_tables_are_dropped_before_metadata(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(document('x' * 200, tables=[{'markdown': 'y' * 150}], metadata={'author': 'z' * 20}))
            ),
            root=tmp_path,
            max_output_bytes=300,
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.omitted == ['tables']
        assert result.metadata == {'author': 'z' * 20}
        assert len(result.model_dump_json().encode()) <= 300

    async def test_metadata_goes_last(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(document('x' * 100, tables=[{'markdown': 'y' * 100}], metadata={'a': 'z' * 150}))
            ),
            root=tmp_path,
            max_output_bytes=250,
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert (result.omitted, result.truncated) == (['tables', 'metadata'], True)
        assert 0 < result.content.count('x') < 100
        assert len(result.model_dump_json().encode()) <= 250

    async def test_detected_languages_go_after_metadata(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('', metadata={'a': 'zz'}, detected_languages=['l' * 120]))),
            root=tmp_path,
            max_output_bytes=200,
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.omitted == ['metadata', 'detected_languages']
        assert (result.content, result.truncated) == ('', False)
        assert len(result.model_dump_json().encode()) <= 200

    @pytest.mark.parametrize('content', ['x' * 50, ''])
    async def test_a_cap_below_the_identity_is_refused(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path, content: str
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document(content))), root=tmp_path, max_output_bytes=1
        )
        with pytest.raises(ModelRetry, match="'report.pdf' cannot be returned within max_output_bytes"):
            await toolset.extract(run_context, 'report.pdf')

    async def test_content_is_cut_again_once_a_part_is_shed(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('x' * 1_000, tables=[{'markdown': 'y' * 200}]))),
            root=tmp_path,
            max_output_bytes=300,
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert (result.omitted, result.truncated) == (['tables'], True)
        assert result.content.count('x') >= 100
        assert len(result.model_dump_json().encode()) <= 300

    @pytest.mark.parametrize('part', ['tables', 'metadata', 'detected_languages'])
    async def test_a_part_larger_than_the_cap_is_shed_before_the_content(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path, part: str
    ) -> None:
        oversized: dict[str, JsonValue] = {
            'tables': [{'markdown': 't' * 30_000}],
            'metadata': {'blob': 'm' * 30_000},
            'detected_languages': ['l' * 30_000],
        }
        answer = extraction(document('hello', **{part: oversized[part]}))
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path, max_output_bytes=20_000)
        result = await toolset.extract(run_context, 'report.pdf')

        assert (result.content, result.truncated, result.omitted) == ('hello', False, [part])

    async def test_a_cap_below_the_marker_keeps_the_flag(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        bare = XbergDocument(source='report.pdf', mime_type='application/pdf', content='', truncated=True)
        cap = len(bare.model_dump_json().encode()) + 5
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('x' * 50))), root=tmp_path, max_output_bytes=cap
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert (result.content, result.truncated, result.omitted) == ('', True, [])
        assert len(result.model_dump_json().encode()) <= cap

    async def test_content_that_was_not_cut_is_not_marked_truncated(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('', tables=[{'markdown': 'y' * 400}]))),
            root=tmp_path,
            max_output_bytes=300,
        )
        result = await toolset.extract(run_context, 'report.pdf')

        assert (result.content, result.truncated, result.omitted) == ('', False, ['tables'])

    async def test_a_document_that_fits_is_untouched(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document('short'))), root=tmp_path)
        result = await toolset.extract(run_context, 'report.pdf')

        assert (result.content, result.truncated, result.omitted) == ('short', False, [])


class TestFileAccess:
    async def test_refuses_a_path_outside_the_root(self, run_context: RunContext[None], tmp_path: Path) -> None:
        outside = tmp_path.parent / 'outside.pdf'
        outside.write_bytes(b'%PDF')
        toolset = XbergToolset[None](root=tmp_path / 'inside')
        with pytest.raises(ModelRetry, match='resolves outside the extraction root'):
            await toolset.extract(run_context, '../outside.pdf')

    async def test_refuses_a_symlink_that_escapes_the_root(self, run_context: RunContext[None], tmp_path: Path) -> None:
        root = tmp_path / 'root'
        root.mkdir()
        secret = tmp_path / 'secret.pdf'
        secret.write_bytes(b'%PDF')
        (root / 'link.pdf').symlink_to(secret)
        toolset = XbergToolset[None](root=root)
        with pytest.raises(ModelRetry, match='resolves outside the extraction root'):
            await toolset.extract(run_context, 'link.pdf')

    async def test_reports_a_missing_file(self, run_context: RunContext[None], tmp_path: Path) -> None:
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry, match="Could not read 'nope.pdf'"):
            await toolset.extract(run_context, 'nope.pdf')

    async def test_refuses_a_directory(self, run_context: RunContext[None], tmp_path: Path) -> None:
        (tmp_path / 'papers').mkdir()
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry, match="'papers' is not a regular file"):
            await toolset.extract(run_context, 'papers')

    async def test_refuses_an_oversized_file(self, run_context: RunContext[None], tmp_path: Path, pdf: Path) -> None:
        toolset = XbergToolset[None](root=tmp_path, max_upload_bytes=4)
        with pytest.raises(ModelRetry, match="'report.pdf' is 15 bytes, over the 4-byte upload limit"):
            await toolset.extract(run_context, 'report.pdf')

    async def test_the_upload_limit_bounds_a_batch_as_a_whole(
        self, run_context: RunContext[None], tmp_path: Path
    ) -> None:
        for name in ('a.pdf', 'b.pdf'):
            (tmp_path / name).write_bytes(b'%PDF-1')
        toolset = XbergToolset[None](root=tmp_path, max_upload_bytes=10)
        with pytest.raises(
            ModelRetry, match="'b.pdf' takes this request past the 10-byte upload limit. Split the batch."
        ):
            await toolset.extract_batch(run_context, ['a.pdf', 'b.pdf'])

    async def test_refuses_a_symlink_met_at_the_open(self, run_context: RunContext[None], tmp_path: Path) -> None:
        (tmp_path / 'loop.pdf').symlink_to(tmp_path / 'loop.pdf')
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry, match="'loop.pdf' is a symlink loop, or became a symlink after it was checked"):
            await toolset.extract(run_context, 'loop.pdf')

    @pytest.mark.skipif(sys.platform != 'linux', reason='the fallback learns where a file was opened from /proc')
    async def test_the_fallback_open_reads_a_file_inside_the_root(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded: list[httpx.Request],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path)
        monkeypatch.setattr(os, 'supports_dir_fd', set[object]())

        await toolset.extract(run_context, 'report.pdf')

        assert uploaded_filenames(recorded[0]) == ['report.pdf']

    @pytest.mark.skipif(sys.platform != 'linux', reason='the fallback learns where a file was opened from /proc')
    async def test_the_fallback_open_refuses_a_file_a_swap_moved_outside_the_root(
        self, run_context: RunContext[None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root, outside = tmp_path / 'root', tmp_path / 'outside'
        root.mkdir()
        outside.mkdir()
        (outside / 'report.pdf').write_bytes(b'%PDF-1.7 secret')
        (root / 'docs').symlink_to(outside, target_is_directory=True)
        toolset = XbergToolset[None](root=root)
        monkeypatch.setattr(os, 'supports_dir_fd', set[object]())
        monkeypatch.setattr(os.path, 'realpath', os.path.abspath)

        with pytest.raises(
            ModelRetry, match="'docs/report.pdf' is a symlink loop, or became a symlink after it was checked"
        ):
            await toolset.extract(run_context, 'docs/report.pdf')

    async def test_follows_a_symlink_that_stays_inside_the_root(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path
    ) -> None:
        (tmp_path / 'real').mkdir()
        (tmp_path / 'real' / 'report.pdf').write_bytes(b'%PDF-1.7')
        (tmp_path / 'link').symlink_to(tmp_path / 'real')
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path)
        result = await toolset.extract(run_context, 'link/report.pdf')

        assert result.source == 'link/report.pdf'

    async def test_uploads_a_symlink_under_the_name_the_model_asked_for(
        self, run_context: RunContext[None], api_client: ClientFactory, recorded: list[httpx.Request], tmp_path: Path
    ) -> None:
        (tmp_path / 'blob').write_bytes(b'%PDF-1.7')
        (tmp_path / 'report.pdf').symlink_to(tmp_path / 'blob')
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path)
        await toolset.extract(run_context, 'report.pdf')

        assert uploaded_filenames(recorded[0]) == ['report.pdf']
        assert b'Content-Type: application/pdf' in recorded[0].content

    async def test_refuses_a_path_that_is_not_valid_utf8(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path)
        with pytest.raises(ModelRetry, match='is not valid UTF-8, which the upload cannot carry: ') as info:
            await toolset.extract(run_context, 'report\udcff.pdf')

        assert str(info.value).isascii()

    async def test_refuses_a_symlink_loop_in_a_directory_component(
        self, run_context: RunContext[None], tmp_path: Path
    ) -> None:
        (tmp_path / 'loopdir').symlink_to(tmp_path / 'loopdir')
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry, match="'loopdir/report.pdf'(: Not a directory| is a symlink loop)"):
            await toolset.extract(run_context, 'loopdir/report.pdf')

    async def test_the_root_itself_is_not_a_regular_file(self, run_context: RunContext[None], tmp_path: Path) -> None:
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry, match="'\\.' is not a regular file"):
            await toolset.extract(run_context, '.')

    async def test_refuses_a_request_whose_framing_exceeds_the_upload_limit(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path, max_upload_bytes=15)
        with pytest.raises(ModelRetry, match='bytes with its multipart framing, over the 15-byte upload limit'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_refuses_a_path_with_an_embedded_nul(self, run_context: RunContext[None], tmp_path: Path) -> None:
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry, match='is not a valid path'):
            await toolset.extract(run_context, 'report\x00.pdf')


class TestMetadataTools:
    async def test_detect_mime_type(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                httpx.Response(200, json={'mime_type': 'application/pdf', 'filename': 'report.pdf'})
            ),
            root=tmp_path,
        )
        assert await toolset.detect_mime_type(run_context, 'report.pdf') == 'application/pdf'
        assert str(recorded[0].url) == 'http://127.0.0.1:8000/detect'
        assert uploaded_filenames(recorded[0]) == ['report.pdf']

    @pytest.mark.parametrize(
        ('answer', 'complaint'),
        [
            pytest.param(
                {'mime_type': 'application/pdf', 'filename': 'other.pdf'},
                "identified 'other.pdf', not the uploaded 'report.pdf'",
                id='another-file',
            ),
            pytest.param({'mime_type': 'application/pdf'}, 'unexpected response: 1 validation error', id='no-filename'),
        ],
    )
    async def test_refuses_a_detection_that_is_not_of_the_upload(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        answer: dict[str, JsonValue],
        complaint: str,
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, json=answer)), root=tmp_path)
        with pytest.raises(ModelRetry, match=complaint):
            await toolset.detect_mime_type(run_context, 'report.pdf')

    @pytest.mark.parametrize(
        ('name', 'wire'),
        [
            pytest.param('a"b.pdf', 'a%22b.pdf', id='quote'),
            pytest.param('a\nb.pdf', 'a%0Ab.pdf', id='newline'),
            pytest.param('a\\b.pdf', 'a\\\\b.pdf', id='backslash'),
        ],
    )
    async def test_accepts_the_name_as_multipart_carried_it(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, name: str, wire: str
    ) -> None:
        (tmp_path / name).write_bytes(b'%PDF-1.7')
        answer = {'mime_type': 'application/pdf', 'filename': wire}
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, json=answer)), root=tmp_path)

        assert await toolset.detect_mime_type(run_context, name) == 'application/pdf'

    async def test_list_formats(
        self, run_context: RunContext[None], api_client: ClientFactory, recorded: list[httpx.Request]
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                httpx.Response(
                    200,
                    json=[
                        {'extension': 'pdf', 'mime_type': 'application/pdf'},
                        {'extension': 'md', 'mime_type': 'text/markdown'},
                    ],
                )
            )
        )
        assert await toolset.list_formats(run_context) == {'pdf': 'application/pdf', 'md': 'text/markdown'}
        assert recorded[0].method == 'GET'
        assert str(recorded[0].url) == 'http://127.0.0.1:8000/formats'

    async def test_a_format_list_over_the_cap_is_refused(
        self, run_context: RunContext[None], api_client: ClientFactory
    ) -> None:
        formats = [{'extension': f'x{n}', 'mime_type': 'application/x'} for n in range(20)]
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, json=formats)), max_output_bytes=100)
        with pytest.raises(ModelRetry, match='answered with a format list of [0-9]+ bytes, over the 100-byte cap'):
            await toolset.list_formats(run_context)

    async def test_a_format_list_over_the_value_limit_is_refused_unparsed(
        self, run_context: RunContext[None], api_client: ClientFactory
    ) -> None:
        formats = [{'extension': f'x{n}', 'mime_type': 'application/x'} for n in range(20)]
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, json=formats)), max_response_values=50)
        with pytest.raises(ModelRetry, match='answered with more than 50 JSON values, over the response value limit'):
            await toolset.list_formats(run_context)

    async def test_a_mime_type_over_the_cap_is_refused(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, json={'mime_type': 'x' * 200, 'filename': 'report.pdf'})),
            root=tmp_path,
            max_output_bytes=100,
        )
        with pytest.raises(ModelRetry, match='answered with a MIME type of 202 bytes, over the 100-byte cap'):
            await toolset.detect_mime_type(run_context, 'report.pdf')

    async def test_an_encoded_base_path_is_kept_as_written(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())), root=tmp_path, url='http://gateway/tenant%2Fblue'
        )
        await toolset.extract(run_context, 'report.pdf')

        assert recorded[0].url.raw_path == b'/tenant%2Fblue/extract'


class TestFailures:
    async def test_quotes_a_xberg_error_envelope(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(
                httpx.Response(422, json={'error_type': 'ParsingError', 'message': 'cannot parse', 'status_code': 422})
            ),
            root=tmp_path,
        )
        with pytest.raises(ModelRetry, match=r'Xberg ParsingError \(422\): cannot parse'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_quotes_a_body_that_is_not_an_envelope(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(502, text='<html>bad gateway</html>')), root=tmp_path
        )
        with pytest.raises(ModelRetry, match='answered 502: <html>bad gateway</html>'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_reports_an_unreachable_server_without_its_credentials(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        url = f'http://user:secret@127.0.0.1:{closed_port()}/?api_key=hunter2'
        toolset = XbergToolset[None](root=tmp_path, url=url, timeout=5.0)
        with pytest.raises(ModelRetry, match='Could not reach the Xberg API server at http://127.0.0.1:') as info:
            await toolset.extract(run_context, 'report.pdf')
        assert 'secret' not in str(info.value)
        assert 'hunter2' not in str(info.value)

    async def test_clips_the_url_an_unreachable_server_is_named_by(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        url = f'http://127.0.0.1:{closed_port()}/' + 'p' * 10_000
        toolset = XbergToolset[None](root=tmp_path, url=url, timeout=5.0)
        with pytest.raises(ModelRetry, match='Could not reach the Xberg API server at http://127.0.0.1:') as info:
            await toolset.extract(run_context, 'report.pdf')

        assert 'p' * 501 not in str(info.value)
        assert '...: ' in str(info.value)

    async def test_refuses_a_client_that_starts_framing_requests_later(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        client = api_client(extraction(document()))
        toolset = XbergToolset[None](http_client=client, root=tmp_path)
        client.headers['Content-Length'] = '1'
        with pytest.raises(UserError, match='`http_client` cannot set content-length'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_quotes_a_long_path_clipped(self, run_context: RunContext[None], tmp_path: Path) -> None:
        with pytest.raises(ModelRetry, match="Could not read 'x{500}\\.\\.\\.'") as info:
            await XbergToolset[None](root=tmp_path).extract(run_context, 'x' * 100_000)

        assert len(str(info.value)) < 700

    async def test_the_owned_client_ignores_proxy_variables(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ('HTTP_PROXY', 'ALL_PROXY'):
            monkeypatch.setenv(name, f'http://127.0.0.1:{closed_port()}')
        for name in ('NO_PROXY', 'no_proxy'):
            monkeypatch.delenv(name, raising=False)
        payload = extraction(document('direct')).content

        async def serve(stream: SocketStream) -> None:
            """Answer once the whole request is in, reading it in small pieces so no segment size is assumed."""
            async with stream:
                received = bytearray()
                while b'\r\n\r\n' not in received:
                    received.extend(await stream.receive(64))
                head, _, body = bytes(received).partition(b'\r\n\r\n')
                lines = head.split(b'\r\n')
                length = next(
                    int(line.split(b':', 1)[1]) for line in lines if line.lower().startswith(b'content-length:')
                )
                while len(body) < length:
                    body += await stream.receive(64)
                await stream.send(
                    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n'
                    b'Content-Length: %d\r\n\r\n%s' % (len(payload), payload)
                )

        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        url = f'http://127.0.0.1:{listener.extra(SocketAttribute.local_port)}'
        toolset = XbergToolset[None](root=tmp_path, url=url, timeout=5.0)
        async with listener, anyio.create_task_group() as tg:
            tg.start_soon(listener.serve, serve)
            result = await toolset.extract(run_context, 'report.pdf')
            tg.cancel_scope.cancel()
            assert result.content == 'direct'

    async def test_clips_a_transport_error(self, run_context: RunContext[None], tmp_path: Path, pdf: Path) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError('illegal header line: ' + 'x' * 10_000)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='at http://127.0.0.1:8000: illegal header line') as info:
                await toolset.extract(run_context, 'report.pdf')

        message = str(info.value)
        assert 'x' * 479 + '...' in message
        assert 'x' * 480 not in message

    @pytest.mark.parametrize('declared', ['abc', '1, 1', '-1'])
    async def test_refuses_an_answer_whose_declared_length_is_not_a_number(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path, declared: str
    ) -> None:
        answer = extraction(document())
        answer.headers['content-length'] = declared
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match=f'unexpected response: Content-Length {declared!r} is not a length'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_refuses_a_declared_length_too_long_to_convert(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        answer = extraction(document())
        answer.headers['content-length'] = '1' * 5_000
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match='unexpected response: Content-Length ') as info:
            await toolset.extract(run_context, 'report.pdf')

        assert len(str(info.value)) < 700

    async def test_quotes_only_the_head_of_a_long_error_page(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        answer = httpx.Response(502, content=('é' * 5_000).encode())
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match='answered 502: ') as info:
            await toolset.extract(run_context, 'report.pdf')

        assert str(info.value).endswith('é' * 500 + '...')
        assert 'é' * 501 not in str(info.value)

    async def test_refuses_an_answer_declared_over_the_response_limit(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document('x' * 1_000))), root=tmp_path, max_response_bytes=100
        )
        with pytest.raises(ModelRetry, match='over the 100-byte response limit. Extract fewer files in one call.'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_refuses_a_streamed_answer_over_the_response_limit(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b'x' * 60
            yield b'x' * 60

        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=chunks())), root=tmp_path, max_response_bytes=100
        )
        with pytest.raises(ModelRetry, match='over the 100-byte response limit'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_refuses_a_single_chunk_over_the_response_limit(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b'x' * 120

        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=chunks())), root=tmp_path, max_response_bytes=100
        )
        with pytest.raises(ModelRetry, match='over the 100-byte response limit'):
            await toolset.extract(run_context, 'report.pdf')

    @pytest.mark.parametrize('number', [b'NaN', b'1e999', b'[-Infinity]'], ids=['nan', 'overflow', 'nested'])
    async def test_refuses_metadata_with_a_number_that_is_not_finite(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path, number: bytes
    ) -> None:
        body = b'{"results": [{"content": "hi", "mime_type": "application/pdf", "metadata": {"a": ' + number + b'}}]}'
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, content=body)), root=tmp_path)
        with pytest.raises(
            ModelRetry, match='(?s)unexpected response: 1 validation error.*metadata holds a number that is not finite'
        ):
            await toolset.extract(run_context, 'report.pdf')

    async def test_refuses_an_answer_over_the_value_limit_before_parsing_it(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        body = b'{"results": [' + b'1, -2.5e3, true, null, [], {}, "s", ' * 200
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=body)), root=tmp_path, max_response_values=1_000
        )
        with pytest.raises(ModelRetry, match='more than 1000 JSON values, over the response value limit') as info:
            await toolset.extract(run_context, 'report.pdf')

        assert 'unexpected response' not in str(info.value)

    @pytest.mark.parametrize('literal', [b'NaN', b'Infinity', b'-Infinity'], ids=['nan', 'infinity', 'negative'])
    async def test_counts_the_non_finite_literals_the_parser_admits(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path, literal: bytes
    ) -> None:
        body = b'{"results": [{"content": "hi", "mime_type": "x", "metadata": {"a": [' + (literal + b', ') * 200
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=body)), root=tmp_path, max_response_values=100
        )
        with pytest.raises(ModelRetry, match='more than 100 JSON values, over the response value limit'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_counts_no_values_inside_strings(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        content = '[{"1,true,null:' * 200 + '\\"\\\\\n'
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document(content))), root=tmp_path, max_response_values=30
        )
        assert (await toolset.extract(run_context, 'report.pdf')).content == content

    async def test_an_unterminated_string_is_scanned_once(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        body = b'{"results": ["' + b'\\"' * 5_000
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=body)), root=tmp_path, max_response_values=100
        )
        with pytest.raises(ModelRetry, match='returned an unexpected response'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_the_credentials_a_gateway_echoes(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        page = b'upstream failed for /?api_key=hunter2 as user:secret via Basic dXNlcjpzZWNyZXQ= with t0k3n and k3y'

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=page)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={'X-API-Key': 'k3y'}) as client:
            toolset = XbergToolset[None](
                http_client=client,
                root=tmp_path,
                url='http://user:secret@xberg.test:8000/?api_key=hunter2',
                headers={'X-Tenant-Key': 't0k3n'},
            )
            with pytest.raises(ModelRetry, match='answered 502: ') as info:
                await toolset.extract(run_context, 'report.pdf')

        message = str(info.value)
        assert message.endswith(
            'upstream failed for /?[redacted] as [redacted] via [redacted] with [redacted] and [redacted]'
        )
        assert not any(secret in message for secret in ('hunter2', 'secret', 'dXNlcjpzZWNyZXQ=', 't0k3n', 'k3y'))

    async def test_redacts_each_value_of_a_repeated_header(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b'rejected secret-one')

        repeated = [('X-Key', 'secret-one'), ('X-Key', 'secret-two')]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=repeated) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: rejected \\[redacted\\]$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'secret-one' not in str(info.value)

    async def test_redacts_a_credential_longer_than_the_clip(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        key = 'k' * 4_000

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=f'bad key {key} rejected'.encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={'X-Key': key}) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: bad key \\[redacted\\] rejected$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'kkkk' not in str(info.value)

    async def test_redacts_a_cookie_a_redirect_set(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/extract':
                return httpx.Response(302, headers={'Location': '/gateway/extract', 'Set-Cookie': 'session=abc123'})
            value = request.headers['cookie'].partition('=')[2]
            return httpx.Response(500, content=b'rejected cookie ' + value.encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 500: rejected cookie \\[redacted\\]$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'abc123' not in str(info.value)

    async def test_redacts_what_a_redirect_added_when_the_transport_then_fails(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/extract':
                headers = {'Location': '/gateway/extract?token=redirect-secret', 'Set-Cookie': 'session=abc123'}
                return httpx.Response(302, headers=headers)
            raise httpx.RemoteProtocolError(f'bad response for {request.url} with {request.headers["cookie"]}')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='/gateway/extract\\?\\[redacted\\] with \\[redacted\\]\\.') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'redirect-secret' not in str(info.value) and 'abc123' not in str(info.value)

    async def test_reports_a_transport_error_a_hook_raised(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        async def hook(_: httpx.Request) -> None:
            raise httpx.ConnectError('hook refused')

        async with httpx.AsyncClient(event_hooks={'request': [hook]}) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='at http://127.0.0.1:8000: hook refused'):
                await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_a_credential_a_redirect_added_to_the_url(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/extract':
                return httpx.Response(302, headers={'Location': '/gateway/extract?token=redirect-secret'})
            return httpx.Response(500, content=f'failed at {request.url}'.encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='/gateway/extract\\?\\[redacted\\]$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'redirect-secret' not in str(info.value)

    async def test_redacts_what_every_hop_of_a_redirect_chain_added(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/extract':
                return httpx.Response(302, headers={'Location': '/mid?token=redirect-secret'})
            if request.url.path == '/mid':
                return httpx.Response(302, headers={'Location': '/final'})
            return httpx.Response(502, content=b'upstream rejected redirect-secret')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: upstream rejected \\[redacted\\]$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'redirect-secret' not in str(info.value)

    async def test_redacts_a_credential_an_auth_flow_rotated(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        class Rotating(httpx.Auth):
            requires_response_body = True

            def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
                request.headers['Authorization'] = 'Bearer first-secret'
                yield request
                request.headers['Authorization'] = 'Bearer second-secret'
                yield request

        def handler(request: httpx.Request) -> httpx.Response:
            if request.headers['Authorization'] == 'Bearer first-secret':
                return httpx.Response(401, content=b'challenge')
            return httpx.Response(502, content=b'upstream rejected first-secret and second-secret')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=Rotating()) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: upstream rejected \\[redacted\\] and \\[redacted\\]$'):
                await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_a_credential_that_equals_a_value_httpx_generates(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b'credential keep-alive rejected')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={'X-Key': 'keep-alive'}) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: credential \\[redacted\\] rejected$'):
                await toolset.extract(run_context, 'report.pdf')

    async def test_keeps_the_values_httpx_generates_readable(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        page = f'sent keep-alive and */* as python-httpx/{httpx.__version__} with identity'.encode()
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(502, content=page)), root=tmp_path)
        with pytest.raises(ModelRetry, match='answered 502: sent keep-alive and \\*/\\* as python-httpx/') as info:
            await toolset.extract(run_context, 'report.pdf')

        assert '[redacted]' not in str(info.value)

    async def test_redacts_a_credential_glued_to_other_text(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b'prefix-top-secret-suffix')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'X-Key': 'top-secret'})
            with pytest.raises(ModelRetry, match='answered 502: prefix-\\[redacted\\]-suffix$'):
                await toolset.extract(run_context, 'report.pdf')

    @pytest.mark.parametrize(
        ('secret', 'page', 'shown'),
        [
            pytest.param('k3y', b'prefix-k3y-suffix', 'prefix-[redacted]-suffix', id='glued'),
            pytest.param(
                'on',
                b'status on, region london',
                'status [redacted], regi[redacted] l[redacted]d[redacted]',
                id='inside-words',
            ),
            pytest.param('dact', b'token dact rejected', 'token [redacted] rejected', id='marker-safe'),
        ],
    )
    async def test_redacts_a_short_secret_wherever_it_appears(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, secret: str, page: bytes, shown: str
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=page)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'X-Key': secret})
            with pytest.raises(ModelRetry) as info:
                await toolset.extract(run_context, 'report.pdf')

        assert str(info.value).endswith(f'answered 502: {shown}')

    @pytest.mark.parametrize(
        ('name', 'value', 'page', 'shown'),
        [
            pytest.param(
                'Accept-Encoding', 'TOPSECRET', b'prefix-TOPSECRET-suffix', 'prefix-[redacted]-suffix', id='encoding'
            ),
            pytest.param(
                'Content-Type',
                'multipart/form-data; boundary=TOPSECRET',
                b'boundary TOPSECRET rejected',
                'boundary [redacted] rejected',
                id='boundary',
            ),
            pytest.param(
                'Content-Type',
                'multipart/form-data; boundary="TOPSECRET"',
                b'boundary TOPSECRET rejected',
                'boundary [redacted] rejected',
                id='quoted-boundary',
            ),
            pytest.param(
                'Content-Type', 'text/TOPSECRET', b'type text/TOPSECRET rejected', 'type [redacted] rejected', id='type'
            ),
            pytest.param('Content-Length', '4242', b'length 4242 rejected', 'length [redacted] rejected', id='length'),
            pytest.param(
                'Transfer-Encoding',
                'TOPSECRET',
                b'coding TOPSECRET rejected',
                'coding [redacted] rejected',
                id='coding',
            ),
        ],
    )
    async def test_redacts_a_reserved_header_a_hook_replaced(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, name: str, value: str, page: bytes, shown: str
    ) -> None:
        async def hook(request: httpx.Request) -> None:
            request.headers[name] = value

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=page)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), event_hooks={'request': [hook]}) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry) as info:
                await toolset.extract(run_context, 'report.pdf')

        assert str(info.value).endswith(f'answered 502: {shown}')

    async def test_keeps_the_framing_httpx_wrote_readable(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            framing = f'sent {request.headers["content-type"]} of {request.headers["content-length"]} bytes'
            return httpx.Response(502, content=framing.encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(
                ModelRetry, match='answered 502: sent multipart/form-data; boundary=[0-9a-f]+ of \\d+ bytes$'
            ) as info:
                await toolset.extract(run_context, 'report.pdf')

        assert '[redacted]' not in str(info.value)

    async def test_redacts_a_header_a_hook_added_that_the_transport_refused(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        async def serve(stream: SocketStream) -> None:
            async with stream:
                with contextlib.suppress(anyio.EndOfStream):
                    while True:
                        await stream.receive(64)

        async def hook(request: httpx.Request) -> None:
            request.headers['X-Token'] = 'hooked\n'

        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        url = f'http://127.0.0.1:{listener.extra(SocketAttribute.local_port)}'
        async with listener, anyio.create_task_group() as tg:
            tg.start_soon(listener.serve, serve)
            async with httpx.AsyncClient(event_hooks={'request': [hook]}) as client:
                toolset = XbergToolset[None](http_client=client, root=tmp_path, url=url)
                with pytest.raises(ModelRetry, match="Illegal header value b'\\[redacted\\]'") as info:
                    await toolset.extract(run_context, 'report.pdf')
                assert 'hooked' not in str(info.value)
            tg.cancel_scope.cancel()

    @pytest.mark.parametrize(
        ('header', 'complaint'),
        [
            pytest.param('Content-Encoding', "answered with '\\[redacted\\]' content encoding", id='encoding'),
            pytest.param('Content-Length', "Content-Length '\\[redacted\\]' is not a length", id='length'),
        ],
    )
    async def test_redacts_a_credential_echoed_in_a_response_header(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, header: str, complaint: str
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'{}', headers={header: 'top-secret'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'X-Key': 'top-secret'})
            with pytest.raises(ModelRetry, match=complaint) as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'top-secret' not in str(info.value)

    async def test_redacts_an_uppercase_credential_echoed_as_the_content_encoding(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'{}', headers={'Content-Encoding': 'TOP-SECRET'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'X-Key': 'TOP-SECRET'})
            with pytest.raises(ModelRetry, match="answered with '\\[redacted\\]' content encoding") as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'top-secret' not in str(info.value).lower()

    async def test_redacts_the_host_a_supplied_client_sets(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b'host top-secret rejected')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={'Host': 'top-secret'}) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: host \\[redacted\\] rejected$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'top-secret' not in str(info.value)

    async def test_keeps_the_host_the_url_derives_readable(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        response = httpx.Response(502, content=b'host 127.0.0.1:8000 rejected')
        toolset = XbergToolset[None](http_client=api_client(response), root=tmp_path)
        with pytest.raises(ModelRetry, match='answered 502: host 127.0.0.1:8000 rejected$'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_quotes_an_error_body_over_the_value_limit_rather_than_parsing_it(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        body = b'{"error_type": "X", "message": "bad", "extra": [1, 2, 3, 4, 5, 6]}'
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(500, content=body)), root=tmp_path, max_response_values=2
        )
        with pytest.raises(ModelRetry, match='answered 500: \\{"error_type": "X", "message": "bad"') as info:
            await toolset.extract(run_context, 'report.pdf')

        assert 'Xberg X' not in str(info.value)

    async def test_quotes_an_error_body_too_large_to_be_an_envelope(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        body = b'{"error_type": "GatewayError", "message": "' + b'x' * (1 << 20) + b'"}'
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(502, content=body)), root=tmp_path)
        with pytest.raises(ModelRetry, match='answered 502: \\{"error_type": "GatewayError", "message": "x+\\.\\.\\.$'):
            await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_a_credential_the_clip_would_cut(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        failure = {'error_type': 'GatewayError', 'message': 'x' * 489 + ' top-secret-token ' + 'y' * 100_000}

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, json=failure)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'X-Key': 'top-secret-token'})
            with pytest.raises(ModelRetry, match='GatewayError \\(502\\): x{489} \\[redacted\\]\\.\\.\\.$') as info:
                await toolset.extract(run_context, 'report.pdf')

        assert 'top-secret' not in str(info.value)

    @pytest.mark.parametrize(
        ('entry', 'echo'),
        [
            pytest.param(
                'http://proxy-user:proxy-s%40cret@proxy.test:3128',
                'auth cHJveHktdXNlcjpwcm94eS1zQGNyZXQ= refused',
                id='url-blob',
            ),
            pytest.param(
                'http://proxy-user:proxy-s%40cret@proxy.test:3128',
                'login proxy-user:proxy-s@cret refused',
                id='url-pair',
            ),
            pytest.param('proxy-user:proxy-s@cret', 'auth cHJveHktdXNlcjpwcm94eS1zQGNyZXQ= refused', id='pair-blob'),
            pytest.param('proxy-user:proxy-s@cret', 'password proxy-s@cret refused', id='pair-password'),
            pytest.param('sec\nret', 'token sec\nret refused', id='not-a-url'),
        ],
    )
    async def test_redacts_a_credential_listed_in_redact(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, entry: str, echo: str
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(407, content=echo.encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, redact=[entry])
            with pytest.raises(ModelRetry, match='answered 407: \\w+ \\[redacted\\] refused$'):
                await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_a_quoted_cookie_value_without_its_quotes(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b'cookie top-secret rejected')

        cookie = {'Cookie': 'session="top-secret"'}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=cookie) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: cookie \\[redacted\\] rejected$'):
                await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_a_credential_under_a_plain_header_name(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b'agent top-secret-api-key rejected')

        agent = {'User-Agent': 'top-secret-api-key'}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=agent) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: agent \\[redacted\\] rejected$'):
                await toolset.extract(run_context, 'report.pdf')

    @pytest.mark.parametrize(
        ('credential', 'echo'),
        [
            pytest.param('Bearer t0k3n', 'token t0k3n rejected', id='bearer-token'),
            pytest.param('Basic dXNlcjpzZWNyZXQ=', 'credential dXNlcjpzZWNyZXQ= rejected', id='basic-blob'),
            pytest.param('Basic dXNlcjpzZWNyZXQ=', 'login user:secret rejected', id='basic-pair'),
            pytest.param('Basic dXNlcjpzZWNyZXQ', 'password secret rejected', id='basic-unpadded-password'),
            pytest.param('Basic a', 'token a rejected', id='basic-undecodable'),
            pytest.param('t0k3n', 'token t0k3n rejected', id='no-scheme'),
        ],
    )
    async def test_redacts_the_parts_of_an_authorization_header(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, credential: str, echo: str
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=echo.encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'Authorization': credential})
            with pytest.raises(ModelRetry, match='answered 502: \\w+ \\[redacted\\] rejected$'):
                await toolset.extract(run_context, 'report.pdf')

    async def test_redacts_the_credentials_added_during_the_send(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            echoed = f'rejected {request.headers["authorization"]} and {request.headers["x-token"]}'
            return httpx.Response(502, content=echoed.encode())

        async def add_token(request: httpx.Request) -> None:
            request.headers['X-Token'] = 't0k3n'

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=httpx.BasicAuth('user', 'secret'),
            event_hooks={'request': [add_token]},
        )
        async with client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(ModelRetry, match='answered 502: ') as info:
                await toolset.extract(run_context, 'report.pdf')

        message = str(info.value)
        assert message.endswith('rejected [redacted] and [redacted]')
        assert 'dXNlcjpzZWNyZXQ=' not in message and 't0k3n' not in message

    async def test_a_client_timeout_is_a_timeout_not_an_unreachable_server(
        self,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout('timed out')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path)
            with pytest.raises(
                ModelRetry, match='did not finish answering within the client timeout \\(ReadTimeout\\)'
            ):
                await toolset.extract(recorded_context(False), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert attributes['xberg.refusal'] == 'timeout'

    async def test_an_answer_within_the_response_limit_is_read_whole(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            body = extraction(document('streamed')).content
            yield body[:10]
            yield body[10:]

        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, content=chunks())), root=tmp_path)
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.content == 'streamed'

    async def test_refuses_an_encoded_answer(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=b'x', headers={'content-encoding': 'gzip'})),
            root=tmp_path,
        )
        with pytest.raises(ModelRetry, match="answered with 'gzip' content encoding"):
            await toolset.extract(run_context, 'report.pdf')

    async def test_a_long_content_coding_is_clipped(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=b'x', headers={'content-encoding': 'z' * 5_000})),
            root=tmp_path,
        )
        with pytest.raises(ModelRetry, match="answered with 'z{500}\\.\\.\\.' content encoding"):
            await toolset.extract(run_context, 'report.pdf')

    async def test_asks_for_an_unencoded_answer(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path)
        await toolset.extract(run_context, 'report.pdf')

        assert recorded[0].headers.get_list('accept-encoding') == ['identity']

    async def test_accepts_the_identity_coding_in_any_case(
        self, run_context: RunContext[None], api_client: ClientFactory, tmp_path: Path, pdf: Path
    ) -> None:
        answer = extraction(document('plain'))
        answer.headers['content-encoding'] = 'Identity'
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        result = await toolset.extract(run_context, 'report.pdf')

        assert result.content == 'plain'

    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    async def test_the_timeout_bounds_the_whole_request(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        async def trickle(stream: SocketStream) -> None:
            """Answer forever, a byte at a time, so no single read ever times out."""
            async with stream:
                await stream.receive()
                await stream.send(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n')
                with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                    while True:
                        await stream.send(b'1\r\nx\r\n')
                        await anyio.sleep(0.05)

        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        url = f'http://127.0.0.1:{listener.extra(SocketAttribute.local_port)}'
        toolset = XbergToolset[None](root=tmp_path, url=url, timeout=0.5)
        async with listener, anyio.create_task_group() as tg:
            tg.start_soon(listener.serve, trickle)
            with pytest.raises(ModelRetry, match='did not finish answering within 0.5 seconds'):
                await toolset.extract(run_context, 'report.pdf')
            tg.cancel_scope.cancel()

    async def test_a_supplied_client_keeps_its_own_timeout(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        client = api_client(extraction(document()))
        client.timeout = httpx.Timeout(30.0)
        toolset = XbergToolset[None](http_client=client, root=tmp_path, timeout=0.001)
        await toolset.extract(run_context, 'report.pdf')

        assert recorded[0].extensions['timeout'] == {
            'connect': 30.0,
            'pool': 30.0,
            'read': 30.0,
            'write': 30.0,
        }

    async def test_headers_are_sent(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())),
            root=tmp_path,
            headers={'x-gateway-token': 'secret'},
        )
        await toolset.extract(run_context, 'report.pdf')

        assert recorded[0].headers['x-gateway-token'] == 'secret'

    async def test_a_trailing_slash_does_not_double_up(
        self,
        run_context: RunContext[None],
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
    ) -> None:
        toolset = XbergToolset[None](
            http_client=api_client(extraction(document())),
            root=tmp_path,
            url='http://xberg.test:8000/',
        )
        await toolset.extract(run_context, 'report.pdf')

        assert str(recorded[0].url) == 'http://xberg.test:8000/extract'


class TestCancellation:
    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    @pytest.mark.parametrize('supplied', [False, True], ids=['owned-client', 'supplied-client'])
    @pytest.mark.parametrize('phase', ['request', 'response'])
    @pytest.mark.filterwarnings(
        # Trio warns when it finalizes the multipart body generator the cancel abandoned.
        "ignore:Async generator 'httpx._multipart.MultipartStream.__aiter__' was garbage collected:ResourceWarning"
    )
    async def test_a_cancelled_call_releases_its_connection(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path, supplied: bool, phase: str
    ) -> None:
        started = anyio.Event()
        peer_closed = anyio.Event()

        async def hold(stream: SocketStream) -> None:
            """Take the request in, answer partly or not at all, then wait for the peer to hang up."""
            async with stream:
                await stream.receive()
                if phase == 'response':
                    await stream.send(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{"\r\n')
                started.set()
                with contextlib.suppress(anyio.EndOfStream, anyio.BrokenResourceError):
                    while True:
                        await stream.receive()
                peer_closed.set()

        async def cancel_once_started(scope: anyio.CancelScope) -> None:
            await started.wait()
            scope.cancel()

        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        url = f'http://127.0.0.1:{listener.extra(SocketAttribute.local_port)}'
        async with httpx.AsyncClient() as owned_by_the_test, listener, anyio.create_task_group() as tg:
            toolset = XbergToolset[None](root=tmp_path, url=url, http_client=owned_by_the_test if supplied else None)
            tg.start_soon(listener.serve, hold)
            with anyio.CancelScope() as scope:
                tg.start_soon(cancel_once_started, scope)
                await toolset.extract(run_context, 'report.pdf')
            assert scope.cancelled_caught
            with anyio.fail_after(5):
                await peer_closed.wait()
            tg.cancel_scope.cancel()

    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    @pytest.mark.filterwarnings(
        "ignore:Async generator 'httpx._multipart.MultipartStream.__aiter__' was garbage collected:ResourceWarning"
    )
    async def test_a_cancelled_call_names_its_cancellation_on_the_span(
        self,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        started = anyio.Event()

        async def hold(stream: SocketStream) -> None:
            async with stream:
                await stream.receive()
                started.set()
                with contextlib.suppress(anyio.EndOfStream, anyio.BrokenResourceError):
                    while True:
                        await stream.receive()

        async def cancel_once_started(scope: anyio.CancelScope) -> None:
            await started.wait()
            scope.cancel()

        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        url = f'http://127.0.0.1:{listener.extra(SocketAttribute.local_port)}'
        async with listener, anyio.create_task_group() as tg:
            toolset = XbergToolset[None](root=tmp_path, url=url)
            tg.start_soon(listener.serve, hold)
            with anyio.CancelScope() as scope:
                tg.start_soon(cancel_once_started, scope)
                await toolset.extract(recorded_context(False), 'report.pdf')
            assert scope.cancelled_caught
            tg.cancel_scope.cancel()

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert attributes['xberg.exception_type'] == anyio.get_cancelled_exc_class().__name__


class TestUploadCancellation:
    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    async def test_a_cancelled_call_does_not_wait_for_an_upload_slot(
        self, run_context: RunContext[None], tmp_path: Path, pdf: Path
    ) -> None:
        """The upload phase honors an outer cancel while its worker thread is still owed a slot."""
        limiter = anyio.to_thread.current_default_thread_limiter()
        tokens = limiter.total_tokens
        occupied = anyio.Event()
        release = threading.Event()

        def occupy() -> None:
            anyio.from_thread.run_sync(occupied.set)
            release.wait()

        async def cancel_once_waiting(scope: anyio.CancelScope) -> None:
            while limiter.statistics().tasks_waiting == 0:
                await anyio.sleep(0)
            scope.cancel()

        limiter.total_tokens = 1
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(anyio.to_thread.run_sync, occupy)
                await occupied.wait()
                toolset = XbergToolset[None](root=tmp_path, url='http://127.0.0.1:9')
                with anyio.CancelScope() as scope:
                    tg.start_soon(cancel_once_waiting, scope)
                    await toolset.extract(run_context, 'report.pdf')
                assert scope.cancelled_caught
                release.set()
        finally:
            limiter.total_tokens = tokens


class TestTelemetry:
    async def test_span_records_the_request_and_the_answer(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        (tmp_path / 'second.pdf').write_bytes(b'%PDF')
        toolset = XbergToolset[None](
            http_client=api_client(
                extraction(
                    document('x' * 5_000),
                    errors=[{'index': 1, 'source': 'second.pdf', 'error_type': 'OcrError', 'message': 'no text'}],
                )
            ),
            root=tmp_path,
            max_output_bytes=500,
        )
        await toolset.extract_batch(recorded_context(False), ['report.pdf', 'second.pdf'])

        spans = {span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()}
        assert spans['xberg_extract'] == {
            'xberg.url': 'http://127.0.0.1:8000',
            'xberg.max_output_bytes': 500,
            'xberg.output_format': 'markdown',
            'xberg.inputs': 2,
            'xberg.documents': 1,
            'xberg.errors': 1,
            'xberg.truncated_documents': 1,
        }

    async def test_url_credentials_and_path_prefix_reach_the_request_but_not_the_span(
        self,
        api_client: ClientFactory,
        recorded: list[httpx.Request],
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        url = 'http://user:secret@xberg.test:8000/gw/?api_key=hunter2'
        for include_content in (False, True):
            toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path, url=url)
            await toolset.extract(recorded_context(include_content), 'report.pdf')

        assert (recorded[0].url.path, recorded[0].url.params['api_key']) == ('/gw/extract', 'hunter2')
        urls = [dict(span.attributes or {})['xberg.url'] for span in exporter.get_finished_spans()]
        assert urls == ['http://xberg.test:8000', 'http://xberg.test:8000/gw']

    async def test_metadata_tools_record_their_result_size(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        formats = httpx.Response(200, json=[{'extension': 'pdf', 'mime_type': 'application/pdf'}])
        await XbergToolset[None](http_client=api_client(formats)).list_formats(recorded_context(False))
        detection = httpx.Response(200, json={'mime_type': 'application/pdf', 'filename': 'report.pdf'})
        toolset = XbergToolset[None](http_client=api_client(detection), root=tmp_path)
        await toolset.detect_mime_type(recorded_context(True), 'report.pdf')

        spans = {span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()}
        assert spans['xberg_formats'] == {
            'xberg.url': 'http://127.0.0.1:8000',
            'xberg.max_output_bytes': 20_000,
            'xberg.result_bytes': len(b'{"pdf":"application/pdf"}'),
        }
        assert spans['xberg_detect'] == {
            'xberg.url': 'http://127.0.0.1:8000',
            'xberg.max_output_bytes': 20_000,
            'xberg.sources': ('report.pdf',),
            'xberg.result_bytes': len(b'"application/pdf"'),
        }

    async def test_a_refused_result_records_its_size_beside_the_cap(
        self,
        api_client: ClientFactory,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        formats = [{'extension': f'x{n}', 'mime_type': 'application/x'} for n in range(20)]
        toolset = XbergToolset[None](http_client=api_client(httpx.Response(200, json=formats)), max_output_bytes=100)
        with pytest.raises(ModelRetry):
            await toolset.list_formats(recorded_context(False))

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        size = attributes['xberg.result_bytes']
        assert attributes['xberg.max_output_bytes'] == 100
        assert isinstance(size, int) and size > 100
        assert attributes['xberg.exception_type'] == 'ModelRetry'

    async def test_a_single_file_failure_is_recorded_on_the_span(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        answer = extraction(
            errors=[{'index': 0, 'source': 'report.pdf', 'error_type': 'OcrError', 'message': 'no text'}]
        )
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match='Xberg could not extract'):
            await toolset.extract(recorded_context(False), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert attributes['xberg.errors'] == 1
        assert attributes['xberg.exception_type'] == 'ModelRetry'

    async def test_a_refused_batch_is_recorded_on_the_span(
        self,
        tmp_path: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        toolset = XbergToolset[None](root=tmp_path, max_batch_inputs=2)
        with pytest.raises(ModelRetry, match='Split the batch'):
            await toolset.extract_batch(recorded_context(False), ['a.pdf', 'b.pdf', 'c.pdf'])

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert attributes['xberg.inputs'] == 3
        assert attributes['xberg.exception_type'] == 'ModelRetry'
        assert 'xberg.documents' not in attributes

    async def test_a_refusal_names_its_reason_and_sizes(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path, max_upload_bytes=4)
        with pytest.raises(ModelRetry, match='over the 4-byte upload limit'):
            await toolset.extract(recorded_context(False), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert (attributes['xberg.refusal'], attributes['xberg.limit'], attributes['xberg.measured']) == (
            'upload_limit',
            4,
            len(pdf.read_bytes()),
        )
        assert 'report.pdf' not in str(attributes)

    async def test_a_refused_answer_records_the_response_limit(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        answer = extraction(document('x' * 1_000))
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path, max_response_bytes=100)
        with pytest.raises(ModelRetry, match='over the 100-byte response limit'):
            await toolset.extract(recorded_context(False), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert (attributes['xberg.refusal'], attributes['xberg.limit']) == ('response_limit', 100)
        assert attributes['xberg.measured'] == len(answer.content)

    async def test_a_refused_answer_records_the_value_limit(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        body = b'[' + b'true,' * 30
        toolset = XbergToolset[None](
            http_client=api_client(httpx.Response(200, content=body)), root=tmp_path, max_response_values=20
        )
        with pytest.raises(ModelRetry, match='over the response value limit'):
            await toolset.extract(recorded_context(False), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert (attributes['xberg.refusal'], attributes['xberg.limit'], attributes['xberg.measured']) == (
            'value_limit',
            20,
            21,
        )

    async def test_a_server_error_records_its_status(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        answer = httpx.Response(503, json={'error_type': 'Busy', 'message': 'later'})
        toolset = XbergToolset[None](http_client=api_client(answer), root=tmp_path)
        with pytest.raises(ModelRetry, match='Xberg Busy \\(503\\): later'):
            await toolset.extract(recorded_context(False), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert (attributes['xberg.refusal'], attributes['xberg.status_code']) == ('server_error', 503)
        assert 'xberg.limit' not in attributes

    async def test_a_failure_records_only_the_exception_type(
        self,
        tmp_path: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry):
            await toolset.extract(recorded_context(False), 'nope.pdf')

        span = exporter.get_finished_spans()[0]
        attributes = dict(span.attributes or {})
        assert attributes['xberg.exception_type'] == 'ModelRetry'
        assert not span.events
        assert 'nope.pdf' not in str(attributes)

    async def test_a_failure_is_recorded_when_content_is_traced(
        self,
        tmp_path: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        toolset = XbergToolset[None](root=tmp_path)
        with pytest.raises(ModelRetry):
            await toolset.extract(recorded_context(True), 'nope.pdf')

        span = exporter.get_finished_spans()[0]
        assert [event.name for event in span.events] == ['exception']

    async def test_a_recorded_failure_leaves_out_the_exceptions_it_was_raised_from(
        self,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('failed top-secret')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            toolset = XbergToolset[None](http_client=client, root=tmp_path, headers={'X-Key': 'top-secret'})
            with pytest.raises(ModelRetry):
                await toolset.extract(recorded_context(True), 'report.pdf')

        (event,) = exporter.get_finished_spans()[0].events
        attributes = dict(event.attributes or {})
        assert attributes['exception.type'] == 'pydantic_ai.exceptions.ModelRetry'
        assert 'failed [redacted]' in str(attributes['exception.message'])
        assert 'Traceback' in str(attributes['exception.stacktrace'])
        assert not any('top-secret' in str(value) for value in attributes.values())

    async def test_sources_are_content(
        self,
        api_client: ClientFactory,
        tmp_path: Path,
        pdf: Path,
        recorded_context: Callable[[bool], RunContext[None]],
        exporter: InMemorySpanExporter,
    ) -> None:
        toolset = XbergToolset[None](http_client=api_client(extraction(document())), root=tmp_path)
        await toolset.extract(recorded_context(True), 'report.pdf')

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert attributes['xberg.sources'] == ('report.pdf',)

"""Shared fixtures for the Xberg capability tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import replace

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

ClientFactory = Callable[[httpx.Response], httpx.AsyncClient]
"""Builds a client that answers every request with the given response, recording each request."""


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def run_context() -> RunContext[None]:
    """Minimal `RunContext` for invoking toolset methods directly in tests."""
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)


@pytest.fixture
def recorded() -> list[httpx.Request]:
    """Every request the clients from `api_client` received, in order."""
    return []


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def recorded_context(
    run_context: RunContext[None], exporter: InMemorySpanExporter
) -> Iterator[Callable[[bool], RunContext[None]]]:
    """A `RunContext` whose tracer records, with content inclusion switchable."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def build(include_content: bool) -> RunContext[None]:
        return replace(run_context, tracer=provider.get_tracer('xberg-tests'), trace_include_content=include_content)

    yield build
    provider.shutdown()


@pytest.fixture
async def api_client(recorded: list[httpx.Request]) -> AsyncIterator[ClientFactory]:
    """Factory for clients backed by `httpx.MockTransport`, closed at teardown."""
    clients: list[httpx.AsyncClient] = []

    def build(response: httpx.Response) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            recorded.append(request)
            return response

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    yield build
    for client in clients:
        await client.aclose()


def extraction(
    *results: dict[str, JsonValue],
    errors: Sequence[dict[str, JsonValue]] = (),
) -> httpx.Response:
    """A `/extract` response in the server's own envelope."""
    return httpx.Response(
        200,
        json={
            'results': list(results),
            'errors': list(errors),
            'summary': {'inputs': len(results) + len(errors), 'results': len(results), 'errors': len(errors)},
        },
    )


def document(content: str = 'hello', **fields: JsonValue) -> dict[str, JsonValue]:
    """One `/extract` result, with the server's own defaults for anything unnamed."""
    return {'content': content, 'mime_type': 'application/pdf', **fields}


def _parts(request: httpx.Request) -> list[tuple[str, str | None, str]]:
    """Each multipart part of a request as `(name, filename, value)`."""
    boundary = request.headers['content-type'].split('boundary=')[1]
    parts: list[tuple[str, str | None, str]] = []
    for part in request.content.decode('utf-8', errors='replace').split(f'--{boundary}'):
        if 'name="' not in part:
            continue
        headers, _, value = part.partition('\r\n\r\n')
        filename = headers.split('filename="', 1)[1].split('"', 1)[0] if 'filename="' in headers else None
        parts.append((headers.split('name="', 1)[1].split('"', 1)[0], filename, value.rsplit('\r\n', 1)[0]))
    return parts


def form_fields(request: httpx.Request) -> dict[str, str]:
    """The non-file multipart fields of a request, by name."""
    return {name: value for name, filename, value in _parts(request) if filename is None}


def uploaded_filenames(request: httpx.Request) -> list[str]:
    """Every filename the request uploaded, in order."""
    return [filename for _, filename, _ in _parts(request) if filename is not None]


def config_of(request: httpx.Request) -> JsonValue:
    raw = form_fields(request).get('config')
    return None if raw is None else json.loads(raw)

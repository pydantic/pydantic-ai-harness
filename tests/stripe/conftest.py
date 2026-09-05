"""Test configuration for the Stripe capability."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import pytest
from pydantic import JsonValue

pytest.importorskip('fastmcp')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class StripeServer:
    """Protocol-level Stripe MCP fake and the requests it received."""

    headers: list[dict[str, str]] = field(default_factory=lambda: [])
    urls: list[str] = field(default_factory=lambda: [])
    follow_redirects: list[bool] = field(default_factory=lambda: [])
    status_code: int | None = None
    redirect_to: str | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(dict(request.headers))
        self.urls.append(str(request.url))
        if self.redirect_to is not None:
            return httpx.Response(307, headers={'Location': self.redirect_to})
        if self.status_code is not None:
            return httpx.Response(self.status_code)
        message: JsonValue = json.loads(request.content)
        if not isinstance(message, dict):  # pragma: no cover
            return httpx.Response(400)

        method = message.get('method')
        request_id = message.get('id')
        if method == 'notifications/initialized':
            return httpx.Response(202)
        if method == 'initialize':
            result: JsonValue = {
                'protocolVersion': '2025-06-18',
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'stripe-fake', 'version': '1'},
            }
        elif method == 'tools/list':
            result = {'tools': [_tool(name) for name in _TOOL_NAMES]}
        elif method == 'tools/call':
            params = message.get('params')
            name = params.get('name') if isinstance(params, dict) else None
            mode = 'write' if name == 'stripe_api_write' else 'read'
            result = {'content': [{'type': 'text', 'text': f'{{"mode":"{mode}"}}'}]}
        else:  # pragma: no cover
            return httpx.Response(400)
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': request_id, 'result': result})


_TOOL_NAMES = (
    'get_stripe_account_info',
    'stripe_api_search',
    'stripe_api_details',
    'stripe_api_read',
    'stripe_api_write',
    'search_stripe_documentation',
    'fetch_stripe_documentation',
    'send_stripe_feedback',
)


def _tool(name: str) -> dict[str, JsonValue]:
    return {
        'name': name,
        'description': f'Fake {name}',
        'inputSchema': {'type': 'object', 'additionalProperties': True},
    }


@pytest.fixture
async def stripe_server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[StripeServer]:
    """Route the canonical Stripe URL to a protocol-level MCP fake."""
    server = StripeServer()
    real_async_client = httpx.AsyncClient

    def make_client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = False,
    ) -> httpx.AsyncClient:
        server.follow_redirects.append(follow_redirects)
        return real_async_client(
            transport=httpx.MockTransport(server.handle),
            base_url='https://mcp.stripe.com',
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
        )

    monkeypatch.setattr(httpx, 'AsyncClient', make_client)
    yield server

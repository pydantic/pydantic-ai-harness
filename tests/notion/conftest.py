"""Fixtures for the Notion hosted MCP integration."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ._support import NotionState

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

collect_ignore = (
    ['test_notion.py', 'test_example.py']
    if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('fastmcp') is None
    else []
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


@pytest.fixture
def notion_state() -> NotionState:
    return {
        'calls': [],
        'ai_search_status': 'available',
        'missing_access_tools': set(),
        'mutation_error': False,
        'page_content': 'Old launch plan',
        'user_id': 'user-1',
        'unavailable_tools': set(),
        'workspace_id': 'workspace-1',
    }


@pytest.fixture
def notion_server(notion_state: NotionState) -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-fake')

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> dict[str, object]:
        """Fetch a Notion object or the current connection identity."""
        notion_state['calls'].append(('notion-fetch', {'id': id}))
        if id == 'self':
            current_tool_access = {
                'ai_search': {'status': notion_state['ai_search_status']},
                'create_database': {'status': 'available'},
                'fetch': {'status': 'available'},
                'get_users': {'status': 'available'},
                'query_meeting_notes': {'status': 'available'},
                'search': {'status': 'available'},
                'update_page': {'status': 'available'},
            }
            current_tool_access.update({name: {'status': 'not_enabled'} for name in notion_state['unavailable_tools']})
            for name in notion_state['missing_access_tools']:
                current_tool_access.pop(name, None)
            return {
                'self': {
                    'workspace': {'id': notion_state['workspace_id'], 'name': 'Acme'},
                    'user': {
                        'id': notion_state['user_id'],
                        'name': 'Ada',
                        'type': 'person',
                        'email': 'ada@example.com',
                    },
                    'current_tool_access': current_tool_access,
                }
            }
        return {
            'id': id,
            'url': f'https://notion.so/{id}',
            'path': 'Acme / Launch plan',
            'content': notion_state['page_content'],
        }

    @server.tool(name='notion-search')
    def search(query: str) -> list[dict[str, str]]:
        """Search Notion by exact keyword."""
        notion_state['calls'].append(('notion-search', {'query': query}))
        return [{'id': 'page-1', 'title': query, 'url': 'https://notion.so/page-1'}]

    @server.tool(name='notion-ai-search')
    def ai_search(query: str) -> list[dict[str, str]]:
        """Search Notion semantically."""
        notion_state['calls'].append(('notion-ai-search', {'query': query}))
        return [{'id': 'page-1', 'title': 'Launch plan', 'url': 'https://notion.so/page-1'}]

    @server.tool(name='notion-get-users')
    def get_users() -> list[dict[str, str]]:  # pragma: no cover
        """List users."""
        return [{'id': 'user-1', 'name': 'Ada'}]

    @server.tool(name='notion-query-meeting-notes')
    def query_meeting_notes(query: str) -> list[dict[str, str]]:  # pragma: no cover
        """Query meeting notes."""
        return [{'id': 'meeting-1', 'title': query}]

    @server.tool(name='notion-update-page')
    def update_page(page_id: str, command: str, new_str: str) -> dict[str, str]:
        """Update a Notion page."""
        notion_state['calls'].append(
            ('notion-update-page', {'page_id': page_id, 'command': command, 'new_str': new_str})
        )
        notion_state['page_content'] = new_str
        if notion_state['mutation_error']:
            raise RuntimeError('ambiguous provider failure')
        return {'id': page_id, 'url': f'https://notion.so/{page_id}'}

    @server.tool(name='notion-create-database')
    def create_database(title: str) -> dict[str, str]:  # pragma: no cover
        """Create a database."""
        return {'id': 'database-1', 'title': title}

    @server.tool(name='notion-delete-workspace')
    def delete_workspace() -> str:  # pragma: no cover
        """A hypothetical server tool that the integration must not expose."""
        return 'deleted'

    return server


@pytest.fixture
def attribution_error_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-attribution-error')

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> str:
        """Fail connection identity lookup."""
        raise RuntimeError(f'identity unavailable for {id}')

    return server


@pytest.fixture
def oversized_attribution_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-oversized-attribution')

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> str:
        """Return an identity response above the integration limit."""
        return f'{id}:{"x" * 17_000}'

    return server


@pytest.fixture
def malformed_attribution_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-malformed-attribution')

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> dict[str, object]:
        """Return success without the required identity fields."""
        return {'id': id}

    return server


@pytest.fixture
def non_text_attribution_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415
    from mcp.types import ImageContent  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-non-text-attribution')

    @server.tool(name='notion-fetch')
    def fetch() -> object:
        """Return a non-text identity response."""
        return ImageContent(type='image', data='eA==', mimeType='image/png')

    return server


@pytest.fixture
def oversized_identity_field_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-oversized-identity-field')

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> dict[str, object]:
        """Return an individual identity field above its limit."""
        return {
            'self': {
                'workspace': {'id': 'workspace-1', 'name': 'x' * 300},
                'user': {'id': 'user-1', 'name': 'Ada', 'type': 'person'},
                'current_tool_access': {'ai_search': {'status': 'available'}},
            }
        }

    return server


@pytest.fixture
def attributed_server_with_meta() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-attribution-meta')

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> dict[str, object]:
        """Return valid identity plus untrusted envelope and nested metadata."""
        return {
            'self': {
                'workspace': {'id': 'workspace-1', 'name': 'Acme', '_meta': 'workspace-secret'},
                'user': {'id': 'user-1', 'name': 'Ada', 'type': 'person', '_meta': 'user-secret'},
                'current_tool_access': {
                    'ai_search': {'status': 'available', '_meta': 'access-secret'},
                    'fetch': {'status': 'available'},
                },
                '_meta': 'self-secret',
            },
            '_meta': 'envelope-secret',
        }

    return server


@pytest.fixture
def rotating_identity_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('notion-rotating-identity')
    fetch_count = 0

    @server.tool(name='notion-fetch')
    def fetch(id: str) -> dict[str, object]:
        """Change workspace identity after initial discovery."""
        nonlocal fetch_count
        fetch_count += 1
        return {
            'self': {
                'workspace': {'id': f'workspace-{fetch_count}', 'name': 'Acme'},
                'user': {'id': 'user-1', 'name': 'Ada', 'type': 'person'},
                'current_tool_access': {
                    'ai_search': {'status': 'available'},
                    'fetch': {'status': 'available'},
                    'search': {'status': 'available'},
                    'update_page': {'status': 'available'},
                },
            }
        }

    @server.tool(name='notion-update-page')
    def update_page(page_id: str, command: str, new_str: str) -> dict[str, str]:  # pragma: no cover
        """Return an update result that must not be reached."""
        return {'page_id': page_id, 'command': command, 'new_str': new_str}

    @server.tool(name='notion-search')
    def search(query: str) -> list[dict[str, str]]:  # pragma: no cover
        """Return a result that must not be reached after identity changes."""
        return [{'id': 'page-1', 'title': query}]

    return server

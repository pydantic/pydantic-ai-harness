"""Google Workspace connection composition and tool selection."""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.google_workspace import GoogleWorkspace

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestGoogleWorkspace:
    @pytest.mark.parametrize(('services', 'message'), [([], 'at least one'), (['mail'], 'Unknown Google')])
    def test_unknown_services(self, services: list[str], message: str) -> None:
        with pytest.raises(UserError, match=message):
            GoogleWorkspace(services=services)  # pyright: ignore[reportArgumentType]

    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GOOGLE_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='needs a token'):
            GoogleWorkspace('gmail').get_toolset()

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(GoogleWorkspace('gmail', auth='secret-token'))

    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"gmail_read_item":"read","gmail_write_item":"written","gmail_unmarked_item":"unmarked"}'),
            (True, '{"gmail_read_item":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(
        self, connections: list[tuple[str, httpx.Auth | str | None]], read_only: bool, expected: str
    ) -> None:
        agent = Agent(TestModel(), capabilities=[GoogleWorkspace('gmail', auth='token', read_only=read_only)])
        assert (await agent.run('Use the tools')).output == expected

    async def test_product_connections_preserve_falsey_auth(
        self, connections: list[tuple[str, httpx.Auth | str | None]]
    ) -> None:
        class FalseyAuth(httpx.BasicAuth):
            def __bool__(self) -> bool:
                return False

        auth = FalseyAuth('user', 'secret')
        assert not auth
        agent = Agent(TestModel(call_tools=[]), capabilities=[GoogleWorkspace(['gmail', 'calendar'], auth=auth)])
        await agent.run('Hello')
        assert connections == [
            ('https://gmailmcp.googleapis.com/mcp/v1', auth),
            ('https://calendarmcp.googleapis.com/mcp/v1', auth),
        ]

    def test_environment_token(
        self, connections: list[tuple[str, httpx.Auth | str | None]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'environment-token')
        GoogleWorkspace('people').get_toolset()
        assert connections == [('https://people.googleapis.com/mcp/v1', 'environment-token')]

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(
        self, connections: list[tuple[str, httpx.Auth | str | None]], include: bool
    ) -> None:
        capability = GoogleWorkspace('gmail', auth='token', include_instructions=include)
        result = await Agent(TestModel(call_tools=[]), capabilities=[capability]).run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Google instructions.' in (request.instructions or '')) is include

    async def test_duplicate_services(self, connections: list[tuple[str, httpx.Auth | str | None]]) -> None:
        capability = GoogleWorkspace(['gmail', 'gmail'], auth='token', read_only=True)
        result = await Agent(TestModel(), capabilities=[capability]).run('Read')
        assert result.output == '{"gmail_read_item":"read"}'

"""Shared fixtures for the GitHub integration tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

collect_ignore = (
    ['test_github.py'] if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('fastmcp') is None else []
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
def github_calls() -> list[tuple[str, dict[str, object]]]:
    return []


@pytest.fixture
def github_server(github_calls: list[tuple[str, dict[str, object]]]) -> FastMCP:  # noqa: C901
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415
    from mcp.types import ToolAnnotations  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('github-fake')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_file_contents(owner: str, repo: str, path: str) -> dict[str, str]:
        """Read a file."""
        arguments: dict[str, object] = {'owner': owner, 'repo': repo, 'path': path}
        github_calls.append(('get_file_contents', arguments))
        return {'owner': owner, 'repo': repo, 'path': path, 'content': 'print("hello")'}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_code(query: str) -> str:
        """Search code."""
        return query

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_commits(query: str) -> str:
        """Search commits."""
        return query

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_issues(query: str, owner: str | None = None, repo: str | None = None) -> dict[str, str | None]:
        """Search issues."""
        return {'query': query, 'owner': owner, 'repo': repo}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_pull_requests(query: str) -> str:
        """Search pull requests."""
        return query

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def pull_request_read(owner: str, repo: str, pullNumber: int, method: str = 'get') -> dict[str, object]:
        """Read a pull request."""
        arguments: dict[str, object] = {
            'owner': owner,
            'repo': repo,
            'pullNumber': pullNumber,
            'method': method,
        }
        github_calls.append(('pull_request_read', arguments))
        return arguments

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def list_teams(org: str) -> str:
        """List organization teams."""
        return org

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_me() -> str:
        """Get the current user."""
        return 'octocat'  # pragma: no cover

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def fork_repository(owner: str, repo: str, organization: str | None = None) -> dict[str, str | None]:
        """Fork a repository to an optional destination organization."""
        return {'owner': owner, 'repo': repo, 'organization': organization}  # pragma: no cover

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def repository_ruleset_read(
        level: str,
        owner: str | None = None,
        repo: str | None = None,
        org: str | None = None,
        enterprise: str | None = None,
    ) -> dict[str, str | None]:
        """Read a ruleset selected by a target-level discriminator."""
        return {  # pragma: no cover
            'level': level,
            'owner': owner,
            'repo': repo,
            'org': org,
            'enterprise': enterprise,
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def issue_dependency_write(
        method: str,
        type: str,
        owner: str,
        repo: str,
        issue_number: int,
        related_issue_number: int,
        related_owner: str | None = None,
        related_repo: str | None = None,
    ) -> dict[str, object]:
        """Write an issue dependency that can target another repository."""
        arguments: dict[str, object] = {
            'method': method,
            'type': type,
            'owner': owner,
            'repo': repo,
            'issue_number': issue_number,
            'related_issue_number': related_issue_number,
            'related_owner': related_owner,
            'related_repo': related_repo,
        }
        github_calls.append(('issue_dependency_write', arguments))
        return arguments

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def issue_write(
        method: str,
        owner: str,
        repo: str,
        issue_number: int | None = None,
        parent_issue_number: int | None = None,
        parent_owner: str | None = None,
        parent_repo: str | None = None,
        title: str | None = None,
    ) -> dict[str, object]:
        """Create an issue with an optional cross-repository parent."""
        arguments: dict[str, object] = {
            'method': method,
            'owner': owner,
            'repo': repo,
            'issue_number': issue_number,
            'parent_issue_number': parent_issue_number,
            'title': title,
            'parent_owner': parent_owner,
            'parent_repo': parent_repo,
        }
        github_calls.append(('issue_write', arguments))
        return arguments

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def list_issue_fields(owner: str, repo: str | None = None) -> dict[str, str | None]:
        """List repository or organization issue fields."""
        arguments: dict[str, object] = {'owner': owner, 'repo': repo}
        github_calls.append(('list_issue_fields', arguments))
        return {'owner': owner, 'repo': repo}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def sub_issue_write(owner: str, repo: str, issue_number: int, sub_issue_id: int) -> dict[str, object]:
        """Change a sub-issue relationship using an opaque target ID."""
        return {  # pragma: no cover
            'owner': owner,
            'repo': repo,
            'issue_number': issue_number,
            'sub_issue_id': sub_issue_id,
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def add_sub_issue(owner: str, repo: str, issue_number: int, sub_issue_id: int) -> dict[str, object]:
        """Add a sub-issue using an opaque target ID."""
        return {  # pragma: no cover
            'owner': owner,
            'repo': repo,
            'issue_number': issue_number,
            'sub_issue_id': sub_issue_id,
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def remove_sub_issue(owner: str, repo: str, issue_number: int, sub_issue_id: int) -> dict[str, object]:
        """Remove a sub-issue using an opaque target ID."""
        return {  # pragma: no cover
            'owner': owner,
            'repo': repo,
            'issue_number': issue_number,
            'sub_issue_id': sub_issue_id,
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def reprioritize_sub_issue(owner: str, repo: str, issue_number: int, sub_issue_id: int) -> dict[str, object]:
        """Reprioritize a sub-issue using an opaque target ID."""
        return {  # pragma: no cover
            'owner': owner,
            'repo': repo,
            'issue_number': issue_number,
            'sub_issue_id': sub_issue_id,
        }

    @server.tool()
    def unclassified_tool(owner: str, repo: str) -> str:
        """A tool without safety annotations."""
        return f'{owner}/{repo}'  # pragma: no cover

    return server

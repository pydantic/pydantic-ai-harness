"""In-process Atlassian MCP boundary for tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

collect_ignore = ['test_atlassian.py'] if importlib.util.find_spec('mcp') is None else []


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def run_context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)


@pytest.fixture
def atlassian_calls() -> list[str]:
    return []


@pytest.fixture
def atlassian_server(atlassian_calls: list[str]) -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('atlassian-fake')

    @server.tool()
    def atlassianUserInfo() -> dict[str, str]:
        """Return the authenticated Atlassian user."""
        atlassian_calls.append('atlassianUserInfo')
        return {'accountId': 'user-1'}

    @server.tool()
    def getJiraIssue(cloudId: str, issueIdOrKey: str) -> dict[str, str]:
        """Get one Jira work item."""
        atlassian_calls.append('getJiraIssue')
        return {'cloudId': cloudId, 'key': issueIdOrKey, 'summary': 'Fix login'}

    @server.tool()
    def searchJiraIssuesUsingJql(cloudId: str, jql: str) -> list[dict[str, str]]:
        """Search Jira work items."""
        atlassian_calls.append('searchJiraIssuesUsingJql')
        return [{'cloudId': cloudId, 'key': 'ENG-42', 'jql': jql}]

    @server.tool()
    def createJiraIssue(cloudId: str, projectKey: str, summary: str) -> dict[str, str]:
        """Create one Jira work item."""
        atlassian_calls.append('createJiraIssue')
        return {'cloudId': cloudId, 'key': f'{projectKey}-43', 'summary': summary}

    @server.tool()
    def deleteJiraIssue(cloudId: str, issueIdOrKey: str) -> dict[str, str]:
        """Permanently delete one Jira work item."""
        atlassian_calls.append('deleteJiraIssue')
        return {'cloudId': cloudId, 'deleted': issueIdOrKey}

    @server.tool()
    def getConfluenceContent(cloudId: str, contentId: str) -> dict[str, str]:
        """Get one Confluence content item."""
        atlassian_calls.append('getConfluenceContent')
        return {'cloudId': cloudId, 'id': contentId}

    @server.tool()
    def createConfluenceContent(cloudId: str, title: str) -> dict[str, str]:
        """Create one Confluence content item."""
        atlassian_calls.append('createConfluenceContent')
        return {'cloudId': cloudId, 'title': title}

    @server.tool()
    def getJsmOpsAlerts(cloudId: str) -> list[dict[str, str]]:
        """Get Jira Service Management alerts."""
        atlassian_calls.append('getJsmOpsAlerts')
        return [{'cloudId': cloudId, 'id': 'alert-1'}]

    @server.tool()
    def updateJsmOpsAlert(cloudId: str, alertId: str) -> dict[str, str]:
        """Update one Jira Service Management alert."""
        atlassian_calls.append('updateJsmOpsAlert')
        return {'cloudId': cloudId, 'id': alertId}

    @server.tool()
    def getBitbucketRepository(cloudId: str, workspace: str, repoSlug: str) -> dict[str, str]:
        """Get one Bitbucket repository."""
        atlassian_calls.append('getBitbucketRepository')
        return {'cloudId': cloudId, 'workspace': workspace, 'slug': repoSlug}

    @server.tool()
    def createBitbucketRepoPullRequest(cloudId: str, workspace: str, repoSlug: str) -> dict[str, str]:
        """Create one Bitbucket pull request."""
        atlassian_calls.append('createBitbucketRepoPullRequest')
        return {'cloudId': cloudId, 'workspace': workspace, 'slug': repoSlug}

    # An exact allowlist is expected never to execute this fake server tool.
    @server.tool()
    def futureUnreviewedAtlassianMutation(cloudId: str) -> dict[str, str]:  # pragma: no cover
        """A future server tool that Harness has not reviewed."""
        return {'cloudId': cloudId}

    return server


@pytest.fixture
def unavailable_atlassian_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('atlassian-unavailable-fake')

    # A fail-closed allowlist is expected never to execute this fake server tool.
    @server.tool()
    def futureUnreviewedAtlassianMutation(cloudId: str) -> dict[str, str]:  # pragma: no cover
        """A future server tool that Harness has not reviewed."""
        return {'cloudId': cloudId}

    return server


@pytest.fixture
def jira_only_atlassian_server() -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('atlassian-jira-only-fake')

    @server.tool()
    def getJiraIssue(cloudId: str, issueIdOrKey: str) -> dict[str, str]:  # pragma: no cover
        """A Jira tool that cannot run while another selected product is unavailable."""
        return {'cloudId': cloudId, 'key': issueIdOrKey}

    return server

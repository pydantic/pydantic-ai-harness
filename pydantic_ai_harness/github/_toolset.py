"""GitHub remote MCP policy and transport.

External contract, verified 2026-09-04:

- GitHub's official hosted server uses streamable HTTP at
  `https://api.githubcopilot.com/mcp/`.
- `X-MCP-Toolsets` selects comma-separated toolsets. The default server
  toolsets are `context`, `repos`, `issues`, `pull_requests`, and `users`.
- `X-MCP-Readonly: true` removes write tools and takes precedence over tool
  selection. Every tool registration in the official server must explicitly
  declare its MCP `readOnlyHint` annotation.
- `X-MCP-Tools`, `X-MCP-Features`, and `X-MCP-Insiders` can expand the exposed
  tool surface, so caller-supplied values are not accepted.
- URL selectors such as `/x/all` and `/insiders` can also expand that surface.
  Built-in connections therefore accept only the public GitHub MCP host or a
  GitHub Enterprise Cloud data-residency host at the plain `/mcp` path.
- The consolidated and granular sub-issue mutation tools accept an opaque
  `sub_issue_id` that does not identify its repository, so repository scope
  cannot be enforced for those tools and they are not exposed.
- Generic MCP hosts can authenticate with a PAT bearer token. OAuth requires
  the host to configure a GitHub App or OAuth App.

Sources:
https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md
https://github.com/github/github-mcp-server/blob/main/docs/server-configuration.md
https://github.com/github/github-mcp-server/blob/main/docs/feature-flags.md
https://github.com/github/github-mcp-server/blob/main/README.md
https://github.com/github/github-mcp-server/blob/main/pkg/toolvalidation/readonlyhint.go
https://github.github.com/gh-aw/troubleshooting/debug-ghe/

Re-check the endpoint and headers in the remote server, server configuration,
and feature flag docs. Confirm the sub-issue tool names and schemas in the tool
catalog and that the annotation check still covers all registrations before
changing safety classification.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import ToolsetTool

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the GitHub capability. Install it with: uv add "pydantic-ai-slim[mcp]"'
    ) from _import_error

__all__ = ['GITHUB_MCP_URL', 'AccessMode', 'GitHubToolset', 'MCPToolsetClient']

GITHUB_MCP_URL = 'https://api.githubcopilot.com/mcp/'
"""GitHub's official remote MCP endpoint."""

AccessMode = Literal['read', 'write']
"""Whether GitHub exposes read tools only or also exposes mutation tools."""

_DEFAULT_TOOLSETS = ('repos', 'issues', 'pull_requests')
_SUPPORTED_TOOLSETS = frozenset(_DEFAULT_TOOLSETS)
_OPAQUE_TARGET_TOOL_NAMES = frozenset(
    {'add_sub_issue', 'remove_sub_issue', 'reprioritize_sub_issue', 'sub_issue_write'}
)
_RESERVED_HEADERS = frozenset({'x-mcp-features', 'x-mcp-insiders', 'x-mcp-readonly', 'x-mcp-tools', 'x-mcp-toolsets'})
_SEARCH_TOOL_NAMES = frozenset({'search_code', 'search_commits', 'search_issues', 'search_pull_requests'})
_QUALIFIER_RE = re.compile(r'(?<![A-Za-z0-9_])(repo|org|user):([^\s)]+)', re.IGNORECASE)
_BOOLEAN_OR_RE = re.compile(r'\bOR\b')
_GITHUB_MCP_HOST_RE = re.compile(
    r'^(?:api\.githubcopilot\.com|copilot-api\.(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+ghe\.com)$', re.IGNORECASE
)
_TOOLSET_RE = re.compile(r'^[a-z0-9_]+$')
_SCOPE_COMPONENT_RE = re.compile(r'^[^\s/:]+$')
_OBJECT_MAPPING_ADAPTER = TypeAdapter(dict[str, object])


def validate_scope(repository: str | None, organization: str | None) -> tuple[str, str | None]:
    if (repository is None) == (organization is None):
        raise UserError('GitHub requires exactly one of `repository` or `organization`.')
    if repository is not None:
        parts = repository.split('/')
        if len(parts) != 2 or any(not _SCOPE_COMPONENT_RE.fullmatch(part) for part in parts):
            raise UserError('`repository` must use the `owner/repo` form.')
        return parts[0], parts[1]
    assert organization is not None
    if not _SCOPE_COMPONENT_RE.fullmatch(organization):
        raise UserError('`organization` must be one GitHub organization login.')
    return organization, None


def validate_access(access: str) -> AccessMode:
    if access not in ('read', 'write'):
        raise UserError('`access` must be `read` or `write`.')
    return access


def _validate_url(url: str) -> None:
    parts = urlsplit(url)
    if (
        parts.scheme.lower() != 'https'
        or _GITHUB_MCP_HOST_RE.fullmatch(parts.hostname or '') is None
        or parts.netloc.casefold() != (parts.hostname or '').casefold()
        or parts.path not in ('/mcp', '/mcp/')
        or parts.query
        or parts.fragment
    ):
        raise UserError('`url` must be an official HTTPS GitHub MCP endpoint with path `/mcp` or `/mcp/`.')


def _validate_toolsets(toolsets: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(toolsets)
    if (
        not resolved
        or any(not _TOOLSET_RE.fullmatch(name) for name in resolved)
        or not set(resolved) <= _SUPPORTED_TOOLSETS
    ):
        supported = ', '.join(sorted(_SUPPORTED_TOOLSETS))
        raise UserError(f'`toolsets` must contain one or more supported GitHub MCP toolsets: {supported}.')
    return resolved


def _object_mapping(value: object) -> dict[str, object] | None:
    try:
        return _OBJECT_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _is_read_only(tool: ToolsetTool[AgentDepsT]) -> bool:
    metadata = tool.tool_def.metadata
    annotations: object = metadata.get('annotations') if metadata is not None else None
    parsed = _object_mapping(annotations)
    return parsed is not None and parsed.get('readOnlyHint') is True


class GitHubToolset(MCPToolset[AgentDepsT]):
    """GitHub's hosted MCP tools with access, approval, and target-scope policy."""

    def __init__(
        self,
        *,
        repository: str | None = None,
        organization: str | None = None,
        access: AccessMode = 'read',
        require_approval: bool = True,
        toolsets: Sequence[str] = _DEFAULT_TOOLSETS,
        url: str = GITHUB_MCP_URL,
        auth: httpx.Auth | Literal['oauth'] | str | None = None,
        headers: Mapping[str, str] | None = None,
        client: MCPToolsetClient | None = None,
        id: str | None = None,
    ) -> None:
        owner, repo = validate_scope(repository, organization)
        access = validate_access(access)
        resolved_toolsets = _validate_toolsets(toolsets)
        supplied_headers = dict(headers or {})
        reserved = sorted(name for name in supplied_headers if name.lower() in _RESERVED_HEADERS)
        if reserved:
            raise UserError(f'GitHub manages the {reserved[0]!r} header from its access and toolset policy.')
        if auth is not None and any(name.lower() == 'authorization' for name in supplied_headers):
            raise UserError('Pass GitHub authentication through either `auth` or an `Authorization` header, not both.')
        self.repository = repository
        self.organization = organization
        self.access = access
        self.require_approval = require_approval
        self._owner = owner
        self._repo = repo
        if client is not None:
            if auth is not None or supplied_headers:
                raise UserError('A prebuilt `client` owns authentication and transport headers; configure them on it.')
            if url != GITHUB_MCP_URL or resolved_toolsets != _DEFAULT_TOOLSETS:
                raise UserError('A prebuilt `client` owns its URL and GitHub toolset selection; configure them on it.')
            super().__init__(client, id=id)
        else:
            _validate_url(url)
            if auth == 'oauth':
                raise UserError(
                    'GitHub OAuth requires a host-configured GitHub App or OAuth App; pass a prebuilt `client`.'
                )
            if auth is None and not any(name.lower() == 'authorization' for name in supplied_headers):
                raise UserError(
                    'GitHub remote MCP authentication is required through `auth` or an `Authorization` header.'
                )
            supplied_headers['X-MCP-Toolsets'] = ','.join(resolved_toolsets)
            supplied_headers['X-MCP-Readonly'] = 'true' if access == 'read' else 'false'
            super().__init__(url, id=id, auth=auth, headers=supplied_headers)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        return {
            name: tool
            for name, tool in tools.items()
            if self._tool_matches_scope(tool) and (self.access == 'write' or _is_read_only(tool))
        }

    def _tool_matches_scope(self, tool: ToolsetTool[AgentDepsT]) -> bool:
        if tool.tool_def.name in _OPAQUE_TARGET_TOOL_NAMES:
            return False
        properties = tool.tool_def.parameters_json_schema.get('properties')
        parsed_properties = _object_mapping(properties)
        property_names = set(parsed_properties) if parsed_properties is not None else set[str]()
        target_names = property_names & {'owner', 'repo', 'org', 'organization', 'enterprise'}
        if 'enterprise' in target_names or (
            {'owner', 'repo'} & target_names and {'org', 'organization'} & target_names
        ):
            return False
        if tool.tool_def.name in _SEARCH_TOOL_NAMES and 'query' in property_names:
            return target_names <= {'owner', 'repo'}
        if self._repo is not None:
            return {'owner', 'repo'} <= property_names
        return 'owner' in property_names or bool({'org', 'organization'} & property_names)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, object],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> object:
        scoped_args = self._scope_args(name, tool_args)
        if self.access == 'write' and self.require_approval and not _is_read_only(tool) and not ctx.tool_call_approved:
            raise ApprovalRequired
        return await super().call_tool(name, scoped_args, ctx, tool)

    def _scope_args(self, name: str, tool_args: dict[str, object]) -> dict[str, object]:
        scoped = dict(tool_args)
        self._validate_secondary_targets(name, scoped)
        if name in _SEARCH_TOOL_NAMES:
            self._validate_scope_arguments(name, scoped, require_relevant=False)
            query = scoped.get('query')
            if not isinstance(query, str):
                raise ModelRetry(f'`{name}` requires a string `query` within the configured GitHub scope.')
            if _BOOLEAN_OR_RE.search(query):
                raise ModelRetry(f'`{name}` cannot use boolean `OR` within a scoped GitHub search; split the search.')
            qualifier_name = 'repo' if self._repo is not None else 'org'
            qualifier_value = f'{self._owner}/{self._repo}' if self._repo is not None else self._owner
            for found_name, found_value in _QUALIFIER_RE.findall(query):
                if found_name.lower() != qualifier_name or found_value.casefold() != qualifier_value.casefold():
                    raise ModelRetry(f'`{name}` cannot search outside the configured GitHub scope {qualifier_value!r}.')
            scoped['query'] = f'{query} {qualifier_name}:{qualifier_value}'
            return scoped

        self._validate_scope_arguments(name, scoped, require_relevant=True)
        return scoped

    def _validate_secondary_targets(self, name: str, scoped: Mapping[str, object]) -> None:
        for owner_key, repo_key, require_pair in (
            ('parent_owner', 'parent_repo', True),
            ('related_owner', 'related_repo', False),
        ):
            owner_present = owner_key in scoped
            repo_present = repo_key in scoped
            if require_pair and owner_present != repo_present:
                raise ModelRetry(f'`{name}` must provide `{owner_key}` and `{repo_key}` together.')
            if owner_present:
                value = scoped[owner_key]
                if not isinstance(value, str) or value.casefold() != self._owner.casefold():
                    target = self.repository or self.organization
                    raise ModelRetry(f'`{name}` must keep its secondary target within GitHub scope {target!r}.')
            if repo_present:
                value = scoped[repo_key]
                if not isinstance(value, str) or (self._repo is not None and value.casefold() != self._repo.casefold()):
                    target = self.repository or self.organization
                    raise ModelRetry(f'`{name}` must keep its secondary target within GitHub scope {target!r}.')

    def _validate_scope_arguments(self, name: str, scoped: Mapping[str, object], *, require_relevant: bool) -> None:
        if ('owner' in scoped) != ('repo' in scoped) and (self._repo is not None or 'repo' in scoped):
            target = self.repository or self.organization
            raise ModelRetry(f'`{name}` must identify both owner and repository within GitHub scope {target!r}.')
        expected = {'owner': self._owner}
        if self._repo is not None:
            expected['repo'] = self._repo
        else:
            expected.update({'org': self._owner, 'organization': self._owner})
        relevant = False
        for key, expected_value in expected.items():
            if key not in scoped:
                continue
            relevant = True
            value = scoped[key]
            if not isinstance(value, str) or value.casefold() != expected_value.casefold():
                target = self.repository or self.organization
                raise ModelRetry(f'`{name}` must stay within the configured GitHub scope {target!r}.')
        if require_relevant and (not relevant or (self._repo is not None and not set(expected) <= scoped.keys())):
            target = self.repository or self.organization
            raise ModelRetry(f'`{name}` does not identify the configured GitHub scope {target!r}.')

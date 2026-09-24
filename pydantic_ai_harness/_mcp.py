"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from httpx import Auth
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolsetClient

T = TypeVar('T')
DepsT = TypeVar('DepsT')

MCPAuth: TypeAlias = Auth | str
"""A bearer token or API key, or an `httpx.Auth`."""

MCPAuthFunc: TypeAlias = Callable[[RunContext[AgentDepsT]], 'MCPAuth | None | Awaitable[MCPAuth | None]']
"""Returns the credential for the current run, usually from `ctx.deps`, or `None` when the run has none."""

MCPClientFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]], 'MCPToolsetClient | None | Awaitable[MCPToolsetClient | None]'
]
"""Returns the MCP client or transport for the current run, or `None` when the run has none."""


def per_run(
    func: Callable[[RunContext[DepsT]], Awaitable[T | None]],
    build: Callable[[T], AbstractToolset[DepsT]],
    *,
    id: str,
) -> AbstractToolset[DepsT]:
    """Build a toolset for each run from the value `func` returns for that run's context.

    Each run gets its own toolset, so hosted connections do not share an identity across runs. When `func`
    returns `None`, the run has no tools from this toolset.
    """

    async def toolset_for_run(ctx: RunContext[DepsT]) -> AbstractToolset[DepsT] | None:
        value = await func(ctx)
        return None if value is None else build(value)

    return DynamicToolset(toolset_for_run, per_run_step=False, id=id)


def per_run_auth(
    auth: MCPAuth | MCPAuthFunc[DepsT] | None,
    build: Callable[[MCPAuth | None], AbstractToolset[DepsT]],
    *,
    id: str,
) -> AbstractToolset[DepsT]:
    """Build a hosted MCP toolset with a fixed credential, or with the credential an `auth` function returns per run.

    A fixed or unset `auth` is built immediately, so every run shares one connection and one identity; `build`
    receives `None` only when `auth` is unset, to apply its environment fallback. A function's `None` omits the
    tools instead, so a run without a credential cannot fall back to the deployment's token.

    `'oauth'` is rejected in either form: Pydantic AI's `MCPToolset` reads it as FastMCP's browser login, which
    would open a browser on the host and wait for a callback that a server never receives.
    """
    # An `httpx.Auth` subclass may define `__call__`; it is still a fixed credential.
    if isinstance(auth, Auth) or not callable(auth):
        return build(_no_browser_login(auth))
    func = auth

    async def credential(ctx: RunContext[DepsT]) -> MCPAuth | None:
        result = func(ctx)
        return _no_browser_login(await result if inspect.isawaitable(result) else result)

    return per_run(credential, build, id=id)


def _no_browser_login(auth: MCPAuth | None) -> MCPAuth | None:
    if auth == 'oauth':
        raise UserError('Browser OAuth is not supported; pass an API key, a token, or an `httpx.Auth` as `auth`.')
    return auth


def per_run_client(
    client: MCPToolsetClient | MCPClientFunc[DepsT],
    build: Callable[[MCPToolsetClient], AbstractToolset[DepsT]],
    *,
    id: str,
) -> AbstractToolset[DepsT]:
    """Build a toolset from a fixed MCP client now, or from the client a function returns for each run."""
    if not callable(client):
        return build(client)
    func = client

    async def client_for_run(ctx: RunContext[DepsT]) -> MCPToolsetClient | None:
        result = func(ctx)
        return await result if inspect.isawaitable(result) else result

    return per_run(client_for_run, build, id=id)


def is_read_only(tool: ToolDefinition) -> bool:
    """Whether the server explicitly marks a tool read-only."""
    match (tool.metadata or {}).get('annotations'):
        case {'readOnlyHint': True}:
            return True
        case _:
            return False

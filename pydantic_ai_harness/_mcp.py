"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from os import environ
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

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
    value: T | Callable[[RunContext[DepsT]], T | None | Awaitable[T | None]],
    build: Callable[[T], AbstractToolset[DepsT]],
    *,
    id: str,
) -> AbstractToolset[DepsT]:
    """Build a toolset now from a fixed value, or for each run from the value a function returns.

    A fixed value gives every run the same connection. A function is called at the start of each run, so
    concurrent runs never share a connection or an identity; when it returns `None`, that run has no tools.
    """
    # An `httpx.Auth` subclass may define `__call__`; it is still a fixed value.
    if isinstance(value, Auth) or not callable(value):
        return build(cast(T, value))
    func = value

    async def toolset_for_run(ctx: RunContext[DepsT]) -> AbstractToolset[DepsT] | None:
        result = func(ctx)
        resolved = cast('T | None', await result if inspect.isawaitable(result) else result)
        return None if resolved is None else build(resolved)

    return DynamicToolset(toolset_for_run, per_run_step=False, id=id)


def credential(auth: MCPAuth | None, *, env: str | None, service: str) -> MCPAuth:
    """The credential to connect with: `auth`, else the `env` variable.

    Browser OAuth (`'oauth'`) is rejected: it opens a browser on the host and waits for a callback, which
    hangs an agent running on a server.
    """
    if auth is None and env is not None:
        auth = environ.get(env)
    if auth is None:
        raise UserError(
            f'Set `{env}` or pass `auth` to connect to {service}.' if env else f'Pass `auth` to connect to {service}.'
        )
    if auth == 'oauth':
        raise UserError('Browser OAuth is not supported; pass an API key, a token, or an `httpx.Auth` as `auth`.')
    return auth


def is_read_only(tool: ToolDefinition) -> bool:
    """Whether the server explicitly marks a tool read-only."""
    match (tool.metadata or {}).get('annotations'):
        case {'readOnlyHint': True}:
            return True
        case _:
            return False

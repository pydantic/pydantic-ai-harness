"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from os import environ
from typing import TypeAlias

from httpx import Auth
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

MCPAuth: TypeAlias = Auth | str
"""A bearer token or API key, or an `httpx.Auth`."""

MCPAuthFunc: TypeAlias = Callable[[RunContext[AgentDepsT]], MCPAuth | None | Awaitable[MCPAuth | None]]
"""Returns the credential for the current run, usually from `ctx.deps`, or `None` when the run has none."""


def per_run(
    auth: MCPAuth | MCPAuthFunc[AgentDepsT] | None,
    connect: Callable[[MCPAuth | None], AbstractToolset[AgentDepsT]],
    *,
    id: str,
) -> AbstractToolset[AgentDepsT]:
    """Connect now with a fixed credential, or at the start of each run with the one `auth` returns.

    Concurrent runs never share a connection or an identity. When the function returns `None`, that run
    has no tools.
    """
    if auth is None or isinstance(auth, Auth | str):
        return connect(auth)
    func = auth

    async def toolset_for_run(ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        result = func(ctx)
        resolved = await result if inspect.isawaitable(result) else result
        return None if resolved is None else connect(resolved)

    return DynamicToolset(toolset_for_run, per_run_step=False, id=id)


def credential(auth: MCPAuth | None, *, env: str | None, service: str) -> MCPAuth:
    """The credential to connect with: `auth`, else the `env` variable. An empty string counts as unset.

    Browser OAuth (`'oauth'`) is rejected: it opens a browser on the host and waits for a callback, which
    hangs an agent running on a server.
    """
    if auth is None and env is not None:
        auth = environ.get(env) or None
    if auth is None or auth == '':
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

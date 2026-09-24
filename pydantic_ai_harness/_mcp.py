"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

from os import environ

from httpx import Auth
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition


def credential(auth: Auth | str | None, *, env: str | None, service: str) -> Auth | str:
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

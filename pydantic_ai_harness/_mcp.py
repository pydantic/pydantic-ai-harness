"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

from collections.abc import Sequence
from os import environ
from typing import Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

CapabilityT = TypeVar('CapabilityT', bound=AbstractCapability[Any])


def credential(auth: str | None, *, env: str | None, service: str) -> str:
    """The API key or token to connect with: `auth`, else the `env` variable. An empty string counts as unset."""
    if not auth and env is not None:
        auth = environ.get(env)
    if not auth:
        raise UserError(
            f'Set `{env}` or pass `auth` to connect to {service}.' if env else f'Pass `auth` to connect to {service}.'
        )
    return auth


def is_read_only(tool: ToolDefinition) -> bool:
    """Whether the server explicitly marks a tool read-only."""
    match (tool.metadata or {}).get('annotations'):
        case {'readOnlyHint': True}:
            return True
        case _:
            return False


def one_connection(capabilities: Sequence[CapabilityT]) -> CapabilityT:
    """Resolve hosted MCP capabilities that share an `id`.

    The same configuration stated twice is one connection. Different ones are an error: merging them
    field by field could send one account's credential to another's server or drop `read_only`.
    """
    first = capabilities[0]
    if all(capability == first for capability in capabilities[1:]):
        return first
    raise UserError(
        f'Two `{type(first).__name__}` capabilities share the id {first.id!r}. Give each its own `id` and wrap '
        'them in `PrefixTools`, since their tool names are the same.'
    )

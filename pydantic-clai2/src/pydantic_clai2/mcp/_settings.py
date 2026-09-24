"""Server configuration models shared by the user file, project file, catalog, and plugin settings."""

import os
from collections.abc import Mapping
from string import Template
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

ServerName = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9]*$')]
"""Underscores are reserved for the `server_tool` separator, so two pairs cannot produce one name."""


class ServerSettings(BaseModel):
    """Common server options at the trusted configuration boundary."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    enabled: bool = True


class StdioServer(ServerSettings):
    """A local program, launched without a shell by the MCP client."""

    transport: Literal['stdio']
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


class HTTPServer(ServerSettings):
    """A Streamable HTTP MCP endpoint."""

    transport: Literal['http']
    url: HttpUrl
    headers: dict[str, str] | None = None
    auth: Literal['oauth'] | None = None
    """`oauth` lets FastMCP run the browser sign-in; tokens stay in memory."""


Server = Annotated[StdioServer | HTTPServer, Field(discriminator='transport')]
Servers = dict[ServerName, Server]


class MCPSettings(BaseModel):
    """Plugin settings. Server names also prefix tool names to avoid cross-server collisions."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer])


def http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    """Do not let a configured endpoint redirect MCP requests to another server.

    FastMCP passes `follow_redirects=True` to every client factory; it is accepted and overridden.
    """
    del follow_redirects
    return httpx.AsyncClient(
        headers=headers, timeout=timeout or httpx.Timeout(30, read=300), auth=auth, follow_redirects=False
    )


def references(server: Server) -> list[str]:
    """Environment variables named by `$VAR` in `env` values or `headers`, in first-use order."""
    values = server.env if isinstance(server, StdioServer) else server.headers
    names = (name for value in (values or {}).values() for name in Template(value).get_identifiers())
    return list(dict.fromkeys(names))


def missing(server: Server) -> list[str]:
    """Referenced variables the current environment does not set."""
    return [name for name in references(server) if name not in os.environ]


def resolve(values: Mapping[str, str] | None) -> dict[str, str] | None:
    """Fill `$VAR` references from the environment at connect time, so saved files hold no secrets."""
    if values is None:
        return None
    return {key: Template(value).safe_substitute(os.environ) for key, value in values.items()}


def target(server: Server) -> str:
    """What the server runs or contacts, for listings. Env, headers, URL credentials, and queries are not shown."""
    if isinstance(server, StdioServer):
        return ' '.join([server.command, *server.args])
    url = server.url
    port = f':{url.port}' if url.port not in (None, 80, 443) else ''
    return f'{url.scheme}://{url.host}{port}{url.path or ""}'

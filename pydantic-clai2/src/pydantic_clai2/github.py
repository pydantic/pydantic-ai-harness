"""The built-in `github` plugin: GitHub's hosted MCP tools, through harness `GitHub`."""

import os

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.github import GitHub

from .api_keys import load_keys
from .plugins import DepsT, PluginHost

TOKEN_NAME = 'GITHUB_TOKEN'
"""Both the environment variable and the `/keys` entry the token is read from."""


class GitHubSettings(BaseModel):
    """The JSON a `github` declaration may carry. The token never goes here; settings are stored in plaintext."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(
        default=True, description="Offer only GitHub's read tools, so the agent cannot change repositories."
    )


def activate(host: PluginHost[DepsT]) -> None:
    """Connect as the account behind `GITHUB_TOKEN`, refusing to load without one."""
    settings = host.settings(GitHubSettings)
    host.add(GitHub[DepsT](auth=_token(), read_only=settings.read_only))


def _token() -> str:
    # The environment wins over the saved key, as `GH_TOKEN` does over `gh auth login`,
    # so one launch can use another account without touching `/keys`.
    if token := os.environ.get(TOKEN_NAME, '').strip():
        return token
    if saved := load_keys().get(TOKEN_NAME):
        return saved.get_secret_value()
    raise UserError(
        f'GitHub needs a token: set `{TOKEN_NAME}`, or save an API key named {TOKEN_NAME} '
        'with `/set api_key` or `/keys`, then run `/plugins enable github`.'
    )

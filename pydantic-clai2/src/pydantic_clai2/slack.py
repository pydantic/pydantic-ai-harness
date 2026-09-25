"""The built-in `slack` plugin: harness `Slack`, connected with a user token from the environment or `/keys`."""

import os

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.slack import Slack

from .api_keys import load_keys
from .plugins import DepsT, PluginHost

TOKEN_NAME = 'SLACK_USER_TOKEN'
"""The environment variable harness `Slack` reads, and the `/keys` name CLAI falls back to."""


class SlackSettings(BaseModel):
    """The JSON a `slack` declaration may carry. The token never goes here: settings are stored in plaintext."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(
        default=True,
        description='Keep only the tools Slack marks read-only, so the agent cannot post or edit as you.',
    )


def activate(host: PluginHost[DepsT]) -> None:
    """Resolve the token now, so a missing one fails `/plugins enable slack` instead of the first turn."""
    settings = host.settings(SlackSettings)
    host.add(Slack[DepsT](auth=_user_token(), read_only=settings.read_only))


def _user_token() -> str:
    token = os.environ.get(TOKEN_NAME) or _saved_token()
    if not token:
        raise UserError(
            f'Slack needs a user token (xoxp-). Set {TOKEN_NAME}, or save it under that name with /keys, '
            'then run /plugins enable slack again.'
        )
    return token


def _saved_token() -> str | None:
    saved = load_keys().get(TOKEN_NAME)
    return saved.get_secret_value() if saved is not None else None

"""The built-in `slack` plugin: harness `Slack`, connected with a user token from the environment or `/keys`."""

import os

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.slack import Slack

from .api_keys import load_keys
from .plugins import DepsT, PluginHost, SessionStart

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
    """Connect once the plugin loads, so a missing token fails `/plugins enable slack` instead of the first turn."""
    settings = host.settings(SlackSettings)

    @host.on('session_start')
    async def connect(event: SessionStart) -> None:
        # `/keys` takes a cross-process lock that can wait up to 20 seconds; keep it off the event loop.
        token = os.environ.get(TOKEN_NAME) or await to_thread.run_sync(_saved_token)
        if not token:
            raise UserError(
                f'Slack needs a user token (xoxp-). Set {TOKEN_NAME}, or save it under that name with /keys, '
                'then run /plugins enable slack again.'
            )
        host.add(Slack[DepsT](auth=token, read_only=settings.read_only))


def _saved_token() -> str | None:
    saved = load_keys().get(TOKEN_NAME)
    return saved.get_secret_value() if saved is not None else None

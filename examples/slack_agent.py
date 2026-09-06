"""Run an agent people can message in Slack.

Set `SLACK_INSTALLATION_DIR` to a private directory for OAuth credentials,
then start from this file's directory with:
`uv run uvicorn slack_agent:build_app --factory --host 0.0.0.0 --port 8000 --workers 1`.
See `docs/slack.md` for credentials and Slack app settings.
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore
from slack_sdk.oauth.state_store import FileOAuthStateStore

from pydantic_ai_harness.slack import register_slack

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[None, str]:
    """Create the agent that answers Slack messages."""
    return Agent(model)


def build_app() -> AsyncSlackRequestHandler:
    """Create the web app for Slack messages and account connections."""
    installation_dir = Path(os.environ['SLACK_INSTALLATION_DIR']).expanduser()
    installation_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    installation_dir.chmod(0o700)
    installation_store = FileInstallationStore(
        base_dir=str(installation_dir),
        client_id=os.environ['SLACK_CLIENT_ID'],
    )
    oauth_settings = AsyncOAuthSettings(
        client_id=os.environ['SLACK_CLIENT_ID'],
        client_secret=os.environ['SLACK_CLIENT_SECRET'],
        scopes=[
            'app_mentions:read',
            'assistant:write',
            'channels:history',
            'chat:write',
            'groups:history',
            'im:history',
        ],
        user_scopes=os.environ['SLACK_USER_SCOPES'],
        redirect_uri=os.environ['SLACK_REDIRECT_URI'],
        installation_store=installation_store,
        state_store=FileOAuthStateStore(
            expiration_seconds=600,
            base_dir=str(installation_dir / 'oauth-state'),
            client_id=os.environ['SLACK_CLIENT_ID'],
        ),
    )
    bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
    register_slack(bolt, build_agent())
    return AsyncSlackRequestHandler(bolt, path='/slack/events')


if __name__ == '__main__':
    raise SystemExit(
        "From this file's directory, run: "
        'uv run uvicorn slack_agent:build_app --factory --host 0.0.0.0 --port 8000 --workers 1'
    )

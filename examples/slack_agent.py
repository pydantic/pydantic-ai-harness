"""Build a one-worker ASGI Slack agent with caller-owned Bolt OAuth.

Set `SLACK_INSTALLATION_DIR` to a private directory for OAuth credentials,
then expose `build_app()` through an ASGI server. Bolt serves its OAuth
install and redirect routes as well as `/slack/events`.
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore

from pydantic_ai_harness.slack import Slack, register_slack

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[None, str]:
    """Build an agent with Slack's native hosted MCP capability."""
    return Agent(model, capabilities=[Slack()])


def build_app() -> AsyncSlackRequestHandler:
    """Build the caller-owned Bolt app as a one-worker ASGI handler."""
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
        installation_store_bot_only=False,
    )
    bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
    register_slack(bolt, build_agent())
    return AsyncSlackRequestHandler(bolt, path='/slack/events')


if __name__ == '__main__':
    raise SystemExit('Expose build_app() to your one-worker ASGI server.')

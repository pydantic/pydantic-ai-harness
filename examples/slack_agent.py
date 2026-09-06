"""Run a Slack agent as a one-worker ASGI application.

Set ``SLACK_INSTALLATION_DIR`` to a private credential directory, then run the
returned ASGI app with an ASGI server. Bolt serves ``/slack/install`` and
``/slack/oauth_redirect`` alongside ``/slack/events`` for onboarding.
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore

from pydantic_ai_harness.slack import FileConversationStore, Slack, SlackApp

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[None, str]:
    """Build an agent with Slack's native hosted MCP toolset."""
    return Agent(model, capabilities=[Slack()])


def build_app() -> AsyncSlackRequestHandler:
    """Build the one-worker ASGI application and its caller-owned Bolt app."""
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
            'mpim:history',
        ],
        user_scopes=os.environ['SLACK_USER_SCOPES'],
        redirect_uri=os.environ['SLACK_REDIRECT_URI'],
        installation_store=installation_store,
        installation_store_bot_only=False,
    )
    bolt = AsyncApp(oauth_settings=oauth_settings, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
    SlackApp(
        build_agent(),
        app=bolt,
        allowed_users={os.environ['SLACK_TEAM_ID']: {os.environ['SLACK_USER_ID']}},
        store=FileConversationStore(os.environ['SLACK_CONVERSATION_DIR']),
        install_url=os.environ.get('SLACK_INSTALL_URL'),
    )
    return AsyncSlackRequestHandler(bolt, path='/slack/events')


if __name__ == '__main__':
    raise SystemExit('Expose build_app() to your one-worker ASGI server.')

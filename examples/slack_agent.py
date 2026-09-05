"""A Slack agent with user-scoped workspace search and thread history.

Set `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_MCP_TOKEN`,
`SLACK_ALLOWED_USER_IDS`, and your model's API key, then:

    uv run --with 'pydantic-ai-harness[slack,anthropic]' examples/slack_agent.py

Create the Slack app from the manifest in `pydantic_ai_harness/slack/README.md`,
invite the bot to a channel, and mention it. Socket Mode connects outward, so
there is no public URL to host.
"""

import os

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness.slack import FileConversationStore, Slack, SlackApp

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-sonnet-4-6')


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[None, str]:
    """Build an agent with Slack's curated read-only MCP tools."""
    return Agent(model, capabilities=[Slack()])


def main() -> None:
    """Start the bot and serve Slack until interrupted."""
    SlackApp(build_agent(), store=FileConversationStore('~/.slack-agent')).run()


if __name__ == '__main__':
    main()

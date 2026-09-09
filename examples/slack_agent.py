"""Put a Pydantic AI agent on Slack.

Set `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`, then run `uvicorn slack_agent:app` for
the HTTP Events API, or set `SLACK_APP_TOKEN` too and run this file for Socket Mode.
"""

import os

import anyio
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness.slack import Slack, SlackApp

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[None, str]:
    """Build the agent that answers Slack messages; `Slack()` gives it Slack tools as the bot."""
    return Agent(model, capabilities=[Slack()])


agent = build_agent()
app = SlackApp(agent)

if __name__ == '__main__':
    anyio.run(app.serve)

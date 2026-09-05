"""A Slack agent that reports as it works and asks before anything risky.

Set `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, and your model's API key, then:

    uv run --with 'pydantic-ai-harness[slack,anthropic]' examples/slack_agent.py

Create the Slack app from the manifest in `pydantic_ai_harness/slack/README.md`,
invite the bot to a channel, and mention it. Socket Mode connects outward, so
there is no public URL to host.
"""

import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.models import Model

from pydantic_ai_harness.slack import (
    FileConversationStore,
    SlackApprovals,
    SlackChatToolset,
    SlackInteractions,
    SlackThread,
)
from pydantic_ai_harness.slack.app import SlackAgent

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-sonnet-4-6')

INSTRUCTIONS = """\
You work in a Slack thread, so the people reading you are watching it happen.

Post a plan with `post_plan` before any work that takes more than one step, and
update it as each step finishes. Say what you found along the way with
`post_message` rather than saving it all for the end.

Keep your final answer short. Anything long -- a report, a table, a diff --
belongs in a file you send with `upload_file`, with a couple of lines saying
what is in it.

Decide what you can decide. Use `ask_user` only when the choice is genuinely
theirs to make.
"""


# One registry per process: the toolset asks through it and the app resolves
# button clicks against it, so both halves must be the same object.
INTERACTIONS = SlackInteractions()


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[SlackThread, str]:
    """A Slack-native agent with one tool that needs a person's approval."""
    agent = Agent(
        model,
        deps_type=SlackThread,
        instructions=INSTRUCTIONS,
        toolsets=[SlackChatToolset(interactions=INTERACTIONS, file_root='./workspace')],
        capabilities=[HandleDeferredToolCalls(handler=SlackApprovals(INTERACTIONS))],
    )

    @agent.tool_plain(requires_approval=True)
    def send_invoice(customer: str, amount_usd: float) -> str:  # pyright: ignore[reportUnusedFunction]
        """Send an invoice. A person approves this in Slack before it runs."""
        return f'Invoiced {customer} ${amount_usd:.2f}'

    return agent


def main() -> None:
    """Start the bot and serve Slack until interrupted."""
    SlackAgent(
        build_agent(),
        interactions=INTERACTIONS,
        store=FileConversationStore('~/.slack-agent'),
    ).run()


if __name__ == '__main__':
    main()

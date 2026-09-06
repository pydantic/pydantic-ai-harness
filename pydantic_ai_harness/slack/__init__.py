"""Give an agent native Slack MCP access."""

from pydantic_ai_harness.slack._bolt import register_slack
from pydantic_ai_harness.slack._capability import Slack
from pydantic_ai_harness.slack._context import SlackContext, SlackFile, current_slack_context

__all__ = ['Slack', 'SlackContext', 'SlackFile', 'current_slack_context', 'register_slack']

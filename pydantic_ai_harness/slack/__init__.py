"""Give an agent native Slack MCP access."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.slack._capability import Slack
from pydantic_ai_harness.slack._context import SlackContext, SlackFile, current_slack_context

if TYPE_CHECKING:
    from pydantic_ai_harness.slack._bolt import register_slack

__all__ = ['Slack', 'SlackContext', 'SlackFile', 'current_slack_context', 'register_slack']


def __getattr__(name: str) -> object:
    if name == 'register_slack':
        from pydantic_ai_harness.slack._bolt import register_slack

        return register_slack
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

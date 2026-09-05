"""The ready-made Slack app. Separate module so importing it is what pulls in `slack-bolt`."""

from pydantic_ai_harness.slack._app import DEFAULT_ERROR_REPLY, SlackAgent

__all__ = ['DEFAULT_ERROR_REPLY', 'SlackAgent']

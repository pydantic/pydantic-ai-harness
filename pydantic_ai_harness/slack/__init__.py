"""Give an agent typed Slack MCP tools and serve it in Slack with Bolt."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.slack._capability import Slack
from pydantic_ai_harness.slack._context import SlackContext, SlackFile, current_slack_context
from pydantic_ai_harness.slack._store import (
    ConversationStore,
    FileConversationStore,
    InMemoryConversationStore,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.slack._app import SlackApp

__all__ = [
    'Slack',
    'SlackApp',
    'SlackContext',
    'SlackFile',
    'ConversationStore',
    'InMemoryConversationStore',
    'FileConversationStore',
    'current_slack_context',
]

_BOLT_EXPORTS = {'SlackApp'}


def __getattr__(name: str) -> object:
    # Imported on demand so the rest of the package stays usable without
    # `slack-bolt` installed.
    if name in _BOLT_EXPORTS:
        from pydantic_ai_harness.slack import _app

        return getattr(_app, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

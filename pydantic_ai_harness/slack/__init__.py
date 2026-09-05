"""Give an agent typed Slack MCP tools and serve it in Slack with Bolt."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.slack._access import SlackAccess
from pydantic_ai_harness.slack._approvals import SlackApprovals
from pydantic_ai_harness.slack._capability import Slack
from pydantic_ai_harness.slack._context import SlackContext, SlackContextEntity, current_slack_context
from pydantic_ai_harness.slack._interactions import (
    SlackInteractions,
    SlackPromptError,
)
from pydantic_ai_harness.slack._mcp import SlackTool, SlackTools
from pydantic_ai_harness.slack._store import (
    ConversationStore,
    FileConversationStore,
    InMemoryConversationStore,
)
from pydantic_ai_harness.slack._thread import SlackThread

if TYPE_CHECKING:
    from pydantic_ai_harness.slack._app import SlackApp

__all__ = [
    'ConversationStore',
    'FileConversationStore',
    'InMemoryConversationStore',
    'SlackApp',
    'SlackAccess',
    'SlackApprovals',
    'Slack',
    'SlackContext',
    'SlackContextEntity',
    'SlackTool',
    'SlackTools',
    'SlackInteractions',
    'SlackPromptError',
    'SlackThread',
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

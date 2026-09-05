"""Primitives for building a Pydantic AI agent that lives in Slack.

Slack is both the front door and one of the tools. `SlackChat` is a capability
you add to any agent, whatever its deps: it gives the model a way to report,
ask, and send files, either in the thread it is answering or in channels you
name. `SlackBot` puts that agent behind Slack, over Socket Mode or the Events
API. Every piece is usable on its own.

`SlackBot` needs `slack-bolt`, so it is imported lazily: naming it is what
pulls Bolt in. Everything else needs only `slack-sdk`.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.slack._approvals import APPROVE, DENY, SlackApprovals
from pydantic_ai_harness.slack._capability import DEFAULT_INSTRUCTIONS, SlackChat
from pydantic_ai_harness.slack._client import SlackClient, default_client
from pydantic_ai_harness.slack._interactions import (
    DEFAULT_PROMPT_TIMEOUT_SECONDS,
    PROMPT_ACTION_PREFIX,
    SlackInteractions,
    SlackPromptError,
)
from pydantic_ai_harness.slack._store import (
    ConversationStore,
    FileConversationStore,
    InMemoryConversationStore,
)
from pydantic_ai_harness.slack._thread import (
    SlackThread,
    ThreadResolver,
    bind_thread,
    conversation_key,
    current_thread,
)
from pydantic_ai_harness.slack._toolset import MAX_MESSAGE_CHARS, PlanStep, SlackChatToolset, StepStatus

if TYPE_CHECKING:
    from pydantic_ai_harness.slack._app import DEFAULT_ERROR_REPLY, SlackBot

__all__ = [
    'APPROVE',
    'DEFAULT_ERROR_REPLY',
    'DEFAULT_INSTRUCTIONS',
    'DEFAULT_PROMPT_TIMEOUT_SECONDS',
    'DENY',
    'MAX_MESSAGE_CHARS',
    'PROMPT_ACTION_PREFIX',
    'ConversationStore',
    'FileConversationStore',
    'InMemoryConversationStore',
    'PlanStep',
    'SlackBot',
    'SlackApprovals',
    'SlackChat',
    'SlackChatToolset',
    'SlackClient',
    'SlackInteractions',
    'SlackPromptError',
    'SlackThread',
    'StepStatus',
    'ThreadResolver',
    'bind_thread',
    'conversation_key',
    'current_thread',
    'default_client',
]

_BOLT_EXPORTS = {'DEFAULT_ERROR_REPLY', 'SlackBot'}


def __getattr__(name: str) -> object:
    # Imported on demand so the rest of the package stays usable without
    # `slack-bolt` installed.
    if name in _BOLT_EXPORTS:
        from pydantic_ai_harness.slack import _app

        return getattr(_app, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

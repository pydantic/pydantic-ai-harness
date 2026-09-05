"""Primitives for building a Pydantic AI agent that lives in Slack.

Slack is both the front door and one of the tools. `SlackChatToolset` gives the
model a way to talk to the thread it is running in, `SlackApprovals` gates
dangerous tools behind a button, and `SlackAgent` wires both to a Socket Mode
app so a working bot is a few lines. Every piece is usable on its own.

`SlackAgent` needs `slack-bolt`, so it is imported from
`pydantic_ai_harness.slack.app` rather than from here; the rest of the package
has no dependency beyond Pydantic AI.
"""

from pydantic_ai_harness.slack._approvals import APPROVE, DENY, SlackApprovals
from pydantic_ai_harness.slack._client import SlackClient, SlackResponse
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
from pydantic_ai_harness.slack._thread import SlackThread, conversation_key
from pydantic_ai_harness.slack._toolset import PlanStep, SlackChatToolset, StepStatus

__all__ = [
    'APPROVE',
    'DENY',
    'DEFAULT_PROMPT_TIMEOUT_SECONDS',
    'PROMPT_ACTION_PREFIX',
    'ConversationStore',
    'FileConversationStore',
    'InMemoryConversationStore',
    'PlanStep',
    'SlackApprovals',
    'SlackChatToolset',
    'SlackClient',
    'SlackInteractions',
    'SlackPromptError',
    'SlackResponse',
    'SlackThread',
    'StepStatus',
    'conversation_key',
]

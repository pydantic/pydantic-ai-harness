"""The Slack chat capability: tools for talking to a thread, plus how to use them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness.slack._interactions import SlackInteractions
from pydantic_ai_harness.slack._thread import SlackThread
from pydantic_ai_harness.slack._toolset import SlackChatToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

DEFAULT_INSTRUCTIONS = """\
You work in a Slack thread, so the people reading you are watching it happen.

Post a plan with `post_plan` before any work that takes more than one step. It
returns a `plan_id`; pass that back with every step to tick them off in place.
Say what you found along the way with `post_message` rather than saving it all
for the end.

Keep your final answer short. Anything long -- a report, a table, a diff --
belongs in a file you send with `upload_file`, with a couple of lines saying
what is in it.
"""
"""Guidance that makes the tools behave like a colleague rather than a log."""

_ASK_USER_GUIDANCE = """
Decide what you can decide. Use `ask_user` only when the choice is genuinely
theirs to make, because it stops your turn until someone clicks.
"""


@dataclass
class SlackChat(AbstractCapability[SlackThread]):
    """Let an agent report progress, ask, and send files in the Slack thread it is running in.

    The agent's `deps` must be a [`SlackThread`][pydantic_ai_harness.slack.SlackThread]
    naming where the run is talking, so one `Agent` serves every conversation.

    Ships the tools and the guidance together. Without instructions a model
    treats `post_message` as optional and says nothing until it finishes, which
    is the thing a Slack agent exists not to do.
    """

    interactions: SlackInteractions | None = None
    """Shared prompt registry backing `ask_user`.

    Leave it unset for an agent that should never block waiting for a person;
    `ask_user` is then not registered. Pass the same instance to
    [`SlackApprovals`][pydantic_ai_harness.slack.SlackApprovals] and to the
    application, so a click reaches the run waiting on it.
    """

    file_root: str | Path | None = None
    """Directory `upload_file` may send from. Paths outside it are refused.

    Leave it unset and `upload_file` is not registered: there is no directory to
    judge a model-supplied path against, and sending an arbitrary path off the
    host is not a sensible default.
    """

    instructions: str | None = None
    """Replaces [`DEFAULT_INSTRUCTIONS`][pydantic_ai_harness.slack.DEFAULT_INSTRUCTIONS] verbatim.

    Set it to `''` to add none, for an agent whose own instructions already say
    how to behave in a thread.
    """

    def get_toolset(self) -> SlackChatToolset:
        """Build the chat toolset this capability configures."""
        return SlackChatToolset(interactions=self.interactions, file_root=self.file_root)

    def get_instructions(self) -> AgentInstructions[SlackThread] | None:
        """Tell the model how to use the tools, or nothing when told to add none."""
        if self.instructions is not None:
            return self.instructions or None
        # The sentence about `ask_user` is dropped when the tool is not there, so
        # the instructions never describe a tool the model cannot call.
        if self.interactions is None:
            return DEFAULT_INSTRUCTIONS
        return f'{DEFAULT_INSTRUCTIONS}{_ASK_USER_GUIDANCE}'

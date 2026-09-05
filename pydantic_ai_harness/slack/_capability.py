"""The Slack chat capability: tools for talking to Slack, plus how to use them."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext

from pydantic_ai_harness.slack._approvals import SlackApprovals
from pydantic_ai_harness.slack._client import SlackClient, default_client
from pydantic_ai_harness.slack._interactions import SlackInteractions
from pydantic_ai_harness.slack._thread import SlackThread, ThreadResolver
from pydantic_ai_harness.slack._toolset import SlackChatToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

DEFAULT_INSTRUCTIONS = """\
You work in Slack, so the people reading you are watching it happen.

Post a plan with `post_plan` before any work that takes more than one step. It
returns a `plan_id`; pass that back with every step to tick them off in place.
Say what you found along the way with `post_message` rather than saving it all
for the end.

Keep your final answer short. Anything long -- a report, a table, a diff --
belongs in a file you send with `upload_file`, with a couple of lines saying
what is in it.
"""
"""Guidance that makes the tools behave like a colleague rather than a log."""

_THREAD_ADAPTER = TypeAdapter(SlackThread)
"""Validates the mapping form a spec writes a thread in."""

_ASK_USER_GUIDANCE = """
Decide what you can decide. Use `ask_user` only when the choice is genuinely
theirs to make, because it stops your turn until someone clicks.
"""


@dataclass
class SlackChat(AbstractCapability[AgentDepsT]):
    """Let an agent report progress, ask, and send files in Slack.

    Add it to any agent. It does not touch the agent's `deps`, so an agent you
    already have keeps the deps it already has:

    ```python {test="skip"}
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SlackChat()])
    ```

    Where messages go is settled per run, in this order: the channel the model
    named, if `channels` lists one it may name; then the thread the run is bound
    to, which [`SlackBot`][pydantic_ai_harness.slack.SlackBot] binds for an
    inbound message; then the single channel in `channels`.

    So the same capability covers an agent answering in the thread it was
    mentioned in, and an agent with no Slack front door at all that reports into
    `#alerts` when a cron job runs it.

    Ships the tools and the guidance together. Without instructions a model
    treats `post_message` as optional and says nothing until it finishes, which
    is the thing a Slack agent exists not to do.

    Authentication comes from the `SLACK_BOT_TOKEN` environment variable by
    default; pass `token` or `client` to configure it explicitly. The token is
    only read when a tool first posts, so constructing this needs no credentials.
    """

    channels: Sequence[str] = ()
    """Channels the model may post to, by `#name` or id.

    Empty means the agent can only talk in the thread it is running in. With one
    channel, that is where messages go when the model does not say. With several,
    the model picks and anything not listed is refused.

    This bounds what the model can name, not what the token can reach: the bot
    can still post anywhere it has been added. Real scoping is the app's install.
    """

    ask_user: bool = False
    """Register `ask_user`, which posts a multiple-choice question and waits.

    Off by default: it stops the turn until someone clicks, which is wrong for an
    agent that should never block on a person, and needs something routing button
    clicks back -- `SlackBot` does, a cron job does not.
    """

    approvals: bool = False
    """Ask in Slack before any tool marked `requires_approval` runs.

    Posts each pending call with Approve and Deny buttons and continues the run
    when someone answers. Unlike `ask_user` this is not the model's choice: the
    gate is on the tool. Needs button clicks routed back, same as `ask_user`.
    """

    approver_ids: Collection[str] | None = None
    """Who may answer approval prompts. Defaults to whoever started the run.

    Set it to a reviewer group when the person asking should not approve their
    own agent's actions.
    """

    file_root: str | Path | None = None
    """Directory `upload_file` may send from. Paths outside it are refused.

    Leave it unset and `upload_file` is not registered: there is no directory to
    judge a model-supplied path against, and sending an arbitrary path off the
    host is not a sensible default.
    """

    token: str | None = None
    """Slack bot token (`xoxb-`). Defaults to `SLACK_BOT_TOKEN`."""

    client: SlackClient | None = None
    """Slack client to call through, instead of one built from a token."""

    thread: SlackThread | ThreadResolver[AgentDepsT] | None = None
    """Fix the thread to post in, or work it out from the run context.

    Omit it and the tools use the thread bound by
    [`bind_thread`][pydantic_ai_harness.slack.bind_thread]. Set it when the
    binding cannot reach the run -- under durable execution, where the worker
    rebuilds the capability in another process.
    """

    interactions: SlackInteractions | None = None
    """Prompt registry backing `ask_user` and approvals. One is made if you omit it.

    Pass your own to change the answer timeout, or to share one registry between
    several capabilities.
    """

    instructions: str | None = None
    """Replaces [`DEFAULT_INSTRUCTIONS`][pydantic_ai_harness.slack.DEFAULT_INSTRUCTIONS] verbatim.

    Set it to `''` to add none, for an agent whose own instructions already say
    how to behave in a thread.
    """

    _resolved_client: SlackClient | None = field(default=None, init=False, repr=False, compare=False)
    _resolved_interactions: SlackInteractions | None = field(default=None, init=False, repr=False, compare=False)
    _resolved_toolset: SlackChatToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> SlackChat[Any]:
        """Build from an agent spec, where everything arrives as plain data.

        A spec can set `channels`, `ask_user`, `approvals`, `approver_ids`,
        `file_root`, `token`, `instructions`, and `thread` as a mapping of
        `SlackThread` fields. `client` and `interactions` are live objects, so
        they can only be passed in code.
        """
        for live in ('client', 'interactions'):
            if live in kwargs:
                raise ValueError(
                    f'{live} cannot be set from an agent spec because it is a live object. '
                    f'Leave it out and set it in code, or set token to authenticate from the spec.'
                )
        thread: object = kwargs.get('thread')
        if isinstance(thread, Mapping):
            # Pydantic leaves it as a mapping: the field's union with a resolver
            # callable stops it coercing to `SlackThread` on its own.
            kwargs['thread'] = _THREAD_ADAPTER.validate_python(thread)
        return cls(*args, **kwargs)

    def resolve_client(self) -> SlackClient:
        """The Slack client these tools call through, built from the token on first use."""
        if self._resolved_client is None:
            self._resolved_client = self.client if self.client is not None else default_client(self.token)
        return self._resolved_client

    def resolve_interactions(self) -> SlackInteractions:
        """The prompt registry backing `ask_user` and approvals, made on first use."""
        if self._resolved_interactions is None:
            self._resolved_interactions = self.interactions if self.interactions is not None else SlackInteractions()
        return self._resolved_interactions

    def resolve_prompt(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Route a button click back to the run waiting on it.

        [`SlackBot`][pydantic_ai_harness.slack.SlackBot] finds this capability on
        the agent and calls it, so nothing has to be wired up by hand. Returns
        `False` when the click changed nothing: an expired prompt, a repeat click,
        a person not allowed to answer, or an agent that asks nothing.
        """
        if not (self.ask_user or self.approvals):
            return False
        return self.resolve_interactions().resolve(block_id=block_id, value=value, user_id=user_id)

    def get_toolset(self) -> SlackChatToolset[AgentDepsT]:
        """Build the chat toolset this capability configures."""
        if self._resolved_toolset is None:
            self._resolved_toolset = SlackChatToolset[AgentDepsT](
                # Resolved on first post, not now: adding this capability to an
                # agent must not require a token to be configured yet.
                self.resolve_client,
                channels=self.channels,
                thread=self.thread,
                interactions=self.resolve_interactions() if self.ask_user else None,
                file_root=self.file_root,
            )
        return self._resolved_toolset

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        """Ask in Slack about tools that require approval, when `approvals` is on."""
        if not self.approvals:
            return None
        # Built per round rather than cached: it holds no state between rounds,
        # and everything it needs is already resolved once and kept.
        approvals = SlackApprovals[AgentDepsT](
            self.resolve_client(),
            self.resolve_interactions(),
            thread=self.thread,
            allowed_user_ids=self.approver_ids,
        )
        return await approvals(ctx, requests)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model how to use the tools, or nothing when told to add none."""
        if self.instructions is not None:
            return self.instructions or None
        parts = [DEFAULT_INSTRUCTIONS]
        # The sentence about `ask_user` is dropped when the tool is not there, so
        # the instructions never describe a tool the model cannot call.
        if self.ask_user:
            parts.append(_ASK_USER_GUIDANCE)
        if len(self.channels) > 1:
            listed = ', '.join(self.channels)
            parts.append(
                f'\nYou can post to these channels by naming one: {listed}. '
                'Leave the channel unset to post where the conversation is happening.\n'
            )
        return ''.join(parts)

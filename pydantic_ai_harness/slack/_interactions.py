"""Ask a person a question in Slack and wait for the answer.

One object serves both the `ask_user` tool and approval prompts, so a run never
has two sets of buttons live in the same thread at once.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import anyio

from pydantic_ai_harness.slack._thread import SlackThread

PROMPT_ACTION_PREFIX = 'pydantic_ai_harness_slack_prompt:'
"""Prefix on the `action_id` of every button these prompts post.

Register one Slack action handler matching this prefix and forward the click to
[`SlackInteractions.resolve`][pydantic_ai_harness.slack.SlackInteractions.resolve].
"""

DEFAULT_PROMPT_TIMEOUT_SECONDS = 600.0
"""How long a prompt waits before giving up. Ten minutes suits a working thread:
long enough for someone to come back from a meeting, short enough that a run does
not sit open overnight holding an agent turn."""

_MAX_OPTIONS = 25
"""Slack's cap on elements in one `actions` block."""

_MAX_VALUE_CHARS = 2000
"""Slack's cap on a button `value`."""


class SlackPromptError(RuntimeError):
    """A prompt could not be posted or was rejected before anyone could answer."""


@dataclass(slots=True)
class _Pending:
    options: tuple[str, ...]
    allowed_user_ids: frozenset[str]
    event: anyio.Event = field(default_factory=anyio.Event)
    answer: str | None = None
    answered_by: str | None = None


class SlackInteractions:
    """Post a question with buttons into a thread and wait for someone to click.

    Create one per application and share it between the chat toolset and the
    approval handler. Prompts for the same thread are serialized, so a second
    question waits for the first to be answered rather than posting a competing
    set of buttons.

    A click is only accepted from an allowed user. When a prompt does not name
    any, the person whose message started the run is the only one who can answer.
    """

    def __init__(self, *, timeout_seconds: float = DEFAULT_PROMPT_TIMEOUT_SECONDS) -> None:
        """Wait `timeout_seconds` for an answer before treating a prompt as unanswered."""
        if timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive')
        self._timeout_seconds = timeout_seconds
        self._pending: dict[str, _Pending] = {}
        self._locks: dict[str, anyio.Lock] = {}
        self._next_token = 0

    async def ask(
        self,
        thread: SlackThread,
        question: str,
        options: Sequence[str],
        *,
        allowed_user_ids: Collection[str] | None = None,
    ) -> str | None:
        """Post `question` with one button per option and wait for a click.

        Returns the chosen option, or `None` when nobody answered in time. The
        posted message is edited either way, so a thread never keeps live buttons
        that no longer do anything.

        Raises:
            ValueError: If `options` is empty, exceeds Slack's limits, or contains
                duplicates that would make the answer ambiguous.
            SlackPromptError: If Slack rejected the message.
        """
        choices = tuple(options)
        if not choices:
            raise ValueError('options must contain at least one option')
        if len(choices) > _MAX_OPTIONS:
            raise ValueError(f'Slack allows at most {_MAX_OPTIONS} buttons in one prompt, got {len(choices)}')
        if len(set(choices)) != len(choices):
            raise ValueError('options must be unique so the answer is unambiguous')
        for choice in choices:
            if not choice or len(choice) > _MAX_VALUE_CHARS:
                raise ValueError(f'each option must be between 1 and {_MAX_VALUE_CHARS} characters')

        allowed = frozenset(allowed_user_ids) if allowed_user_ids is not None else frozenset({thread.user_id})
        lock = self._locks.setdefault(thread.key, anyio.Lock())
        async with lock:
            return await self._ask_once(thread, question, choices, allowed)

    async def _ask_once(
        self,
        thread: SlackThread,
        question: str,
        choices: tuple[str, ...],
        allowed: frozenset[str],
    ) -> str | None:
        self._next_token += 1
        token = f'{thread.key}#{self._next_token}'
        response = await thread.client.chat_postMessage(
            channel=thread.channel_id,
            thread_ts=thread.thread_ts,
            text=question,
            blocks=_prompt_blocks(token, question, choices),
        )
        timestamp = response.get('ts')
        if not isinstance(timestamp, str):
            raise SlackPromptError('Slack did not return a timestamp for the prompt message')

        # No `await` separates the post from this registration, so a click cannot
        # arrive while the prompt is unregistered.
        pending = _Pending(options=choices, allowed_user_ids=allowed)
        self._pending[token] = pending
        try:
            with anyio.move_on_after(self._timeout_seconds):
                await pending.event.wait()
        finally:
            del self._pending[token]

        answer = pending.answer
        await thread.client.chat_update(
            channel=thread.channel_id,
            ts=timestamp,
            text=_settled_text(question, answer, pending.answered_by),
            blocks=[],
        )
        return answer

    def resolve(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Record a button click. Call this from the application's action handler.

        `block_id` and `value` come straight off the Slack action payload. Returns
        `False` when the click changes nothing -- an expired prompt, a value that
        is not one of its options, a second click on a prompt already answered, or
        a user who is not allowed to answer this one. Tell the person that rather
        than letting the click disappear.
        """
        pending = self._pending.get(block_id)
        if pending is None or pending.answer is not None:
            return False
        if user_id not in pending.allowed_user_ids or value not in pending.options:
            return False
        pending.answer = value
        pending.answered_by = user_id
        pending.event.set()
        return True


def _prompt_blocks(token: str, question: str, choices: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {'type': 'section', 'text': {'type': 'mrkdwn', 'text': question}},
        {
            'type': 'actions',
            'block_id': token,
            'elements': [
                {
                    'type': 'button',
                    'action_id': f'{PROMPT_ACTION_PREFIX}{index}',
                    'text': {'type': 'plain_text', 'text': choice[:75]},
                    'value': choice,
                }
                for index, choice in enumerate(choices)
            ],
        },
    ]


def _settled_text(question: str, answer: str | None, answered_by: str | None) -> str:
    if answer is None:
        return f'{question}\n\n_No answer, so this prompt expired._'
    return f'{question}\n\n_<@{answered_by}> chose *{answer}*._'

"""Ask a person a question in Slack and wait for the answer.

One object serves both the `ask_user` tool and approval prompts, so a run never
has two sets of buttons live in the same thread at once.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from weakref import WeakValueDictionary

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

_MAX_OPTION_CHARS = 75
"""Slack's cap on button text. Longer options are refused rather than shortened,
so two options can never render as the same button."""

logger = logging.getLogger(__name__)


class SlackPromptError(RuntimeError):
    """Slack posted the prompt but did not identify the message.

    Without a message id there is nothing to edit once the prompt settles.
    """


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
        # Weak, so a thread's lock is collected once no prompt holds it. A busy
        # workspace would otherwise accumulate one lock per thread forever.
        self._locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()

    async def ask(
        self,
        thread: SlackThread,
        question: str,
        options: Sequence[str],
        *,
        allowed_user_ids: Collection[str] | None = None,
    ) -> str | None:
        """Post `question` with one button per option and wait for a click.

        Returns the chosen option, or `None` when nobody answered in time. Either
        way the posted message is edited to record what happened, so the thread
        does not keep buttons that no longer do anything. Cancelling the run
        instead skips that edit and leaves the buttons in place.

        Raises:
            ValueError: If `options` is empty, exceeds Slack's limits, contains
                duplicates that would make the answer ambiguous, or
                `allowed_user_ids` is a string rather than a collection of ids.
            SlackPromptError: If Slack did not identify the message it posted.
        """
        choices = tuple(options)
        if not choices:
            raise ValueError('options must contain at least one option')
        if len(choices) > _MAX_OPTIONS:
            raise ValueError(f'Slack allows at most {_MAX_OPTIONS} buttons in one prompt, got {len(choices)}')
        if len(set(choices)) != len(choices):
            raise ValueError('options must be unique so the answer is unambiguous')
        for choice in choices:
            if not choice or len(choice) > _MAX_OPTION_CHARS:
                raise ValueError(f'each option must be between 1 and {_MAX_OPTION_CHARS} characters')

        if isinstance(allowed_user_ids, str):
            raise ValueError('allowed_user_ids must be a collection of user ids, not a string')
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
        # Random, not a counter: a counter restarts with the process, so a button
        # left over from a previous run would resolve the first prompt of the new
        # one -- an old Approve click answering an unrelated question.
        token = secrets.token_urlsafe(16)
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
        try:
            await thread.client.chat_update(
                channel=thread.channel_id,
                ts=timestamp,
                text=_settled_text(question, answer, pending.answered_by),
                blocks=[],
            )
        except Exception:
            # The answer is already in hand, so failing to tidy the buttons away is
            # not a reason to fail the run. The prompt is deregistered either way,
            # so those buttons now resolve to nothing.
            logger.warning('Could not update the settled Slack prompt in %s', thread.key, exc_info=True)
        return answer

    def resolve(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Record a button click. Call this from the application's action handler.

        `block_id` and `value` come straight off the Slack action payload. Returns
        `False` when the click changes nothing -- an expired prompt, a value that
        is not one of its options, a second click on a prompt already answered, or
        a user who is not allowed to answer this one.
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
                    'text': {'type': 'plain_text', 'text': choice},
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

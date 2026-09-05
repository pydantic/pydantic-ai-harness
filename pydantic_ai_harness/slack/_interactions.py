"""Ask approvers a question in Slack and wait for the answer.

One registry routes button clicks to pending runs and serializes approval rounds
from the same source thread.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Collection
from dataclasses import dataclass, field
from weakref import WeakValueDictionary

import anyio
from pydantic import BaseModel, ValidationError

from pydantic_ai_harness.slack._client import SlackClient, SlackResponse
from pydantic_ai_harness.slack._thread import SlackThread

PROMPT_ACTION_PREFIX = 'pydantic_ai_harness_slack_prompt:'
"""Prefix on the `action_id` of every button these prompts post.

`SlackApp` registers the matching Bolt action handler.
"""

DEFAULT_PROMPT_TIMEOUT_SECONDS = 600.0
"""How long a prompt waits before giving up. Ten minutes suits a working thread:
long enough for someone to come back from a meeting, short enough that a run does
not sit open overnight holding an agent turn."""

# Slack-owned Block Kit limits were verified 2026-09-05 against:
# https://docs.slack.dev/reference/methods/conversations.open/
# https://docs.slack.dev/reference/block-kit/blocks/section-block/
# https://docs.slack.dev/reference/block-kit/blocks/actions-block/
# https://docs.slack.dev/reference/block-kit/block-elements/button-element/
# Re-check these pages before changing the question, option-count, or button limits.
MAX_QUESTION_CHARS = 3000
"""Longest question a prompt can ask. Slack's cap on one section block, which is
what the question is rendered as. The settled message that replaces the prompt is
plain text under a 4000 cap, so it needs no margin here. Longer questions are
refused rather than truncated: half a question is one nobody can answer."""

APPROVE = 'Approve'
DENY = 'Deny'
_APPROVAL_OPTIONS = (APPROVE, DENY)

_SETTLE_RETRY_DELAYS = (0.25, 1.0)
"""Local bounded backoff for removing buttons after a prompt settles."""

logger = logging.getLogger(__name__)


class SlackPromptError(RuntimeError):
    """Slack posted the prompt but did not identify the message.

    Without a message id there is nothing to edit once the prompt settles.
    """


@dataclass(slots=True)
class _Pending:
    allowed_user_ids: frozenset[str]
    event: anyio.Event = field(default_factory=anyio.Event)
    answer: str | None = None
    answered_by: str | None = None


@dataclass(frozen=True, slots=True)
class _PostedPrompt:
    channel_id: str
    timestamp: str


class _Conversation(BaseModel):
    id: str


class SlackInteractions:
    """Post private approval buttons to each approver and wait for a click.

    One registry per capability routes the app's action listener to pending
    approval prompts. Prompts for the same thread are serialized, so a second
    question waits for the first to be answered rather than posting a competing
    set of buttons.

    A click is only accepted from an allowed user. When a prompt does not name
    any, the person whose message started the run is the only one who can answer.
    """

    def __init__(self, *, timeout_seconds: float = DEFAULT_PROMPT_TIMEOUT_SECONDS) -> None:
        """Wait `timeout_seconds` for an answer before treating a prompt as unanswered."""
        self._timeout_seconds = timeout_seconds
        self._pending: dict[str, _Pending] = {}
        # Weak, so a thread's lock is collected once no prompt holds it. A busy
        # workspace would otherwise accumulate one lock per thread forever.
        self._locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()

    async def ask(
        self,
        client: SlackClient,
        thread: SlackThread,
        question: str,
        *,
        allowed_user_ids: Collection[str] | None = None,
    ) -> str | None:
        """DM `question` with Approve and Deny buttons to each approver.

        Returns the chosen option, or `None` when nobody answered in time. Either
        way each private message is edited to record what happened, so approver
        DMs do not keep buttons that no longer do anything. Cleanup is shielded
        from run cancellation.

        Raises:
            SlackPromptError: If Slack cannot identify or settle the posted message.
        """
        if allowed_user_ids is not None:
            allowed = frozenset(allowed_user_ids)
        elif thread.user_id is not None:
            allowed = frozenset({thread.user_id})
        else:  # pragma: no cover - SlackApp always supplies the invoking user
            raise RuntimeError('Slack approval prompts require an invoking user or explicit approvers')
        lock = self._locks.setdefault(thread.key, anyio.Lock())
        async with lock:
            return await self._ask_once(client, thread, question, allowed)

    async def _ask_once(
        self,
        client: SlackClient,
        thread: SlackThread,
        question: str,
        allowed: frozenset[str],
    ) -> str | None:
        # Random, not a counter: a counter restarts with the process, so a button
        # left over from a previous run would resolve the first prompt of the new
        # one -- an old Approve click answering an unrelated question.
        token = secrets.token_urlsafe(16)
        pending = _Pending(allowed_user_ids=allowed)
        self._pending[token] = pending
        posted: list[_PostedPrompt] = []
        settlement_error: Exception | None = None
        waiting = False
        expired = False
        try:
            for user_id in sorted(allowed):
                dm = await client.conversations_open(users=user_id)
                channel_id = _conversation_id(dm)
                response = await client.chat_postMessage(
                    channel=channel_id,
                    text=question,
                    blocks=_prompt_blocks(token, question),
                    mrkdwn=False,
                )
                posted_timestamp = response.get('ts')
                if not isinstance(posted_timestamp, str):
                    raise SlackPromptError('Slack did not return a timestamp for the prompt message')
                posted.append(_PostedPrompt(channel_id=channel_id, timestamp=posted_timestamp))

            await _set_status(client, thread, 'suspended')
            waiting = True
            with anyio.move_on_after(self._timeout_seconds):
                await pending.event.wait()
            expired = pending.answer is None
            return pending.answer
        finally:
            # Cleanup must finish when the run is cancelled. Keep the prompt
            # registered until its buttons have been removed, so a click during
            # the update cannot race a prompt that already looks settled.
            with anyio.CancelScope(shield=True):
                for prompt in posted:
                    for attempt in range(len(_SETTLE_RETRY_DELAYS) + 1):
                        try:
                            await client.chat_update(
                                channel=prompt.channel_id,
                                ts=prompt.timestamp,
                                text=_settled_text(question, pending.answer, pending.answered_by, expired=expired),
                                blocks=[],
                                mrkdwn=False,
                            )
                            break
                        except Exception as error:
                            if attempt == len(_SETTLE_RETRY_DELAYS):
                                logger.warning(
                                    'Could not update a settled Slack prompt for %s after %d attempts',
                                    thread.key,
                                    len(_SETTLE_RETRY_DELAYS) + 1,
                                    exc_info=True,
                                )
                                settlement_error = error
                            else:
                                await anyio.sleep(_SETTLE_RETRY_DELAYS[attempt])
                if waiting:
                    await _set_status(client, thread, 'processing')
            del self._pending[token]
            if settlement_error is not None:
                raise SlackPromptError(
                    'Slack did not remove the approval buttons, so the answer was not accepted'
                ) from settlement_error

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
        if user_id not in pending.allowed_user_ids or value not in _APPROVAL_OPTIONS:
            return False
        pending.answer = value
        pending.answered_by = user_id
        pending.event.set()
        return True


def _conversation_id(response: SlackResponse) -> str:
    """Return the DM channel ID from `conversations.open`, or fail closed."""
    try:
        return _Conversation.model_validate(response.get('channel')).id
    except ValidationError as error:
        raise SlackPromptError('Slack did not return a channel ID for the approval conversation') from error


async def _set_status(client: SlackClient, thread: SlackThread, status: str) -> None:
    """Reflect approval waiting in Slack without making the decision depend on status UI."""
    if thread.thread_ts is None:  # pragma: no cover - SlackApp approval runs always have a thread root
        return
    try:
        await client.agents_sessions_setStatus(
            channel_id=thread.channel_id,
            thread_ts=thread.thread_ts,
            status=status,
        )
    except Exception:
        logger.debug('Could not update the Slack approval status for %s', thread.key, exc_info=True)


def _prompt_blocks(token: str, question: str) -> list[dict[str, object]]:
    return [
        {'type': 'section', 'text': {'type': 'plain_text', 'text': question}},
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
                for index, choice in enumerate(_APPROVAL_OPTIONS)
            ],
        },
    ]


def _settled_text(question: str, answer: str | None, answered_by: str | None, *, expired: bool) -> str:
    if answer is None:
        outcome = 'No answer, so this prompt expired.' if expired else 'Approval was abandoned before a decision.'
        return f'{question}\n\n{outcome}'
    return f'{question}\n\n{answered_by} chose {answer}.'

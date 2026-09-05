from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import TypeAdapter
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse
from typing_extensions import TypedDict

from pydantic_ai_harness.slack import SlackThread, bind_thread


class _ActionBlock(TypedDict):
    """The `actions` block of a posted prompt, as far as the tests read it."""

    block_id: str
    elements: list[dict[str, object]]


_BLOCKS_ADAPTER = TypeAdapter(list[dict[str, object]])
_ACTION_BLOCK_ADAPTER = TypeAdapter(_ActionBlock)


@dataclass
class SlackCall:
    """One recorded Slack Web API call."""

    method: str
    kwargs: dict[str, object]


@dataclass
class _Recorder:
    """Mutable state the fake carries, kept off `AsyncWebClient`'s own attributes."""

    calls: list[SlackCall] = field(default_factory=list[SlackCall])
    next_ts: str = '1700000000.000100'
    post_response: dict[str, object] | None = None
    post_error: Exception | None = None
    update_error: Exception | None = None
    status_error: Exception | None = None


class FakeSlackClient(AsyncWebClient):
    """A real `AsyncWebClient` whose calls are recorded rather than sent.

    Subclassing rather than duck-typing is what keeps the fake honest: Pyright
    checks each override against the SDK's own signature, so a method the SDK
    renames or retypes fails here instead of against live Slack.
    """

    def __init__(self) -> None:
        super().__init__(token='xoxb-fake')  # pyright: ignore[reportUnknownMemberType]
        self.recorder = _Recorder()

    @property
    def calls(self) -> list[SlackCall]:
        return self.recorder.calls

    @property
    def next_ts(self) -> str:
        return self.recorder.next_ts

    def method_calls(self, method: str) -> list[SlackCall]:
        return [call for call in self.calls if call.method == method]

    def _record(self, method: str, kwargs: dict[str, object], payload: dict[str, object]) -> AsyncSlackResponse:
        self.calls.append(SlackCall(method, kwargs))
        return AsyncSlackResponse(
            client=self,
            http_verb='POST',
            api_url=f'https://slack.com/api/{method}',
            req_args={},
            data=payload,
            headers={},
            status_code=200,
        )

    async def chat_postMessage(
        self,
        *,
        channel: str | None = None,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
        thread_ts: str | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        if self.recorder.post_error is not None:
            raise self.recorder.post_error
        return self._record(
            'chat_postMessage',
            {'channel': channel, 'text': text, 'blocks': blocks, 'thread_ts': thread_ts},
            self.recorder.post_response or {'ok': True, 'ts': self.next_ts},
        )

    async def chat_update(
        self,
        *,
        channel: str | None = None,
        ts: str | None = None,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        if self.recorder.update_error is not None:
            raise self.recorder.update_error
        return self._record(
            'chat_update',
            {'channel': channel, 'ts': ts, 'text': text, 'blocks': blocks},
            {'ok': True, 'ts': ts},
        )

    async def files_upload_v2(
        self,
        *,
        channel: str | None = None,
        file: str | bytes | object = None,
        title: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        return self._record(
            'files_upload_v2',
            {
                'channel': channel,
                'file': file,
                'title': title,
                'initial_comment': initial_comment,
                'thread_ts': thread_ts,
            },
            {'ok': True},
        )

    async def assistant_threads_setStatus(
        self,
        *,
        channel_id: str | None = None,
        thread_ts: str | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        if self.recorder.status_error is not None:
            raise self.recorder.status_error
        return self._record(
            'assistant_threads_setStatus',
            {'channel_id': channel_id, 'thread_ts': thread_ts, 'status': status},
            {'ok': True},
        )


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def slack_client() -> FakeSlackClient:
    return FakeSlackClient()


@pytest.fixture
def thread() -> SlackThread:
    return SlackThread(
        channel_id='C123',
        thread_ts='1700000000.000001',
        user_id='U0ASKER',
        team_id='T1',
    )


@pytest.fixture
def bound_thread(thread: SlackThread) -> Iterator[SlackThread]:
    """A thread bound for the test, the way `SlackBot` binds one around a run."""
    with bind_thread(thread):
        yield thread


def prompt_block_id(client: FakeSlackClient, index: int = 0) -> str:
    """Block id of the nth prompt posted, which is what a button click carries back."""
    return _prompt_action_block(client, index)['block_id']


def prompt_buttons(client: FakeSlackClient, index: int = 0) -> list[dict[str, object]]:
    """The button elements of the nth prompt posted."""
    return _prompt_action_block(client, index)['elements']


def _prompt_action_block(client: FakeSlackClient, index: int) -> _ActionBlock:
    posts = client.method_calls('chat_postMessage')
    blocks = _BLOCKS_ADAPTER.validate_python(posts[index].kwargs['blocks'])
    return _ACTION_BLOCK_ADAPTER.validate_python(blocks[1])

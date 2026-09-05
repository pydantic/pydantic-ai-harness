from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai_harness.slack import SlackThread


class _ActionBlock(TypedDict):
    """The `actions` block of a posted prompt, as far as the tests read it."""

    block_id: str
    elements: list[dict[str, object]]


_BLOCKS_ADAPTER = TypeAdapter(list[dict[str, object]])
_ACTION_BLOCK_ADAPTER = TypeAdapter(_ActionBlock)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class SlackCall:
    """One recorded Slack Web API call."""

    method: str
    kwargs: dict[str, object]


class FakeSlackResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def get(self, key: str, default: object = None) -> object:
        return self._payload.get(key, default)


@dataclass
class FakeSlackClient:
    """Records calls and hands back canned responses."""

    calls: list[SlackCall] = field(default_factory=list[SlackCall])
    next_ts: str = '1700000000.000100'
    post_response: dict[str, object] | None = None
    status_error: Exception | None = None

    def _record(self, method: str, kwargs: dict[str, object]) -> FakeSlackResponse:
        self.calls.append(SlackCall(method, kwargs))
        return FakeSlackResponse({'ok': True, 'ts': self.next_ts})

    def method_calls(self, method: str) -> list[SlackCall]:
        return [call for call in self.calls if call.method == method]

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
        thread_ts: str | None = None,
    ) -> FakeSlackResponse:
        self.calls.append(
            SlackCall('chat_postMessage', {'channel': channel, 'text': text, 'blocks': blocks, 'thread_ts': thread_ts})
        )
        if self.post_response is not None:
            return FakeSlackResponse(self.post_response)
        return FakeSlackResponse({'ok': True, 'ts': self.next_ts})

    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
    ) -> FakeSlackResponse:
        return self._record('chat_update', {'channel': channel, 'ts': ts, 'text': text, 'blocks': blocks})

    async def files_upload_v2(
        self,
        *,
        channel: str,
        file: str,
        title: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> FakeSlackResponse:
        return self._record(
            'files_upload_v2',
            {
                'channel': channel,
                'file': file,
                'title': title,
                'initial_comment': initial_comment,
                'thread_ts': thread_ts,
            },
        )

    async def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
    ) -> FakeSlackResponse:
        if self.status_error is not None:
            raise self.status_error
        return self._record(
            'assistant_threads_setStatus', {'channel_id': channel_id, 'thread_ts': thread_ts, 'status': status}
        )


@pytest.fixture
def slack_client() -> FakeSlackClient:
    return FakeSlackClient()


@pytest.fixture
def thread(slack_client: FakeSlackClient) -> SlackThread:
    return SlackThread(
        client=slack_client,
        channel_id='C123',
        thread_ts='1700000000.000001',
        user_id='U0ASKER',
        team_id='T1',
    )


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

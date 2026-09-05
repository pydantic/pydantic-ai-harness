from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pydantic_ai.mcp as mcp_module
import pytest
from pydantic import TypeAdapter
from pydantic_ai import FunctionToolset
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse
from typing_extensions import TypedDict

from pydantic_ai_harness.slack import SlackTool


async def _fake_read_channel(channel_id: str) -> str:
    return channel_id


async def _fake_read_thread(channel_id: str, message_ts: str) -> str:
    return f'{channel_id}:{message_ts}'


async def _fake_read_file(file_id: str) -> str:
    return file_id


async def _fake_operation() -> str:
    return 'ok'


def fake_mcp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit: SlackTool | None = None,
    extra_names: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Replace the remote Slack boundary with the same public tool catalog."""
    authorizations: list[dict[str, str]] = []

    def build(_url: str, *, id: str, headers: dict[str, str]) -> FunctionToolset[object]:
        assert id == 'slack-mcp'
        authorizations.append(headers)
        toolset = FunctionToolset[object](id=id)
        for slack_tool in SlackTool:
            if slack_tool is omit:
                continue
            if slack_tool is SlackTool.READ_CHANNEL:
                toolset.add_function(_fake_read_channel, name=slack_tool.value, description='Slack operation')
            elif slack_tool is SlackTool.READ_THREAD:
                toolset.add_function(_fake_read_thread, name=slack_tool.value, description='Slack operation')
            elif slack_tool is SlackTool.READ_FILE:
                toolset.add_function(_fake_read_file, name=slack_tool.value, description='Slack operation')
            else:
                toolset.add_function(_fake_operation, name=slack_tool.value, description='Slack operation')
        for name in extra_names:

            async def provider_operation() -> str:
                return 'ok'

            toolset.add_function(provider_operation, name=name, description='New Slack provider operation')
        return toolset

    monkeypatch.setattr(mcp_module, 'MCPToolset', build)
    return authorizations


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
    update_attempts: int = 0
    status_error: Exception | None = None
    open_response: dict[str, object] | None = None
    open_error_user_id: str | None = None


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

    async def conversations_open(
        self,
        *,
        channel: str | None = None,
        return_im: bool | None = None,
        users: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        user_id = users if isinstance(users, str) else '-'.join(users or ())
        if self.recorder.open_error_user_id == user_id:
            raise RuntimeError(f'Could not open a DM with {user_id}')
        return self._record(
            'conversations_open',
            {'channel': channel, 'return_im': return_im, 'users': users},
            self.recorder.open_response or {'ok': True, 'channel': {'id': f'D-{user_id}'}},
        )

    async def chat_postMessage(
        self,
        *,
        channel: str | None = None,
        text: str | None = None,
        markdown_text: str | None = None,
        blocks: Sequence[object] | None = None,
        thread_ts: str | None = None,
        mrkdwn: bool | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        if self.recorder.post_error is not None:
            raise self.recorder.post_error
        response = self._record(
            'chat_postMessage',
            {
                'channel': channel,
                'text': text,
                'markdown_text': markdown_text,
                'blocks': blocks,
                'thread_ts': thread_ts,
                'mrkdwn': mrkdwn,
            },
            self.recorder.post_response or {'ok': True, 'ts': self.next_ts},
        )
        return response

    async def chat_update(
        self,
        *,
        channel: str | None = None,
        ts: str | None = None,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
        mrkdwn: bool | None = None,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        self.recorder.update_attempts += 1
        if self.recorder.update_error is not None:
            raise self.recorder.update_error
        return self._record(
            'chat_update',
            {'channel': channel, 'ts': ts, 'text': text, 'blocks': blocks, 'mrkdwn': mrkdwn},
            {'ok': True, 'ts': ts},
        )

    async def agents_sessions_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str | None = None,
        status: str,
        **kwargs: Any,
    ) -> AsyncSlackResponse:
        if self.recorder.status_error is not None:
            raise self.recorder.status_error
        return self._record(
            'agents_sessions_setStatus',
            {'channel_id': channel_id, 'thread_ts': thread_ts, 'status': status},
            {'ok': True},
        )


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def slack_client() -> FakeSlackClient:
    return FakeSlackClient()


def prompt_block_id(client: FakeSlackClient, index: int = 0) -> str:
    """Block id of the nth prompt posted, which is what a button click carries back."""
    return _prompt_action_block(client, index)['block_id']


def _prompt_action_block(client: FakeSlackClient, index: int) -> _ActionBlock:
    posts = client.method_calls('chat_postMessage')
    blocks = _BLOCKS_ADAPTER.validate_python(posts[index].kwargs['blocks'])
    return _ACTION_BLOCK_ADAPTER.validate_python(blocks[1])

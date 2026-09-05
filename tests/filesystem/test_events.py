"""Events emitted by the FileSystem capability."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.filesystem import (
    DirectoryListedEvent,
    FileReadEvent,
    FileSystem,
    FileWrittenEvent,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _has_tool_result(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, (RetryPromptPart, ToolReturnPart)) for message in messages for part in message.parts)


def _tool_model(tool_name: str, json_args: str) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if not _has_tool_result(messages):
            yield {0: DeltaToolCall(name=tool_name, json_args=json_args, tool_call_id='call_1')}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


async def _run_and_collect(
    root: Path, tool_name: str, json_args: str, *, denied_patterns: list[str] | None = None
) -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    capability = FileSystem(root_dir=root, denied_patterns=denied_patterns or [])
    await Agent(_tool_model(tool_name, json_args), capabilities=[capability]).run('go', event_stream_handler=handler)
    return events


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


class TestFileSystemEvents:
    async def test_read_emits_normalized_path_and_hash(self, tmp_path: Path) -> None:
        content = 'hello\n'
        (tmp_path / 'target.txt').write_text(content)

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"sub/../target.txt"}')

        assert [event for event in events if isinstance(event, FileReadEvent)] == [
            FileReadEvent(
                path='target.txt',
                content_hash=_hash(content),
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='read_file',
            )
        ]

    async def test_binary_read_emits_raw_content_hash(self, tmp_path: Path) -> None:
        content = b'hello\x00world'
        (tmp_path / 'binary.bin').write_bytes(content)

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"binary.bin"}')

        read_events = [event for event in events if isinstance(event, FileReadEvent)]
        assert len(read_events) == 1
        assert read_events[0].path == 'binary.bin'
        assert read_events[0].content_hash == hashlib.sha256(content).hexdigest()[:12]

    async def test_list_emits_normalized_path_and_entry_count(self, tmp_path: Path) -> None:
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'one.txt').write_text('one')
        (sub / 'two.txt').write_text('two')

        events = await _run_and_collect(tmp_path, 'list_directory', '{"path":"other/../sub"}')

        assert [event for event in events if isinstance(event, DirectoryListedEvent)] == [
            DirectoryListedEvent(
                path='sub',
                entry_count=2,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='list_directory',
            )
        ]

    @pytest.mark.parametrize(
        ('tool_name', 'json_args', 'content'),
        [
            ('write_file', '{"path":"sub/../target.txt","content":"new\\n"}', 'new\n'),
            (
                'edit_file',
                '{"path":"sub/../target.txt","old_text":"old","new_text":"new"}',
                'new\n',
            ),
        ],
    )
    async def test_write_and_edit_emit_one_written_event(
        self, tmp_path: Path, tool_name: str, json_args: str, content: str
    ) -> None:
        (tmp_path / 'target.txt').write_text('old\n')

        events = await _run_and_collect(tmp_path, tool_name, json_args)

        assert [event for event in events if isinstance(event, FileWrittenEvent)] == [
            FileWrittenEvent(
                path='target.txt',
                content_hash=_hash(content),
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name=tool_name,
            )
        ]

    async def test_denied_operation_emits_no_capability_event(self, tmp_path: Path) -> None:
        (tmp_path / 'secret.txt').write_text('hidden')

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"secret.txt"}', denied_patterns=['secret.txt'])

        assert not any(isinstance(event, (FileReadEvent, DirectoryListedEvent, FileWrittenEvent)) for event in events)

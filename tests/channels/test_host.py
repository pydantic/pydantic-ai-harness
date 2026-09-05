from __future__ import annotations

import asyncio
from collections.abc import Sequence

import anyio
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult

from pydantic_ai_harness.channels import (
    ChannelEvent,
    ChannelHost,
    InMemoryConversationStore,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _RecordingAdapter:
    def __init__(self, error: Exception | None = None) -> None:
        self.replies: list[tuple[ChannelEvent, str]] = []
        self.error = error

    async def reply(self, event: ChannelEvent, text: str) -> None:
        self.replies.append((event, text))
        if self.error is not None:
            raise self.error


class _RecordingStore:
    def __init__(self, *, save_error: Exception | None = None) -> None:
        self.delegate = InMemoryConversationStore()
        self.loads: list[str] = []
        self.saves: list[tuple[str, Sequence[ModelMessage]]] = []
        self.save_error = save_error

    async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
        self.loads.append(conversation_id)
        return await self.delegate.load(conversation_id)

    async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        self.saves.append((conversation_id, messages))
        if self.save_error is not None:
            raise self.save_error
        await self.delegate.save(conversation_id, messages)


class _FalsyStore(_RecordingStore):
    def __bool__(self) -> bool:
        return False


def _event(event_id: str, conversation_id: str = 'conversation') -> ChannelEvent:
    return ChannelEvent(
        event_id=event_id,
        conversation_id=conversation_id,
        sender_id='sender',
        text=f'prompt {event_id}',
        reply_to_id=f'message {event_id}',
    )


class TestChannelHost:
    async def test_runs_public_agent_replies_and_saves_history(self) -> None:
        adapter = _RecordingAdapter()
        store = _RecordingStore()
        agent: Agent[None, str] = Agent('test', name='channel-agent')
        host = ChannelHost(agent, adapter, store=store)

        first = await host.handle(_event('one'))
        second = await host.handle(_event('two'))

        assert isinstance(first, AgentRunResult)
        assert [text for _, text in adapter.replies] == ['success (no tool calls)', 'success (no tool calls)']
        assert store.loads == ['conversation', 'conversation']
        assert [len(messages) for _, messages in store.saves] == [2, 4]
        assert len(second.all_messages()) == 4
        assert all(message.conversation_id == 'conversation' for message in second.all_messages())

    async def test_default_store_keeps_conversations_separate(self) -> None:
        adapter = _RecordingAdapter()
        host = ChannelHost(Agent('test'), adapter)

        left = await host.handle(_event('left', 'left'))
        right = await host.handle(_event('right', 'right'))

        assert len(left.all_messages()) == 2
        assert len(right.all_messages()) == 2

    async def test_uses_a_falsy_conversation_store(self) -> None:
        adapter = _RecordingAdapter()
        store = _FalsyStore()
        assert not store
        host = ChannelHost(Agent('test'), adapter, store=store)

        await host.handle(_event('one'))

        assert store.loads == ['conversation']
        assert len(store.saves) == 1

    async def test_passes_per_event_dependencies(self) -> None:
        adapter = _RecordingAdapter()
        seen: list[str] = []

        def instructions(ctx: RunContext[str]) -> str:
            seen.append(ctx.deps)
            return f'Dependency: {ctx.deps}'

        agent = Agent('test', deps_type=str, instructions=instructions)
        host = ChannelHost(agent, adapter)

        await host.handle(_event('one'), deps='tenant-specific')

        assert seen == ['tenant-specific']

    async def test_serializes_one_conversation_but_allows_different_conversations(self) -> None:
        active_by_conversation: dict[str, int] = {}
        maximum_by_conversation: dict[str, int] = {}
        total_active = 0
        maximum_total = 0
        two_models_started = asyncio.Event()
        release_models = asyncio.Event()

        async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal total_active, maximum_total
            conversation_id = messages[-1].conversation_id
            assert conversation_id is not None
            active_by_conversation[conversation_id] = active_by_conversation.get(conversation_id, 0) + 1
            maximum_by_conversation[conversation_id] = max(
                maximum_by_conversation.get(conversation_id, 0), active_by_conversation[conversation_id]
            )
            total_active += 1
            maximum_total = max(maximum_total, total_active)
            if total_active == 2:
                two_models_started.set()
            await release_models.wait()
            active_by_conversation[conversation_id] -= 1
            total_active -= 1
            return ModelResponse(parts=[TextPart('done')])

        host = ChannelHost(Agent(FunctionModel(model)), _RecordingAdapter())
        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(host.handle, _event('same-1', 'same'))
                group.start_soon(host.handle, _event('same-2', 'same'))
                group.start_soon(host.handle, _event('other', 'other'))
                await two_models_started.wait()
                assert maximum_by_conversation == {'same': 1, 'other': 1}
                assert maximum_total == 2
                release_models.set()

        assert maximum_by_conversation == {'same': 1, 'other': 1}
        assert maximum_total == 2

    async def test_does_not_retry_adapter_failure_or_save_history(self) -> None:
        calls = 0

        def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart('done')])

        adapter = _RecordingAdapter(RuntimeError('send failed'))
        store = _RecordingStore()
        host = ChannelHost(Agent(FunctionModel(model)), adapter, store=store)

        with pytest.raises(RuntimeError, match='send failed'):
            await host.handle(_event('one'))
        adapter.error = None
        result = await host.handle(_event('two'))

        assert calls == 2
        assert len(result.all_messages()) == 2
        assert len(store.saves) == 1

    async def test_store_failure_happens_after_reply_and_is_not_retried(self) -> None:
        adapter = _RecordingAdapter()
        store = _RecordingStore(save_error=RuntimeError('save failed'))
        host = ChannelHost(Agent('test'), adapter, store=store)

        with pytest.raises(RuntimeError, match='save failed'):
            await host.handle(_event('one'))
        store.save_error = None
        result = await host.handle(_event('two'))

        assert len(adapter.replies) == 2
        assert [len(messages) for _, messages in store.saves] == [2, 2]
        assert len(result.all_messages()) == 2

    async def test_store_load_failure_stops_processing_and_releases_lock(self) -> None:
        class FailingLoadStore(_RecordingStore):
            load_error: Exception | None = RuntimeError('load failed')

            async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
                self.loads.append(conversation_id)
                if self.load_error is not None:
                    raise self.load_error
                return await self.delegate.load(conversation_id)

        calls = 0

        def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart('done')])

        adapter = _RecordingAdapter()
        store = FailingLoadStore()
        host = ChannelHost(Agent(FunctionModel(model)), adapter, store=store)

        with pytest.raises(RuntimeError, match='load failed'):
            await host.handle(_event('one'))
        assert calls == 0
        assert adapter.replies == []
        assert store.saves == []

        store.load_error = None
        result = await host.handle(_event('two'))
        assert len(result.all_messages()) == 2

    async def test_duplicate_event_ids_are_caller_owned(self) -> None:
        adapter = _RecordingAdapter()
        host = ChannelHost(Agent('test'), adapter)
        event = _event('duplicate')

        await host.handle(event)
        await host.handle(event)

        assert len(adapter.replies) == 2

    async def test_agent_failure_does_not_reply_or_save_and_releases_lock(self) -> None:
        calls = 0

        def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError('model failed')
            return ModelResponse(parts=[TextPart('done')])

        adapter = _RecordingAdapter()
        store = _RecordingStore()
        host = ChannelHost(Agent(FunctionModel(model)), adapter, store=store)

        with pytest.raises(RuntimeError, match='model failed'):
            await host.handle(_event('one'))
        await host.handle(_event('two'))

        assert len(adapter.replies) == 1
        assert len(store.saves) == 1

    async def test_cancellation_does_not_reply_or_save_and_releases_lock(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart('done')])

        adapter = _RecordingAdapter()
        store = _RecordingStore()
        host = ChannelHost(Agent(FunctionModel(model)), adapter, store=store)
        task = asyncio.create_task(host.handle(_event('cancelled')))
        with anyio.fail_after(5):
            await started.wait()

        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=5)
        assert done == {task}
        assert not pending
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        with anyio.fail_after(5):
            await host.handle(_event('next'))

        assert len(adapter.replies) == 1
        assert len(store.saves) == 1

    async def test_cancellation_while_waiting_for_lock_does_not_start_run(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return ModelResponse(parts=[TextPart('done')])

        adapter = _RecordingAdapter()
        store = _RecordingStore()
        host = ChannelHost(Agent(FunctionModel(model)), adapter, store=store)
        first = asyncio.create_task(host.handle(_event('first')))
        with anyio.fail_after(5):
            await started.wait()
        waiting = asyncio.create_task(host.handle(_event('cancelled')))
        await asyncio.sleep(0)

        waiting.cancel()
        done, pending = await asyncio.wait({waiting}, timeout=5)
        assert done == {waiting}
        assert not pending
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release.set()
        with anyio.fail_after(5):
            await first
            await host.handle(_event('next'))

        assert calls == 2
        assert len(adapter.replies) == 2
        assert len(store.saves) == 2

    async def test_cancellation_during_load_stops_processing_and_releases_lock(self) -> None:
        load_started = asyncio.Event()
        release_load = asyncio.Event()

        class BlockingLoadStore(_RecordingStore):
            block = True

            async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
                self.loads.append(conversation_id)
                if self.block:
                    load_started.set()
                    await release_load.wait()
                return await self.delegate.load(conversation_id)

        adapter = _RecordingAdapter()
        store = BlockingLoadStore()
        host = ChannelHost(Agent('test'), adapter, store=store)
        task = asyncio.create_task(host.handle(_event('cancelled')))
        with anyio.fail_after(5):
            await load_started.wait()

        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=5)
        assert done == {task}
        assert not pending
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adapter.replies == []
        assert store.saves == []

        store.block = False
        with anyio.fail_after(5):
            result = await host.handle(_event('next'))
        assert len(result.all_messages()) == 2

    async def test_cancellation_during_reply_leaves_history_uncommitted_and_releases_lock(self) -> None:
        reply_started = asyncio.Event()
        release_reply = asyncio.Event()

        class BlockingAdapter(_RecordingAdapter):
            block = True

            async def reply(self, event: ChannelEvent, text: str) -> None:
                self.replies.append((event, text))
                if self.block:
                    reply_started.set()
                    await release_reply.wait()

        adapter = BlockingAdapter()
        store = _RecordingStore()
        host = ChannelHost(Agent('test'), adapter, store=store)
        task = asyncio.create_task(host.handle(_event('cancelled')))
        with anyio.fail_after(5):
            await reply_started.wait()

        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=5)
        assert done == {task}
        assert not pending
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(adapter.replies) == 1
        assert store.saves == []

        adapter.block = False
        with anyio.fail_after(5):
            result = await host.handle(_event('next'))
        assert len(result.all_messages()) == 2

    async def test_cancellation_during_save_happens_after_reply_and_releases_lock(self) -> None:
        save_started = asyncio.Event()
        release_save = asyncio.Event()

        class BlockingStore(_RecordingStore):
            async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
                self.saves.append((conversation_id, messages))
                save_started.set()
                await release_save.wait()
                await self.delegate.save(conversation_id, messages)

        adapter = _RecordingAdapter()
        store = BlockingStore()
        host = ChannelHost(Agent('test'), adapter, store=store)
        task = asyncio.create_task(host.handle(_event('cancelled')))
        with anyio.fail_after(5):
            await save_started.wait()

        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=5)
        assert done == {task}
        assert not pending
        with pytest.raises(asyncio.CancelledError):
            await task
        release_save.set()
        with anyio.fail_after(5):
            result = await host.handle(_event('next'))

        assert len(adapter.replies) == 2
        assert [len(messages) for _, messages in store.saves] == [2, 2]
        assert len(result.all_messages()) == 2

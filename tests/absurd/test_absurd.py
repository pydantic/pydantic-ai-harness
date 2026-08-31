"""Tests for the `AbsurdDurability` capability.

Behavior is driven through `Agent(..., capabilities=[AbsurdDurability()])` inside an in-memory
`FakeAsyncTaskContext` (see `conftest.py`) so there is no Postgres or Docker dependency. The two
production behaviors the capability relies on -- encounter-order step-name disambiguation and a
replay that serves stored checkpoints without re-running `fn` -- are reproduced faithfully by the
fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractContextManager

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import JsonValue
from pydantic_ai import Agent
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.capabilities import AbstractCapability, ResolveModelId, durable_operation
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset

from pydantic_ai_harness.absurd import AbsurdDurability

from ._helpers import FakeAsyncTaskContext, FakeSyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


def _text_model(counter: dict[str, int] | None = None) -> FunctionModel:
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


def _tool_then_done_model(tool_name: str, args: dict[str, JsonValue]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for message in messages for part in message.parts
        )
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


class TestTransparency:
    async def test_run_outside_task_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1


class TestCapabilityOperation:
    async def test_operation_is_checkpointed_and_replayed(self) -> None:
        calls: list[str] = []

        class Recorder(AbstractCapability[object]):
            id = 'recorder'

            async def before_run(self, ctx: RunContext[object]) -> None:
                await self.record(ctx, 'started')

            @durable_operation('record')
            async def record(self, ctx: RunContext[object], value: str) -> None:
                del ctx
                calls.append(value)

        agent = Agent(_text_model(), name='cap', capabilities=[Recorder(), AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        operation_name = 'cap__capability__recorder.record'
        assert operation_name in ctx.stored
        assert calls == ['started']

        replay = ctx.replay()
        with absurd_task_context(replay):
            await agent.run('hi')

        assert calls == ['started']
        assert replay.invoked == []


class TestModelRequestCheckpoint:
    async def test_request_checkpointed_and_replay_serves_cache(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('hi')
        assert 'a__model.request' in ctx.stored
        assert ctx.invoked == ['a__model.request']

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('hi')

        assert counter['calls'] == 1
        assert first.output == second.output == 'ok'
        assert replay.invoked == []


class TestStreaming:
    async def test_stream_checkpointed_and_replayed(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='stream', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            async with agent.run_stream('hi') as result:
                first_out = await result.get_output()
        assert first_out == 'ok'
        assert 'stream__model.request_stream' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            async with agent.run_stream('hi') as result:
                replay_out = await result.get_output()

        assert replay_out == 'ok'
        assert counter['calls'] == 1
        assert replay.invoked == []

    async def test_handler_sees_live_events_and_stream_replays_equal(self) -> None:
        live_events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                live_events.append(event)

        counter = {'calls': 0}
        agent = Agent(
            _text_model(counter), name='stream', capabilities=[AbsurdDurability(event_stream_handler=handler)]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            async with agent.run_stream_events('hi') as stream:
                first_events = [event async for event in stream]

        # The model-stream events were delivered live to the handler inside the request_stream step.
        assert any(isinstance(event, (PartStartEvent, PartDeltaEvent)) for event in live_events)
        assert 'stream__model.request_stream' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            async with agent.run_stream_events('hi') as stream:
                replay_events = [event async for event in stream]

        assert counter['calls'] == 1
        assert replay_events == first_events


class TestFunctionTool:
    async def test_tool_checkpointed_exactly_once_across_replay(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('charge it')
        assert 'pay__function_toolset__billing.call_tool:charge_card' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('charge it')

        assert calls['n'] == 1
        assert first.output == second.output == 'done'
        assert replay.invoked == []


class TestModelRetry:
    async def test_model_retry_crosses_checkpoint_and_replays(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            calls['n'] += 1
            raise ModelRetry('nope, try again')

        agent = Agent(
            _tool_then_done_model('flaky', {}), name='retry', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')
        step = 'retry__function_toolset__tools.call_tool:flaky'
        # The raised `ModelRetry` crossed the checkpoint as a serialized value, not an exception.
        assert ctx.stored[step] == {'message': 'nope, try again', 'kind': 'model_retry'}
        assert first.output == 'done'
        assert calls['n'] == 1

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        # On replay the stored `ModelRetry` is re-raised without re-running the tool.
        assert calls['n'] == 1
        assert second.output == 'done'
        assert replay.invoked == []


class TestCrashMidRunRetry:
    async def test_model_step_served_from_checkpoint_while_failed_tool_reruns(self) -> None:
        # The core value prop: the model step completes and is checkpointed, then a tool raises a
        # real (non-`ModelRetry`) error that fails the task. On retry, Absurd replays: the model
        # step is served from its checkpoint (model not called again) while the tool re-runs.
        model_calls = {'n': 0}
        tool_attempts = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_attempts['n'] += 1
            if tool_attempts['n'] == 1:
                raise RuntimeError('worker died mid-tool')
            return 'recovered'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            model_calls['n'] += 1
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='flaky', args={})])

        agent = Agent(FunctionModel(model_fn), name='crash', toolsets=[toolset], capabilities=[AbsurdDurability()])

        model_step = 'crash__model.request'
        tool_step = 'crash__function_toolset__tools.call_tool:flaky'

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(RuntimeError, match='worker died mid-tool'):
                await agent.run('go')

        # The model step checkpointed before the tool ran; the failed tool step did not.
        assert model_step in ctx.stored
        assert tool_step not in ctx.stored
        assert model_calls['n'] == 1
        assert tool_attempts['n'] == 1

        replay = ctx.replay()
        with absurd_task_context(replay):
            result = await agent.run('go')

        assert result.output == 'done'
        # The first model request was served from its checkpoint (not re-invoked); the tool re-ran
        # and the second model turn is a fresh step.
        assert model_step not in replay.invoked
        assert tool_step in replay.invoked
        assert f'{model_step}#2' in replay.invoked
        assert model_calls['n'] == 2
        assert tool_attempts['n'] == 2


class TestPerToolOptOut:
    async def test_metadata_false_runs_inline_uncheckpointed(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(metadata={'absurd': False})
        def ping() -> str:
            calls['n'] += 1
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='inline', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('ping it')

        assert result.output == 'done'
        assert calls['n'] == 1
        assert not any('call_tool:ping' in name for name in ctx.stored)


class TestToolStepConfigRejected:
    async def test_non_empty_dict_config_raises(self) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(metadata={'absurd': {'retries': 3}})
        def ping() -> str:  # pragma: no cover - rejected before it can run
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='cfg', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='take no per-tool options'):
                await agent.run('ping it')

    async def test_empty_dict_config_is_allowed(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(metadata={'absurd': {}})
        def ping() -> str:
            calls['n'] += 1
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='cfg', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('ping it')

        assert result.output == 'done'
        assert calls['n'] == 1
        assert 'cfg__function_toolset__tools.call_tool:ping' in ctx.stored


class TestModelSelection:
    async def test_registered_model_folds_id_into_step_and_replays(self) -> None:
        primary = {'n': 0}
        cheap = {'n': 0}

        def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            primary['n'] += 1
            return ModelResponse(parts=[TextPart(content='primary')])

        def cheap_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            cheap['n'] += 1
            return ModelResponse(parts=[TextPart(content='cheap')])

        agent = Agent(
            FunctionModel(primary_fn, model_name='primary'),
            name='sw',
            capabilities=[AbsurdDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            default_result = await agent.run('hi')
            cheap_result = await agent.run('hi', model='cheap')

        assert default_result.output == 'primary'
        assert cheap_result.output == 'cheap'
        assert 'sw__model.request' in ctx.stored
        assert 'sw__model.request.cheap' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            replayed = await agent.run('hi', model='cheap')

        assert replayed.output == 'cheap'
        assert cheap['n'] == 1
        assert replay.invoked == []

    async def test_model_id_with_hash_is_rejected(self) -> None:
        with pytest.raises(UserError, match='contains'):
            AbsurdDurability(models={'cheap#2': FunctionModel(lambda m, i: ModelResponse(parts=[]), model_name='c')})

    async def test_runtime_model_id_with_hash_is_rejected_before_checkpoint(self) -> None:
        model = _text_model()
        resolve_model_id = ResolveModelId(lambda ctx, model_id: model if model_id == 'cheap#2' else None)
        agent = Agent(model, name='runtime', capabilities=[resolve_model_id, AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='contains'):
                await agent.run('hi', model='cheap#2')

        assert ctx.stored == {}

    async def test_string_default_model_gets_unsuffixed_step_name(self) -> None:
        agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        assert 'strdef__model.request' in ctx.stored
        assert not any(name.endswith('.test') for name in ctx.stored)

    async def test_string_default_model_id_with_hash_is_rejected(self) -> None:
        with pytest.raises(UserError, match='contains'):
            Agent('test#2', name='strdef', capabilities=[AbsurdDurability()])


class TestRuntimeToolsets:
    async def test_runtime_executing_toolset_rejected_inside_task(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        late = FunctionToolset(id='late')

        @late.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - rejected before it can run
            return value

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=r'cannot be passed to `run\(toolsets=...\)` at runtime'):
                await agent.run('hi', toolsets=[late])

    async def test_non_executing_runtime_toolset_allowed_inside_task(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi', toolsets=[ExternalToolset(tool_defs=[])])
        assert result.output == 'ok'

    async def test_decorated_toolset_added_after_binding_is_checkpointed(self) -> None:
        calls = {'factory': 0, 'tool': 0}
        agent = Agent(
            _tool_then_done_model('greet', {'name': 'Ada'}), name='decorated', capabilities=[AbsurdDurability()]
        )

        @agent.toolset(id='decorated-tools')
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:
            del ctx
            calls['factory'] += 1
            toolset = FunctionToolset[object](id='inner')

            @toolset.tool_plain
            def greet(name: str) -> str:
                calls['tool'] += 1
                return f'hello {name}'

            return toolset

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('greet')

        first_factory_calls = calls['factory']
        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('greet')

        assert first.output == second.output == 'done'
        assert calls == {'factory': first_factory_calls, 'tool': 1}
        assert replay.invoked == []


class TestConcurrentRuns:
    async def test_same_task_namespace_cannot_overlap(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart(content='ok')])

        agent = Agent(FunctionModel(model_fn), name='shared', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = asyncio.create_task(agent.run('first'))
            await asyncio.wait_for(started.wait(), timeout=1)
            try:
                with pytest.raises(UserError, match='Concurrent Absurd agent runs'):
                    await asyncio.wait_for(agent.run('second'), timeout=1)
            finally:
                release.set()
                await asyncio.wait_for(first, timeout=1)


class TestNonWrappedLeaf:
    async def test_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset(tool_defs=[])
        agent = Agent(_text_model(), name='a', toolsets=[external], capabilities=[AbsurdDurability()])
        assert any(leaf is external for leaf in agent.toolsets)


class TestBindingErrors:
    async def test_unnamed_agent_raises(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_text_model(), capabilities=[AbsurdDurability()])

    async def test_duplicate_toolset_ids_raise(self) -> None:
        first = FunctionToolset(id='dup')

        @first.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value

        second = FunctionToolset(id='dup')

        @second.tool_plain
        def shout(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value.upper()

        with pytest.raises(UserError, match='same `id`'):
            Agent(_text_model(), name='a', toolsets=[first, second], capabilities=[AbsurdDurability()])

    async def test_idless_leaf_function_toolset_raises(self) -> None:
        toolset = FunctionToolset()

        @toolset.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value

        with pytest.raises(UserError, match='need to have a unique `id`'):
            Agent(_text_model(), name='a', toolsets=[toolset], capabilities=[AbsurdDurability()])


class TestSyncContext:
    async def test_sync_task_context_raises(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        with absurd_task_context(FakeSyncTaskContext()):
            with pytest.raises(UserError, match='requires an async Absurd task context'):
                await agent.run('hi')


class TestParallelExecutionMode:
    async def test_parallel_execution_mode_applied_during_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent(
            _text_model(), name='a', capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')]
        )
        recorded: list[ParallelExecutionMode] = []
        real = agent.parallel_tool_call_execution_mode

        def spy(mode: ParallelExecutionMode = 'parallel') -> AbstractContextManager[None]:
            recorded.append(mode)
            return real(mode)

        monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        assert 'parallel_ordered_events' in recorded

    async def test_parallel_execution_mode_untouched_outside_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Outside a task the capability is transparent, so it must not override the agent's
        # configured tool-call execution mode.
        agent = Agent(
            _text_model(), name='a', capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')]
        )
        recorded: list[ParallelExecutionMode] = []
        real = agent.parallel_tool_call_execution_mode

        def spy(mode: ParallelExecutionMode = 'parallel') -> AbstractContextManager[None]:
            recorded.append(mode)
            return real(mode)

        monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)

        # Inside a task the override applies; outside one it must not touch the configured mode.
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')
        assert recorded == ['parallel_ordered_events']

        recorded.clear()
        await agent.run('hi')
        assert recorded == []


class TestRepeatedStepNames:
    async def test_two_runs_in_one_task_disambiguate_by_encounter_order(self) -> None:
        # A single task handler that runs the agent twice: the second run's model step reuses the
        # same step name, so Absurd's encounter-order counter records it under a `#2` suffix.
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('hi')
            second = await agent.run('hi again')

        assert first.output == second.output == 'ok'
        assert 'a__model.request' in ctx.stored
        assert 'a__model.request#2' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            await agent.run('hi')
            await agent.run('hi again')

        assert counter['calls'] == 2
        assert replay.invoked == []

    async def test_same_tool_called_twice_in_one_response(self) -> None:
        calls: list[int] = []
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def charge(amount: int) -> str:
            calls.append(amount)
            return f'charged {amount}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='charge', args={'amount': 1}, tool_call_id='c1'),
                    ToolCallPart(tool_name='charge', args={'amount': 2}, tool_call_id='c2'),
                ]
            )

        agent = Agent(FunctionModel(model_fn), name='pay', toolsets=[toolset], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('charge both')

        assert result.output == 'done'
        assert calls == [1, 2]
        step = 'pay__function_toolset__tools.call_tool:charge'
        assert step in ctx.stored
        assert f'{step}#2' in ctx.stored


class TestParallelOrderedEventsDeterminism:
    async def test_name_assignment_follows_scheduling_order_not_completion(self) -> None:
        # The adversarial case behind excluding `'parallel'` but keeping `'parallel_ordered_events'`:
        # two concurrent calls of the SAME tool, where the first-scheduled call completes LAST.
        # Absurd assigns the `#1`/`#2` checkpoint slot at `ctx.step(...)` entry, which happens before
        # the tool body runs, so assignment follows call-scheduling order (the model's tool-call
        # order), not completion order. If it followed completion order the slots -- and the cached
        # results served on replay -- would swap.
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        async def record(marker: str, delay_ms: int) -> str:
            await asyncio.sleep(delay_ms / 1000)
            return marker

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    # First-scheduled call sleeps longest, so it completes last.
                    ToolCallPart(tool_name='record', args={'marker': 'first', 'delay_ms': 40}, tool_call_id='r1'),
                    ToolCallPart(tool_name='record', args={'marker': 'second', 'delay_ms': 0}, tool_call_id='r2'),
                ]
            )

        agent = Agent(
            FunctionModel(model_fn),
            name='par',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')],
        )

        step = 'par__function_toolset__tools.call_tool:record'

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')

        assert first.output == 'done'
        # `#1` is the first-scheduled call ('first'), even though it completed last.
        assert ctx.stored[step] == {'result': 'first', 'kind': 'tool_return'}
        assert ctx.stored[f'{step}#2'] == {'result': 'second', 'kind': 'tool_return'}

        # On replay the slots must map the same way: results are served from the checkpoint, not
        # swapped, regardless of which call finishes first.
        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        assert second.output == 'done'
        assert replay.stored[step] == {'result': 'first', 'kind': 'tool_return'}
        assert replay.stored[f'{step}#2'] == {'result': 'second', 'kind': 'tool_return'}
        assert replay.invoked == []


class TestEventStreamHandler:
    async def test_handler_events_are_checkpointed(self) -> None:
        events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(event)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:
                yield 'done'

        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def greet() -> str:
            return 'hello'

        agent = Agent(
            FunctionModel(stream_function=stream_fn, model_name='fn'),
            name='ev',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(event_stream_handler=handler)],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'done'
        assert any(isinstance(event, FunctionToolCallEvent) for event in events)
        assert 'ev__event_stream_handler' in ctx.stored


class TestCancelSuspendedResponse:
    async def test_cancel_suspended_response_is_checkpointed(self) -> None:
        # Drive the cancel through a real run: the model first returns a `'suspended'` response, so
        # the agent re-issues it as a continuation; the continuation request then fails, and the
        # graph tears down the suspended job via `cancel_suspended_response`, which Absurd checkpoints.
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(m.parts and getattr(m, 'state', None) == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')

        assert len(cancelled) == 1
        assert cancelled[0].state == 'suspended'
        assert 'a__model.cancel_suspended_response' in ctx.stored


class TestCheckpointFormat:
    async def test_hand_written_model_request_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[AbsurdDurability()])

        # A hand-authored checkpoint payload pins the persistence format for a model request.
        payload: JsonValue = {
            'parts': [{'content': 'golden-response', 'part_kind': 'text'}],
            'model_name': 'fn',
            'kind': 'response',
        }
        ctx = FakeAsyncTaskContext(store={'gold__model.request': payload})
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'golden-response'
        assert counter['calls'] == 0

    async def test_hand_written_stream_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[AbsurdDurability()])

        # A hand-authored stream checkpoint pins the `{response, events}` payload shape.
        payload: JsonValue = {
            'response': {
                'parts': [{'content': 'golden-stream', 'part_kind': 'text'}],
                'model_name': 'fn',
                'kind': 'response',
            },
            'events': [
                {'index': 0, 'part': {'content': 'golden-stream', 'part_kind': 'text'}, 'event_kind': 'part_start'}
            ],
        }
        ctx = FakeAsyncTaskContext(store={'gold__model.request_stream': payload})
        with absurd_task_context(ctx):
            async with agent.run_stream('hi') as result:
                out = await result.get_output()

        assert out == 'golden-stream'
        assert counter['calls'] == 0

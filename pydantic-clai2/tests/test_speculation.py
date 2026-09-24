"""Speculative Code Mode: the switch, the tool fold, the counters, and the pinned row."""

import io
import json
import re
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, AgentRunResultEvent, ModelRetry, PartStartEvent, RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.code_mode import (
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallMissedEvent,
)
from pydantic_ai_harness.coder import Coder
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import StreamRenderer, theme
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import Settings, resolve_settings
from pydantic_clai2.customization import customization_guide
from pydantic_clai2.eager_timing import EagerExecutionCompletedEvent
from pydantic_clai2.image_input import ImageInput
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.sandbox_calls import SandboxCallFinishedEvent, SandboxCallOrder, SandboxCallStartedEvent
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.speculation import Speculation, SpeculationCounters
from pydantic_clai2.speculative_mode import (
    GUIDANCE,
    NATIVE_TOOLS,
    SPECULATIVE_TOOLS,
    SpeculativeExecution,
    speculative_capabilities,
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def plain(row: str) -> str:
    return Text.from_ansi(row).plain


def speculation(tmp_path: Path, console: Console | None = None) -> Speculation:
    store = SettingsStore(tmp_path / 'config.db')
    context = CommandContext(
        settings=store.load(),
        store=store,
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    return Speculation(context=context, console=console or Console(file=io.StringIO()))


class TestSwitch:
    def test_off_by_default_and_older_settings_load(self) -> None:
        assert Settings().speculative_code_mode is False
        assert resolve_settings({'display.thinking': False}).speculative_code_mode is False
        assert resolve_settings({'run.speculative_code_mode': True}).speculative_code_mode is True

    def test_toggle_persists_and_shows_row_only_while_on(self, tmp_path: Path) -> None:
        switch = speculation(tmp_path)
        assert switch.row() == ''
        assert switch.capabilities() == []

        assert switch.toggle() == 'Speculative execution on from the next turn. Ctrl+X Ctrl+S toggles it.'
        assert SettingsStore(tmp_path / 'config.db').overrides() == {'run.speculative_code_mode': True}
        assert plain(switch.row()).startswith('Speculative Execution  0 hits')
        assert [type(capability).__name__ for capability in switch.capabilities()] == [
            'CodeMode',
            'EagerTiming',
            'SpeculativeExecution',
            'ShowSandboxCalls',
        ]

        switch.counters.hits = 2
        assert switch.toggle().startswith('Speculative execution off')
        assert switch.row() == ''
        assert switch.capabilities() == []
        switch.toggle()
        assert plain(switch.row()).startswith('Speculative Execution  2 hits')

    def test_missing_sandbox_dependency_warns_and_runs_natively(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = io.StringIO()
        switch = speculation(tmp_path, Console(file=output, width=200))
        switch.toggle()
        monkeypatch.setitem(sys.modules, 'pydantic_clai2.speculative_mode', None)
        assert switch.capabilities() == []
        assert 'Speculative execution is unavailable' in output.getvalue()


class TestRow:
    def test_matches_code_puppy_layout(self) -> None:
        counters = SpeculationCounters(hits=29, misses=1, wasted=0, speculative_ms=520, eager_ms=6_549)
        assert plain(counters.row()) == (
            'Speculative Execution  29 hits \u00b7 1 miss \u00b7 0 wasted    saved \u2265 7.0s   spec 0.5s \u00b7 eager 6.5s'
        )

    def test_counts_light_up_only_when_nonzero(self) -> None:
        idle = SpeculationCounters().row()
        assert theme.sgr(theme.SUCCESS, bold=True) not in idle
        busy = SpeculationCounters(hits=1, wasted=1, eager_ms=100).row()
        assert f'{theme.sgr(theme.SUCCESS, bold=True)}1 hit' in busy
        assert f'{theme.sgr(theme.ERROR, bold=True)}1 wasted' in busy
        assert f'{theme.sgr(theme.MUTED)}0 misses' in busy
        assert f'{theme.sgr(theme.SUCCESS, bold=True)}saved \u2265 0.1s' in busy


def streamed(respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse]) -> FunctionModel:
    """Speculative runs stream, so replay each response as one delta per part."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        for index, part in enumerate(respond(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    return FunctionModel(stream_function=stream)


def fold_agent(model: FunctionModel, counters: SpeculationCounters) -> Agent[None, str]:
    agent: Agent[None, str] = Agent(model, capabilities=speculative_capabilities(counters))

    @agent.tool_plain
    def read_file(path: str) -> str:
        """Read."""
        return f'contents of {path}'

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        """Write."""
        return 'written'  # pragma: no cover -- only the tool surface is inspected.

    @agent.tool_plain
    def edit_file(path: str) -> str:
        """Edit."""
        return 'edited'  # pragma: no cover -- only the tool surface is inspected.

    return agent


class TestFold:
    async def test_shipped_tool_names_match_the_allowlists(self) -> None:
        seen: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(tool.name for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[Coder[None](repo_context=False), customization_guide()],
        )
        await agent.run('hi')
        assert {*SPECULATIVE_TOOLS, *NATIVE_TOOLS} <= set(seen)

    async def test_coder_shell_folds_into_run_code(self) -> None:
        seen: list[AgentInfo] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            streamed(respond),
            capabilities=[Coder(repo_context=False), *speculative_capabilities(SpeculationCounters())],
        )
        await agent.run('hi')
        [info] = seen
        tools = {tool.name: tool for tool in info.function_tools}
        assert sorted(tools) == ['edit_file', 'run_code', 'write_file']
        assert 'async def shell(' in (tools['run_code'].description or '')

    async def test_writes_stay_native_and_guidance_rides_along(self) -> None:
        seen: list[AgentInfo] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            return ModelResponse(parts=[TextPart('done')])

        await fold_agent(streamed(respond), SpeculationCounters()).run('hi')
        [info] = seen
        assert sorted(tool.name for tool in info.function_tools) == ['edit_file', 'run_code', 'write_file']
        assert GUIDANCE.strip() in (info.instructions or '')
        assert (info.model_settings or {}).get('anthropic_eager_input_streaming') is True

    async def test_counts_speculative_hits_and_eager_overlap(self) -> None:
        counters = SpeculationCounters()
        probe_started, stream_finished = anyio.Event(), anyio.Event()
        head = 'text = await read_file(path="a.py")\nwaited = await probe()\npad = 0\n'
        tail = 'done = 1\n'

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': head + tail})
            yield {1: DeltaToolCall(name='run_code')}
            split = args.index('done = 1')
            for offset in range(0, split, 8):
                yield {1: DeltaToolCall(json_args=args[offset : min(offset + 8, split)])}
                await anyio.sleep(0)
            await probe_started.wait()
            yield {1: DeltaToolCall(json_args=args[split:])}
            stream_finished.set()

        agent = fold_agent(FunctionModel(stream_function=stream), counters)

        @agent.tool_plain
        async def probe() -> str:
            """Hold the stream open until this call has started."""
            probe_started.set()
            await stream_finished.wait()
            return 'probed'

        async with agent.run_stream_events('go') as events:
            async for _ in events:
                pass
        assert (counters.hits, counters.misses, counters.wasted) == (1, 0, 0)
        assert counters.eager_ms > 0


class TestEagerTiming:
    async def test_unstreamed_and_restarted_snippets_report_nothing(self) -> None:
        counters = SpeculationCounters()
        calls = iter(
            [
                ToolCallPart('run_code', {'code': 'name = "a"\nx = await read_file(path=name)\nx'}),
                ToolCallPart('run_code', {'code': 'y = 1', 'restart': True}),
            ]
        )

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[next(calls, TextPart('done'))])

        await fold_agent(streamed(respond), counters).run('go')
        assert counters.eager_ms == 0
        assert counters.misses == 1


def claimed(*, ready: bool, elapsed_ms: float) -> SpeculativeCallClaimedEvent:
    return SpeculativeCallClaimedEvent(
        launch_id='l',
        nested_tool_call_id='c__1',
        wrapped_tool_name='read_file',
        ready_at_claim=ready,
        elapsed_ms=elapsed_ms,
    )


class TestSpeculativeExecution:
    async def test_partial_hits_count_without_time(self) -> None:
        counters = SpeculationCounters()
        report = SpeculativeExecution[None](counters)
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        for event in (
            claimed(ready=True, elapsed_ms=250),
            claimed(ready=False, elapsed_ms=900),
            SpeculativeCallMissedEvent(sandbox_function='grep', wrapped_tool_name='grep', nested_tool_call_id='c__2'),
            SpeculativeCallEvictedEvent(launch_id='l2', wrapped_tool_name='grep', state='ready'),
            EagerExecutionCompletedEvent(saved_ms=1_000),
            PartStartEvent(index=0, part=TextPart('ignored')),
        ):
            await report.on_event(ctx, event=event)
        assert counters == SpeculationCounters(hits=2, misses=1, wasted=1, speculative_ms=250, eager_ms=1_000)


SNIPPET = """\
text = await read_file(path="a.py")
name = "b.py"
other = await read_file(path=name)
if text == "nope":
    skipped = await read_file(path="never.py")
try:
    await flaky()
except Exception:
    pass
try:
    await broken()
except Exception:
    pass
"""


class TestSandboxCallDisplay:
    async def test_calls_inside_run_code_render_like_direct_calls(self) -> None:
        counters = SpeculationCounters()

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': SNIPPET})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 8):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 8])}
                await anyio.sleep(0)

        agent = fold_agent(FunctionModel(stream_function=stream), counters)

        @agent.tool_plain
        def flaky() -> str:
            """Ask for a retry."""
            raise ModelRetry('try again')

        @agent.tool_plain
        def broken() -> str:
            """Fail outright."""
            raise RuntimeError('kaboom')

        output = io.StringIO()
        renderer = StreamRenderer(Console(file=output, width=120), stop_loading=lambda: None)
        reports: list[SandboxCallStartedEvent | SandboxCallFinishedEvent] = []
        async with agent.run_stream_events('go') as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    continue
                if isinstance(event, (SandboxCallStartedEvent, SandboxCallFinishedEvent)):
                    reports.append(event)
                await renderer.on_stream_event(event)
        await renderer.finish()

        assert (counters.hits, counters.misses, counters.wasted) == (1, 1, 1)
        started = [event.call for event in reports if isinstance(event, SandboxCallStartedEvent)]
        assert [(call.tool_name, call.args) for call in started] == [
            ('read_file', {'path': 'a.py'}),
            ('read_file', {'path': 'b.py'}),
            ('flaky', {}),
            ('broken', {}),
        ]
        finished = [event.result for event in reports if isinstance(event, SandboxCallFinishedEvent)]
        assert [call.tool_call_id for call in started] == [result.tool_call_id for result in finished]
        assert all(re.fullmatch(r'.+__\d+', call.tool_call_id) for call in started)
        assert [type(result).__name__ for result in finished] == [
            'ToolReturnPart',
            'ToolReturnPart',
            'RetryPromptPart',
            'RetryPromptPart',
        ]
        headers = [line for line in output.getvalue().splitlines() if line.startswith('\u25cf')]
        assert headers == [
            '\u25cf run_code',
            "\u25cf read_file 'a.py' offset=0 limit=2000 lines",
            "\u25cf read_file 'b.py' offset=0 limit=2000 lines",
            '\u25cf flaky',
            '\u25cf broken',
        ]

    async def test_plugin_renderers_see_sandbox_calls(self) -> None:
        seen: list[str] = []

        def plugin(event: AgentStreamEvent) -> str | None:
            if isinstance(event, FunctionToolCallEvent):
                seen.append(event.part.tool_name)
                return 'drawn by plugin'
            return None

        output = io.StringIO()
        renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, renderers=[plugin])
        call = ToolCallPart(tool_name='read_file', args={'path': 'a.py'}, tool_call_id='parent__1')
        await renderer.on_stream_event(SandboxCallStartedEvent(tool_call_id='parent__1', call=call))
        assert seen == ['run_code', 'read_file']
        assert output.getvalue().count('drawn by plugin') == 2

    def test_each_run_code_header_renders_once(self) -> None:
        order = SandboxCallOrder()
        call = ToolCallPart(tool_name='read_file', tool_call_id='eager__1')
        assert len(order.tool_events(SandboxCallStartedEvent(call=call)) or []) == 2
        assert len(order.tool_events(SandboxCallStartedEvent(call=call)) or []) == 1
        late = FunctionToolCallEvent(ToolCallPart(tool_name='run_code', tool_call_id='eager'))
        assert order.tool_events(late) == []

        after_stream = FunctionToolCallEvent(ToolCallPart(tool_name='run_code', tool_call_id='parent'))
        assert order.tool_events(after_stream) is None
        call = ToolCallPart(tool_name='read_file', tool_call_id='parent__1')
        assert order.tool_events(SandboxCallStartedEvent(call=call)) == [FunctionToolCallEvent(call)]
        direct = FunctionToolCallEvent(ToolCallPart(tool_name='read_file', tool_call_id='direct'))
        assert order.tool_events(direct) is None
        assert order.tool_events(direct) is None
        assert order.tool_events(SandboxCallStartedEvent(call=replace(call, tool_call_id='parent__2'))) == [
            FunctionToolCallEvent(replace(call, tool_call_id='parent__2'))
        ]
        assert order.tool_events(PartStartEvent(index=0, part=TextPart('hi'))) is None


class TestChord:
    async def test_ctrl_x_ctrl_s_toggles_and_pins_the_row(self) -> None:
        toggles: list[str] = []

        def toggle() -> str:
            toggles.append('toggled')
            return 'Speculative execution on'

        console = Console(file=io.StringIO(), force_terminal=True, width=100, height=24)
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            live = LivePrompt(
                console=console,
                commands=Commands(),
                history=InMemoryHistory(),
                images=ImageInput(),
                interrupts=Interrupts(),
                toolbar=lambda: [('', 'ready')],
                clock=lambda: 0,
                chords={'ctrl-x ctrl-s': toggle},
                pinned=lambda: 'PINNED ROW',
            )
            live.feed('ctrl-x')
            live.feed('ctrl-s')
            frame = [plain(row) for row in live.frame()]
            assert toggles == ['toggled']
            assert frame[-2:] == ['PINNED ROW', 'Speculative execution on']

            live.feed('ctrl-x')
            live.feed('a')
            assert toggles == ['toggled']
            assert live.buffer.text == 'a'
            assert plain(live.frame()[-1]).startswith('ready')

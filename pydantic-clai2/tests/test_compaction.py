"""`/compact`, automatic compaction, and the status row's context warning."""

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from rich.console import Console
from test_app_edges import inputs

from pydantic_clai2 import chat
from pydantic_clai2._session import Session
from pydantic_clai2.compaction import Compactor, context_window
from pydantic_clai2.config import Settings
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.status import Status, StatusLine
from pydantic_clai2.theme import WARNING, sgr


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def make_compactor(
    agent: Agent[None, str], *, status: Status | None = None, model: str | None = 'test'
) -> tuple[Compactor[None, str], io.StringIO]:
    session = Session(agent, deps=None)
    session.model = model
    output = io.StringIO()
    compactor = Compactor(
        session=session,
        status=status or Status(),
        console=Console(file=output, width=200),
        fallback_model=lambda: agent.model,
    )
    return compactor, output


def summary_prompt(summary_run: Sequence[ModelMessage]) -> str:
    """The user turn the summariser received, as `capture_run_messages` recorded it."""
    request = summary_run[0]
    assert isinstance(request, ModelRequest)
    [prompt] = request.parts
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
    return prompt.content


def summary_text(session: Session[None, str]) -> str:
    """The one message the history holds after compaction."""
    [summary] = session.messages
    assert isinstance(summary, ModelRequest)
    [part] = summary.parts
    assert isinstance(part, SystemPromptPart)
    return part.content


async def test_compact_sends_history_and_focus_to_the_summariser() -> None:
    compactor, _ = make_compactor(Agent(TestModel()))
    await compactor.session.prompt('hello there')
    await compactor.session.prompt('and again')
    compactor.status.context_tokens = 500
    with capture_run_messages() as summary_run:
        notice = await compactor.command(['the', 'auth', 'work'])
    prompt = summary_prompt(summary_run)
    assert 'User: hello there\nAssistant: success (no tool calls)\nUser: and again' in prompt
    assert prompt.endswith('Give particular weight to: the auth work')
    assert notice.startswith('Compacted 4 messages into a summary; about ') and notice.endswith(' tokens saved.')
    assert summary_text(compactor.session) == 'Summary of previous conversation:\n\nsuccess (no tool calls)'
    assert compactor.status.context_tokens is not None and compactor.status.context_tokens < 500


async def test_compact_without_focus_or_history_or_model() -> None:
    compactor, _ = make_compactor(Agent(TestModel()))
    assert await compactor.command([]) == 'Nothing to compact: the conversation is empty.'
    await compactor.session.prompt('hello')
    with capture_run_messages() as summary_run:
        await compactor.command([])
    assert 'Give particular weight' not in summary_prompt(summary_run)

    compactor, _ = make_compactor(Agent(), model=None)
    compactor.session.replace_messages([ModelRequest.user_text_prompt('hello')])
    with pytest.raises(ValueError, match='Choose a model first'):
        await compactor.command([])


async def test_compact_falls_back_to_the_agent_model() -> None:
    compactor, _ = make_compactor(Agent(TestModel(custom_output_text='the gist')), model=None)
    compactor.session.replace_messages([ModelRequest.user_text_prompt('hello')])
    await compactor.command([])
    assert summary_text(compactor.session) == 'Summary of previous conversation:\n\nthe gist'


@pytest.mark.parametrize(
    ('tokens', 'window', 'compact_at', 'over'),
    [
        (80, 100, 0.8, False),
        (81, 100, 0.8, True),
        (1_000, 100, 0, False),
        (1_000, None, 0.8, False),
        (None, 100, 0.8, False),
    ],
)
def test_over_limit_boundaries(tokens: int | None, window: int | None, compact_at: float, over: bool) -> None:
    assert Status(context_tokens=tokens, context_window=window, compact_at=compact_at).over_limit is over


async def test_auto_compacts_only_past_the_threshold() -> None:
    status = Status(context_tokens=80, context_window=100, compact_at=0.8)
    compactor, output = make_compactor(Agent(TestModel()), status=status)
    await compactor.session.prompt('hello')
    before = compactor.session.messages
    await compactor.auto()
    assert compactor.session.messages == before and output.getvalue() == ''
    status.context_tokens = 81
    await compactor.auto()
    assert 'Context at 81% of 100 tokens; compacting before this turn.' in output.getvalue()
    assert 'Compacted 2 messages' in output.getvalue()
    assert len(compactor.session.messages) == 1 and not status.over_limit


async def test_auto_reports_a_failed_summary_and_keeps_the_history() -> None:
    def boom(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('summariser down')

    status = Status(context_tokens=99, context_window=100, compact_at=0.5)
    compactor, output = make_compactor(Agent(FunctionModel(boom)), status=status, model=None)
    compactor.session.replace_messages([ModelRequest.user_text_prompt('hello')])
    await compactor.auto()
    assert 'Compaction failed: RuntimeError: summariser down' in output.getvalue()
    assert len(compactor.session.messages) == 1


def test_prepare_notes_an_unknown_window_once_per_model() -> None:
    compactor, output = make_compactor(Agent(TestModel()))
    compactor.prepare('local:mystery', window=None, compact_at=0.8)
    compactor.prepare('local:mystery', window=None, compact_at=0.8)
    assert output.getvalue().count('Context window for local:mystery is unknown') == 1
    compactor.prepare('local:other', window=None, compact_at=0)
    compactor.prepare('openai:gpt-4o', window=128_000, compact_at=0.8)
    assert 'local:other' not in output.getvalue() and 'gpt-4o' not in output.getvalue()
    assert compactor.status.context_window == 128_000 and compactor.status.compact_at == 0.8
    compactor.prepare('local:other', window=None, compact_at=0.8)
    assert 'Context window for local:other is unknown' in output.getvalue()


def test_context_window_prefers_the_saved_override() -> None:
    assert context_window('openai:gpt-4o', override=None) == 128_000
    assert context_window('openai:gpt-4o', override=32_000) == 32_000
    assert context_window('test', override=None) is None


def test_toolbar_paints_the_figure_past_the_threshold() -> None:
    status = Status(model='m', context_tokens=90, context_window=100, compact_at=0.8)
    assert status.toolbar() == [('', 'm | context: '), (WARNING, '90'), ('', ' tokens | ~0 streamed tokens | ready')]
    status.compact_at = 0
    assert status.toolbar()[1] == ('', '90')
    assert ''.join(text for _, text in status.toolbar()) == status.text()


async def test_footer_paints_the_figure_past_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor')
    output = io.StringIO()
    status = Status(model='m', context_tokens=90, context_window=100, compact_at=0.8)
    async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), status):
        pass
    painted = output.getvalue()
    assert f'{sgr(WARNING)}9{sgr(WARNING)}0' in painted
    assert f'{sgr(WARNING)}m' not in painted


async def test_shell_compacts_on_command_and_automatically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['first', '/compact', 'second', 'third', '/exit'])
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings('test', {'context_window': 40})
    output = io.StringIO()
    console = Console(file=output, width=200)
    await chat(
        Agent(TestModel()), deps=None, console=console, settings=Settings(model='test', compact_at=0.5), store=store
    )
    text = output.getvalue()
    assert text.count('Compacted') == 2
    assert 'Context at' in text and 'compacting before this turn' in text
    assert 'is unknown' not in text


async def test_shell_notes_an_unknown_window_and_never_compacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs(monkeypatch, ['first', 'second', '/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        console=Console(file=output, width=200),
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    assert text.count('Context window for test is unknown, so automatic compaction is off.') == 1
    assert 'Compacted' not in text


async def test_shell_skips_the_window_lookup_without_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['hello', '/exit'])
    output = io.StringIO()
    await chat(
        Agent(),
        deps=None,
        console=Console(file=output, width=200),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    assert 'Choose a model first' in output.getvalue() and 'is unknown' not in output.getvalue()

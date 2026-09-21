"""The built-in `ask_user` plugin, driven headless."""

import asyncio
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from threading import Event

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.ask_user import (
    DECLINED,
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
    QuestionOption,
)
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from surface_terminal import SurfaceTerminal

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.ask_user_menu import MenuResult, QuestionMenu, TerminalAnswerer, activate, render_answer
from pydantic_clai2.menu_worker import menu_key
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.prompt_surface import PromptSurface


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


APPROACH = Question(
    header='Approach',
    question='How should we do it?',
    options=(QuestionOption(label='Refactor', description='Rewrite the module'), QuestionOption(label='Patch')),
)
TARGETS = Question(
    header='Targets',
    question='Which files?',
    options=(QuestionOption(label='api.py'), QuestionOption(label='db.py')),
    multi_select=True,
)


class ScreenLog:
    """A `FullScreen` that records when the terminal was taken and given back."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @asynccontextmanager
    async def __call__(self) -> AsyncGenerator[None]:
        self.events.append('taken')
        try:
            yield
        finally:
            self.events.append('released')


def test_single_select_and_number_shortcuts() -> None:
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.title == 'Approach'
    assert menu.choose('up') is None
    assert menu.choose('enter') == ('Patch',)
    assert menu.choose('1') == ('Refactor',)
    assert menu.choose('down') is None
    assert menu.choose('tab') is None
    assert menu.choose('enter') == ('Refactor',)


def test_multi_select_enter_toggles_and_done_submits() -> None:
    menu = QuestionMenu(question=TARGETS, position=2, total=3)
    assert menu.title == 'Targets (question 2 of 3)'
    assert menu.choose('3') is None  # Empty answers cannot be submitted.
    assert menu.choose('1') is None
    assert menu.choose('enter') is None
    assert menu.selected == set()
    assert menu.choose('2') is None
    assert menu.choose('1') is None
    assert menu.choose(' ') is None
    assert menu.choose('9') is None
    assert menu.choose('3') == ('api.py', 'db.py')


def test_inline_frame_shows_context_selection_and_navigation() -> None:
    menu = QuestionMenu(question=TARGETS, position=1, total=1)
    menu.choose('1')
    frame = '\n'.join(menu.frame(width=100, height=24))
    assert 'Targets' in frame
    assert '> 1. [x] api.py' in frame
    assert '2. [ ] db.py' in frame
    assert '3. Done' in frame
    assert 'Space' not in frame
    menu.choose('down')
    frame = menu.frame(width=24, height=10)
    assert any('db.py' in row for row in frame)
    assert len(frame) <= 5
    assert all(len(Text.from_ansi(row).plain) <= 24 for row in frame)


@pytest.mark.parametrize(
    'keys, expected',
    [
        (['', 'down', 'enter'], ('Patch',)),
        (['escape'], None),
        (['ctrl-c'], None),
    ],
)
def test_inline_terminal_preserves_transcript(keys: list[str], expected: tuple[str, ...] | None) -> None:
    output = io.StringIO()
    console = Console(file=output, width=80, height=24)
    console.print('Previous conversation stays here')
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.run(console=console, key_source=iter(keys).__next__) == expected
    rendered = output.getvalue()
    assert rendered.startswith('Previous conversation stays here')
    assert 'How should we do it?' in rendered
    assert 'Rewrite the module' in rendered
    assert '\x1b[?1049' not in rendered
    assert '\x1b[2J' not in rendered
    assert '\x1b[3J' not in rendered
    assert rendered.endswith('\x1b[?2026l')
    assert '\x1b[?25h' in rendered


async def test_answers_every_question_on_a_settled_screen() -> None:
    answers = iter([('Patch',), ('api.py', 'db.py'), ('Submit answers',)])
    screen = ScreenLog()
    answerer = TerminalAnswerer(full_screen=screen, runner=lambda menu: next(answers))
    response = await answerer(AskUserRequest(questions=(APPROACH, TARGETS)))
    assert response == AskUserResponse(
        answers=(
            AskUserAnswer(header='Approach', selected=('Patch',)),
            AskUserAnswer(header='Targets', selected=('api.py', 'db.py')),
        )
    )
    assert screen.events == ['taken', 'released']


async def test_parallel_requests_take_the_terminal_one_at_a_time() -> None:
    screen = ScreenLog()
    started = asyncio.Event()
    release = asyncio.Event()

    def slow_choice(menu: QuestionMenu) -> tuple[str, ...]:
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return ('Patch',)

    loop = asyncio.get_running_loop()
    answerer = TerminalAnswerer(full_screen=screen, runner=slow_choice)
    first = asyncio.create_task(answerer(AskUserRequest(questions=(APPROACH,))))
    await started.wait()
    second_started = asyncio.Event()

    async def another() -> AskUserResponse:
        second_started.set()
        return await answerer(AskUserRequest(questions=(APPROACH,)))

    second = asyncio.create_task(another())
    await second_started.wait()
    assert screen.events == ['taken']
    release.set()
    assert (await first).answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert (await second).answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert screen.events == ['taken', 'released', 'taken', 'released']


async def test_escape_declines_the_whole_request() -> None:
    answers = iter([('Patch',), None])
    screen = ScreenLog()
    answerer = TerminalAnswerer(full_screen=screen, runner=lambda menu: next(answers))
    response = await answerer(AskUserRequest(questions=(APPROACH, TARGETS)))
    assert response == AskUserResponse(cancelled=True)
    assert screen.events == ['taken', 'released']


def test_render_answer_lists_picks_or_the_decline() -> None:
    answered = AskUserAnsweredEvent(
        request_id='r1',
        response=AskUserResponse(
            answers=(
                AskUserAnswer(header='Approach', selected=('Patch',)),
                AskUserAnswer(header='Targets', selected=('api.py', 'db.py')),
            )
        ),
    )
    output = io.StringIO()
    console = Console(file=output, width=80)
    console.print(render_answer(answered))
    console.print(render_answer(AskUserAnsweredEvent(request_id='r2', response=AskUserResponse(cancelled=True))))
    text = output.getvalue()
    assert '● Approach: Patch\n● Targets: api.py, db.py\n' in text
    assert '● You declined to answer' in text


async def test_activate_registers_capability_and_renderer() -> None:
    host: PluginHost[None] = PluginHost(name='ask_user', console=Console(file=io.StringIO()), settings={})
    activate(host)
    (capability,) = host.capabilities
    assert isinstance(capability, AskUser)
    (renderer,) = host.renderers
    assert renderer(AskUserAnsweredEvent(request_id='r', response=AskUserResponse(cancelled=True))) is not None
    assert any(plugin.id == 'ask_user' for plugin in DEFAULT_PLUGINS)


async def test_declining_reaches_the_model_through_the_plugin() -> None:
    """The capability's declined result survives the round trip through the real tool call."""
    answers = iter([None])
    capabilities: list[AbstractCapability[None]] = [
        AskUser(answerer=TerminalAnswerer(full_screen=ScreenLog(), runner=lambda menu: next(answers)))
    ]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            questions = [APPROACH.model_dump(), TARGETS.model_dump()]
            return ModelResponse(parts=[ToolCallPart('ask_user_question', {'questions': questions})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=capabilities)
    result = await agent.run('go')
    assert result.output == 'done'
    returns = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)]
    assert len(returns) == 1 and returns[0].content == DECLINED


async def test_default_runner_reuses_editor_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.write('Earlier conversation\n')
    console = Console(file=surface, width=80, height=24)
    monkeypatch.setattr('sys.stdin', io.StringIO('2'))
    response = await TerminalAnswerer(full_screen=ScreenLog(), console=console)(AskUserRequest(questions=(APPROACH,)))
    assert response.answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert 'Earlier conversation' in output.getvalue()
    assert 'How should we do it?' not in '\n'.join(surface.transcript.frame(width=80, height=24).rows)
    assert 'How should we do it?' in output.getvalue()
    assert '\x1b[?1049' not in output.getvalue()


def test_terminal_cleanup_on_input_failure() -> None:
    output = io.StringIO()
    console = Console(file=output, width=80, height=24)

    def fail() -> str:
        raise OSError('input closed')

    with pytest.raises(OSError, match='input closed'):
        QuestionMenu(question=APPROACH, position=1, total=1).run(console=console, key_source=fail)
    assert '\x1b[?25h' in output.getvalue()
    assert '\x1b[r' in output.getvalue()


async def test_cancellation_joins_question_reader_before_releasing_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    stopping = asyncio.Event()
    stopped = Event()
    input_ready = Event()
    allow_exit = Event()
    loop = asyncio.get_running_loop()
    screen = ScreenLog()

    def read_key(*, timeout: float) -> str:
        loop.call_soon_threadsafe(started.set)
        assert input_ready.wait(10)
        return ''

    def run(menu: QuestionMenu) -> tuple[str, ...] | None:
        try:
            while menu_key() != 'ctrl-c':
                pass
        finally:
            loop.call_soon_threadsafe(stopping.set)
            assert allow_exit.wait(10)
            assert screen.events == ['taken']
            stopped.set()
        return None

    async def cancel(scope: anyio.CancelScope) -> None:
        try:
            await started.wait()
            scope.cancel()
            input_ready.set()
            await stopping.wait()
            assert screen.events == ['taken']
        finally:
            input_ready.set()
            allow_exit.set()

    monkeypatch.setattr('pydantic_clai2.menu_worker.read_key', read_key)
    answerer = TerminalAnswerer(full_screen=screen, runner=run)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            scope = anyio.CancelScope()
            tasks.start_soon(cancel, scope)
            with scope:
                await answerer(AskUserRequest(questions=(APPROACH,)))
            assert stopped.is_set()
            assert screen.events == ['taken', 'released']


def test_wrapped_descriptions_and_literal_markup() -> None:
    question = Question(
        header='[red]Literal',
        question='Literal question',
        options=(
            QuestionOption(label='First', description='This description needs several lines to remain readable.'),
            QuestionOption(label='[red]Second'),
        ),
    )
    menu = QuestionMenu(question=question, position=1, total=1)
    rows = menu.frame(width=30, height=30)
    text = '\n'.join(Text.from_ansi(row).plain for row in rows)
    assert 'remain readable.' in text
    assert '[red]Literal' in text
    assert '[red]Second' in text
    assert all(len(Text.from_ansi(row).plain) <= 30 for row in rows)
    menu.choose('down')
    assert any('> 2.' in row for row in menu.frame(width=12, height=6))


def test_option_description_preserves_explicit_line_breaks() -> None:
    question = Question(
        header='Steps',
        question='Which sequence?',
        options=(
            QuestionOption(label='First', description='First step\n\nSecond step'),
            QuestionOption(label='Other'),
        ),
    )
    menu = QuestionMenu(question=question, position=1, total=1)
    rows = [Text.from_ansi(row).plain for row in menu.frame(width=80, height=24)]
    assert rows[2:5] == ['> 1. First - First step', '  ', '  Second step']


def test_wide_characters_wrap_without_losing_choice_text() -> None:
    label = '中文選項需要完整顯示'
    description = '確認修復結果並保留所有文字 🎉🎉🎉'
    question = Question(
        header='Choices',
        question='Which one?',
        options=(QuestionOption(label=label, description=description), QuestionOption(label='Other')),
    )
    menu = QuestionMenu(question=question, position=1, total=1)
    rows = [Text.from_ansi(row).plain for row in menu.frame(width=16, height=80)]
    assert all(cell_len(row) <= 16 for row in rows)
    choices = ''.join(row[2:].strip() for row in rows[1:-1])
    assert label in choices
    assert description.replace(' ', '') in choices.replace(' ', '')


@pytest.mark.parametrize('key, expected', [('left', 'previous'), ('right', 'next')])
def test_question_navigation_keys(key: str, expected: str) -> None:
    menu = QuestionMenu(question=TARGETS, position=1, total=2)
    menu.choose('1')
    assert menu.choose(key) == expected
    assert menu.selected == {0}
    assert 'Left/Right' in menu.hint
    single = QuestionMenu(question=APPROACH, position=1, total=1)
    assert single.choose(key) is None


async def test_navigation_preserves_and_revises_drafts() -> None:
    steps = iter(
        [
            (1, ['left']),
            (1, ['2']),
            (2, ['1', 'left']),
            (1, ['1']),
            (2, ['2', '3']),
            (3, ['right']),
            (3, ['left']),
            (2, ['1', '3']),
            (3, ['2']),
            (1, ['right']),
            (2, ['right']),
            (3, ['1']),
        ]
    )

    def run(menu: QuestionMenu) -> MenuResult:
        position, keys = next(steps)
        assert menu.position == position
        if position == 3:
            assert menu.prompt is not None and 'Approach: Refactor' in menu.prompt
        result: MenuResult = None
        for key in keys:
            result = menu.choose(key)
        return result

    response = await TerminalAnswerer(full_screen=ScreenLog(), runner=run)(
        AskUserRequest(questions=(APPROACH, TARGETS))
    )
    assert response.answers == (
        AskUserAnswer(header='Approach', selected=('Refactor',)),
        AskUserAnswer(header='Targets', selected=('db.py',)),
    )


async def test_review_requires_every_question_and_can_cancel() -> None:
    steps = iter([(1, 'right'), (2, 'right'), (3, '1'), (1, '1'), (2, 'right'), (3, '1'), (2, 'escape')])

    def run(menu: QuestionMenu) -> MenuResult:
        position, key = next(steps)
        assert menu.position == position
        if position == 3:
            assert menu.prompt is not None and '(unanswered)' in menu.prompt
        return menu.run(console=Console(file=io.StringIO(), width=100), key_source=lambda: key)

    assert await TerminalAnswerer(full_screen=ScreenLog(), runner=run)(
        AskUserRequest(questions=(APPROACH, TARGETS))
    ) == AskUserResponse(cancelled=True)


def test_review_supports_large_answer_summaries() -> None:
    question = Question(
        header='x' * 25,
        question='Which?',
        options=(QuestionOption(label='a' * 50), QuestionOption(label='b' * 50)),
        multi_select=True,
    )
    menus = [QuestionMenu(question=question, position=i + 1, total=10, selected={0, 1}) for i in range(10)]
    review = TerminalAnswerer.review_menu(menus)
    assert review.prompt is not None and len(review.prompt) > 500
    assert review.choose('1') == ('Submit answers',)


@pytest.mark.parametrize('finish', ['submit', 'escape', 'failure'])
async def test_batch_repaints_in_place_without_polluting_history(monkeypatch: pytest.MonkeyPatch, finish: str) -> None:
    terminal = SurfaceTerminal(width=100, height=30)
    surface = PromptSurface(output=terminal, size=lambda: (100, 30))
    surface.write('Earlier conversation\n')
    console = Console(file=surface, width=100, height=30)
    steps = iter(
        [
            ('Approach (question 1 of 2)', 'left'),
            ('Approach (question 1 of 2)', '2'),
            ('Targets (question 2 of 2)', '1'),
            ('Targets (question 2 of 2)', 'left'),
            ('Approach (question 1 of 2)', 'right'),
            ('Targets (question 2 of 2)', '3'),
            ('Review answers', 'right'),
            ('Review answers', 'finish'),
        ]
    )
    frames: list[str] = []

    def read_key(*, timeout: float) -> str:
        title, key = next(steps)
        visible = '\n'.join(terminal.lines())
        frames.append(visible)
        assert title in visible
        assert 'Earlier conversation' in '\n'.join(terminal.history + terminal.lines())
        assert 'How should we do it?' not in '\n'.join(terminal.history)
        assert 'Which files?' not in '\n'.join(terminal.history)
        assert '\x1b[?25h' not in terminal.getvalue()
        if title.startswith('Approach'):
            assert 'How should we do it?' in visible and 'Which files?' not in visible
        elif title.startswith('Targets'):
            assert 'Which files?' in visible and 'How should we do it?' not in visible
        else:
            assert 'Approach: Patch' in visible and 'Targets: api.py' in visible
        if key != 'finish':
            return key
        if finish == 'failure':
            raise OSError('reader failed')
        return '1' if finish == 'submit' else 'escape'

    monkeypatch.setattr('pydantic_clai2.menu_worker.read_key', read_key)
    answerer = TerminalAnswerer(full_screen=ScreenLog(), console=console)
    request = AskUserRequest(questions=(APPROACH, TARGETS))
    if finish == 'failure':
        with pytest.raises(OSError, match='reader failed'):
            await answerer(request)
    else:
        response = await answerer(request)
        assert response.cancelled == (finish == 'escape')
        if finish == 'submit':
            assert response.answers == (
                AskUserAnswer(header='Approach', selected=('Patch',)),
                AskUserAnswer(header='Targets', selected=('api.py',)),
            )
    assert frames[0] == frames[1]
    assert frames[-1] == frames[-2]
    assert terminal.getvalue().count('\x1b[?25h') == 1
    assert terminal.getvalue().count('\x1b[?2004h') == 1
    transcript = '\n'.join(surface.transcript.frame(width=100, height=30).rows)
    assert transcript == 'Earlier conversation\n'
    assert not any('question ' in line for line in terminal.lines())


def test_long_question_still_leaves_room_for_a_choice() -> None:
    menu = QuestionMenu(question=APPROACH, position=1, total=1, prompt='Long question ' * 100)
    rows = menu.frame(width=20, height=10)
    assert len(rows) <= 5
    assert any('> 1.' in row for row in rows)
    assert all(cell_len(Text.from_ansi(row).plain) <= 20 for row in rows)

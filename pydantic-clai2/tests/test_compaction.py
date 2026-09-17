"""The built-in `compaction` plugin, driven through a `PluginHost` and one `chat()` run."""

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from rich.console import Console
from test_app_edges import inputs

from pydantic_clai2 import Session, chat
from pydantic_clai2.compaction import activate
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.plugins import PluginHost, Transcript
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def make_host(conversation: Transcript | Session[None, str] | None = None, **settings: JsonValue) -> PluginHost[None]:
    host = PluginHost[None](
        name='compaction', console=Console(file=io.StringIO()), settings=dict(settings), conversation=conversation
    )
    activate(host)
    return host


def summary_prompt(summary_run: Sequence[ModelMessage]) -> str:
    """The user turn the summariser received, as `capture_run_messages` recorded it."""
    request = summary_run[0]
    assert isinstance(request, ModelRequest)
    [prompt] = request.parts
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
    return prompt.content


async def test_compact_sends_the_history_and_focus_to_the_summariser() -> None:
    transcript = Transcript(
        messages=[ModelRequest.user_text_prompt('hello there'), ModelRequest.user_text_prompt('and again')],
        model=TestModel(custom_output_text='the gist'),
    )
    host = make_host(transcript, keep_messages=0)
    host.status.context_alert = True
    with capture_run_messages() as summary_run:
        notice = await host.commands.execute_async('/compact the auth work')
    prompt = summary_prompt(summary_run)
    assert 'User: hello there\nUser: and again' in prompt and prompt.endswith(
        'Give particular weight to: the auth work'
    )
    assert notice.startswith('Compacted 2 messages down to 2; about ') and notice.endswith(' tokens saved.')
    summary, first = transcript.messages
    assert isinstance(summary, ModelRequest) and isinstance(first, ModelRequest)
    [summary_part], [first_part] = summary.parts, first.parts
    assert isinstance(summary_part, SystemPromptPart) and isinstance(first_part, UserPromptPart)
    assert summary_part.content == 'Summary of previous conversation:\n\nthe gist'
    assert first_part.content == 'hello there'
    assert not host.status.context_alert


async def test_compact_says_when_there_is_nothing_to_do() -> None:
    assert await make_host().commands.execute_async('/compact') == 'Nothing to compact: the conversation is empty.'
    short = Transcript(messages=[ModelRequest.user_text_prompt('hi')], model='test')
    host = make_host(short)
    with capture_run_messages() as summary_run:
        notice = await host.commands.execute_async('/compact')
    assert notice == 'Nothing to compact: the last 20 messages are always kept.' and not summary_run
    with pytest.raises(ValueError, match='Choose a model first'):
        await make_host(
            Transcript(messages=[ModelRequest.user_text_prompt('hi')]), keep_messages=0
        ).commands.execute_async('/compact')


async def test_settings_are_validated_on_activation() -> None:
    with pytest.raises(ValidationError):
        make_host(max_fraction=0)
    with pytest.raises(ValidationError):
        make_host(compact_at=0.5)


async def test_gauge_paints_the_status_row_when_still_over_the_threshold() -> None:
    session = Session(Agent(TestModel()), deps=None)
    host = make_host(session, context_window=10)
    session.plugins = host.capabilities
    await session.prompt('hello there, this is longer than ten tokens')
    assert host.status.context_alert
    roomy = make_host(session, context_window=100_000)
    session.plugins = roomy.capabilities
    await session.prompt('again')
    assert not roomy.status.context_alert


async def test_shell_loads_the_plugin_and_compacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['first', '/compact', '/plugins list', '/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        console=Console(file=output, width=200),
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(
            PluginSettings(id='compaction', factory='pydantic_clai2.compaction', settings={'keep_messages': 0}),
        ),
    )
    text = output.getvalue()
    assert 'Compacted 2 messages down to 2' in text
    assert 'compaction' in text and 'pydantic_clai2.compaction (built-in)' in text

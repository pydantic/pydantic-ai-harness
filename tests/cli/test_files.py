"""The bridge renders `file_system.*` events: diffs before a change, match counts after a search."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import CapabilityEvent, ModelMessage, RetryPromptPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from termflow.ansi import DIM_OFF, DIM_ON, RESET, fg_color, visible  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import CliBridge, CliDeps, DeclineAll, Verdict, allow_all
from pydantic_ai_harness.filesystem import FilesSearchedEvent, FileSystem

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _tool_results(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, (RetryPromptPart, ToolReturnPart))
    ]


def _tool_model(tool_name: str, **args: object) -> FunctionModel:
    """Call one tool, then answer `done`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if not _tool_results(messages):
            yield {0: DeltaToolCall(name=tool_name, json_args=json.dumps(args), tool_call_id='call_0')}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


def _agent(tmp_path: Path, bridge: CliBridge[CliDeps], model: FunctionModel) -> Agent[CliDeps, str]:
    files = FileSystem[CliDeps](root_dir=tmp_path)
    return Agent(model, deps_type=CliDeps, capabilities=[files, bridge])


@dataclass
class _Recording:
    """Allow everything, remembering what the bridge would have shown the user."""

    descriptions: list[str] = field(default_factory=list[str])

    async def __call__(self, event: CapabilityEvent, *, description: str) -> Verdict:
        self.descriptions.append(description)
        return Verdict(allowed=True)


class TestFileChangeRendering:
    async def test_edit_shows_the_colored_diff_before_the_result(self, tmp_path: Path) -> None:
        (tmp_path / 'notes.txt').write_text('keep\nold\n')
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=allow_all)
        agent = _agent(tmp_path, bridge, _tool_model('edit_file', path='notes.txt', old_text='old', new_text='new'))

        await agent.run('go', deps=CliDeps(approver=allow_all))

        new_hash = hashlib.sha256(b'keep\nnew\n').hexdigest()[:12]
        assert visible(buffer.getvalue()).splitlines() == [
            '> edit_file {"path": "notes.txt", "old_text": "old", "new_text": "new"}',
            '--- a/notes.txt',
            '+++ b/notes.txt',
            '@@ -1,2 +1,2 @@',
            ' keep',
            '-old',
            '+new',
            f'< edit_file Edited notes.txt. [hash:{new_hash}]',
            'done',
        ]
        raw = buffer.getvalue()
        assert f'{fg_color("red")}-old{RESET}' in raw
        assert f'{fg_color("green")}+new{RESET}' in raw
        assert f'{fg_color("cyan")}@@ -1,2 +1,2 @@{RESET}' in raw
        assert '\n keep\n' in raw
        assert (tmp_path / 'notes.txt').read_text() == 'keep\nnew\n'

    async def test_declined_write_is_cancelled_with_the_reason(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=DeclineAll(reason='not today'))
        agent = _agent(tmp_path, bridge, _tool_model('write_file', path='new.txt', content='hi\n'))

        result = await agent.run('go', deps=CliDeps(approver=allow_all))

        assert _tool_results(result.all_messages()) == ["['new.txt' was not written: not today]"]
        assert not (tmp_path / 'new.txt').exists()
        assert visible(buffer.getvalue()).splitlines()[1:] == [
            '--- a/new.txt',
            '+++ b/new.txt',
            '@@ -0,0 +1 @@',
            '+hi',
            "< write_file ['new.txt' was not written: not today]",
            'done',
        ]

    async def test_create_directory_asks_without_a_diff(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        approver = _Recording()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=approver)
        agent = _agent(tmp_path, bridge, _tool_model('create_directory', path='made'))

        await agent.run('go', deps=CliDeps(approver=allow_all))

        assert approver.descriptions == ['create directory made']
        assert visible(buffer.getvalue()).splitlines() == [
            '> create_directory {"path": "made"}',
            '< create_directory Created directory: made',
            'done',
        ]

    async def test_a_truncated_diff_says_so(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=allow_all)
        content = ''.join(f'line {i}\n' for i in range(2000))
        agent = _agent(tmp_path, bridge, _tool_model('write_file', path='big.txt', content=content))

        await agent.run('go', deps=CliDeps(approver=allow_all))

        assert f'{DIM_ON}(diff truncated){DIM_OFF}' in buffer.getvalue()


class TestSearchRendering:
    @pytest.mark.parametrize(
        ('tool_name', 'args', 'summary'),
        [
            ('search_files', {'pattern': 'x', 'path': 'src'}, "2 matches for 'x' in src"),
            ('find_files', {'pattern': '*.py', 'path': 'src'}, "2 matches for '*.py' in src"),
        ],
    )
    async def test_search_result_line_is_a_match_count(
        self, tmp_path: Path, tool_name: str, args: dict[str, str], summary: str
    ) -> None:
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.py').write_text('x = 1\n')
        (tmp_path / 'src' / 'b.py').write_text('x = 2\n')
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=allow_all)
        agent = _agent(tmp_path, bridge, _tool_model(tool_name, **args))

        await agent.run('go', deps=CliDeps(approver=allow_all))

        lines = visible(buffer.getvalue()).splitlines()
        assert lines[1:] == [summary, 'done']
        assert not any(line.startswith(f'< {tool_name}') for line in lines)

    async def test_a_search_reported_by_a_hook_renders_without_a_tool_call(self) -> None:
        buffer = io.StringIO()
        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=type(None),
            capabilities=[_SearchedInHook(), CliBridge(output=buffer, width=80)],
        )

        await agent.run('go')

        assert visible(buffer.getvalue()).splitlines() == ["0 matches for 'needle' in .", 'done']

    async def test_one_match_and_a_capped_search_read_naturally(self, tmp_path: Path) -> None:
        for name in ('a', 'b', 'c'):
            (tmp_path / f'{name}.py').write_text('x = 1\n')
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=allow_all)
        files = FileSystem[CliDeps](root_dir=tmp_path, max_search_results=1)
        agent = Agent(_tool_model('search_files', pattern='x'), deps_type=CliDeps, capabilities=[files, bridge])

        await agent.run('go', deps=CliDeps(approver=allow_all))

        assert visible(buffer.getvalue()).splitlines()[1] == "1 match for 'x' in ., truncated"


class _SearchedInHook(AbstractCapability[None]):
    """A host-side emitter: a search event with no tool call behind it."""

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        await ctx.emit(
            FilesSearchedEvent(
                path='.', root_dir='/work', pattern='needle', search='grep', match_count=0, truncated=False
            )
        )
        return request_context

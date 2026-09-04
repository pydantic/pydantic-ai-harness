"""Tests for the RepoContext capability."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import LocalSandbox, Sandbox, UnavailableSandbox
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.repo_context import (
    AgentContextInventory,
    ContextFile,
    RepoContext,
    RepoContextToolset,
)
from pydantic_ai_harness.repo_context._inventory import scan_assets
from pydantic_ai_harness.repo_context._loader import (
    discover_instruction_files,
    find_dir_context_file,
    render_context_files,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[Sandbox]:
    async with LocalSandbox(root=tmp_path) as backend:
        yield Sandbox.wrap(backend)


def _run_context(sandbox: Sandbox) -> RunContext[object]:
    return RunContext[object](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0, sandbox=sandbox
    )


def _call(tool_name: str, **args: str) -> tuple[ToolCallPart, ToolDefinition, dict[str, str]]:
    return ToolCallPart(tool_name=tool_name, args=args), ToolDefinition(name=tool_name), args


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def _render_capability_instructions(capability: RepoContext[object], ctx: RunContext[object]) -> str | None:
    instructions = capability.get_instructions()
    assert callable(instructions)
    rendered: object = instructions(ctx)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert isinstance(rendered, str) or rendered is None
    return rendered


class TestDiscoverInstructionFiles:
    async def test_walk_up_ancestor_first(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'root')
        workspace = tmp_path / 'a' / 'b'
        _write(workspace / 'CLAUDE.md', 'leaf')
        files = await discover_instruction_files(sandbox, workspace, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['root', 'leaf']

    async def test_home_none_only_workspace(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'root')
        workspace = tmp_path / 'a'
        _write(workspace / 'CLAUDE.md', 'leaf')
        files = await discover_instruction_files(sandbox, workspace, None, ('CLAUDE.md',))
        assert [f.content for f in files] == ['leaf']

    async def test_home_equals_workspace(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'only')
        files = await discover_instruction_files(sandbox, tmp_path, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['only']

    async def test_home_not_ancestor_falls_back_to_workspace(self, tmp_path: Path, sandbox: Sandbox) -> None:
        workspace = tmp_path / 'a'
        _write(workspace / 'CLAUDE.md', 'leaf')
        unrelated = tmp_path / 'other'
        unrelated.mkdir()
        files = await discover_instruction_files(sandbox, workspace, unrelated, ('CLAUDE.md',))
        assert [f.content for f in files] == ['leaf']

    async def test_both_filenames_within_dir_order(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'claude')
        _write(tmp_path / 'AGENTS.md', 'agents')
        files = await discover_instruction_files(sandbox, tmp_path, None, ('CLAUDE.md', 'AGENTS.md'))
        assert [f.content for f in files] == ['claude', 'agents']

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    async def test_symlink_deduped_by_content_hash(self, tmp_path: Path, sandbox: Sandbox) -> None:
        # Symlink-realpath dedup is dropped with the sandbox migration (isolation is
        # the sandbox's job); the content-hash dedup still catches shared bytes.
        _write(tmp_path / 'CLAUDE.md', 'shared')
        (tmp_path / 'AGENTS.md').symlink_to(tmp_path / 'CLAUDE.md')
        files = await discover_instruction_files(sandbox, tmp_path, None, ('CLAUDE.md', 'AGENTS.md'))
        assert len(files) == 1

    async def test_identical_content_deduped_by_hash(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'same')
        workspace = tmp_path / 'a'
        _write(workspace / 'CLAUDE.md', 'same')
        files = await discover_instruction_files(sandbox, workspace, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['same']

    async def test_missing_files_skipped(self, tmp_path: Path, sandbox: Sandbox) -> None:
        files = await discover_instruction_files(sandbox, tmp_path, None, ('CLAUDE.md',))
        assert files == []

    @pytest.mark.parametrize('error', [FileNotFoundError(), NotADirectoryError()])
    async def test_unreadable_model_path_does_not_block_capability_setup(self, tmp_path: Path, error: OSError) -> None:
        sandbox = MagicMock(spec=Sandbox)
        sandbox.resolve = AsyncMock(return_value=tmp_path.as_posix())
        sandbox.stat = AsyncMock(side_effect=error)
        cap = RepoContext[object](workspace_dir=tmp_path, expose_inventory_tool=False)
        ctx = _run_context(sandbox=sandbox)

        await cap.before_run(ctx)

        assert cap.get_instructions() is not None

    async def test_duplicate_filename_is_read_once(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'once')

        files = await discover_instruction_files(sandbox, tmp_path, None, ('CLAUDE.md', 'CLAUDE.md'))

        assert [file.content for file in files] == ['once']

    async def test_non_utf8_file_does_not_crash(self, tmp_path: Path, sandbox: Sandbox) -> None:
        (tmp_path / 'CLAUDE.md').write_bytes(b'caf\xe9 instructions')
        files = await discover_instruction_files(sandbox, tmp_path, None, ('CLAUDE.md',))
        assert len(files) == 1
        assert 'instructions' in files[0].content


class TestFindDirContextFile:
    async def test_first_existing_wins(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'AGENTS.md', 'agents')
        found = await find_dir_context_file(sandbox, tmp_path, ('CLAUDE.md', 'AGENTS.md'))
        assert found is not None
        assert found.content == 'agents'

    async def test_none_when_absent(self, tmp_path: Path, sandbox: Sandbox) -> None:
        assert await find_dir_context_file(sandbox, tmp_path, ('CLAUDE.md',)) is None

    async def test_non_utf8_file_does_not_crash(self, tmp_path: Path, sandbox: Sandbox) -> None:
        (tmp_path / 'CLAUDE.md').write_bytes(b'caf\xe9 instructions')
        found = await find_dir_context_file(sandbox, tmp_path, ('CLAUDE.md',))
        assert found is not None
        assert 'instructions' in found.content


class TestRender:
    def test_label_outside_workspace_falls_back_to_posix(self, tmp_path: Path) -> None:
        outside = _write(tmp_path / 'outer' / 'CLAUDE.md', 'x')
        cf = ContextFile(directory=outside.parent, path=outside, content='x')
        rendered = render_context_files([cf], relative_to=tmp_path / 'inner')
        assert outside.as_posix() in rendered

    def test_trailing_newline_does_not_open_a_gap_before_the_closing_tag(self, tmp_path: Path) -> None:
        # The terminator goes; the two spaces before it stay, being a hard line break in Markdown.
        path = _write(tmp_path / 'CLAUDE.md', 'be nice  \n')
        cf = ContextFile(directory=path.parent, path=path, content=path.read_text(encoding='utf-8'))
        assert (
            render_context_files([cf], relative_to=tmp_path)
            == '<context-file path="CLAUDE.md">\nbe nice  \n</context-file>'
        )


class TestInstructions:
    async def test_includes_files_and_inventory_hint(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'be nice')
        cap = RepoContext[object](workspace_dir=tmp_path)
        ctx = _run_context(sandbox=sandbox)
        await cap.before_run(ctx)
        instructions = _render_capability_instructions(cap, ctx)
        assert isinstance(instructions, str)
        assert 'be nice' in instructions
        assert 'inventory_agent_context' in instructions

    def test_none_when_all_disabled(self, tmp_path: Path) -> None:
        cap = RepoContext[object](workspace_dir=tmp_path, autoload_instructions=False, expose_inventory_tool=False)
        assert cap.get_instructions() is None

    async def test_autoload_off_keeps_inventory_hint(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'ignored')
        cap = RepoContext[object](workspace_dir=tmp_path, autoload_instructions=False)
        await cap.before_run(_run_context(sandbox=sandbox))
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'ignored' not in instructions
        assert 'inventory_agent_context' in instructions

    async def test_autoload_off_does_not_require_sandbox_file_access(self, tmp_path: Path) -> None:
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[RepoContext[object](workspace_dir=tmp_path, autoload_instructions=False)],
        )

        result = await agent.run('go', sandbox=UnavailableSandbox('sandbox file access is disabled'))

        assert result.output is not None

    async def test_no_files_no_inventory_is_none(self, tmp_path: Path, sandbox: Sandbox) -> None:
        cap = RepoContext[object](workspace_dir=tmp_path, expose_inventory_tool=False)
        ctx = _run_context(sandbox=sandbox)
        await cap.before_run(ctx)
        assert _render_capability_instructions(cap, ctx) is None

    async def test_files_cached_across_calls(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'CLAUDE.md', 'first')
        cap = RepoContext[object](workspace_dir=tmp_path)
        ctx = _run_context(sandbox=sandbox)
        await cap.before_run(ctx)
        first = _render_capability_instructions(cap, ctx)
        assert first is not None and 'first' in first
        _write(tmp_path / 'CLAUDE.md', 'second')
        second = _render_capability_instructions(cap, ctx)
        # Read-once: `before_run` loaded the file, so subsequent edits are not picked up.
        assert second is not None and 'second' not in second

    async def test_autoload_without_sandbox_raises(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'host instructions')
        agent = Agent(TestModel(), capabilities=[RepoContext[object](workspace_dir=tmp_path)])

        with pytest.raises(UserError, match='No sandbox is attached'):
            await agent.run('go')


class TestToolset:
    def test_get_toolset_none_when_disabled(self, tmp_path: Path) -> None:
        assert RepoContext[object](workspace_dir=tmp_path, expose_inventory_tool=False).get_toolset() is None

    def test_get_toolset_present(self, tmp_path: Path) -> None:
        assert isinstance(RepoContext[object](workspace_dir=tmp_path).get_toolset(), RepoContextToolset)

    async def test_inventory_tool_runs_through_agent(self, tmp_path: Path) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 'skill')
        agent = Agent(
            TestModel(call_tools=['inventory_agent_context']),
            capabilities=[RepoContext[object](workspace_dir=tmp_path)],
        )
        async with LocalSandbox(root=tmp_path) as backend:
            result = await agent.run('go', sandbox=backend)
        assert 'inventory_agent_context' in result.output


class TestScanAssets:
    async def test_full_shape(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 's')
        _write(tmp_path / '.claude' / 'agents' / 'bar.md', 'a')
        _write(tmp_path / '.claude' / 'settings.json', '{}')
        inv = await scan_assets(sandbox, tmp_path, ('.claude', '.agents', '.codex', '.grok'))
        by_root = {r.root: r for r in inv.roots}
        claude = by_root['.claude']
        assert claude.exists
        assert claude.skills == ['.claude/skills/foo/SKILL.md']
        assert claude.agents == ['.claude/agents/bar.md']
        assert claude.settings == '.claude/settings.json'
        assert by_root['.agents'].exists is False
        assert by_root['.codex'].notes is not None
        assert by_root['.grok'].notes is not None

    async def test_existing_root_without_settings(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 's')
        inv = await scan_assets(sandbox, tmp_path, ('.claude',))
        assert inv.roots[0].settings is None
        assert inv.roots[0].notes is None

    async def test_root_without_skills_directory(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / '.claude' / 'agents' / 'helper.md', 'agent')
        inv = await scan_assets(sandbox, tmp_path, ('.claude',))
        assert inv.roots[0].skills == []
        assert inv.roots[0].agents == ['.claude/agents/helper.md']

    @pytest.mark.parametrize('root', ('/outside', '../outside'))
    async def test_asset_roots_must_be_relative_to_workspace(self, tmp_path: Path, sandbox: Sandbox, root: str) -> None:
        with pytest.raises(ValueError, match='relative to the workspace'):
            await scan_assets(sandbox, tmp_path, (root,))

    async def test_skill_walk_has_a_depth_bound(self, tmp_path: Path, sandbox: Sandbox) -> None:
        near = tmp_path / '.claude' / 'skills'
        for index in range(8):
            near /= f'level-{index}'
        _write(near / 'SKILL.md', 'near')
        _write(near / 'level-8' / 'SKILL.md', 'deep')

        inventory = await scan_assets(sandbox, tmp_path, ('.claude',))

        assert inventory.roots[0].skills == [f'.claude/skills/{"/".join(f"level-{i}" for i in range(8))}/SKILL.md']

    async def test_file_at_asset_root_is_not_a_directory(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / '.claude', 'not a directory')

        inv = await scan_assets(sandbox, tmp_path, ('.claude',))

        assert inv.roots[0].exists is False

    async def test_unrecognized_inventory_entries_are_ignored(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / '.claude' / 'README.md', 'not an agent')
        _write(tmp_path / '.claude' / 'skills' / 'README.md', 'not a skill')
        _write(tmp_path / '.claude' / 'settings.json', '{}')

        inv = await scan_assets(sandbox, tmp_path, ('.claude',))

        assert inv.roots[0].settings == '.claude/settings.json'
        assert inv.roots[0].skills == []

    async def test_returns_model(self, tmp_path: Path, sandbox: Sandbox) -> None:
        assert isinstance(await scan_assets(sandbox, tmp_path, ()), AgentContextInventory)

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    async def test_symlinked_asset_uses_confined_display_path(self, tmp_path: Path, sandbox: Sandbox) -> None:
        workspace = tmp_path / 'ws'
        outside = _write(tmp_path / 'outside' / 'foo' / 'SKILL.md', 's')
        link = workspace / '.claude' / 'skills' / 'foo' / 'SKILL.md'
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        inv = await scan_assets(sandbox, workspace, ('.claude',))
        claude = inv.roots[0]
        assert claude.exists
        assert len(claude.skills) == 1
        assert claude.skills == ['.claude/skills/foo/SKILL.md']


class TestNestedTraversal:
    async def test_off_by_default_returns_result(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](workspace_dir=tmp_path)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='listing'
        )
        assert out == 'listing'

    async def test_pointer_appended_on_first_traversal(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='listing'
        )
        assert out.startswith('listing')
        assert 'sub/CLAUDE.md' in out
        assert 'nested' not in out

    async def test_second_traversal_no_reappend(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        ctx = _run_context(sandbox=sandbox)
        first = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result='one')
        second = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result='two')
        assert 'CLAUDE.md' in first
        assert second == 'two'

    async def test_concurrent_traversal_injects_directory_once(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        ctx = _run_context(sandbox=sandbox)

        outputs = await asyncio.gather(
            cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result='one'),
            cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result='two'),
        )

        assert sum('CLAUDE.md' in output for output in outputs) == 1

    async def test_tool_name_not_matched(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('write_file', path='sub')
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='r'
        )
        assert out == 'r'

    async def test_non_str_path_arg_ignored(self, tmp_path: Path, sandbox: Sandbox) -> None:
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call = ToolCallPart(tool_name='list_directory', args={'path': 123})
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox),
            call=call,
            tool_def=ToolDefinition(name='list_directory'),
            args={'path': 123},
            result='r',
        )
        assert out == 'r'

    async def test_dir_without_context_file_untouched(self, tmp_path: Path, sandbox: Sandbox) -> None:
        (tmp_path / 'sub').mkdir()
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='r'
        )
        assert out == 'r'

    async def test_read_file_uses_parent_dir(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        target = _write(tmp_path / 'sub' / 'code.py', 'x = 1')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('read_file', path=str(target))
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='file body'
        )
        assert 'CLAUDE.md' in out

    @pytest.mark.parametrize('error', [FileNotFoundError(), NotADirectoryError()])
    async def test_unreadable_tool_path_leaves_result_unchanged(self, tmp_path: Path, error: OSError) -> None:
        sandbox = MagicMock(spec=Sandbox)
        sandbox.resolve = AsyncMock(side_effect=(tmp_path.as_posix(), (tmp_path / 'gone').as_posix()))
        sandbox.stat = AsyncMock(side_effect=error)
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='gone')

        result = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='listing'
        )

        assert result == 'listing'

    async def test_contents_mode_inlines_body(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'NESTED BODY')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True, nested_inject='contents')
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='r'
        )
        assert 'NESTED BODY' in out

    async def test_label_falls_back_when_dir_outside_workspace(self, tmp_path: Path, sandbox: Sandbox) -> None:
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        outside = _write(tmp_path / 'outside' / 'CLAUDE.md', 'nested').parent
        cap = RepoContext[object](workspace_dir=workspace, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path=str(outside))
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='r'
        )
        assert outside.as_posix() in out

    async def test_non_str_result_returned_unchanged(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](
            workspace_dir=tmp_path, nested_traversal=True, traversal_tool_names=frozenset({'list_dir', 'read_file'})
        )
        call, tool_def, args = _call('list_dir', path='sub')
        listing = [{'name': 'CLAUDE.md'}, {'name': 'code.py'}]
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result=listing
        )
        assert out is listing

    async def test_string_result_still_gets_note(self, tmp_path: Path, sandbox: Sandbox) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[object](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(
            _run_context(sandbox=sandbox), call=call, tool_def=tool_def, args=args, result='listing'
        )
        assert out.startswith('listing')
        assert 'CLAUDE.md' in out


class TestForRunAndMisc:
    async def test_agent_reuse_isolates_instruction_state_between_sandboxes(self, tmp_path: Path) -> None:
        first_root = tmp_path / 'first'
        second_root = tmp_path / 'second'
        _write(first_root / 'CLAUDE.md', 'first sandbox instructions')
        _write(second_root / 'CLAUDE.md', 'second sandbox instructions')
        captured: list[list[ModelMessage]] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            captured.append(messages)
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(model),
            capabilities=[RepoContext[object](workspace_dir=Path('.'), expose_inventory_tool=False)],
        )

        instructions: list[str] = []
        for root in (first_root, second_root):
            async with LocalSandbox(root=root) as backend:
                await agent.run('go', sandbox=backend)
            first_request = captured[-1][0]
            assert isinstance(first_request, ModelRequest)
            instructions.append(first_request.instructions or '')

        assert 'first sandbox instructions' in instructions[0]
        assert 'second sandbox instructions' not in instructions[0]
        assert 'second sandbox instructions' in instructions[1]
        assert 'first sandbox instructions' not in instructions[1]

    def test_serialization_name(self) -> None:
        assert RepoContext.get_serialization_name() == 'RepoContext'

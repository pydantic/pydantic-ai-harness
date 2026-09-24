"""Tests for disk-loaded sub-agents, effort floor, and model inheritance."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import AbstractModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, UnavailableWorkspace, Workspace

from pydantic_ai_harness._workspace import workspace_attached
from pydantic_ai_harness.subagents import (
    MINIMUM_EFFORT_FLOOR,
    AgentOverride,
    SubAgent,
    SubAgents,
    clamp_effort,
)
from pydantic_ai_harness.subagents._disk import (
    ParsedAgent,
    host_folders,
    parse_agent_markdown,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _delegate_then_finish(agent_name: str) -> FunctionModel:
    """A parent model that delegates to `agent_name` once, then replies with text."""
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': agent_name, 'task': 'do it'}, tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


def _delegate_returns(result: Any) -> list[str]:
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
    ]


def _write_agent(folder: Path, filename: str, content: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding='utf-8')


class TestClampEffort:
    def test_none_becomes_floor(self) -> None:
        assert clamp_effort(None) == MINIMUM_EFFORT_FLOOR

    def test_false_becomes_floor(self) -> None:
        assert clamp_effort(False) == MINIMUM_EFFORT_FLOOR

    def test_true_unchanged(self) -> None:
        assert clamp_effort(True) is True

    def test_below_floor_raised(self) -> None:
        assert clamp_effort('minimal') == 'low'

    def test_at_floor_unchanged(self) -> None:
        assert clamp_effort('low') == 'low'

    def test_above_floor_unchanged(self) -> None:
        assert clamp_effort('high') == 'high'

    def test_custom_floor(self) -> None:
        assert clamp_effort('low', floor='high') == 'high'
        assert clamp_effort('xhigh', floor='high') == 'xhigh'


class TestParseAgentMarkdown:
    def test_full_frontmatter(self) -> None:
        text = '---\nname: researcher\ndescription: Researches topics\ntools: Read, Grep\ncolor: blue\n---\nBody here.'
        parsed = parse_agent_markdown(text)
        assert parsed == ParsedAgent(
            name='researcher', description='Researches topics', tools=('Read', 'Grep'), body='Body here.'
        )

    def test_no_frontmatter_uses_whole_text_as_body(self) -> None:
        parsed = parse_agent_markdown('Just a body, no frontmatter.\n')
        assert parsed == ParsedAgent(None, None, (), 'Just a body, no frontmatter.')

    def test_empty_text(self) -> None:
        assert parse_agent_markdown('') == ParsedAgent(None, None, (), '')

    def test_unclosed_frontmatter_is_all_body(self) -> None:
        parsed = parse_agent_markdown('---\nname: x\nstill no close')
        assert parsed == ParsedAgent(None, None, (), '---\nname: x\nstill no close')

    def test_block_list_tools(self) -> None:
        # Includes a blank line (skipped) and a `- ` item with an empty value (dropped).
        text = '---\nname: a\n\ntools:\n  - Read\n  - Edit\n  - \n---\nBody'
        parsed = parse_agent_markdown(text)
        assert parsed.tools == ('Read', 'Edit')

    def test_allowed_tools_key(self) -> None:
        parsed = parse_agent_markdown('---\nname: a\nallowed-tools: Bash(git:*), Read\n---\nB')
        assert parsed.tools == ('Bash(git:*)', 'Read')

    def test_quoted_scalar_values(self) -> None:
        parsed = parse_agent_markdown('---\nname: "quoted"\ndescription: \'single\'\n---\nB')
        assert parsed.name == 'quoted'
        assert parsed.description == 'single'

    def test_non_scalar_name_falls_back_to_none(self) -> None:
        # `name:` with an empty value parses as a (empty) list, which is not a str.
        parsed = parse_agent_markdown('---\nname:\ndescription: d\n---\nB')
        assert parsed.name is None
        assert parsed.description == 'd'

    def test_stray_dash_line_without_list_key_is_ignored(self) -> None:
        # A `- item` line with no preceding list key has no colon, so it resets and is skipped.
        parsed = parse_agent_markdown('---\n- orphan\nname: a\n---\nB')
        assert parsed.name == 'a'
        assert parsed.tools == ()

    def test_no_tools_key(self) -> None:
        assert parse_agent_markdown('---\nname: a\n---\nB').tools == ()


class TestHostFolders:
    def test_str_is_the_home_convention_with_claude_fallback(self, tmp_path: Path) -> None:
        # The project folder is read through the run workspace, so only home is a host folder.
        assert host_folders('agents', tmp_path) == [tmp_path / '.claude' / 'agents']
        (tmp_path / '.agents').mkdir()
        assert host_folders('reviewers', tmp_path) == [tmp_path / '.agents' / 'reviewers']

    def test_sequence_used_verbatim(self, tmp_path: Path) -> None:
        paths = [tmp_path / 'a', tmp_path / 'b']
        assert host_folders(paths, tmp_path) == paths

    def test_duplicate_paths_in_sequence_deduped(self, tmp_path: Path) -> None:
        assert host_folders([tmp_path / 'a', tmp_path / 'a'], tmp_path) == [tmp_path / 'a']


class TestDiskLoading:
    def test_auto_loads_from_convention_by_default(self) -> None:
        # The isolation fixture points the home root at an empty dir; populate its
        # conventional folder and the default `SubAgents()` picks it up with no config.
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', '---\nname: planner\n---\nPlan.')
        cap: SubAgents[object] = SubAgents()
        assert 'planner' in cap._by_name

    def test_host_cwd_is_not_read(self) -> None:
        # Project definitions are the workspace's; the host process's cwd is not a source.
        _write_agent(Path.cwd() / '.agents' / 'agents', 'planner.md', '---\nname: planner\n---\nPlan.')
        cap: SubAgents[object] = SubAgents()
        assert cap._by_name == {}

    def test_none_disables_loading(self) -> None:
        cap: SubAgents[object] = SubAgents(agent_folders=None)
        assert cap._by_name == {}
        assert cap.get_toolset() is None

    def test_loads_agents_from_folder(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'researcher.md', '---\nname: researcher\ndescription: Researches\n---\nResearch well.')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        assert 'researcher' in cap._by_name
        agent = cap._by_name['researcher'].agent
        assert agent.name == 'researcher'
        assert agent.model is None  # inherits the parent model at delegation

    def test_name_falls_back_to_filename_stem(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'planner.md', 'No frontmatter, just a body.')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        assert 'planner' in cap._by_name

    def test_missing_folder_is_skipped(self, tmp_path: Path) -> None:
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path / 'does-not-exist'])
        assert cap._by_name == {}

    def test_undecodable_file_is_skipped_with_warning(self, tmp_path: Path) -> None:
        # A non-UTF-8 `.md` file must not abort loading: it is skipped with a warning
        # and every valid definition in the same folder still loads.
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'broken.md').write_bytes(b'---\nname: broken\n---\n\xff\xfe not utf-8')
        _write_agent(tmp_path, 'valid.md', '---\nname: valid\n---\nWork.')
        with pytest.warns(UserWarning, match='Skipping unreadable disk sub-agent file'):
            cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        assert 'valid' in cap._by_name
        assert 'broken' not in cap._by_name

    def test_listing_uses_description(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'r.md', '---\nname: r\ndescription: Researches\n---\nB')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert '- r: Researches' in instructions


class TestPrecedence:
    def test_explicit_shadows_disk(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'worker.md', '---\nname: worker\ndescription: from disk\n---\nB')
        explicit = Agent(TestModel(), name='worker', description='from code')
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed"):
            cap: SubAgents[object] = SubAgents(agents=[SubAgent(explicit)], agent_folders=[tmp_path])
        assert cap._by_name['worker'].agent is explicit

    def test_earlier_folder_shadows_later(self, tmp_path: Path) -> None:
        project = tmp_path / 'project'
        home = tmp_path / 'home'
        _write_agent(project, 'worker.md', '---\nname: worker\ndescription: project\n---\nB')
        _write_agent(home, 'worker.md', '---\nname: worker\ndescription: home\n---\nB')
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed"):
            cap: SubAgents[object] = SubAgents(agent_folders=[project, home])
        listing = cap.get_instructions()
        assert isinstance(listing, str)
        assert 'project' in listing
        assert 'home' not in listing


class TestOverrides:
    def test_model_and_effort_override(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\n---\nB')
        model = TestModel()
        cap: SubAgents[object] = SubAgents(
            agent_folders=[tmp_path],
            agent_overrides={'w': AgentOverride(model=model, effort='high')},
        )
        agent = cap._by_name['w'].agent
        assert agent.model is model

    def test_effort_floored_without_override(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        # No override -> effort defaults to the floor on the built agent's settings.
        agent = cap._by_name['w'].agent
        assert isinstance(agent, Agent)
        settings = agent.model_settings
        assert isinstance(settings, dict)
        assert settings == {'thinking': MINIMUM_EFFORT_FLOOR}


class TestToolResolver:
    def test_resolver_attaches_tools(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\ntools: search\n---\nB')

        toolset: FunctionToolset[object] = FunctionToolset()

        def resolver(name: str) -> Sequence[AgentToolset[object]] | None:
            return [toolset] if name == 'search' else None

        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path], tool_resolver=resolver)
        # The resolved toolset is attached to the built agent.
        assert toolset in cap._by_name['w'].agent.toolsets

    def test_unknown_tool_warns_and_skips(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\ntools: mystery\n---\nB')

        def resolver(name: str) -> Sequence[AgentToolset[object]] | None:
            return None

        with pytest.warns(UserWarning, match="Unknown tool 'mystery'"):
            SubAgents(agent_folders=[tmp_path], tool_resolver=resolver)

    def test_no_resolver_ignores_frontmatter_tools(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\ntools: Read, Edit\n---\nB')
        # Without a resolver, no warning and no own tools -- inheritance is the path.
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        assert 'w' in cap._by_name


class TestModelInheritance:
    async def test_disk_agent_inherits_parent_model(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'worker.md', '---\nname: worker\n---\nDo the work.')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        disk_agent = cap._by_name['worker'].agent
        assert isinstance(disk_agent, Agent)

        # `RunContext.model` is an `AbstractModel`; the inherited model captured here is the
        # parent's request-response model, asserted by identity below.
        captured: dict[str, AbstractModel] = {}

        @disk_agent.instructions
        def _capture(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            captured['model'] = ctx.model
            return ''

        parent_model = _delegate_then_finish('worker')
        parent: Agent[object, str] = Agent(parent_model, capabilities=[cap])
        result = await parent.run('go')
        assert result.output == 'all done'
        # The model-less disk agent ran on the parent's resolved model.
        assert captured['model'] is parent_model
        assert _delegate_returns(result) == ['all done']


def _recording_model(seen: list[tuple[str | None, list[str]]], delegate_to: str | None = None) -> FunctionModel:
    """A parent model that records each step's instructions and tool names, delegating once if asked."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        seen.append((request.instructions, [tool.name for tool in info.function_tools]))
        if delegate_to is not None and len(seen) == 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': delegate_to, 'task': 'do it'}, tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


class TestWorkspaceDiscovery:
    """The project folder is read through `ctx.workspace` at the start of each run."""

    async def test_project_agents_come_from_the_workspace(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\ndescription: Works\n---\nWork.')
        # Neither a non-markdown file nor a directory is a definition.
        _write_agent(tmp_path / '.agents' / 'agents', 'notes.txt', 'not an agent')
        (tmp_path / '.agents' / 'agents' / 'nested.md').mkdir()
        cap: SubAgents[object] = SubAgents()
        assert cap._by_name == {}, 'nothing is read from the workspace before a run'

        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen, delegate_to='worker'), capabilities=[cap])
        result = await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert _delegate_returns(result) == ['all done']
        instructions, tools = seen[0]
        assert instructions is not None and '- worker: Works' in instructions
        assert tools == ['delegate_task']
        # The disk child inherits this model, so it records the middle step; the parent's
        # later step sees the same listing and tool as its first.
        assert seen[2] == seen[0]

    async def test_claude_fallback_and_stem_name(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan things.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- planner' in seen[0][0]

    async def test_agents_root_without_the_leaf_folder(self, tmp_path: Path) -> None:
        # `.agents/` exists, so `.claude/` is not consulted even though only it has the leaf.
        (tmp_path / '.agents').mkdir()
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0] == (None, [])

    async def test_no_workspace_skips_project_discovery(self) -> None:
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', '---\nname: planner\n---\nPlan.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        await parent.run('go')
        # The home agent still loads; the run just has no project folder to read.
        assert seen[0][0] is not None and '- planner' in seen[0][0]

    async def test_runs_over_unchanged_files_are_identical(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents()
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[cap])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0] == seen[1]
        # The delegate is built once and reused, so its agent does not change between runs.
        assert len(cap._built) == 1
        assert cap._by_name == {}, 'per-run discovery does not leak into the shared instance'

    async def test_project_shadows_home_and_identical_home_is_dropped(self, tmp_path: Path) -> None:
        home = Path.home() / '.agents' / 'agents'
        _write_agent(home, 'worker.md', '---\nname: worker\ndescription: home\n---\nB')
        _write_agent(home, 'same.md', '---\nname: same\n---\nB')
        project = tmp_path / '.agents' / 'agents'
        _write_agent(project, 'worker.md', '---\nname: worker\ndescription: project\n---\nB')
        _write_agent(project, 'same.md', '---\nname: same\n---\nB')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed") as record:
            await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1, 'the identical `same` definition is not reported as shadowed'
        instructions = seen[0][0]
        assert instructions is not None
        assert '- worker: project' in instructions
        assert instructions.count('- same') == 1

    async def test_explicit_shadows_project(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\ndescription: disk\n---\nB')
        explicit = Agent(TestModel(), name='worker', description='code')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(
            _recording_model(seen), capabilities=[SubAgents(agents=[SubAgent(explicit)])]
        )
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed"):
            await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- worker: code' in seen[0][0]

    async def test_undecodable_workspace_file_is_skipped_with_warning(self, tmp_path: Path) -> None:
        folder = tmp_path / '.agents' / 'agents'
        folder.mkdir(parents=True)
        (folder / 'broken.md').write_bytes(b'---\nname: broken\n---\n\xff\xfe not utf-8')
        _write_agent(folder, 'valid.md', '---\nname: valid\n---\nWork.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        with pytest.warns(UserWarning, match='Skipping unreadable disk sub-agent file'):
            await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        instructions = seen[0][0]
        assert instructions is not None and '- valid' in instructions and 'broken' not in instructions

    async def test_explicit_folders_are_static(self, tmp_path: Path) -> None:
        # A path sequence is host configuration: loaded once, no per-run copy, no workspace read.
        _write_agent(tmp_path / 'defs', 'w.md', '---\nname: w\n---\nB')
        _write_agent(tmp_path / '.agents' / 'agents', 'project.md', '---\nname: project\n---\nB')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path / 'defs'])
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[cap])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        instructions = seen[0][0]
        assert instructions is not None and '- w' in instructions and 'project' not in instructions

    async def test_merged_per_run_copies_keep_both_rosters(self, tmp_path: Path) -> None:
        # Run-level capabilities under the shared id are combined after `for_run`, on the per-run copies.
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\n---\nB')
        first: SubAgents[object] = SubAgents(agents=[SubAgent(Agent(TestModel(), name='alpha'))])
        second: SubAgents[object] = SubAgents(agents=[SubAgent(Agent(TestModel(), name='beta'))])
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen))
        await parent.run('go', capabilities=[first, second], workspace=LocalWorkspaceBackend(tmp_path))
        instructions = seen[0][0]
        assert instructions is not None
        assert all(f'- {name}' in instructions for name in ('alpha', 'beta', 'worker'))


class TestWorkspaceAttached:
    def test_real_and_wrapped_workspaces_are_attached(self, tmp_path: Path) -> None:
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        assert workspace_attached(workspace)
        assert workspace_attached(ReadOnlyWorkspace(workspace))

    def test_placeholder_is_not_attached(self) -> None:
        assert not workspace_attached(Workspace(UnavailableWorkspace('none')))

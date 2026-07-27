"""Tests for the `Checkpoints` capability and its shadow-git `CheckpointStore`.

The shadow-git layer is exercised directly through the public `CheckpointStore`
(re-exported from the capability package). The capability's public behavior --
snapshotting before a mutating tool runs -- is driven through
`Agent(..., capabilities=[...])` with a `FunctionModel` that issues tool calls.
Every test uses a `tmp_path` project and an isolated `state_dir`, so nothing
touches the developer's real home directory or git config.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.checkpoints import (
    Checkpoint,
    CheckpointError,
    Checkpoints,
    CheckpointStore,
    CheckpointWarning,
    RestoreCheckpointToolset,
)
from pydantic_ai_harness.checkpoints._shadow import shadow_dir_for

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / 'proj'
    proj.mkdir()
    (proj / 'note.txt').write_text('original\n')
    return proj


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / 'state'


def _store(project: Path, state_dir: Path) -> CheckpointStore:
    return CheckpointStore(project_root=project, shadow_dir=shadow_dir_for(project.resolve(), state_dir))


# --- Direct shadow-git layer -------------------------------------------------


class TestCheckpointStore:
    def test_snapshot_then_restore_brings_file_back(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        cp = store.snapshot(tool_name='write_file', run_id='run-1')
        assert cp.tool_name == 'write_file'
        assert cp.files_changed == ['note.txt']

        (project / 'note.txt').write_text('mutated\n')
        store.restore(cp.id)
        assert (project / 'note.txt').read_text() == 'original\n'

    def test_paths_scoped_restore(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        (project / 'other.txt').write_text('keep\n')
        cp = store.snapshot(tool_name='write_file')

        (project / 'note.txt').write_text('mutated\n')
        (project / 'other.txt').write_text('also mutated\n')
        store.restore(cp.id, paths=['note.txt'])

        assert (project / 'note.txt').read_text() == 'original\n'
        assert (project / 'other.txt').read_text() == 'also mutated\n'

    def test_debounce_reuses_last_checkpoint(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        first = store.snapshot(tool_name='write_file')
        again = store.snapshot(tool_name='edit_file')
        assert again.id == first.id
        assert [cp.id for cp in store.list_checkpoints()] == [first.id]

    def test_second_snapshot_after_change_is_new(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        first = store.snapshot(tool_name='write_file')
        (project / 'note.txt').write_text('changed\n')
        (project / 'new.txt').write_text('new\n')
        second = store.snapshot(tool_name='edit_file')
        assert second.id != first.id
        assert sorted(second.files_changed) == ['new.txt', 'note.txt']
        assert [cp.id for cp in store.list_checkpoints()] == [first.id, second.id]

    def test_gitignore_is_respected(self, project: Path, state_dir: Path) -> None:
        (project / '.gitignore').write_text('build/\n')
        (project / 'build').mkdir()
        (project / 'build' / 'artifact.o').write_text('binary\n')
        store = _store(project, state_dir)
        cp = store.snapshot(tool_name='write_file')
        assert all('build' not in changed for changed in cp.files_changed)

    def test_non_git_project(self, project: Path, state_dir: Path) -> None:
        assert not (project / '.git').exists()
        store = _store(project, state_dir)
        cp = store.snapshot(tool_name='write_file')
        (project / 'note.txt').write_text('mutated\n')
        store.restore(cp.id)
        assert (project / 'note.txt').read_text() == 'original\n'
        assert not (project / '.git').exists()

    def test_user_git_repo_is_untouched(self, project: Path, state_dir: Path) -> None:
        subprocess.run(['git', 'init', '-q'], cwd=project, check=True)
        subprocess.run(['git', 'config', 'user.email', 'u@u'], cwd=project, check=True)
        subprocess.run(['git', 'config', 'user.name', 'u'], cwd=project, check=True)
        subprocess.run(['git', 'add', 'note.txt'], cwd=project, check=True)
        subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=project, check=True)
        head_before = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=project, capture_output=True, text=True, check=True
        ).stdout.strip()

        store = _store(project, state_dir)
        store.snapshot(tool_name='write_file')
        (project / 'note.txt').write_text('mutated\n')
        store.snapshot(tool_name='edit_file')

        head_after = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=project, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert head_before == head_after
        log = subprocess.run(
            ['git', 'log', '--format=%an %ae'], cwd=project, capture_output=True, text=True, check=True
        ).stdout
        assert 'pydantic-ai-harness checkpoints' not in log

    def test_empty_project_gets_baseline_checkpoint(self, tmp_path: Path, state_dir: Path) -> None:
        empty = tmp_path / 'empty'
        empty.mkdir()
        store = _store(empty, state_dir)
        cp = store.snapshot(tool_name='write_file')
        assert cp.files_changed == []
        assert store.list_checkpoints() == [cp]

    def test_manual_snapshot_has_no_tool_name(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        cp = store.snapshot()
        assert cp.tool_name is None

    def test_head_and_list_empty_before_any_snapshot(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        store.ensure_initialized()
        assert store.head() is None
        assert store.list_checkpoints() == []

    def test_restore_unknown_id_raises(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        store.snapshot(tool_name='write_file')
        with pytest.raises(CheckpointError, match='unknown checkpoint id'):
            store.restore('deadbeef')

    def test_git_command_failure_raises_checkpoint_error(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        cp = store.snapshot(tool_name='write_file')
        # Restoring a path the checkpoint never captured makes the underlying git
        # command fail, which surfaces as a CheckpointError.
        with pytest.raises(CheckpointError, match='shadow git'):
            store.restore(cp.id, paths=['never-existed.txt'])

    def test_slug_sanitizes_special_characters(self, tmp_path: Path, state_dir: Path) -> None:
        weird = tmp_path / 'my project (v2)'
        weird.mkdir()
        (weird / 'a.txt').write_text('x\n')
        store = _store(weird, state_dir)
        store.snapshot(tool_name='write_file')
        assert store.shadow_dir.name.startswith('my-project--v2-')


# --- Capability through an Agent ---------------------------------------------


def _mutating_agent(project: Path, capability: Checkpoints[None], *, tool_name: str = 'write_file') -> Agent[None, str]:
    """Agent whose model calls `tool_name` once (writing a file), then finishes."""
    state = {'i': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        if i == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name, {'content': 'changed'})])
        return ModelResponse(parts=[TextPart('done')])

    def mutate(content: str) -> str:
        (project / 'note.txt').write_text(content + '\n')
        return 'written'

    toolset = FunctionToolset[None]()
    toolset.add_function(mutate, name=tool_name)
    return Agent(FunctionModel(model_fn), deps_type=type(None), capabilities=[capability], toolsets=[toolset])


async def test_snapshot_taken_before_mutating_tool(project: Path, state_dir: Path) -> None:
    cap = Checkpoints[None](project_root=project, state_dir=state_dir)
    agent = _mutating_agent(project, cap)
    result = await agent.run('go')
    assert result.output == 'done'
    assert (project / 'note.txt').read_text() == 'changed\n'

    checkpoints = cap.checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0].tool_name == 'write_file'
    # The checkpoint captured the pre-write state, so restoring undoes the tool.
    cap.restore(checkpoints[0].id)
    assert (project / 'note.txt').read_text() == 'original\n'


def test_manual_snapshot_via_capability(project: Path, state_dir: Path) -> None:
    cap = Checkpoints[None](project_root=project, state_dir=state_dir)
    cp = cap.snapshot()
    assert cp.tool_name is None
    assert [c.id for c in cap.checkpoints()] == [cp.id]


async def test_non_mutating_tool_takes_no_snapshot(project: Path, state_dir: Path) -> None:
    cap = Checkpoints[None](project_root=project, state_dir=state_dir)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(p, ToolCallPart) for m in messages for p in getattr(m, 'parts', [])):
            return ModelResponse(parts=[ToolCallPart('read_note', {})])
        return ModelResponse(parts=[TextPart('done')])

    def read_note() -> str:
        return (project / 'note.txt').read_text()

    agent = Agent(FunctionModel(model_fn), deps_type=type(None), capabilities=[cap], tools=[read_note])
    await agent.run('go')
    assert cap.checkpoints() == []


async def test_mutating_tools_config_respected(project: Path, state_dir: Path) -> None:
    # A custom set that does NOT include write_file: no snapshot should be taken.
    cap = Checkpoints[None](project_root=project, state_dir=state_dir, mutating_tools={'apply_patch'})
    agent = _mutating_agent(project, cap)
    await agent.run('go')
    assert cap.checkpoints() == []


async def test_snapshot_before_bash(project: Path, state_dir: Path) -> None:
    cap = Checkpoints[None](
        project_root=project,
        state_dir=state_dir,
        mutating_tools=set(),
        snapshot_before_bash=True,
        bash_tools={'run_command'},
    )
    agent = _mutating_agent(project, cap, tool_name='run_command')
    await agent.run('go')
    assert len(cap.checkpoints()) == 1


async def test_agent_level_debounce(project: Path, state_dir: Path) -> None:
    """Two mutating calls that change nothing produce a single checkpoint."""
    cap = Checkpoints[None](project_root=project, state_dir=state_dir)
    state = {'i': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        if i < 2:
            return ModelResponse(parts=[ToolCallPart('write_file', {})])
        return ModelResponse(parts=[TextPart('done')])

    def write_file() -> str:  # no-op: does not change any file
        return 'noop'

    agent = Agent(FunctionModel(model_fn), deps_type=type(None), capabilities=[cap], tools=[write_file])
    await agent.run('go')
    assert len(cap.checkpoints()) == 1


async def test_snapshot_failure_warns_and_continues(project: Path, tmp_path: Path) -> None:
    # Point state_dir at a regular file so the shadow repo cannot be created.
    blocking_file = tmp_path / 'not-a-dir'
    blocking_file.write_text('x')
    cap = Checkpoints[None](project_root=project, state_dir=blocking_file)
    agent = _mutating_agent(project, cap)
    with pytest.warns(CheckpointWarning, match='Could not snapshot'):
        result = await agent.run('go')
    # Best-effort: the tool still ran even though the snapshot failed.
    assert result.output == 'done'
    assert (project / 'note.txt').read_text() == 'changed\n'


# --- Optional model-facing tools ---------------------------------------------


class TestRestoreTool:
    def test_get_toolset_none_by_default(self, project: Path, state_dir: Path) -> None:
        cap = Checkpoints[None](project_root=project, state_dir=state_dir)
        assert cap.get_toolset() is None

    def test_get_toolset_when_exposed(self, project: Path, state_dir: Path) -> None:
        cap = Checkpoints[None](project_root=project, state_dir=state_dir, expose_tool=True)
        assert isinstance(cap.get_toolset(), RestoreCheckpointToolset)

    async def test_tool_lists_and_restores(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        cp = store.snapshot(tool_name='write_file')
        toolset = RestoreCheckpointToolset[None](store)

        listing = await toolset.list_checkpoints()
        assert cp.id in listing
        assert 'write_file' in listing

        (project / 'note.txt').write_text('mutated\n')
        message = await toolset.restore_checkpoint(cp.id)
        assert cp.id in message
        assert (project / 'note.txt').read_text() == 'original\n'

    async def test_tool_paths_scoped_restore_message(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        cp = store.snapshot(tool_name='write_file')
        toolset = RestoreCheckpointToolset[None](store)
        message = await toolset.restore_checkpoint(cp.id, paths=['note.txt'])
        assert 'note.txt' in message

    async def test_tool_lists_nothing_before_any_snapshot(self, project: Path, state_dir: Path) -> None:
        store = _store(project, state_dir)
        store.ensure_initialized()
        toolset = RestoreCheckpointToolset[None](store)
        assert await toolset.list_checkpoints() == 'No checkpoints recorded yet.'


def test_checkpoint_is_frozen_dataclass() -> None:
    cp = Checkpoint(id='abc', time=__import__('datetime').datetime.now(), tool_name='write_file', files_changed=[])
    with pytest.raises(AttributeError):
        cp.id = 'x'  # pyright: ignore[reportAttributeAccessIssue]

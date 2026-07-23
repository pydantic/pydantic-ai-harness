"""Tests for the Harbor adapter, driven by a fake environment (no Docker)."""

from __future__ import annotations

from pathlib import Path

from conftest import FakeEnvironment, _ExecResult
from harbor.models.agent.context import AgentContext

from pydantic_ai_harness_terminal_bench.harbor_agent import PydanticAITerminalBenchAgent
from pydantic_ai_harness_terminal_bench.smoke import SMOKE_MARKER_PATH, SMOKE_TEXT, ScriptedSmokeAgent


def test_name_is_leaderboard_entry() -> None:
    assert PydanticAITerminalBenchAgent.name() == 'pydantic-ai-harness'


def test_version_is_a_string(tmp_path: Path) -> None:
    agent = PydanticAITerminalBenchAgent(logs_dir=tmp_path)
    assert isinstance(agent.version(), str)


def test_resolve_model_maps_harbor_name(tmp_path: Path) -> None:
    agent = PydanticAITerminalBenchAgent(logs_dir=tmp_path, model_name='anthropic/claude-opus-4-6')
    assert agent.resolve_model() == 'anthropic:claude-opus-4-6'


def test_resolve_model_defaults_when_unset(tmp_path: Path) -> None:
    agent = PydanticAITerminalBenchAgent(logs_dir=tmp_path)
    assert agent.resolve_model() == 'anthropic:claude-opus-4-6'


async def test_setup_is_a_noop(tmp_path: Path) -> None:
    agent = PydanticAITerminalBenchAgent(logs_dir=tmp_path)
    await agent.setup(FakeEnvironment())  # type: ignore[arg-type]


async def test_run_round_trips_exec_and_populates_context(tmp_path: Path) -> None:
    """The scripted agent's commands reach the environment; usage lands on the context."""
    environment = FakeEnvironment(
        responses={f'cat {SMOKE_MARKER_PATH}': _ExecResult(stdout=SMOKE_TEXT, return_code=0)},
    )
    agent = ScriptedSmokeAgent(logs_dir=tmp_path)
    context = AgentContext()

    await agent.run('write the file', environment, context)  # type: ignore[arg-type]

    assert environment.calls == [
        f'echo {SMOKE_TEXT} > {SMOKE_MARKER_PATH}',
        f'cat {SMOKE_MARKER_PATH}',
    ]
    assert context.metadata is not None
    assert context.metadata['tool_calls'] == 2
    assert context.n_output_tokens is not None


async def test_run_populates_context_on_failure(tmp_path: Path) -> None:
    """A run that raises still records whatever usage accrued (the finally path)."""

    class _Boom(ScriptedSmokeAgent):
        def build_agent(self):  # type: ignore[override]
            raise RuntimeError('build failed')

    agent = _Boom(logs_dir=tmp_path)
    context = AgentContext()
    try:
        await agent.run('x', FakeEnvironment(), context)  # type: ignore[arg-type]
    except RuntimeError:
        pass
    # result was None, so context stays empty rather than crashing in the finally.
    assert context.is_empty()

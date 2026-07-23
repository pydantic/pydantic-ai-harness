"""Tests for the live-model agent: Gateway routing by env, span-wrapped run."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeEnvironment, _ExecResult
from harbor.models.agent.context import AgentContext
from pydantic_ai.models import Model

from pydantic_ai_harness_terminal_bench.config import GATEWAY_ENV_VAR
from pydantic_ai_harness_terminal_bench.live import LiveBenchAgent
from pydantic_ai_harness_terminal_bench.smoke import SMOKE_MARKER_PATH, SMOKE_TEXT, scripted_model


def test_resolve_model_direct_without_gateway_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GATEWAY_ENV_VAR, raising=False)
    agent = LiveBenchAgent(logs_dir=tmp_path, model_name='anthropic/claude-sonnet-4-6')
    assert agent.resolve_model() == 'anthropic:claude-sonnet-4-6'


def test_resolve_model_routes_through_gateway_when_key_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GATEWAY_ENV_VAR, 'pag-secret')
    agent = LiveBenchAgent(logs_dir=tmp_path, model_name='anthropic/claude-sonnet-4-6')
    assert agent.resolve_model() == 'gateway/anthropic:claude-sonnet-4-6'


async def test_run_delegates_through_the_trial_span(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a Logfire token the span is a no-op, and `run` still drives the seam."""
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)

    class _ScriptedLiveAgent(LiveBenchAgent):
        def resolve_model(self) -> str | Model:
            return scripted_model()

    environment = FakeEnvironment(
        responses={f'cat {SMOKE_MARKER_PATH}': _ExecResult(stdout=SMOKE_TEXT, return_code=0)},
    )
    agent = _ScriptedLiveAgent(logs_dir=tmp_path)
    context = AgentContext()

    await agent.run('write the file', environment, context)  # type: ignore[arg-type]

    assert environment.calls == [
        f'echo {SMOKE_TEXT} > {SMOKE_MARKER_PATH}',
        f'cat {SMOKE_MARKER_PATH}',
    ]
    assert context.metadata is not None
    assert context.metadata['tool_calls'] == 2

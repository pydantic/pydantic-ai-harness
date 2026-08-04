"""Tests for the public `Scheduling` capability surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.scheduling import (
    InMemoryScheduleStore,
    ScheduleRunner,
    Scheduling,
    SqliteScheduleStore,
)


@pytest.fixture
def anyio_backend() -> str:
    """Run agent integration tests on the backend used by Pydantic AI's graph."""
    return 'asyncio'


class TestSchedulingCapability:
    def test_default_store_is_owned_and_shared_with_runner(self) -> None:
        capability = Scheduling[None]()
        assert isinstance(capability.resolved_store, InMemoryScheduleStore)
        agent: Agent[None, str] = Agent(TestModel(), deps_type=type(None), capabilities=[capability])
        ScheduleRunner(agent, deps=None)

    def test_each_default_store_is_fresh(self) -> None:
        assert Scheduling[None]().resolved_store is not Scheduling[None]().resolved_store

    def test_explicit_falsey_store_wins(self) -> None:
        store = _FalseyScheduleStore()
        assert not store
        ScheduleRunner(Agent(TestModel()), deps=None, store=store)

    def test_runner_resolves_wrapped_scheduling_store(self) -> None:
        capability = Scheduling[None]()
        agent: Agent[None, str] = Agent(
            TestModel(),
            deps_type=type(None),
            capabilities=[PrefixTools(wrapped=capability, prefix='sched')],
        )
        ScheduleRunner(agent, deps=None)

    def test_invalid_timezone_rejected(self) -> None:
        with pytest.raises(ValueError, match='Unknown IANA timezone'):
            Scheduling(timezone='not/a-zone')

    def test_runner_store_resolution_errors_are_actionable(self) -> None:
        with pytest.raises(ValueError, match='pass `store=` explicitly'):
            ScheduleRunner(Agent(TestModel()), deps=None)
        agent: Agent[None, str] = Agent(
            TestModel(), deps_type=type(None), capabilities=[Scheduling[None](), Scheduling[None]()]
        )
        with pytest.raises(ValueError, match='multiple Scheduling'):
            ScheduleRunner(agent, deps=None)

    def test_from_spec_all_branches(self, tmp_path: Path) -> None:
        assert isinstance(Scheduling.from_spec().resolved_store, InMemoryScheduleStore)
        sqlite = Scheduling.from_spec(backend='sqlite', database=str(tmp_path / 'schedules.db'))
        assert isinstance(sqlite.resolved_store, SqliteScheduleStore)
        with pytest.raises(ValueError, match='database is only valid'):
            Scheduling.from_spec(database='custom.db')
        with pytest.raises(ValueError, match='Unknown scheduling backend'):
            Scheduling.from_spec(backend='cloud')  # type: ignore[arg-type]

    def test_serialization_name(self) -> None:
        assert Scheduling.get_serialization_name() == 'Scheduling'

    def test_agent_spec_roundtrip(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [{'Scheduling': {'backend': 'memory', 'timezone': 'Asia/Kolkata'}}],
            },
            custom_capability_types=[Scheduling],
        )
        loaded = [cap for cap in agent.root_capability.capabilities if isinstance(cap, Scheduling)]
        assert len(loaded) == 1
        assert loaded[0].timezone == 'Asia/Kolkata'

    async def test_guidance_states(self) -> None:
        for guidance, expected in ((None, 'create_schedule'), ('custom guidance', 'custom guidance'), ('', None)):
            model = TestModel(call_tools=[])
            agent: Agent[None, str] = Agent(
                model, deps_type=type(None), capabilities=[Scheduling[None](guidance=guidance)]
            )
            result = await agent.run('hello')
            parameters = model.last_model_request_parameters
            assert parameters is not None
            parts = parameters.instruction_parts or []
            instructions = InstructionPart.join(parts) if parts else None
            if expected is None:
                assert instructions is None
            else:
                assert instructions is not None
                assert expected in instructions
            assert result.output == 'success (no tool calls)'


class _FalseyScheduleStore(InMemoryScheduleStore):
    def __bool__(self) -> bool:
        return False

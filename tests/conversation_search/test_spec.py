"""`ConversationSearch` spec support: the published `AgentSpec` schema entry and `from_spec`.

The bare `source: HistorySource` field has no JSON representation, and one such field
used to erase the capability's whole schema entry, short form included. `from_spec`
names the spec-expressible parameters instead, with `backend`/`directory`/`database`
mirroring `StepPersistence.from_spec` so a spec can point both capabilities at the
same store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore, RunRecord, SqliteStepStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestSchema:
    def test_schema_publishes_the_capability_entry(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([ConversationSearch])
        params: dict[str, Any] = schema['$defs']['spec_params_ConversationSearch']['properties']
        assert {
            'backend',
            'directory',
            'database',
            'scope',
            'tool_id',
            'max_matches',
            'context_lines',
            'bm25_k1',
            'bm25_b',
            'add_instructions',
            'id',
            'description',
            'defer_loading',
        } <= set(params)
        assert 'source' not in params

    def test_short_form_takes_the_backend(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([ConversationSearch])
        short: dict[str, Any] = schema['$defs']['short_spec_ConversationSearch']['properties']['ConversationSearch']
        assert short['enum'] == ['file', 'sqlite']


class TestFromSpec:
    async def test_sqlite_backend_reads_the_named_database(self, tmp_path: Path) -> None:
        database = tmp_path / 'sessions.db'
        await SqliteStepStore(database=database).register_run(RunRecord(run_id='r1'))

        capability = ConversationSearch.from_spec('sqlite', database=str(database), scope='all')

        assert isinstance(capability.source, SnapshotHistorySource)
        assert [run.run_id for run in await capability.source.list_runs()] == ['r1']

    async def test_file_backend_reads_the_named_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / 'runs'
        await FileStepStore(directory).register_run(RunRecord(run_id='r1'))

        capability = ConversationSearch.from_spec('file', directory=str(directory), scope='all')

        assert [run.run_id for run in await capability.source.list_runs()] == ['r1']

    def test_search_options_are_forwarded(self, tmp_path: Path) -> None:
        capability = ConversationSearch.from_spec(
            'sqlite',
            database=str(tmp_path / 'sessions.db'),
            scope='conversation',
            tool_id='recall',
            max_matches=3,
            context_lines=1,
            bm25_k1=1.2,
            bm25_b=0.5,
            add_instructions=False,
            id='cs',
            description='recall tool',
            defer_loading=True,
        )
        assert capability.scope == 'conversation'
        assert capability.tool_id == 'recall'
        assert capability.max_matches == 3
        assert capability.context_lines == 1
        assert capability.bm25_k1 == 1.2
        assert capability.bm25_b == 0.5
        assert capability.add_instructions is False
        assert capability.id == 'cs'
        assert capability.description == 'recall tool'
        assert capability.defer_loading is True

    def test_scope_left_unset_warns_like_the_constructor(self, tmp_path: Path) -> None:
        with pytest.warns(HarnessDeprecationWarning):
            ConversationSearch.from_spec('sqlite', database=str(tmp_path / 'sessions.db'))

    async def test_loads_through_an_agent_spec(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        agent = Agent.from_spec(
            {
                'capabilities': [
                    {
                        'ConversationSearch': {
                            'backend': 'sqlite',
                            'database': str(tmp_path / 'sessions.db'),
                            'scope': 'all',
                        }
                    },
                ],
            },
            custom_capability_types=[ConversationSearch],
            model=model,
        )
        await agent.run('go')
        assert model.last_model_request_parameters is not None
        tool_names = [tool.name for tool in model.last_model_request_parameters.function_tools]
        assert 'search_conversation_history' in tool_names


class TestFromSpecRejections:
    def test_a_live_source_is_rejected_by_name(self) -> None:
        bad: dict[str, Any] = {'backend': 'sqlite', 'source': object()}
        with pytest.raises(UserError, match='cannot be built from a spec with `source`'):
            ConversationSearch.from_spec(**bad)

    def test_unknown_fields_are_rejected(self) -> None:
        bad: dict[str, Any] = {'backend': 'sqlite', 'corpus': 'everything'}
        with pytest.raises(UserError, match=r"no spec field\(s\) \['corpus'\]"):
            ConversationSearch.from_spec(**bad)

    def test_the_memory_backend_is_refused(self) -> None:
        bad: dict[str, Any] = {'backend': 'memory'}
        with pytest.raises(UserError, match='no `memory`'):
            ConversationSearch.from_spec(**bad)

    def test_directory_requires_the_file_backend(self, tmp_path: Path) -> None:
        with pytest.raises(UserError, match='directory is only valid with backend="file"'):
            ConversationSearch.from_spec('sqlite', directory=str(tmp_path))

    def test_database_requires_the_sqlite_backend(self, tmp_path: Path) -> None:
        with pytest.raises(UserError, match='database is only valid with backend="sqlite"'):
            ConversationSearch.from_spec('file', database=str(tmp_path / 'sessions.db'))

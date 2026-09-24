"""Configuration, catalog, and schema tests for `JevCapabilityComposer`."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.jev import (
    SKILLS_DIRECTORY,
    ComposableCapability,
    Composition,
    JevCapabilityComposer,
    Thinking,
    default_catalog,
)

from ._doubles import CATALOG, Clock, Jev, Notes, composer

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class Undocumented(AbstractCapability[object]):
    pass


class TestConfiguration:
    def test_requires_models(self):
        with pytest.raises(UserError, match='`models`'):
            JevCapabilityComposer(models={}, catalog=CATALOG)

    def test_requires_a_catalog(self):
        with pytest.raises(UserError, match='`catalog`'):
            JevCapabilityComposer(models={'fast': 'test'}, catalog={})

    @pytest.mark.parametrize('threshold', [-0.1, 1.1])
    def test_rejects_threshold_outside_unit_interval(self, threshold: float):
        with pytest.raises(UserError, match='between 0 and 1'):
            JevCapabilityComposer(models={'fast': 'test'}, catalog=CATALOG, confidence_threshold=threshold)

    def test_rejects_an_unsure_model_off_the_menu(self):
        with pytest.raises(UserError, match="unsure_model 'nope' is not a key"):
            JevCapabilityComposer(models={'fast': 'test'}, catalog=CATALOG, unsure_model='nope')

    def test_not_spec_serializable(self):
        assert JevCapabilityComposer.get_serialization_name() is None


class TestCatalogEntries:
    def test_an_entry_is_described_by_its_capability_docstring(self):
        entry = ComposableCapability.of(Clock)

        assert entry == ComposableCapability(description='A second catalog capability.', capability=Clock)

    def test_an_entry_description_and_arguments_can_be_given(self):
        entry = ComposableCapability.of(Notes, description='Keep notes', arguments={'prefix': 'memo'})

        assert entry == CATALOG['notes']

    def test_an_entry_needs_a_description_from_somewhere(self):
        with pytest.raises(UserError, match='Undocumented has no docstring'):
            ComposableCapability.of(Undocumented)


class TestDefaultCatalog:
    def test_every_entry_builds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / SKILLS_DIRECTORY).mkdir(parents=True)
        catalog = default_catalog()

        for entry in catalog.values():
            assert isinstance(entry.capability.from_spec(**entry.arguments), entry.capability)
        assert ('skills' in catalog) == (importlib.util.find_spec('yaml') is not None)
        assert ('code_mode' in catalog) == (importlib.util.find_spec('pydantic_monty') is not None)
        assert ('web_fetch' in catalog) == (importlib.util.find_spec('markdownify') is not None)

    def test_it_leaves_out_what_is_not_available(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / SKILLS_DIRECTORY).mkdir(parents=True)
        find_spec = importlib.util.find_spec

        def without_extras(name: str, package: str | None = None) -> ModuleSpec | None:
            return None if name in ('pydantic_monty', 'ddgs', 'markdownify', 'yaml') else find_spec(name, package)

        monkeypatch.setattr(importlib.util, 'find_spec', without_extras)

        catalog = default_catalog()

        assert list(catalog) == ['filesystem', 'shell', 'planning', 'repo_context', 'pydantic_ai_docs', 'web_search']
        assert catalog['web_search'].arguments == {}

    async def test_local_entries_run_together(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """No two local-tool entries register the same tool name. The web entries are native tools `TestModel` lacks."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / SKILLS_DIRECTORY).mkdir(parents=True)
        local = tuple(key for key in default_catalog() if key not in ('web_search', 'web_fetch'))
        picks = JevCapabilityComposer(models={'fast': TestModel(call_tools=[])}).capability_for(
            Composition(model='fast', thinking=Thinking.low, capabilities=local, confidence={})
        )

        result = await Agent('test', capabilities=[picks]).run('hi')

        assert result.output == 'success (no tool calls)'


class TestSchema:
    async def test_every_option_is_described(self):
        """Jev reads each option's description from the schema, so none may go without one."""
        jev = Jev()
        await composer(jev, []).compose('commit this')

        assert jev.schemas == [
            {
                '$defs': {
                    'Thinking': {
                        'anyOf': [
                            {'const': 'low', 'description': 'Routine or single-step work'},
                            {'const': 'medium', 'description': 'Moderate multi-step work'},
                            {'const': 'high', 'description': 'Hard problems: debugging, design, or large changes'},
                        ],
                        'description': 'How much reasoning effort a request needs.',
                        'title': 'Thinking',
                        'type': 'string',
                    }
                },
                'properties': {
                    'model': {
                        'anyOf': [
                            {'const': 'fast', 'description': 'Quick answers'},
                            {'const': 'strong', 'description': 'Hard problems'},
                        ],
                        'description': 'Which model should handle this request?',
                        'type': 'string',
                    },
                    'thinking': {
                        '$ref': '#/$defs/Thinking',
                        'description': 'How much reasoning effort does this request need?',
                    },
                    'capabilities': {
                        'description': 'Does handling this request need this capability?',
                        'items': {
                            'anyOf': [
                                {'const': 'notes', 'description': 'Keep notes'},
                                {'const': 'clock', 'description': 'Tell the time'},
                            ],
                            'type': 'string',
                        },
                        'type': 'array',
                    },
                },
                'required': ['model', 'thinking', 'capabilities'],
                'title': 'Composition',
                'type': 'object',
            }
        ]

    async def test_a_model_without_a_description_is_described_by_its_name(self):
        jev = Jev(model='fast')
        await JevCapabilityComposer(models={'fast': 'test'}, catalog=CATALOG, jev_model=jev.model_).compose('hi')

        properties = jev.schemas[0]['properties']
        assert isinstance(properties, dict)
        assert properties['model'] == {
            'anyOf': [{'const': 'fast', 'description': 'test'}],
            'description': 'Which model should handle this request?',
            'type': 'string',
        }


class TestValidation:
    async def test_jev_can_only_answer_from_the_menu(self):
        """An off-menu model or capability fails output validation, so it can never be built."""
        with pytest.raises(UnexpectedModelBehavior):
            await composer(Jev(model='bogus'), []).compose('write that down')

        with pytest.raises(UnexpectedModelBehavior):
            await composer(Jev(capabilities=('bogus',)), []).compose('write that down')

"""Tests for the confined-authoring capability."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from pydantic_ai_harness.experimental.confined_authoring import (
    AuthoredSlot,
    ConfinedAuthoring,
    ConfinedAuthoringToolset,
    InjectedFunction,
    SlotParameter,
    SlotStore,
    SlotValidationError,
    validate_tool_slot,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (the shared Monty loop uses asyncio)."""
    return 'asyncio'


# --------------------------------------------------------------------------- #
# Injected functions and fixtures
# --------------------------------------------------------------------------- #


async def weather(ctx: RunContext[object], kwargs: dict[str, object]) -> object:
    return {'temp_c': 21, 'place': kwargs['place']}


async def boom(ctx: RunContext[object], kwargs: dict[str, object]) -> object:
    raise RuntimeError('injected function failed')


async def greet(ctx: RunContext[object], kwargs: dict[str, object]) -> object:
    return f'hello {kwargs["name"]}'


WEATHER = InjectedFunction[object](
    name='weather',
    call=weather,
    parameters={'type': 'object', 'properties': {'place': {'type': 'string'}}, 'required': ['place']},
    returns={'type': 'object'},
    description='Get weather for a place.',
)
BOOM = InjectedFunction[object](
    name='boom',
    call=boom,
    parameters={'type': 'object', 'properties': {}},
)
# `greet` declares no return schema, so its rendered signature is `-> Any`.
GREET = InjectedFunction[object](
    name='greet',
    call=greet,
    parameters={'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
)

POOL = [WEATHER, BOOM, GREET]

FORECAST_CODE = 'w = await weather(place=city)\n{"summary": f"{w[\'place\']}: {w[\'temp_c\']}C", "days": days}'
FORECAST_PARAMS: list[SlotParameter] = [
    SlotParameter(name='city', type='string'),
    SlotParameter(name='days', type='integer'),
]


def _ctx() -> RunContext[object]:
    return RunContext[object](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1)


def _store(tmp_path: Path, functions: list[InjectedFunction[object]] | None = None) -> SlotStore[object]:
    return SlotStore[object](tmp_path, POOL if functions is None else functions)


async def _call_slot(toolset: ConfinedAuthoringToolset[object], name: str, args: dict[str, Any]) -> Any:
    ctx = _ctx()
    fresh = await toolset.for_run(ctx)
    tools = await fresh.get_tools(ctx)
    return await fresh.call_tool(name, args, ctx, tools[name])


def _author(store: SlotStore[object], **overrides: Any) -> AuthoredSlot:
    kwargs: dict[str, Any] = {
        'name': 'forecast',
        'description': 'Summarize the weather for a city.',
        'code': FORECAST_CODE,
        'parameters': FORECAST_PARAMS,
        'uses': ['weather'],
        'returns': 'object',
    }
    kwargs.update(overrides)
    return store.author_tool(**kwargs)


# --------------------------------------------------------------------------- #
# Experimental warning
# --------------------------------------------------------------------------- #


def test_import_emits_experimental_warning() -> None:
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        importlib.reload(importlib.import_module('pydantic_ai_harness.experimental.confined_authoring'))
    assert any(issubclass(w.category, HarnessExperimentalWarning) for w in caught)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_valid_slot_passes(self) -> None:
        validate_tool_slot(
            parameters=FORECAST_PARAMS,
            uses=['weather'],
            code=FORECAST_CODE,
            returns='object',
            functions={'weather': WEATHER},
        )

    def test_no_return_type_skips_annotation(self) -> None:
        validate_tool_slot(parameters=[], uses=[], code='1 + 1', returns=None, functions={})

    def test_invalid_parameter_identifier(self) -> None:
        with pytest.raises(SlotValidationError, match='not a valid Python identifier'):
            validate_tool_slot(parameters=[SlotParameter(name='not ok')], uses=[], code='1', returns=None, functions={})

    def test_duplicate_parameter(self) -> None:
        with pytest.raises(SlotValidationError, match='duplicate parameter'):
            validate_tool_slot(
                parameters=[SlotParameter(name='x'), SlotParameter(name='x')],
                uses=[],
                code='1',
                returns=None,
                functions={},
            )

    def test_parameter_shadows_function(self) -> None:
        with pytest.raises(SlotValidationError, match='shadows the injected function'):
            validate_tool_slot(
                parameters=[SlotParameter(name='weather')],
                uses=['weather'],
                code='1',
                returns=None,
                functions={'weather': WEATHER},
            )

    def test_unknown_used_function(self) -> None:
        with pytest.raises(SlotValidationError, match='not in the capability pool'):
            validate_tool_slot(parameters=[], uses=['missing'], code='1', returns=None, functions={'weather': WEATHER})

    def test_unknown_used_function_empty_pool_lists_none(self) -> None:
        with pytest.raises(SlotValidationError, match=r"Available: \['\(none\)'\]"):
            validate_tool_slot(parameters=[], uses=['missing'], code='1', returns=None, functions={})

    def test_type_error_wrong_argument(self) -> None:
        code = 'w = await weather(place=days)\nw'
        with pytest.raises(SlotValidationError, match='type error'):
            validate_tool_slot(
                parameters=[SlotParameter(name='days', type='integer')],
                uses=['weather'],
                code=code,
                returns='object',
                functions={'weather': WEATHER},
            )

    def test_syntax_error(self) -> None:
        with pytest.raises(SlotValidationError, match='syntax error'):
            validate_tool_slot(parameters=[], uses=[], code='def (:', returns=None, functions={})

    def test_syntax_error_with_declared_return(self) -> None:
        # A declared return type triggers the AST transform, which cannot parse broken
        # source; validation falls back to the original code so Monty reports the syntax error.
        with pytest.raises(SlotValidationError, match='syntax error'):
            validate_tool_slot(parameters=[], uses=[], code='x = (', returns='integer', functions={})

    def test_return_type_mismatch_caught_statically(self) -> None:
        code = 'w = await weather(place=city)\nw'
        with pytest.raises(SlotValidationError, match='type error'):
            validate_tool_slot(
                parameters=[SlotParameter(name='city')],
                uses=['weather'],
                code=code,
                returns='integer',
                functions={'weather': WEATHER},
            )

    def test_declared_return_without_result_expression(self) -> None:
        with pytest.raises(SlotValidationError, match='does not end with a result expression'):
            validate_tool_slot(parameters=[], uses=[], code='x = 1', returns='integer', functions={})

    def test_discarded_async_call_flagged(self) -> None:
        code = 'weather(place=city)\nresult = await weather(place=city)\nresult'
        with pytest.raises(SlotValidationError, match='without `await`'):
            validate_tool_slot(
                parameters=[SlotParameter(name='city')],
                uses=['weather'],
                code=code,
                returns='object',
                functions={'weather': WEATHER},
            )

    def test_gather_and_attribute_calls_not_flagged(self) -> None:
        # A call inside asyncio.gather (an argument to another call) and an awaited
        # call are not discarded statements and must pass the missing-await scan.
        code = 'import asyncio\nrs = await asyncio.gather(weather(place=city), weather(place=city))\nrs'
        validate_tool_slot(
            parameters=[SlotParameter(name='city')],
            uses=['weather'],
            code=code,
            returns=None,
            functions={'weather': WEATHER},
        )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class TestSlotStore:
    def test_author_and_persist(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        record = _author(store)
        assert record.status == 'validated'
        assert record.last_error is None
        assert (tmp_path / 'slots.json').exists()
        assert [r.name for r in store.list_all()] == ['forecast']

    def test_author_invalid_name_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match='invalid slot name'):
            _author(store, name='Not-Valid')
        assert store.list_all() == []

    def test_author_reserved_name_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match='reserved'):
            _author(store, name='list_tool_slots')

    def test_author_validation_failure_persists_draft(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        record = _author(store, code='w = await weather(place=missing_var)\nw')
        assert record.status == 'draft'
        assert record.last_error is not None
        assert store.load_servable() == []

    def test_author_replaces_by_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        _author(store, description='updated')
        records = store.list_all()
        assert len(records) == 1
        assert records[0].description == 'updated'

    def test_disable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        assert store.disable('forecast') is True
        assert store.disable('nope') is False
        assert store.load_servable() == []

    def test_load_servable_activates_then_stable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        first = store.load_servable()
        assert [r.name for r in first] == ['forecast']
        assert store.list_all()[0].status == 'active'
        # Second load: already active with no error, so nothing is rewritten.
        second = store.load_servable()
        assert [r.name for r in second] == ['forecast']

    def test_load_servable_skips_broken_and_records_error(self, tmp_path: Path) -> None:
        _author(_store(tmp_path))
        # A store over the same directory but with an empty pool: the slot's used
        # function is gone, so it fails re-validation and is not served.
        empty_pool_store = _store(tmp_path, functions=[])
        assert empty_pool_store.load_servable() == []
        broken = empty_pool_store.list_all()[0]
        assert broken.last_error is not None
        # Reloading with the same broken state does not rewrite the error.
        assert empty_pool_store.load_servable() == []

    def test_load_manifest_missing(self, tmp_path: Path) -> None:
        assert _store(tmp_path / 'absent').list_all() == []

    def test_load_manifest_corrupt(self, tmp_path: Path) -> None:
        (tmp_path / 'slots.json').write_text('not json{', encoding='utf-8')
        assert _store(tmp_path).list_all() == []

    def test_save_atomic_over_partial_prior_file(self, tmp_path: Path) -> None:
        (tmp_path / 'slots.json').write_text('{"slots": [', encoding='utf-8')
        store = _store(tmp_path)
        _author(store)
        assert [r.name for r in store.list_all()] == ['forecast']
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('slots.') and p.name.endswith('.tmp')]
        assert leftovers == []

    def test_save_cleans_temp_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        store = _store(tmp_path)
        _author(store)

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError('replace failed')

        monkeypatch.setattr(os, 'replace', _boom)
        with pytest.raises(OSError, match='replace failed'):
            _author(store, name='another')
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('slots.') and p.name.endswith('.tmp')]
        assert leftovers == []

    def test_duplicate_injected_function_name_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='names must be unique'):
            SlotStore[object](tmp_path, [WEATHER, WEATHER])

    def test_invalid_injected_function_name_rejected(self, tmp_path: Path) -> None:
        bad = InjectedFunction[object](name='not ok', call=weather, parameters={'type': 'object'})
        with pytest.raises(ValueError, match='not a valid Python identifier'):
            SlotStore[object](tmp_path, [bad])


# --------------------------------------------------------------------------- #
# Serving toolset (direct)
# --------------------------------------------------------------------------- #


class TestServingToolset:
    def _served(
        self, tmp_path: Path, functions: list[InjectedFunction[object]] | None = None
    ) -> ConfinedAuthoringToolset[object]:
        return ConfinedAuthoringToolset[object](store=_store(tmp_path, functions))

    async def test_runs_slot_in_sandbox(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        result = await _call_slot(
            ConfinedAuthoringToolset[object](store=store), 'forecast', {'city': 'Paris', 'days': 3}
        )
        assert result == {'summary': 'Paris: 21C', 'days': 3}

    async def test_default_id(self, tmp_path: Path) -> None:
        assert self._served(tmp_path).id == 'confined_authoring_slots'

    async def test_custom_id(self, tmp_path: Path) -> None:
        ts = ConfinedAuthoringToolset[object](store=_store(tmp_path), toolset_id='custom')
        assert ts.id == 'custom'

    async def test_only_active_slots_served(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        _author(store, name='broken', code='w = await weather(place=missing)\nw')  # draft
        ctx = _ctx()
        fresh = await ConfinedAuthoringToolset[object](store=store).for_run(ctx)
        tools = await fresh.get_tools(ctx)
        assert set(tools) == {'forecast'}

    async def test_served_tool_schema_reflects_parameters(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(
            store,
            name='search',
            code='q',
            parameters=[
                SlotParameter(name='q', type='string', description='the query'),
                SlotParameter(name='limit', type='integer', required=False),
            ],
            uses=[],
            returns=None,
        )
        ctx = _ctx()
        fresh = await ConfinedAuthoringToolset[object](store=store).for_run(ctx)
        schema = (await fresh.get_tools(ctx))['search'].tool_def.parameters_json_schema
        assert schema['properties']['q']['description'] == 'the query'
        assert schema['required'] == ['q']

    async def test_served_tool_forbids_unknown_argument(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        ctx = _ctx()
        fresh = await ConfinedAuthoringToolset[object](store=store).for_run(ctx)
        validator = (await fresh.get_tools(ctx))['forecast'].args_validator
        with pytest.raises(Exception):
            validator.validate_python({'city': 'X', 'days': 1, 'extra': 'nope'})

    async def test_runtime_error_becomes_retry(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store, name='explode', code='x = await boom()\nx', parameters=[], uses=['boom'], returns=None)
        with pytest.raises(ModelRetry, match='raised at runtime'):
            await _call_slot(ConfinedAuthoringToolset[object](store=store), 'explode', {})

    async def test_sandbox_panic_becomes_retry(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        code = 'import asyncio\nf = weather(place=city)\nawait asyncio.gather(f, f)'
        _author(store, name='dup', code=code, parameters=[SlotParameter(name='city')], uses=['weather'], returns=None)
        with pytest.raises(ModelRetry, match='aborted inside the sandbox'):
            await _call_slot(ConfinedAuthoringToolset[object](store=store), 'dup', {'city': 'X'})

    async def test_duration_cap_bounds_compute(self, tmp_path: Path) -> None:
        # A pure-CPU slot with no await would block the event loop until an external kill.
        # `_DEFAULT_LIMITS` caps in-sandbox compute (default 30s); a low `resource_limits`
        # override bounds this runaway as a retry instead of hanging the test.
        store = _store(tmp_path, functions=[])
        code = 'n = 0\nwhile n < 10_000_000_000:\n    n = n + 1\nn'
        _author(store, name='spin', code=code, parameters=[], uses=[], returns=None)
        toolset = ConfinedAuthoringToolset[object](store=store, resource_limits={'max_duration_secs': 0.1})
        with pytest.raises(ModelRetry, match='raised at runtime'):
            await _call_slot(toolset, 'spin', {})

    async def test_non_panic_base_exception_propagates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Boom(BaseException):
            pass

        class _FakeExecutor:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def run(self, _state: object) -> object:
                raise _Boom('not a panic')

        monkeypatch.setattr('pydantic_ai_harness.experimental.confined_authoring._toolset.MontyExecutor', _FakeExecutor)
        store = _store(tmp_path)
        _author(store)
        with pytest.raises(_Boom):
            await _call_slot(ConfinedAuthoringToolset[object](store=store), 'forecast', {'city': 'X', 'days': 1})

    async def test_return_type_mismatch_at_runtime(self, tmp_path: Path) -> None:
        # `greet` has no declared return schema (-> Any), so declaring an `integer`
        # return passes the static check; at runtime greet returns a string, which
        # the return guard rejects.
        store = _store(tmp_path)
        _author(
            store,
            name='as_int',
            code='x = await greet(name=city)\nx',
            parameters=[SlotParameter(name='city')],
            uses=['greet'],
            returns='integer',
        )
        result = await _call_slot(ConfinedAuthoringToolset[object](store=store), 'as_int', {'city': 'Ada'})
        assert isinstance(result, dict)
        assert 'not the declared' in result['error']

    async def test_optional_parameter_bound_to_none_when_absent(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        code = 'suffix if suffix is not None else "default"'
        _author(
            store,
            name='label',
            code=code,
            parameters=[SlotParameter(name='suffix', type='string', required=False)],
            uses=[],
            returns='string',
        )
        toolset = ConfinedAuthoringToolset[object](store=store)
        assert await _call_slot(toolset, 'label', {}) == 'default'
        assert await _call_slot(toolset, 'label', {'suffix': 'given'}) == 'given'

    async def test_print_and_result_shaping(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store, name='chatty', code='print("hi")\n{"x": 1}', parameters=[], uses=[], returns='object')
        result = await _call_slot(ConfinedAuthoringToolset[object](store=store), 'chatty', {})
        assert result == {'output': 'hi\n', 'result': {'x': 1}}

    async def test_print_only_shaping(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store, name='logger', code='print("hi")', parameters=[], uses=[], returns=None)
        result = await _call_slot(ConfinedAuthoringToolset[object](store=store), 'logger', {})
        assert result == {'output': 'hi\n'}

    async def test_none_result_shaping(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store, name='nothing', code='x = 1', parameters=[], uses=[], returns=None)
        result = await _call_slot(ConfinedAuthoringToolset[object](store=store), 'nothing', {})
        assert result == {}


# --------------------------------------------------------------------------- #
# Capability + agent
# --------------------------------------------------------------------------- #


def _authoring(tmp_path: Path, **overrides: Any) -> ConfinedAuthoring[object]:
    kwargs: dict[str, Any] = {'directory': tmp_path, 'functions': POOL}
    kwargs.update(overrides)
    return ConfinedAuthoring[object](**kwargs)


def _author_forecast_response() -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                'author_tool_slot',
                {
                    'name': 'forecast',
                    'description': 'Summarize the weather for a city.',
                    'code': FORECAST_CODE,
                    'parameters': [{'name': 'city', 'type': 'string'}, {'name': 'days', 'type': 'integer'}],
                    'uses': ['weather'],
                    'returns': 'object',
                },
            )
        ]
    )


def _has_return(messages: list[ModelMessage], name: str) -> bool:
    return any(isinstance(p, ToolReturnPart) and p.tool_name == name for m in messages for p in getattr(m, 'parts', []))


class TestCapability:
    def test_default_instructions_include_function_catalog(self, tmp_path: Path) -> None:
        instructions = _authoring(tmp_path).get_instructions()
        assert isinstance(instructions, str)
        assert 'author_tool_slot' in instructions
        assert 'async def weather' in instructions

    def test_default_instructions_without_functions(self, tmp_path: Path) -> None:
        instructions = _authoring(tmp_path, functions=[]).get_instructions()
        assert isinstance(instructions, str)
        assert 'No injected functions are available' in instructions

    def test_custom_guidance(self, tmp_path: Path) -> None:
        assert _authoring(tmp_path, guidance='do it').get_instructions() == 'do it'

    def test_empty_guidance_disables_instructions(self, tmp_path: Path) -> None:
        assert _authoring(tmp_path, guidance='').get_instructions() is None

    def test_store_property(self, tmp_path: Path) -> None:
        assert isinstance(_authoring(tmp_path).store, SlotStore)

    def test_not_spec_serializable(self) -> None:
        assert ConfinedAuthoring.get_serialization_name() is None

    def test_toolset_is_abstract_toolset(self, tmp_path: Path) -> None:
        toolset = _authoring(tmp_path).get_toolset()
        assert isinstance(toolset, AbstractToolset)


class TestAgentFlow:
    async def test_author_then_call_on_next_run(self, tmp_path: Path) -> None:
        def author_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _has_return(messages, 'author_tool_slot'):
                return ModelResponse(parts=[TextPart('done')])
            return _author_forecast_response()

        author_agent: Agent[object, str] = Agent(FunctionModel(author_model), capabilities=[_authoring(tmp_path)])
        await author_agent.run('author it')

        def caller_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _has_return(messages, 'forecast'):
                return ModelResponse(parts=[TextPart('called')])
            return ModelResponse(parts=[ToolCallPart('forecast', {'city': 'Paris', 'days': 2})])

        caller_agent: Agent[object, str] = Agent(FunctionModel(caller_model), capabilities=[_authoring(tmp_path)])
        result = await caller_agent.run('use it')
        forecast_return = next(
            p
            for m in result.all_messages()
            for p in getattr(m, 'parts', [])
            if isinstance(p, ToolReturnPart) and p.tool_name == 'forecast'
        )
        assert forecast_return.content == {'summary': 'Paris: 21C', 'days': 2}

    async def test_author_invalid_slot_reports_error(self, tmp_path: Path) -> None:
        captured: dict[str, str] = {}

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for m in messages:
                for p in getattr(m, 'parts', []):
                    if isinstance(p, ToolReturnPart) and p.tool_name == 'author_tool_slot':
                        captured['result'] = str(p.content)
                        return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'author_tool_slot',
                        {
                            'name': 'bad',
                            'description': 'd',
                            'code': 'w = await weather(place=1)\nw',
                            'uses': ['weather'],
                            'returns': 'object',
                        },
                    )
                ]
            )

        agent: Agent[object, str] = Agent(FunctionModel(model), capabilities=[_authoring(tmp_path)])
        await agent.run('author')
        assert 'failed validation' in captured['result']

    async def test_author_bad_name_retries(self, tmp_path: Path) -> None:
        calls = {'n': 0}

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _has_return(messages, 'author_tool_slot'):
                return ModelResponse(parts=[TextPart('done')])
            calls['n'] += 1
            name = 'Bad Name' if calls['n'] == 1 else 'good_name'
            return ModelResponse(
                parts=[ToolCallPart('author_tool_slot', {'name': name, 'description': 'd', 'code': '1'})]
            )

        agent: Agent[object, str] = Agent(FunctionModel(model), capabilities=[_authoring(tmp_path)])
        await agent.run('author')
        assert calls['n'] == 2

    async def test_list_and_disable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _author(store)
        _author(store, name='broken', code='w = await weather(place=missing)\nw')
        listing: dict[str, str] = {}

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _has_return(messages, 'disable_tool_slot'):
                return ModelResponse(parts=[TextPart('done')])
            if _has_return(messages, 'list_tool_slots'):
                for m in messages:
                    for p in getattr(m, 'parts', []):
                        if isinstance(p, ToolReturnPart) and p.tool_name == 'list_tool_slots':
                            listing['out'] = str(p.content)
                return ModelResponse(parts=[ToolCallPart('disable_tool_slot', {'name': 'forecast'})])
            return ModelResponse(parts=[ToolCallPart('list_tool_slots', {})])

        agent: Agent[object, str] = Agent(FunctionModel(model), capabilities=[_authoring(tmp_path)])
        await agent.run('manage')
        assert 'forecast' in listing['out']
        assert 'ERROR' in listing['out']  # the broken draft slot shows its error
        assert {r.name: r.status for r in store.list_all()}['forecast'] == 'disabled'

    async def test_list_when_empty(self, tmp_path: Path) -> None:
        result: dict[str, str] = {}

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _has_return(messages, 'list_tool_slots'):
                for m in messages:
                    for p in getattr(m, 'parts', []):
                        if isinstance(p, ToolReturnPart) and p.tool_name == 'list_tool_slots':
                            result['out'] = str(p.content)
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('list_tool_slots', {})])

        agent: Agent[object, str] = Agent(FunctionModel(model), capabilities=[_authoring(tmp_path)])
        await agent.run('list')
        assert 'No tool slots authored yet' in result['out']

    async def test_disable_unknown(self, tmp_path: Path) -> None:
        result: dict[str, str] = {}

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _has_return(messages, 'disable_tool_slot'):
                for m in messages:
                    for p in getattr(m, 'parts', []):
                        if isinstance(p, ToolReturnPart) and p.tool_name == 'disable_tool_slot':
                            result['out'] = str(p.content)
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('disable_tool_slot', {'name': 'ghost'})])

        agent: Agent[object, str] = Agent(FunctionModel(model), capabilities=[_authoring(tmp_path)])
        await agent.run('disable')
        assert 'No tool slot named' in result['out']

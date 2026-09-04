"""Tests for the public Modal sandbox capability."""

from __future__ import annotations

import inspect

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import SandboxRef
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness
import pydantic_ai_harness.modal_sandbox as modal_sandbox
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend

from .fake_modal import FakeModal

pytestmark = pytest.mark.anyio(backends=['asyncio'])


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _ctx(run_id: str = 'run-1', conversation_id: str | None = None) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id, conversation_id=conversation_id)


async def _resolved(capability: ModalSandbox[None], ctx: RunContext[None], ref: SandboxRef | None = None):
    """Ask the capability for a backend and touch it, so the create-or-attach actually happens."""
    backend = capability.get_sandbox(ctx, ref=ref)
    assert isinstance(backend, ModalSandboxBackend)
    await backend.sandbox
    return backend


class TestLifecycle:
    async def test_agent_tool_consumes_ctx_sandbox(self, fake_modal: FakeModal) -> None:
        async def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
            returns = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            if not returns:
                return ModelResponse(parts=[ToolCallPart('run_in_sandbox', {})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[ModalSandbox[None]()])

        @agent.tool
        async def run_in_sandbox(ctx: RunContext[None]) -> str:
            return (await ctx.sandbox.run(['echo', 'hello'])).stdout

        result = await agent.run('Use the sandbox.')

        assert result.output == 'done'
        assert len(fake_modal.sandboxes) == 1
        # Nothing terminates it: the conversation may continue in another run, and Modal reaps
        # an idle sandbox at its own `sandbox_timeout`.
        assert fake_modal.sandboxes[0].terminated is False

    async def test_building_a_backend_does_no_io(self, fake_modal: FakeModal) -> None:
        ModalSandbox[None]().get_sandbox(_ctx(), ref=None)

        assert fake_modal.sandboxes == []
        assert fake_modal.attach_ids == []

    async def test_one_conversation_reuses_one_sandbox(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None]()

        first = await _resolved(capability, _ctx('run-1', conversation_id='chat-1'))
        second = await _resolved(capability, _ctx('run-2', conversation_id='chat-1'))

        assert first.ref == second.ref
        assert len(fake_modal.sandboxes) == 1
        assert fake_modal.create_kwargs[0]['name'] is not None

    async def test_different_conversations_use_different_names(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None]()

        await _resolved(capability, _ctx('run-1', conversation_id='chat-1'))
        await _resolved(capability, _ctx('run-2', conversation_id='chat-2'))

        assert fake_modal.create_kwargs[0]['name'] != fake_modal.create_kwargs[1]['name']

    async def test_a_create_race_attaches_to_the_winner(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None]()
        existing = await _resolved(capability, _ctx(conversation_id='chat-1'))
        fake_modal.name_lookup_misses = 1

        connected = await _resolved(capability, _ctx(conversation_id='chat-1'))

        assert connected.ref == existing.ref
        assert fake_modal.name_lookups[-2:] == [
            ('pydantic-ai-harness', fake_modal.create_kwargs[0]['name']),
            ('pydantic-ai-harness', fake_modal.create_kwargs[0]['name']),
        ]

    async def test_a_ref_attaches_to_that_sandbox(self, fake_modal: FakeModal) -> None:
        await _resolved(ModalSandbox[None](), _ctx(), SandboxRef(sandbox_id='sb-existing'))

        assert fake_modal.attach_ids == ['sb-existing']
        assert fake_modal.create_kwargs == []

    async def test_a_configured_sandbox_id_attaches_and_creates_nothing(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None](sandbox_id='sb-existing')

        backend = await _resolved(capability, _ctx())

        assert backend.ref == SandboxRef(sandbox_id='sb-existing')
        assert fake_modal.create_kwargs == []


class TestConfiguration:
    def test_defaults(self) -> None:
        capability = ModalSandbox()

        assert capability.image == 'python:3.12-slim'
        assert capability.app_name == 'pydantic-ai-harness'
        assert capability.sandbox_timeout == 300

    def test_configuration_is_keyword_only(self) -> None:
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(ModalSandbox).parameters.values()
        )

    @pytest.mark.parametrize('timeout', [0, -1])
    def test_rejects_invalid_sandbox_timeout(self, timeout: object) -> None:
        with pytest.raises(ValueError, match='sandbox_timeout'):
            ModalSandbox(sandbox_timeout=timeout)  # type: ignore[arg-type]

    def test_rejects_relative_workdir(self) -> None:
        with pytest.raises(ValueError, match='workdir must be an absolute sandbox path'):
            ModalSandbox(workdir='repo')

    @pytest.mark.parametrize(
        'settings',
        [
            {'image': 'ubuntu:22.04'},
            {'app_name': 'other'},
            {'create_app_if_missing': False},
            {'sandbox_timeout': 600},
            {'workdir': '/work'},
            {'env': {'A': 'b'}},
        ],
    )
    def test_attach_rejects_creation_settings(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValueError, match='only apply when creating'):
            ModalSandbox(sandbox_id='sb-existing', **settings)  # type: ignore[arg-type]

    def test_copies_environment_mapping(self) -> None:
        source = {'A': 'one'}
        capability = ModalSandbox(env=source)
        source['A'] = 'two'

        assert capability.env == {'A': 'one'}

    def test_public_exports_are_narrow(self) -> None:
        assert pydantic_ai_harness.ModalSandbox is ModalSandbox
        assert set(modal_sandbox.__all__) == {
            'ModalSandbox',
            'ModalSandboxAuthError',
            'ModalSandboxBackend',
            'ModalSandboxError',
            'ModalSandboxUnavailableError',
        }

    def test_serialization_name(self) -> None:
        assert ModalSandbox.get_serialization_name() == 'ModalSandbox'

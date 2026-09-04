"""Tests for the Daytona sandbox capability."""

from __future__ import annotations

import inspect

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import SandboxRef
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness
import pydantic_ai_harness.daytona_sandbox as daytona_sandbox
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _ctx(run_id: str = 'run-1', conversation_id: str | None = None) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id, conversation_id=conversation_id)


async def _resolved(
    capability: DaytonaSandbox[None], ctx: RunContext[None], ref: SandboxRef | None = None
) -> DaytonaSandboxBackend:
    """Ask the capability for a backend and touch it, so the create-or-attach actually happens."""
    backend = capability.get_sandbox(ctx, ref=ref)
    assert isinstance(backend, DaytonaSandboxBackend)
    await backend.sandbox
    return backend


class TestLifecycle:
    async def test_building_a_backend_does_no_io(self, fake_daytona: FakeDaytona) -> None:
        DaytonaSandbox[None]().get_sandbox(_ctx(), ref=None)

        assert fake_daytona.sandboxes == []

    async def test_one_conversation_reuses_one_named_sandbox(self, fake_daytona: FakeDaytona) -> None:
        capability = DaytonaSandbox[None]()

        first = await _resolved(capability, _ctx('run-1', conversation_id='chat-1'))
        second = await _resolved(capability, _ctx('run-2', conversation_id='chat-1'))

        assert first.ref == second.ref
        assert first.ref == SandboxRef(sandbox_id=fake_daytona.sandboxes[0].id)
        assert len(fake_daytona.sandboxes) == 1

    async def test_different_conversations_use_different_names(self, fake_daytona: FakeDaytona) -> None:
        capability = DaytonaSandbox[None]()

        await _resolved(capability, _ctx('run-1', conversation_id='chat-1'))
        await _resolved(capability, _ctx('run-2', conversation_id='chat-2'))

        assert fake_daytona.create_params[0].name != fake_daytona.create_params[1].name

    async def test_a_ref_attaches_to_that_sandbox(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox()

        backend = await _resolved(DaytonaSandbox[None](), _ctx(), SandboxRef(sandbox_id=sandbox.id))

        assert backend.ref == SandboxRef(sandbox_id=sandbox.id)
        assert fake_daytona.create_params == []

    async def test_a_configured_sandbox_id_attaches_and_creates_nothing(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.sandbox('existing')

        backend = await _resolved(DaytonaSandbox[None](sandbox_id='existing'), _ctx())

        assert backend.ref == SandboxRef(sandbox_id='existing')
        assert fake_daytona.create_params == []

    async def test_nothing_deletes_the_sandbox_when_a_run_ends(self, fake_daytona: FakeDaytona) -> None:
        # A conversation spans many runs, so the end of one is not the end of the workspace.
        await _resolved(DaytonaSandbox[None](), _ctx(conversation_id='chat-1'))

        assert fake_daytona.sandboxes[0].deleted is False


class TestDeleteById:
    """`delete_by_id` is how an application ends a sandbox itself, without starting it first."""

    async def test_deletes_without_starting(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox('owned')

        await DaytonaSandboxBackend.delete_by_id('owned')

        assert sandbox.start_calls == []
        assert sandbox.deleted is True


class TestConfiguration:
    def test_defaults_and_normalization(self) -> None:
        capability = DaytonaSandbox(workdir='/workspace/../repo')
        assert capability.snapshot is None
        assert capability.auto_stop_minutes == 60
        assert capability.network_block_all is False
        assert capability.workdir == '/repo'

    def test_configuration_is_keyword_only(self) -> None:
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(DaytonaSandbox).parameters.values()
        )

    @pytest.mark.parametrize('minutes', [0, -1])
    def test_rejects_nonpositive_auto_stop(self, minutes: int) -> None:
        with pytest.raises(ValueError, match='auto_stop_minutes'):
            DaytonaSandbox(auto_stop_minutes=minutes)

    def test_rejects_relative_workdir(self) -> None:
        with pytest.raises(ValueError, match='absolute'):
            DaytonaSandbox(workdir='repo')

    @pytest.mark.parametrize(
        'settings',
        [
            {'snapshot': 'base'},
            {'auto_stop_minutes': 30},
            {'env': {'A': 'b'}},
            {'network_block_all': True},
        ],
    )
    def test_attach_rejects_creation_settings(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValueError, match='only apply when creating'):
            DaytonaSandbox(sandbox_id='existing', **settings)  # type: ignore[arg-type]

    def test_copies_environment_mapping(self) -> None:
        source = {'A': 'one'}
        capability = DaytonaSandbox(env=source)
        source['A'] = 'two'
        assert capability.env == {'A': 'one'}

    def test_public_surface(self) -> None:
        assert not hasattr(pydantic_ai_harness, 'DaytonaSandbox')
        assert set(daytona_sandbox.__all__) == {
            'DaytonaSandbox',
            'DaytonaSandboxAuthError',
            'DaytonaSandboxBackend',
            'DaytonaSandboxError',
            'DaytonaSandboxUnavailableError',
        }
        assert DaytonaSandbox.get_serialization_name() == 'DaytonaSandbox'

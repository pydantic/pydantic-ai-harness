"""Tests for the public E2B sandbox capability."""

from __future__ import annotations

import inspect

import anyio
import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import SandboxRef, SandboxUnavailableError
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness
import pydantic_ai_harness.e2b_sandbox as e2b_sandbox
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxAuthError, E2BSandboxBackend

from .fake_e2b import FakeE2B


def _ctx(run_id: str = 'run-1', conversation_id: str | None = None) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id, conversation_id=conversation_id)


async def _resolved(
    capability: E2BSandbox[None], ctx: RunContext[None], ref: SandboxRef | None = None
) -> E2BSandboxBackend:
    """Ask the capability for a backend and touch it, so the create-or-attach actually happens."""
    backend = capability.get_sandbox(ctx, ref=ref)
    assert isinstance(backend, E2BSandboxBackend)
    await backend.sandbox
    return backend


class TestLifecycle:
    async def test_building_a_backend_does_no_io(self, fake_e2b: FakeE2B) -> None:
        E2BSandbox[None]().get_sandbox(_ctx(), ref=None)

        assert fake_e2b.create_calls == []
        assert fake_e2b.list_calls == []
        assert fake_e2b.connect_calls == []

    async def test_one_conversation_reuses_one_sandbox(self, fake_e2b: FakeE2B) -> None:
        capability = E2BSandbox[None]()

        first = await _resolved(capability, _ctx('run-1', conversation_id='chat-1'))
        second = await _resolved(capability, _ctx('run-2', conversation_id='chat-1'))

        assert first.ref == second.ref
        assert len(fake_e2b.create_calls) == 1
        assert fake_e2b.create_calls[0].metadata == {'pydantic-ai-conversation-id': 'chat-1'}

    async def test_user_metadata_is_preserved(self, fake_e2b: FakeE2B) -> None:
        await _resolved(E2BSandbox[None](metadata={'owner': 'tests'}), _ctx(conversation_id='chat-1'))

        assert fake_e2b.create_calls[0].metadata == {
            'owner': 'tests',
            'pydantic-ai-conversation-id': 'chat-1',
        }

    async def test_different_conversations_create_different_sandboxes(self, fake_e2b: FakeE2B) -> None:
        capability = E2BSandbox[None]()

        first = await _resolved(capability, _ctx('run-1', conversation_id='chat-1'))
        second = await _resolved(capability, _ctx('run-2', conversation_id='chat-2'))

        assert first.ref != second.ref
        assert len(fake_e2b.create_calls) == 2

    async def test_the_oldest_matching_sandbox_wins(self, fake_e2b: FakeE2B) -> None:
        metadata = {'pydantic-ai-conversation-id': 'chat-1'}
        oldest = fake_e2b.new_sandbox('oldest', metadata)
        newest = fake_e2b.new_sandbox('newest', metadata)
        latest = fake_e2b.new_sandbox('latest', metadata)
        # E2B 2.34 cannot order this query server-side, so exercise an unordered response.
        fake_e2b.sandboxes[:] = [newest, oldest, latest]

        backend = await _resolved(E2BSandbox[None](), _ctx(conversation_id='chat-1'))

        assert backend.ref == SandboxRef(sandbox_id='oldest')
        assert fake_e2b.list_calls == [(metadata, None)]

    async def test_a_concurrent_creator_loses_to_the_oldest_sandbox(self, fake_e2b: FakeE2B) -> None:
        canonical = fake_e2b.new_sandbox('canonical', {'pydantic-ai-conversation-id': 'chat-1'})
        fake_e2b.list_batches = [[], [canonical]]

        backend = await _resolved(E2BSandbox[None](), _ctx(conversation_id='chat-1'))

        assert backend.ref == SandboxRef(sandbox_id=canonical.sandbox_id)
        assert fake_e2b.sandboxes[1].killed is True

    async def test_a_fresh_create_keeps_its_original_handle(self, fake_e2b: FakeE2B) -> None:
        backend = await _resolved(E2BSandbox[None](), _ctx(conversation_id='chat-1'))

        assert backend.ref == SandboxRef(sandbox_id=fake_e2b.sandboxes[0].sandbox_id)
        assert fake_e2b.connect_calls == []

    @pytest.mark.parametrize('unavailable', [True, False])
    async def test_list_failure_uses_public_error_surface(self, fake_e2b: FakeE2B, unavailable: bool) -> None:
        fake_e2b.list_error = fake_e2b.auth_type('bad key') if unavailable else RuntimeError('offline')

        error_type = SandboxUnavailableError if unavailable else e2b_sandbox.E2BSandboxError
        with pytest.raises(error_type):
            await _resolved(E2BSandbox[None](), _ctx(conversation_id='chat-1'))

    async def test_a_ref_attaches_to_that_sandbox(self, fake_e2b: FakeE2B) -> None:
        await _resolved(E2BSandbox[None](), _ctx(), SandboxRef(sandbox_id='sbx-existing'))

        assert fake_e2b.connect_calls == [('sbx-existing', None)]
        assert fake_e2b.create_calls == []

    async def test_a_configured_sandbox_id_attaches_and_creates_nothing(self, fake_e2b: FakeE2B) -> None:
        backend = await _resolved(E2BSandbox[None](sandbox_id='sbx-existing'), _ctx())

        assert backend.ref == SandboxRef(sandbox_id='sbx-existing')
        assert fake_e2b.create_calls == []

    async def test_nothing_kills_the_sandbox_when_a_run_ends(self, fake_e2b: FakeE2B) -> None:
        # A conversation spans many runs, so the end of one is not the end of the workspace.
        await _resolved(E2BSandbox[None](), _ctx(conversation_id='chat-1'))

        assert fake_e2b.kill_ids == []
        assert fake_e2b.sandboxes[0].killed is False


class TestKillById:
    """`kill_by_id` is how an application ends a sandbox itself, without reconnecting first."""

    async def test_kills_without_reconnecting(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('owned')

        await E2BSandboxBackend.kill_by_id('owned')

        assert fake_e2b.kill_ids == ['owned']
        assert fake_e2b.connect_calls == []
        assert fake_e2b.sandboxes[0].killed is True

    async def test_is_idempotent_when_the_sandbox_is_gone(self, fake_e2b: FakeE2B) -> None:
        await E2BSandboxBackend.kill_by_id('gone')

        assert fake_e2b.kill_ids == ['gone']

    async def test_is_idempotent_when_the_sandbox_was_already_killed(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('owned')

        await E2BSandboxBackend.kill_by_id('owned')
        await E2BSandboxBackend.kill_by_id('owned')

        assert fake_e2b.kill_ids == ['owned', 'owned']

    async def test_is_bounded(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        fake_e2b.kill_hangs = True

        with pytest.raises(e2b_sandbox.E2BSandboxError, match='Timed out'):
            await E2BSandboxBackend.kill_by_id('sbx-hung')

    async def test_auth_failure_is_terminal(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.kill_error = fake_e2b.auth_type('bad key')

        with pytest.raises(E2BSandboxAuthError, match='E2B rejected the credentials'):
            await E2BSandboxBackend.kill_by_id('sbx-owned')

    async def test_completes_under_cancellation(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-owned')
        fake_e2b.kill_gate = anyio.Event()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(E2BSandboxBackend.kill_by_id, 'sbx-owned')
            while not fake_e2b.kill_started:
                await anyio.sleep(0)
            task_group.cancel_scope.cancel()
            fake_e2b.kill_gate.set()

        assert fake_e2b.sandboxes[0].killed is True

    async def test_failure_does_not_replace_cancellation(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.kill_gate = anyio.Event()
        fake_e2b.kill_error = RuntimeError('cleanup failed')

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(E2BSandboxBackend.kill_by_id, 'sbx-owned')
            while not fake_e2b.kill_started:
                await anyio.sleep(0)
            task_group.cancel_scope.cancel()
            fake_e2b.kill_gate.set()


class TestConfiguration:
    def test_defaults(self) -> None:
        capability = E2BSandbox()

        assert capability.template is None
        assert capability.sandbox_timeout == 300
        assert capability.allow_internet_access is True

    def test_configuration_is_keyword_only(self) -> None:
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(E2BSandbox).parameters.values()
        )

    @pytest.mark.parametrize('timeout', [0, -1])
    def test_rejects_invalid_sandbox_timeout(self, timeout: int) -> None:
        with pytest.raises(ValueError, match='sandbox_timeout'):
            E2BSandbox(sandbox_timeout=timeout)

    def test_rejects_relative_workdir(self) -> None:
        with pytest.raises(ValueError, match='absolute'):
            E2BSandbox(workdir='repo')

    def test_normalizes_absolute_workdir(self) -> None:
        assert E2BSandbox(workdir='/workspace/../repo').workdir == '/repo'

    def test_reserves_the_conversation_metadata_key(self) -> None:
        with pytest.raises(ValueError, match='reserved'):
            E2BSandbox(metadata={'pydantic-ai-conversation-id': 'mine'})

    @pytest.mark.parametrize(
        'settings',
        [
            {'template': 'base'},
            {'sandbox_timeout': 600},
            {'env': {'A': 'b'}},
            {'metadata': {'owner': 'tests'}},
            {'allow_internet_access': False},
        ],
    )
    def test_attach_rejects_creation_settings(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValueError, match='only apply when creating'):
            E2BSandbox(sandbox_id='sbx-existing', **settings)  # type: ignore[arg-type]

    def test_copies_input_mappings(self) -> None:
        env = {'A': 'one'}
        metadata = {'owner': 'one'}
        capability = E2BSandbox(env=env, metadata=metadata)
        env['A'] = 'two'
        metadata['owner'] = 'two'

        assert capability.env == {'A': 'one'}
        assert capability.metadata == {'owner': 'one'}

    def test_public_exports_are_narrow(self) -> None:
        assert not hasattr(pydantic_ai_harness, 'E2BSandbox')
        assert set(e2b_sandbox.__all__) == {
            'E2BSandbox',
            'E2BSandboxAuthError',
            'E2BSandboxBackend',
            'E2BSandboxError',
            'E2BSandboxUnavailableError',
        }

    def test_serialization_name(self) -> None:
        assert E2BSandbox.get_serialization_name() == 'E2BSandbox'

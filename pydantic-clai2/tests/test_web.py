"""Exercise the browser's public ASGI endpoints with real plugin composition."""

import asyncio
import io
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Self

import anyio
import httpx2
import pytest
from anyio.lowlevel import checkpoint
from pydantic_ai import ModelRequestContext, ModelRetry, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from rich.console import Console

from pydantic_clai2 import web
from pydantic_clai2.auth import CodexAuth
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.plugin_loader import PluginError
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart, TurnEnd, TurnStart
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.web import web_app

CHAT = {
    'id': 'chat-1',
    'trigger': 'submit-message',
    'messages': [{'id': 'user-1', 'role': 'user', 'parts': [{'type': 'text', 'text': 'hello'}]}],
}


def install_plugin(
    monkeypatch: pytest.MonkeyPatch,
    store: SettingsStore,
    activate: Callable[[PluginHost[None]], None],
    *,
    name: str = 'web_test_plugin',
) -> None:
    module = ModuleType(name)
    monkeypatch.setattr(module, 'activate', activate, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    store.save_plugin(PluginSettings(id=name, factory=name))


async def test_stream_plugins_settings_retries_and_coding_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings('test', {'temperature': 0.25, 'max_tokens': 128})
    output = io.StringIO()
    loop = asyncio.get_running_loop()
    events: list[str] = []
    attempts: list[int] = []

    def activate(host: PluginHost[None]) -> None:
        @host.on('session_start')
        async def start(event: SessionStart) -> None:
            assert asyncio.get_running_loop() is loop
            events.append('start')

        @host.on('session_end')
        async def end(event: SessionEnd) -> None:
            await checkpoint()
            assert asyncio.get_running_loop() is loop
            events.append(event.reason)

        @host.on('before_model_request')
        async def guard(ctx: RunContext[None], request: ModelRequestContext) -> ModelRequestContext:
            assert ctx.model_settings == {'temperature': 0.25, 'max_tokens': 128}
            assert ctx.usage_limits is not None
            assert ctx.usage_limits.request_limit == 50
            events.append('guard')
            return request

        @host.on('prepare_tools')
        async def safe_tools(ctx: RunContext[None], tools: list[ToolDefinition]) -> list[ToolDefinition]:
            names = {tool.name for tool in tools}
            assert {'shell', 'read_file', 'write_file', 'edit_file', 'echo'} <= names
            assert 'ask_user' not in names
            return [tool for tool in tools if tool.name == 'echo']

        async def echo(ctx: RunContext[None]) -> str:
            assert asyncio.get_running_loop() is loop
            attempts.append(ctx.retry)
            if ctx.retry < 2:
                raise ModelRetry('retry echo')
            return 'plugin response'

        async def dynamic(ctx: RunContext[None]) -> Capability[None]:
            return Capability(tools=[echo])

        host.add(dynamic)
        host.add(lambda ctx: Capability(instructions='Use the configured tools.'))
        host.add(lambda ctx: None)

    install_plugin(monkeypatch, store, activate)
    updates = PluginSettings(id='updates', factory='pydantic_clai2.updates')
    store.save_plugin(updates)
    saved = store.plugins()
    async with web_app(
        settings=Settings(model='test', tool_retries=2),
        store=store,
        project=ProjectSettings(),
        console=Console(file=output),
    ) as app:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            config = await client.get('/api/configure')
            assert config.status_code == 200
            assert config.json()['models'][0]['id'] == 'test:test'
            async with client.stream('POST', '/api/chat', json=CHAT) as response:
                assert response.status_code == 200
                assert response.headers['content-type'].startswith('text/event-stream')
                stream = '\n'.join([line async for line in response.aiter_lines()])
            assert 'plugin response' in stream
            assert 'data: [DONE]' in stream
        assert events[0] == 'start'
        assert 'exit' not in events
    assert events[-1] == 'exit'
    assert events.count('guard') >= 2
    assert attempts == [0, 1, 2]
    assert store.plugins() == saved
    for name in ('ask_user', 'persistence', 'notifications', 'updates'):
        assert f'Omitting terminal plugin {name!r}' in output.getvalue()
    assert not (tmp_path / 'sessions.db').exists()


async def test_browser_runs_share_one_execution_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    first_started, second_submitted, release = (anyio.Event() for _ in range(3))
    entered: list[int] = []
    responses: list[httpx2.Response] = []

    def activate(host: PluginHost[None]) -> None:
        async def echo() -> str:
            entered.append(len(entered) + 1)
            if len(entered) == 1:
                first_started.set()
                await release.wait()
            return 'done'

        host.add(Capability(tools=[echo]))

    install_plugin(monkeypatch, store, activate)
    with anyio.fail_after(20):
        async with web_app(
            settings=Settings(model='test'), store=store, project=ProjectSettings(), builtin_plugins=()
        ) as app:
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url='http://127.0.0.1'
            ) as client:

                async def request(second: bool) -> None:
                    if second:
                        second_submitted.set()
                    responses.append(await client.post('/api/chat', json=CHAT))

                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(request, False)
                    await first_started.wait()
                    tasks.start_soon(request, True)
                    await second_submitted.wait()
                    await anyio.wait_all_tasks_blocked()
                    assert entered == [1]
                    release.set()
    assert entered == [1, 2]
    assert len(responses) == 2
    assert all(response.status_code == 200 and 'data: [DONE]' in response.text for response in responses)


async def test_security_and_model_allowlist(tmp_path: Path) -> None:
    async with web_app(
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
        project=ProjectSettings(),
        builtin_plugins=(),
    ) as app:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            for path in ('/', '/api/configure', '/api/health', '/api/chat'):
                assert (await client.get(path, headers={'Host': 'attacker.example'})).status_code == 421
            assert (await client.get('/api/health', headers={'Host': 'localhost'})).status_code == 200
            for content_type in ('text/plain', 'application/x-www-form-urlencoded', 'multipart/form-data', ''):
                response = await client.post('/api/chat', content='{}', headers={'Content-Type': content_type})
                assert response.status_code == 415
            response = await client.options(
                '/api/chat',
                headers={
                    'Origin': 'https://attacker.example',
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'content-type',
                },
            )
            assert not any(key.startswith('access-control-allow-') for key in response.headers)
            response = await client.post('/api/chat', json=CHAT | {'model': 'openai:unconfigured'})
            assert response.status_code == 400
            assert (await client.post('/api/chat', json=CHAT | {'builtinTools': ['web_search']})).status_code == 400


@pytest.mark.parametrize('hook', ['turn_start', 'turn_end', 'widget', 'activation'])
async def test_reject_incompatible_plugins_and_close_loaded_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hook: str,
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    events: list[str] = []

    def first(host: PluginHost[None]) -> None:
        @host.on('session_end')
        async def end(event: SessionEnd) -> None:
            await checkpoint()
            events.append(event.reason)

    def incompatible(host: PluginHost[None]) -> None:
        async def guard(event: TurnStart | TurnEnd) -> None:
            raise AssertionError('Must not serve this plugin')

        if hook == 'turn_start':
            host.on('turn_start')(guard)
        elif hook == 'turn_end':
            host.on('turn_end')(guard)
        elif hook == 'widget':
            host.full_screen()
        else:
            raise ValueError('activation failed')

    install_plugin(monkeypatch, store, first, name='aaa_first')
    # Replacing a UI-only builtin id must not hide a custom guard plugin.
    install_plugin(monkeypatch, store, incompatible, name='notifications')
    with pytest.raises(PluginError, match='notifications'):
        async with web_app(settings=Settings(model='test'), store=store, project=ProjectSettings(), builtin_plugins=()):
            pytest.fail('An incompatible plugin must prevent serving')
    assert events == ['error']
    assert all(plugin.enabled for plugin in store.plugins())


@pytest.mark.parametrize('outcome', ['normal', 'error', 'cancel'])
async def test_context_cleanup_on_failure_and_outer_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    events: list[str] = []

    def activate(host: PluginHost[None]) -> None:
        @host.on('session_end')
        async def end(event: SessionEnd) -> None:
            await checkpoint()
            events.append(event.reason)

    class TrackedModel(TestModel):
        async def __aenter__(self) -> Self:
            events.append('model enter')
            return self

        async def __aexit__(self, *args: object) -> None:
            await checkpoint()
            events.append('model exit')

    async def resolve(name: str, *, auth: CodexAuth) -> TestModel:
        return TrackedModel(call_tools=[])

    monkeypatch.setattr(web, 'resolve_model', resolve)
    install_plugin(monkeypatch, store, activate)
    with pytest.raises(ValueError, match='server failed') if outcome == 'error' else anyio.CancelScope() as scope:
        async with web_app(settings=Settings(model='test'), store=store, project=ProjectSettings(), builtin_plugins=()):
            if outcome == 'error':
                raise ValueError('server failed')
            if outcome == 'cancel':
                assert isinstance(scope, anyio.CancelScope)
                scope.cancel()
                await checkpoint()
    assert events == ['model enter', 'exit' if outcome == 'normal' else 'error', 'model exit']


async def test_project_plugins_need_approval_and_store_overrides_builtins(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='coder', factory='missing', enabled=False))
    project = ProjectSettings(plugins=(PluginSettings(id='untrusted', factory='missing', enabled=False),))
    async with web_app(settings=Settings(model='test'), store=store, project=project) as app:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            response = await client.post('/api/chat', json=CHAT)
    assert '"type":"text-delta"' in response.text
    assert '"type":"error"' not in response.text
    assert '"type":"tool-input-start"' not in response.text
    assert len(store.plugins()) == 1
    assert not store.plugins()[0].enabled


async def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Choose a model'):
        async with web_app(
            settings=Settings(model=None),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
        ):
            pytest.fail('No model configured')

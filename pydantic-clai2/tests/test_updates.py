"""The real unpublished endpoint, protocol edge cases, and plugin-owned workers."""

import asyncio
import io
import json
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from importlib import metadata
from pathlib import Path

import anyio
import httpx2
import pytest
from anyio.lowlevel import checkpoint
from cassetter import Cassetter
from packaging.version import Version
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, updates
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart
from pydantic_clai2.screen import Screen
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.updates import activate, latest_version

READINESS_TIMEOUT = 10


@pytest.fixture(scope='module')
def vcr_config() -> Cassetter:
    return Cassetter(intercept=['httpx2'])


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'pydantic_clai2-0.1.0.dist-info'
    path.mkdir()
    (path / 'METADATA').write_text('Name: pydantic-clai2\nVersion: 0.1.0\n')
    original = metadata.distribution

    def distribution(name: str) -> metadata.Distribution:
        return metadata.PathDistribution(path) if name == 'pydantic-clai2' else original(name)

    monkeypatch.setattr(metadata, 'distribution', distribution)
    return path


@pytest.mark.vcr
async def test_pypi_currently_has_no_published_clai2() -> None:
    assert await latest_version() is None


@pytest.mark.parametrize(
    ('version', 'files', 'expected'),
    [
        ('0.2.0', [{'yanked': False}], '0.2.0'),
        ('1!2.0.post1', [{'yanked': False, 'requires_python': '>=3.11'}], '1!2.0.post1'),
        ('v0.2', [{'yanked': False}], '0.2'),
        ('0.2.0rc1', [{'yanked': False}], None),
        ('0.2.0.dev1', [{'yanked': False}], None),
        ('0.2.0+local', [{'yanked': False}], None),
        ('not a version', [{'yanked': False}], None),
        ('1.' * 100 + '1', [{'yanked': False}], None),
        ('0.2.0', [], None),
        ('0.2.0', [{'yanked': True}], None),
        ('0.2.0', [{'yanked': True}, {'yanked': False}], '0.2.0'),
        ('0.2.0', [{'yanked': False, 'requires_python': '>=999'}], None),
        ('0.2.0', [{'yanked': False, 'requires_python': 'invalid'}], None),
    ],
)
async def test_future_release_metadata(version: str, files: list[dict[str, bool | str]], expected: str | None) -> None:
    body = json.dumps({'info': {'version': version}, 'urls': files}).encode()

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == 'https://pypi.org/pypi/pydantic-clai2/json'
        assert request.headers['accept-encoding'] == 'identity'
        return httpx2.Response(200, stream=httpx2.ByteStream(body))

    result = await latest_version(transport=httpx2.MockTransport(respond))
    assert result == (Version(expected) if expected else None)


@pytest.mark.parametrize('status', [301, 404, 429, 500])
async def test_unsuccessful_status_is_silent(status: int) -> None:
    assert await latest_version(transport=httpx2.MockTransport(lambda _: httpx2.Response(status))) is None


@pytest.mark.parametrize(
    'body',
    [
        b'not json',
        b'\xff',
        b'null',
        b'[]',
        b'{}',
        b'{"info": []}',
        b'{"info":{"version":2},"urls":[]}',
        b'{"info":{"version":"2"},"urls":[{}]}',
        b'{"info":{"version":"2"},"urls":[{"yanked":"false"}]}',
        b' ' * (1024 * 1024 + 1),
    ],
)
async def test_malformed_or_oversized_response_is_silent(body: bytes) -> None:
    assert (
        await latest_version(
            transport=httpx2.MockTransport(lambda _: httpx2.Response(200, stream=httpx2.ByteStream(body)))
        )
        is None
    )


@pytest.mark.parametrize('error', [httpx2.ConnectError('offline'), httpx2.ReadTimeout('slow'), TimeoutError()])
async def test_network_failure_is_silent(error: Exception) -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise error

    assert await latest_version(transport=httpx2.MockTransport(fail)) is None


async def test_total_deadline_closes_a_stalled_response(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = anyio.Event()
    original = anyio.fail_after

    class StalledStream(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{'
            await anyio.sleep_forever()

        async def aclose(self) -> None:
            closed.set()

    def deadline(seconds: float) -> AbstractContextManager[anyio.CancelScope]:
        assert seconds == 5
        return original(0)

    monkeypatch.setattr(anyio, 'fail_after', deadline)
    assert (
        await latest_version(transport=httpx2.MockTransport(lambda _: httpx2.Response(200, stream=StalledStream())))
        is None
    )
    assert closed.is_set()


@pytest.mark.parametrize(
    ('current', 'latest', 'notice'),
    [
        ('0.1.0', '0.2.0', True),
        ('0.9.0', '0.10.0', True),
        ('0.2.0rc1', '0.2.0', True),
        ('0.2.0', '0.2.0', False),
        ('0.3.0', '0.2.0', False),
        ('1!0.1', '9.0', False),
        ('0.2.0+local', '0.2.0', False),
        ('0.1.0', None, False),
    ],
)
async def test_plugin_notice_is_once_per_activation(
    installed: Path, current: str, latest: str | None, notice: bool
) -> None:
    (installed / 'METADATA').write_text(f'Name: pydantic-clai2\nVersion: {current}\n')
    output = io.StringIO()
    host: PluginHost[None] = PluginHost(name='updates', console=Console(file=output, width=300), settings={})
    finished = anyio.Event()

    async def check() -> Version | None:
        finished.set()
        return Version(latest) if latest else None

    activate(host, check=check)
    for handler in host.handlers:
        await handler(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    with anyio.fail_after(READINESS_TIMEOUT):
        await finished.wait()
    for handler in host.handlers:
        await handler(SessionEnd(reason='exit'))
    text = output.getvalue()
    assert text.count('available (installed') == int(notice)
    if notice:
        assert f'CLAI2 {latest} available (installed {current})' in text
        assert 'original installer' in text
    else:
        assert text == ''


@pytest.mark.parametrize('failure', ['missing', 'no-version', 'invalid', 'io', 'offline', 'timeout'])
async def test_missing_metadata_and_failed_check_are_harmless(
    installed: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == 'no-version':
        (installed / 'METADATA').write_text('Name: pydantic-clai2\n')
    elif failure == 'invalid':
        (installed / 'METADATA').write_text('Name: pydantic-clai2\nVersion: broken\n')
    elif failure in ('missing', 'io'):

        def unavailable(name: str) -> metadata.Distribution:
            raise metadata.PackageNotFoundError(name) if failure == 'missing' else OSError('unreadable')

        monkeypatch.setattr(metadata, 'distribution', unavailable)

    async def check() -> Version | None:
        raise httpx2.ConnectError('offline') if failure == 'offline' else TimeoutError()

    output = io.StringIO()
    host: PluginHost[None] = PluginHost(name='updates', console=Console(file=output), settings={})
    activate(host, check=check)
    for handler in host.handlers:
        await handler(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    await anyio.wait_all_tasks_blocked()
    for handler in host.handlers:
        await handler(SessionEnd(reason='exit'))
    assert output.getvalue() == ''


async def test_stopping_an_activated_host_before_start_is_a_noop() -> None:
    output = io.StringIO()
    host: PluginHost[None] = PluginHost(name='updates', console=Console(file=output), settings={})
    activate(host)
    for handler in host.handlers:
        await handler(SessionEnd(reason='error'))
    assert output.getvalue() == ''


@pytest.mark.parametrize('cancel_outer', [False, True])
async def test_loader_owns_workers_and_persists_opt_out(
    installed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_outer: bool
) -> None:
    before = asyncio.all_tasks()
    started = anyio.Event()
    cleaned = anyio.Event()
    calls = 0

    async def check() -> Version | None:
        nonlocal calls
        calls += 1
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                await checkpoint()
                cleaned.set()

    monkeypatch.setattr(updates, 'latest_version', check)
    store = SettingsStore(tmp_path / 'settings.db')
    output = io.StringIO()
    declaration = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'updates')
    assert declaration.enabled
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=output),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=Settings()),
        builtin=[declaration],
    )
    with anyio.CancelScope() as outer:
        try:
            await loader.load_all()
            assert not started.is_set()
            with anyio.fail_after(READINESS_TIMEOUT):
                await started.wait()
            assert 'updates:' in await loader.command(['list'])
            await loader.command(['disable', 'updates'])
            assert cleaned.is_set()
            assert not store.plugins()[0].enabled
            assert asyncio.all_tasks() == before
            started = anyio.Event()
            cleaned = anyio.Event()
            await loader.command(['enable', 'updates'])
            with anyio.fail_after(READINESS_TIMEOUT):
                await started.wait()
            assert calls == 2
            if cancel_outer:
                outer.cancel()
                await checkpoint()
        finally:
            await loader.close('exit')
    assert cleaned.is_set()
    assert asyncio.all_tasks() == before
    assert output.getvalue() == ''


async def test_unload_cancels_a_notice_waiting_for_a_menu(installed: Path) -> None:
    output = io.StringIO()
    console = Console(file=output)
    screen = Screen()
    ready = anyio.Event()

    async def check() -> Version | None:
        ready.set()
        return Version('0.2.0')

    host: PluginHost[None] = PluginHost(
        name='updates', console=console, settings={}, notify=lambda text: screen.notify(text, console=console)
    )
    activate(host, check=check)
    with screen.busy():
        for handler in host.handlers:
            await handler(SessionStart(agent=Agent(TestModel()), settings=Settings()))
        with anyio.fail_after(READINESS_TIMEOUT):
            await ready.wait()
        assert output.getvalue() == ''
        for handler in host.handlers:
            await handler(SessionEnd(reason='exit'))
    assert output.getvalue() == ''

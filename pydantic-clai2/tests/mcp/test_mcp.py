"""MCP approval, plugin ownership, and real stdio calls through Agent."""

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from anyio.abc import SocketAttribute, SocketStream
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from termflow.tui.completion import (  # pyright: ignore[reportMissingTypeStubs]
    CompleteEvent,
    Document,
)
from test_plugin_loader import Harness

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.plugin_loader import PluginError

READY_TIMEOUT = 30


def write_config(directory: Path, *, prefix: str = 'local', port: int = 0) -> Path:
    path = directory / '.mcp.json'
    path.write_text(
        json.dumps(
            {
                'mcpServers': {
                    prefix: {
                        'command': sys.executable,
                        'args': [str(Path(__file__).with_name('mcp_server.py'))],
                        'cwd': str(directory),
                        'env': {
                            'CLAI_MCP_TOKEN': '${CLAI_MCP_SOURCE:-stdio-token}',
                            'CLAI_MCP_PID_FILE': str(directory / 'server.pid'),
                            'CLAI_MCP_READY_PORT': str(port),
                        },
                    }
                }
            }
        )
    )
    return path


def assert_stopped(directory: Path) -> None:
    pid = int((directory / 'server.pid').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.fixture
async def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Harness]:
    monkeypatch.chdir(tmp_path)
    plugin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'mcp')
    value = Harness(tmp_path, builtin=(plugin,))
    await value.loader.load_all()
    try:
        yield value
    finally:
        await value.loader.close('exit')


class TestMCPPlugin:
    async def test_default_and_approval(self, harness: Harness, tmp_path: Path) -> None:
        write_config(tmp_path)
        model = TestModel(call_tools=[])
        agent = Agent(model, deps_type=type(None))
        await agent.run('No connections', capabilities=harness.loader.capabilities())
        assert model.last_model_request_parameters is not None
        assert model.last_model_request_parameters.function_tools == []
        assert 'not loaded' in await harness.commands.execute_async('/mcp')
        assert str(tmp_path / '.mcp.json') in await harness.commands.execute_async('/mcp status')
        warning = await harness.commands.execute_async('/mcp load')
        assert 'execute commands' in warning and '/mcp load --approve' in warning
        assert not (tmp_path / 'server.pid').exists()
        assert harness.store.plugins() == []
        assert 'Loaded 1 MCP server(s)' in await harness.commands.execute_async('/mcp load --approve')
        assert not (tmp_path / 'server.pid').exists()
        assert '1 server(s) loaded' in await harness.commands.execute_async('/mcp status')
        assert harness.store.plugins() == []
        with pytest.raises(ValueError, match='Usage: /mcp'):
            await harness.commands.execute_async('/mcp load relative.json')
        for text, expected in [('/mcp ', ['status', 'load']), ('/mcp load ', ['--approve'])]:
            assert [item.text for item in harness.commands.get_completions(Document(text), CompleteEvent())] == expected

    async def test_stdio_calls_replace_and_plugin_lifecycle(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('CLAI_MCP_SOURCE', 'expanded-token')
        write_config(tmp_path)
        await harness.commands.execute_async('/mcp load --approve')
        model = TestModel(call_tools=['local_context'])
        agent = Agent(model, deps_type=type(None))
        for _ in range(2):
            result = await agent.run('Read context', capabilities=harness.loader.capabilities())
            assert 'expanded-token' in result.output and str(tmp_path) in result.output
            assert model.last_model_request_parameters is not None
            assert {tool.name for tool in model.last_model_request_parameters.function_tools} == {
                'local_context',
                'local_wait',
            }
            assert_stopped(tmp_path)
        write_config(tmp_path, prefix='replacement')
        await harness.commands.execute_async('/mcp load --approve')
        result = await Agent(TestModel(call_tools=['replacement_context']), deps_type=type(None)).run(
            'Use replacement', capabilities=harness.loader.capabilities()
        )
        assert 'replacement_context' in result.output and 'local_context' not in result.output
        assert_stopped(tmp_path)
        await harness.loader.reload('mcp')
        assert 'not loaded' in await harness.commands.execute_async('/mcp status')
        await harness.commands.execute_async('/mcp load --approve')
        await harness.loader.disable('mcp')
        assert harness.loader.capabilities() == []
        with pytest.raises(ValueError, match='Unknown command /mcp'):
            await harness.commands.execute_async('/mcp status')
        await harness.loader.enable('mcp')
        assert 'not loaded' in await harness.commands.execute_async('/mcp status')
        assert harness.store.plugins()[0].settings == {}
        await harness.loader.remove('mcp')
        assert 'not loaded' in await harness.commands.execute_async('/mcp status')

    async def test_stdio_logs_do_not_write_over_the_editor(
        self, harness: Harness, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        write_config(tmp_path)
        loaded = await harness.commands.execute_async('/mcp load --approve')
        logs = Path(loaded.partition('Stdio logs: ')[2])
        assert logs.is_dir()
        await Agent(TestModel(call_tools=['local_context']), deps_type=type(None)).run(
            'Log from the server', capabilities=harness.loader.capabilities()
        )
        assert 'MCP stderr sentinel' in (logs / 'server-1.log').read_text()
        assert 'MCP stderr sentinel' not in capfd.readouterr().err
        assert str(logs) in await harness.commands.execute_async('/mcp status')
        assert_stopped(tmp_path)
        if os.name == 'posix':
            assert logs.stat().st_mode & 0o777 == 0o700
            assert (logs / 'server-1.log').stat().st_mode & 0o777 == 0o600

    async def test_servers_with_the_same_tool_name_have_separate_tools_and_logs(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        (tmp_path / '.mcp.json').write_text(
            json.dumps(
                {
                    'mcpServers': {
                        name: {
                            'command': sys.executable,
                            'args': [str(Path(__file__).with_name('mcp_server.py'))],
                            'env': {
                                'CLAI_MCP_TOKEN': name,
                                'CLAI_MCP_PID_FILE': str(tmp_path / f'{name}.pid'),
                            },
                        }
                        for name in ('first', 'second')
                    }
                }
            )
        )
        loaded = await harness.commands.execute_async('/mcp load --approve')
        logs = Path(loaded.partition('Stdio logs: ')[2])
        result = await Agent(TestModel(call_tools=['first_context', 'second_context']), deps_type=type(None)).run(
            'Use both servers', capabilities=harness.loader.capabilities()
        )
        assert 'first_context' in result.output and 'second_context' in result.output
        for index, name in enumerate(('first', 'second'), start=1):
            assert 'MCP stderr sentinel' in (logs / f'server-{index}.log').read_text()
            with pytest.raises(ProcessLookupError):
                os.kill(int((tmp_path / f'{name}.pid').read_text()), 0)

    async def test_http_configs_load_without_connecting(self, harness: Harness, tmp_path: Path) -> None:
        (tmp_path / '.mcp.json').write_text(
            json.dumps(
                {
                    'mcpServers': {
                        'http': {'url': 'http://127.0.0.1:1/mcp', 'headers': {'Authorization': 'secret-token'}},
                        'sse': {'url': 'http://127.0.0.1:1/sse'},
                    }
                }
            )
        )
        assert 'Loaded 2 MCP server(s)' in await harness.commands.execute_async('/mcp load --approve')
        status = await harness.commands.execute_async('/mcp status')
        assert '2 server(s) loaded' in status
        assert 'secret-token' not in status and '127.0.0.1' not in status

    async def test_core_config_subset(self, harness: Harness, tmp_path: Path) -> None:
        (tmp_path / '.mcp.json').write_text(
            json.dumps(
                {
                    'mcpServers': {
                        'local': {
                            'command': sys.executable,
                            'args': [str(Path(__file__).with_name('mcp_server.py'))],
                            'env': {'CLAI_MCP_PID_FILE': str(tmp_path / 'server.pid')},
                            'cwd': None,
                            'url': 'http://127.0.0.1:1/mcp',
                            'disabled': True,
                            'type': 'http',
                        }
                    }
                }
            )
        )
        await harness.commands.execute_async('/mcp load --approve')
        result = await Agent(TestModel(call_tools=['local_context']), deps_type=type(None)).run(
            'Command wins; disabled and type are ignored', capabilities=harness.loader.capabilities()
        )
        assert str(tmp_path) in result.output
        assert_stopped(tmp_path)

    async def test_failed_turn_closes_stdio(self, harness: Harness, tmp_path: Path) -> None:
        async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert (tmp_path / 'server.pid').exists()
            raise RuntimeError('model failure')

        write_config(tmp_path)
        await harness.commands.execute_async('/mcp load --approve')
        with pytest.raises(RuntimeError, match='model failure'):
            await Agent(FunctionModel(fail), deps_type=type(None)).run(
                'Fail', capabilities=harness.loader.capabilities()
            )
        assert_stopped(tmp_path)

    async def test_approval_does_not_follow_another_repository(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(tmp_path)
        await harness.commands.execute_async('/mcp load --approve')
        other = tmp_path / 'other'
        other.mkdir()
        write_config(other)
        monkeypatch.chdir(other)
        await harness.loader.reload('mcp')
        assert f'not loaded. Config: {other / ".mcp.json"}' in await harness.commands.execute_async('/mcp')
        await Agent(TestModel(), deps_type=type(None)).run('No MCP', capabilities=harness.loader.capabilities())
        assert not (other / 'server.pid').exists()
        assert not (tmp_path / 'server.pid').exists()

    async def test_persistent_absolute_path_and_future_edits(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_config(tmp_path)
        settings = json.dumps({'config_path': str(path)})
        await harness.loader.command(['add', 'mcp', 'pydantic_clai2.mcp', settings])
        assert '1 server(s) loaded' in await harness.commands.execute_async('/mcp')
        assert not (tmp_path / 'server.pid').exists()
        other = tmp_path / 'other'
        other.mkdir()
        write_config(other)
        monkeypatch.chdir(other)
        write_config(tmp_path, prefix='edited')
        await harness.loader.reload('mcp')
        result = await Agent(TestModel(call_tools=['edited_context']), deps_type=type(None)).run(
            'Trusted path only', capabilities=harness.loader.capabilities()
        )
        assert 'edited_context' in result.output
        assert not (other / 'server.pid').exists()
        assert_stopped(tmp_path)
        path.write_text('{"secret-token": false}')
        await harness.loader.reload('mcp')
        assert 'not loaded' in await harness.commands.execute_async('/mcp')
        assert 'Cannot load MCP config' in harness.text
        assert 'secret-token' not in harness.text
        path.write_text('{"mcpServers": {}}')
        assert 'Loaded 0 MCP server(s)' in await harness.commands.execute_async('/mcp load --approve')
        assert '0 server(s) loaded' in await harness.commands.execute_async('/mcp')

    @pytest.mark.parametrize('settings', [{'config_path': '.mcp.json'}, {'config_path': 123}, {'secret-token': True}])
    async def test_reject_unsafe_settings(self, harness: Harness, settings: dict[str, JsonValue]) -> None:
        with pytest.raises(PluginError) as exc:
            await harness.loader.command(['add', 'mcp', 'pydantic_clai2.mcp', json.dumps(settings)])
        assert 'secret-token' not in str(exc.value)
        assert harness.loader.capabilities() == []

    @pytest.mark.parametrize(
        'content',
        [
            b'{"secret-token":',
            b'{"mcpServers": {"secret-token": {"command": 123}}}',
            b'{"mcpServers": {"secret-token": {"headers": {"Authorization": "secret-token"}}}}',
            b'{"mcpServers": {"local": {"command": "${CLAI_MCP_UNDEFINED_SECRET}"}}}',
            b'{"mcpServers": {"local": {"url": "http://localhost", "headers": {"secret-token": 123}}}}',
            b'\xff',
            None,
        ],
    )
    async def test_bad_config_is_redacted_and_atomic(
        self, harness: Harness, tmp_path: Path, content: bytes | None
    ) -> None:
        path = write_config(tmp_path)
        await harness.commands.execute_async('/mcp load --approve')
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
        error = await harness.commands.execute_async('/mcp load --approve')
        assert 'Cannot load MCP config' in error and 'secret-token' not in error
        assert 'CLAI_MCP_UNDEFINED_SECRET' not in error
        assert '1 server(s) loaded' in await harness.commands.execute_async('/mcp')
        result = await Agent(TestModel(call_tools=['local_context']), deps_type=type(None)).run(
            'Still available', capabilities=harness.loader.capabilities()
        )
        assert 'stdio-token' in result.output
        assert_stopped(tmp_path)

    @pytest.mark.parametrize('outer_scope', [False, True], ids=['task', 'outer-scope'])
    async def test_cancel_cleans_up_and_next_turn_reconnects(
        self, harness: Harness, tmp_path: Path, outer_scope: bool
    ) -> None:
        ready = anyio.Event()

        async def receive(stream: SocketStream) -> None:
            async with stream:
                expected = (tmp_path / 'server.pid').read_bytes()
                assert await BufferedByteReceiveStream(stream).receive_exactly(len(expected)) == expected
                ready.set()

        with anyio.fail_after(READY_TIMEOUT):
            async with await anyio.create_tcp_listener(local_host='127.0.0.1') as listener:
                port = listener.extra(SocketAttribute.local_address)[1]
                assert isinstance(port, int)
                write_config(tmp_path, port=port)
                await harness.commands.execute_async('/mcp load --approve')
                async with asyncio.TaskGroup() as group:
                    listening = group.create_task(listener.serve(receive))
                    agent = Agent(TestModel(call_tools=['local_wait']), deps_type=type(None))

                    async def wait() -> None:
                        await agent.run('Wait', capabilities=harness.loader.capabilities())

                    if outer_scope:
                        # A cancelled enclosing scope also cancels every await during the toolset's cleanup.
                        with anyio.CancelScope() as scope:
                            async with anyio.create_task_group() as turn:
                                turn.start_soon(wait)
                                await ready.wait()
                                scope.cancel()
                        assert scope.cancelled_caught
                    else:
                        running = group.create_task(wait())
                        await ready.wait()
                        running.cancel()  # CLAI's Esc/Ctrl-C path cancels the turn task.
                        with pytest.raises(asyncio.CancelledError):
                            await running
                    assert_stopped(tmp_path)
                    listening.cancel()
        result = await Agent(TestModel(call_tools=['local_context']), deps_type=type(None)).run(
            'Reconnect', capabilities=harness.loader.capabilities()
        )
        assert 'stdio-token' in result.output
        assert_stopped(tmp_path)

"""The built-in CodeMode plugin executes Coder tools through Monty."""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, Session
from pydantic_clai2.commands import Commands
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore


async def test_monty_coder_execution_and_plugin_lifecycle(tmp_path: Path) -> None:
    target = tmp_path / 'result.txt'
    sandboxed = True

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        names = {tool.name for tool in info.function_tools}
        assert ('run_code' in names) is sandboxed
        assert ('write_file' in names) is not sandboxed
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield 'done'
        elif sandboxed:
            yield {
                0: DeltaToolCall(
                    name='run_code',
                    json_args=json.dumps(
                        {'code': f'await write_file(path={str(target)!r}, content=str(sum([20, 22])))'}
                    ),
                )
            }
        else:
            yield {
                0: DeltaToolCall(name='write_file', json_args=json.dumps({'path': str(target), 'content': 'native'}))
            }

    agent = Agent(FunctionModel(stream_function=model))
    store = SettingsStore(tmp_path / 'settings.db')
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=agent, settings=store.load()),
        builtin=tuple(plugin for plugin in DEFAULT_PLUGINS if plugin.id in {'coder', 'code_mode'}),
    )
    await loader.load_all()
    try:
        for action in ('initial', 'disable', 'enable', 'reload', 'remove'):
            if action != 'initial':
                await loader.command([action, 'code_mode'])
            sandboxed = action != 'disable'
            session = Session(agent, deps=None, plugins=loader.capabilities())
            assert (await session.prompt('Write the result')).output == 'done'
            assert target.read_text() == ('42' if sandboxed else 'native')
            target.unlink()
        assert store.plugins() == []
    finally:
        await loader.close('exit')

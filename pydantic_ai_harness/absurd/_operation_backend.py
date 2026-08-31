from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, TypeAlias

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec import (
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
)
from pydantic_ai.exceptions import UserError

from ._context import current_async_task_context

DurableConfig: TypeAlias = Mapping[str, Any]
ToolConfig: TypeAlias = DurableConfig | Literal[False]


def resolve_tool_config(operation_id: DurableOperationId, tool: object | None, tool_name: str) -> ToolConfig:
    del operation_id
    if not isinstance(tool, ToolsetTool) or tool.tool_def.metadata is None:
        return {}
    config = tool.tool_def.metadata.get('absurd')
    if config is False:
        return False
    if config:
        raise UserError(
            f'Absurd steps take no per-tool options, so non-empty {"absurd"!r} metadata '
            f'on tool {tool_name!r} would have no effect. Remove the config.'
        )
    return {}


AbsurdOperationConfig = RoleBasedOperationConfig[DurableConfig]


class AbsurdOperationBackend(JournalCallableOperationBackend[DurableConfig]):
    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: DurableConfig,
    ) -> object:
        del operation_id, cache_key
        assert not config
        task_ctx = current_async_task_context()
        assert task_ctx is not None
        return await task_ctx.step(name, body)

"""Progress reporting for `browser_exec`"""
# ruff: noqa: D415

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from typing import Any, Literal

from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
    ToolReturn,
)
from pydantic_ai.tools import RunContext

_TOOL_NAME = 'browser_exec'

Detail = Literal['steps', 'code']
"""'steps' = one label line per call plus errors; 'code' = also the raw Python and output"""

STEP_LABEL_LIMIT = 80

ERROR_MARKER = '[exit code:'
"""Present in tool output whenever the executed code failed"""

CODE_LIMIT = 2_000
OUTPUT_LIMIT = 800
ERROR_LIMIT = 4_000  # tracebacks deserve more room than ordinary output


def step_label(code: str) -> str:
    """Leading `#` comment if the model wrote one, else the first code line"""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            label = stripped.lstrip('#').strip()
            return clip(label, STEP_LABEL_LIMIT) if label else '(browser step)'
        return clip(stripped, STEP_LABEL_LIMIT)
    return '(browser step)'


def narrate_call(sink: Callable[[str], None], code: str, detail: Detail = 'code') -> None:
    """Report one `browser_exec` invocation before it runs"""
    sink(f'* {step_label(code)}')
    if detail == 'code':
        sink(indent(clip(code.rstrip(), CODE_LIMIT), '    '))


def narrate_result(sink: Callable[[str], None], text: str, images: int, detail: Detail = 'code') -> None:
    """Report what one invocation produced; in 'steps' detail only failures surface"""
    failed = ERROR_MARKER in text
    if detail == 'steps' and not failed:
        return
    limit = ERROR_LIMIT if failed else OUTPUT_LIMIT
    sink(indent(clip(text.rstrip(), limit, from_end=failed), '  | '))
    if images and detail == 'code':
        sink(f'  | [{images} screenshot{"s" if images > 1 else ""} attached]')


def narrate_error(sink: Callable[[str], None], message: str) -> None:
    """Report a call the tool rejected or that failed before producing output"""
    sink(indent(clip(message, ERROR_LIMIT), '  ! '))


def indent(text: str, prefix: str) -> str:
    return '\n'.join(prefix + line for line in text.splitlines())


def clip(text: str, limit: int, *, from_end: bool = False) -> str:
    """Truncate to `limit`; `from_end` keeps the tail, where error markers live"""
    if len(text) <= limit:
        return text
    return '... ' + text[-limit:] if from_end else text[:limit] + ' ...'


def browser_progress(
    sink: Callable[[str], None] = print,
    *,
    detail: Detail = 'steps',
    code_limit: int = CODE_LIMIT,
    output_limit: int = OUTPUT_LIMIT,
    error_limit: int = ERROR_LIMIT,
) -> Callable[[RunContext[Any], AsyncIterable[AgentStreamEvent]], Any]:
    """Build an `event_stream_handler` that narrates the agent's browser activity"""

    async def handler(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == _TOOL_NAME:
                code = str(event.part.args_as_dict().get('code', ''))
                sink(f'* {step_label(code)}')
                if detail == 'code':
                    sink(indent(clip(code.rstrip(), code_limit), '    '))
            elif isinstance(event, FunctionToolResultEvent) and event.part.tool_name == _TOOL_NAME:
                if isinstance(event.part, RetryPromptPart):
                    sink(indent(clip(event.part.model_response(), error_limit), '  ! '))
                    continue

                # the agent unwraps ToolReturn before emitting: text on part.content, media on event.content
                content = event.part.content
                extra: list[object] = list(event.content) if isinstance(event.content, list) else []
                if isinstance(content, ToolReturn):  # pragma: no cover - direct-call paths only
                    extra = list(content.content or [])
                    content = content.return_value
                images = sum(1 for item in extra if not isinstance(item, str))
                text = str(content).rstrip()
                failed = ERROR_MARKER in text
                if detail == 'steps' and not failed:
                    continue
                limit = error_limit if failed else output_limit
                sink(indent(clip(text, limit, from_end=failed), '  | '))
                if images and detail == 'code':
                    sink(f'  | [{images} screenshot{"s" if images > 1 else ""} attached]')

    return handler

"""Measure sandbox tool latency that eager `run_code` execution hid behind argument streaming."""

import re
from dataclasses import dataclass, field
from time import perf_counter

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import AgentStreamEvent, CapabilityEvent, PartEndEvent, PartStartEvent, ToolCallPart
from pydantic_ai.tools import AgentDepsT, ToolDefinition

NESTED_CALL = re.compile(r'(?P<parent>.+)__(?P<speculative>spec_)?\d+')
"""Harness CodeMode ids: `parent__N` for a sandbox dispatch, `parent__spec_N` for a speculative launch."""


def is_sandbox_call(tool_call_id: str | None) -> bool:
    """Whether an event belongs to a call made from inside `run_code`, speculative or not."""
    return tool_call_id is not None and NESTED_CALL.fullmatch(tool_call_id) is not None


@dataclass(kw_only=True)
class EagerExecutionCompletedEvent(CapabilityEvent, namespace='pydantic_clai2'):
    """A completed snippet whose sandbox calls overlapped `run_code` argument streaming."""

    saved_ms: float
    """Summed tool-call overlap with generation, not wall-clock speedup."""


@dataclass(kw_only=True)
class _StreamWindow:
    ended: float | None = None
    saved_ms: float = 0.0


@dataclass
class EagerTiming(AbstractCapability[AgentDepsT]):
    """Observe execution hooks without changing tool behavior.

    A window opens when a tool call part starts streaming and closes when it ends. Sandbox
    dispatches (`parent__N`) that run inside their parent's open window are charged the
    overlapping time. Speculative launches (`parent__spec_N`) are excluded because the harness
    already reports them, so the two totals do not double-count.
    """

    _windows: dict[tuple[int, str], _StreamWindow] = field(
        default_factory=dict[tuple[int, str], _StreamWindow], init=False, repr=False
    )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> 'EagerTiming[AgentDepsT]':
        """Keep windows per run."""
        return EagerTiming()

    async def on_event(self, ctx: RunContext[AgentDepsT], *, event: AgentStreamEvent) -> None:
        """Open a window at a tool call's first streamed byte and close it at its last."""
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
            self._windows[(ctx.run_step, event.part.tool_call_id)] = _StreamWindow()
        elif isinstance(event, PartEndEvent) and isinstance(event.part, ToolCallPart):
            self._windows.setdefault((ctx.run_step, event.part.tool_call_id), _StreamWindow()).ended = perf_counter()

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        """Charge nested dispatches to the open window; report the total when `run_code` succeeds."""
        nested = NESTED_CALL.fullmatch(call.tool_call_id)
        if nested is None or nested['speculative']:
            return await self._top_level(ctx, call=call, tool_def=tool_def, args=args, handler=handler)
        window = self._windows.get((ctx.run_step, nested['parent']))
        if window is None or window.ended is not None:
            return await handler(args)
        started = perf_counter()
        try:
            return await handler(args)
        finally:
            finished = perf_counter()
            cutoff = finished if window.ended is None else min(finished, window.ended)
            window.saved_ms += max(0.0, cutoff - started) * 1000

    async def _top_level(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        key = (ctx.run_step, call.tool_call_id)
        window = self._windows.get(key)
        try:
            result = await handler(args)
            if tool_def.name == 'run_code' and window is not None and not args.get('restart') and window.saved_ms > 0:
                await ctx.emit(EagerExecutionCompletedEvent(tool_call_id=call.tool_call_id, saved_ms=window.saved_ms))
            return result
        finally:
            self._windows.pop(key, None)

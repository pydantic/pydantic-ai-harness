"""Eager execution for `CodeMode`: feed streamed `run_code` statements into the live REPL.

Eager execution runs the real program, in order, in the real session, as each top-level
statement closes in the stream. Nothing is predicted, so nothing can miss or be wasted --
and nothing can be rolled back: side effects land before the tool call is committed, which
is why the tier is opt-in.

A failed statement leaves the session exactly as a failed whole-snippet feed does today:
assignments made before the failing line persist and the error surfaces to the model as the
`run_code` result. The only semantic difference from non-eager execution is timing.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from ._events import emit_best_effort
from ._streaming import MAX_SCAN_CHARS, closed_statements, decode_partial_code

if TYPE_CHECKING:
    from ._toolset import CodeModeToolset


@dataclass(kw_only=True)
class _EagerPart:
    """Accumulated state for one streamed `run_code` tool call part under eager execution."""

    tool_call_id: str
    args_text: str = ''
    args_dict: dict[str, Any] | None = None
    consumed: int = 0
    """Top-level statements handed to the feed pump so far."""

    scanned_source: str | None = None
    """The exact completed-lines prefix the last scan parsed.

    Statements only close on line boundaries, so a delta that changes nothing before the
    final newline cannot change the parse and is skipped. Comparing the parse input itself
    (rather than a newline count) also covers dict-delta streams, where a provider can
    replace the whole `code` value without growing it.
    """

    halted: bool = False
    """No further statements will be fed (a `restart` request was seen in the arguments).

    Statements already fed cannot be unfed; halting bounds the damage. The dispatch handles
    a restarted part by resetting the session and running the full snippet.
    """

    fed_line_count: int = 0
    """Source lines covered by pumped statements; the executed prefix ends here."""

    fed_prefix: str = ''
    """Exact source prefix handed to the pump, for divergence detection at execution."""

    queue: deque[str] = field(default_factory=deque[str])
    """Statement sources waiting for the pump, in program order."""

    pump: asyncio.Task[None] | None = None
    error: BaseException | None = None
    feed_count: int = 0
    busy_seconds: float = 0.0
    """Wall-clock the pump spent executing fragments, for the commit event's overlap math."""

    output: str = ''
    """Concatenated print output from completed fragment feeds."""

    output_capped: bool = False
    """Accumulation hit the host-side byte cap; later fragment prints are dropped."""

    output_bytes: int = 0
    """Running UTF-8 size of `output`, tracked incrementally so feeds stay linear."""

    nested_calls: dict[str, ToolCallPart] = field(default_factory=dict[str, ToolCallPart])
    """Tool calls the pumped statements dispatched, keyed by nested tool call id."""

    nested_returns: dict[str, ToolReturnPart] = field(default_factory=dict[str, ToolReturnPart])
    """Their return parts, under the same nested tool call ids."""


@dataclass
class EagerState:
    """Watch streamed `run_code` parts and feed closed statements into the REPL serially.

    One instance per run, owned by the `CodeMode` clone and bound to its `CodeModeToolset`.
    The stream watcher enqueues statement sources; a single pump task per part feeds them
    through the toolset's normal `run_code` pipeline, so budget accounting, error mapping,
    and tool hooks behave exactly as a model-issued snippet would.
    """

    _parts: dict[str, _EagerPart] = field(default_factory=dict[str, '_EagerPart'], init=False)
    _toolset: CodeModeToolset[Any] | None = field(default=None, init=False)
    _pumps: set[asyncio.Task[None]] = field(default_factory=set['asyncio.Task[None]'], init=False)
    """Every live pump task, including ones whose part was already popped for execution.

    `close()` cancels these; tracking them separately from `_parts` matters because an
    executing `run_code` removes its watch before the pump necessarily finishes.
    """

    feed_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    """Serializes fragment feeds run-wide, matching the dispatch path's scheduling.

    One pump exists per streamed part, but a response can stream several `run_code` parts;
    without the lock their fragments would interleave against the single live session even
    when the run is configured for sequential tool execution. The eager dispatch's tail
    feed holds it too, so a later part's pump cannot overlap an earlier part's completion.
    """

    pending_events: list[CapabilityEvent] = field(default_factory=list[CapabilityEvent], init=False)
    """Commit events buffered until the next capability hook, where emission is permitted."""

    async def flush_events(self, ctx: RunContext[Any]) -> None:
        """Emit buffered events; called from `CodeMode`'s hook dispatches."""
        events, self.pending_events = self.pending_events, []
        for event in events:
            await emit_best_effort(ctx, event)

    def bind(self, toolset: CodeModeToolset[Any]) -> None:
        """Attach the owning toolset; feeds go through its `run_code` pipeline."""
        self._toolset = toolset

    # -- stream side ------------------------------------------------------------------------

    async def observe(self, event: AgentStreamEvent, ctx: RunContext[Any]) -> None:
        """Track one stream event, enqueueing newly closed statements for the pump."""
        match event:
            case PartStartEvent(part=ToolCallPart() as part):
                if part.tool_name != 'run_code':
                    return
                part_id = part.tool_call_id
                watch = self._parts.setdefault(part_id, _EagerPart(tool_call_id=part_id))
                args = part.args
                if isinstance(args, str):
                    watch.args_text += args
                elif isinstance(args, dict):
                    watch.args_dict = args
                self._scan(watch, ctx)
            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                watch = self._find_watch(delta.tool_call_id)
                if watch is None or watch.halted:
                    # A halted watch stops accumulating too: past the scan cap (or after a
                    # restart request) the buffered copy of the arguments has no reader.
                    return
                args_delta = delta.args_delta
                if isinstance(args_delta, str):
                    watch.args_text += args_delta
                elif isinstance(args_delta, dict):
                    merged = dict(watch.args_dict or {})
                    merged.update(args_delta)
                    watch.args_dict = merged
                if len(watch.args_text) > MAX_SCAN_CHARS:
                    watch.halted = True
                    return
                self._scan(watch, ctx)
            case _:
                return

    def _find_watch(self, tool_call_id: str | None) -> _EagerPart | None:
        if tool_call_id is not None:
            return self._parts.get(tool_call_id)
        if len(self._parts) == 1:  # pragma: no cover - delta without id, single-part fallback
            return next(iter(self._parts.values()))
        return None

    def _current_code(self, watch: _EagerPart) -> str | None:
        if watch.args_dict is not None:
            code = watch.args_dict.get('code')
            return code if isinstance(code, str) else None
        return decode_partial_code(watch.args_text)

    def _restart_requested(self, watch: _EagerPart) -> bool:
        """Whether the streamed arguments show a `restart` request so far.

        The key can stream after the code, in which case the prefix has already run by the
        time it appears; that residue is documented on the `eager` flag. Checking the raw
        text means a snippet that merely mentions the key in a string also halts feeding,
        which costs eagerness, never correctness.
        """
        if watch.args_dict is not None:
            return bool(watch.args_dict.get('restart'))
        return '"restart"' in watch.args_text

    def _scan(self, watch: _EagerPart, ctx: RunContext[Any]) -> None:
        """Re-parse the streamed code and enqueue any statements that newly closed.

        No `halted` guard here: `observe` drops deltas for halted watches, and every halt
        cause (restart, prefix rewrite, oversized arguments) re-detects itself on a rescan.
        """
        if self._restart_requested(watch):
            watch.halted = True
            return
        code = self._current_code(watch)
        if code is None:
            return
        source_prefix = code[: code.rfind('\n') + 1]
        if source_prefix == watch.scanned_source:
            return
        watch.scanned_source = source_prefix
        lines = code.split('\n')
        if watch.fed_line_count and '\n'.join(lines[: watch.fed_line_count]) != watch.fed_prefix:
            # A dict delta rewrote code that already executed. Keep `fed_prefix` as the text
            # that actually ran and stop feeding; the dispatch's divergence check resets the
            # session and asks the model to resend.
            watch.halted = True
            return
        fresh, watch.consumed = closed_statements(code, watch.consumed)
        # Each queued source is a contiguous line slice from the end of the previous
        # statement, so comments and blank lines between statements feed with them and the
        # concatenation of all slices reproduces the prefix exactly.
        for statement in fresh:
            end = statement.end_lineno or statement.lineno
            source = '\n'.join(lines[watch.fed_line_count : end])
            watch.queue.append(source)
            watch.fed_line_count = end
        watch.fed_prefix = '\n'.join(lines[: watch.fed_line_count])
        # One pump per part: enqueueing happens before this check, so a finished pump
        # always means an empty queue, and a live one will drain what was just added.
        if watch.queue and (watch.pump is None or watch.pump.done()):
            watch.pump = asyncio.create_task(self._run_pump(watch, ctx))
            self._pumps.add(watch.pump)
            watch.pump.add_done_callback(self._pumps.discard)

    async def _run_pump(self, watch: _EagerPart, ctx: RunContext[Any]) -> None:
        """Feed queued statements one at a time; an error stops the part for good."""
        toolset = self._toolset
        if toolset is None:  # pragma: no cover - `__aenter__` binds before any stream event
            raise RuntimeError('`EagerState` is not bound to a `CodeModeToolset`; the toolset was never entered')
        while watch.queue and watch.error is None:
            source = watch.queue.popleft()
            watch.feed_count += 1
            feed_ctx = replace(ctx, tool_call_id=f'{watch.tool_call_id}~e{watch.feed_count}', tool_name='run_code')
            fed_at = time.perf_counter()
            try:
                async with self.feed_lock:
                    await toolset.feed_eager_fragment(watch, source, feed_ctx)
            except Exception as e:  # cancellation propagates; everything else is held for the dispatch
                watch.error = e
                watch.queue.clear()
            finally:
                watch.busy_seconds += time.perf_counter() - fed_at

    # -- execution side ---------------------------------------------------------------------

    def pop_watch(self, tool_call_id: str, code: str) -> _EagerPart | None:
        """Remove and return the watch for an executing part, tolerating provider-rewritten ids.

        Falls back to any part whose executed prefix matches the submitted code, since a
        re-keyed part is still the same program.
        """
        watch = self._parts.pop(tool_call_id, None)
        if watch is not None:
            return watch
        # An empty executed prefix matches any code; requiring one keeps an unrelated
        # just-started part from being adopted (and then executed twice).
        rekeyed = next((pid for pid, c in self._parts.items() if c.fed_prefix and code.startswith(c.fed_prefix)), None)
        return self._parts.pop(rekeyed) if rekeyed is not None else None

    async def drain(self, watch: _EagerPart) -> None:
        """Wait until every enqueued statement has been fed (or the part errored).

        At most one pump exists per part: `_scan` enqueues before ensuring the pump, so a
        finished pump means an empty queue, and no new statements arrive once the part's
        arguments are complete.
        """
        if watch.pump is not None:
            await watch.pump

    async def close(self) -> None:
        """Cancel every live pump and drop all watches at run end.

        Pumps are cancelled from the task set rather than the watches: a part that reached
        execution was already popped from `_parts`, but its pump may still be feeding when
        the run fails, and an abandoned fragment must not keep invoking tools.
        """
        self._parts.clear()
        pumps, self._pumps = list(self._pumps), set()
        for pump in pumps:
            # The done-callback prunes finished pumps, so these are live; `cancel` on one
            # that finished in the meantime is a no-op and the await returns immediately.
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):  # pragma: no cover - cancellation race
                pass

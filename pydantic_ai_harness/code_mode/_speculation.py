"""Speculative launch of sandboxed tool calls while `run_code` arguments stream.

Implements the streaming half of speculative programmatic tool calling (sPTC,
<https://alexzhang13.github.io/blog/2026/spec-ptc/>) for `CodeMode`: while the model is still
emitting the `run_code` tool call, the partial `code` argument is decoded from the accumulated
JSON, parsed into closed top-level statements, and scanned for calls to allowlisted sandbox
functions whose arguments are all literals. Those calls launch immediately as tasks; when the
completed snippet later executes and dispatches the same call, the in-flight task is adopted
instead of starting the tool cold.

The other half of the trick, overlapping independent calls with each other during execution,
already exists: `MontyExecutor` defers external calls as futures and only forces them when the
sandbox needs the value.

Scope is deliberately the blog's "Case 1" (all-literal arguments). Calls whose arguments carry
dependencies would need a shadow interpreter to materialize values; with Monty that shrinks to
forking a session, which is future work and out of scope here.

Only tools named in the allowlist are ever launched early. Launching early is observationally
equivalent to the normal call only for tools without side effects, so the allowlist is an
explicit user promise, mirroring the reference implementation's `speculatable=True, pure=True`
contract. Speculated calls run through the same nested `ToolManager` path as cold calls, so
capability tool hooks fire at launch time rather than at adoption time -- a documented POC
trade-off.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from pydantic_ai_harness.code_mode._events import (
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallLaunchedEvent,
    SpeculativeCallMissedEvent,
    SpeculativeCallSettledEvent,
    SpeculativeCodeUpdateEvent,
    emit_best_effort,
)
from pydantic_ai_harness.code_mode._streaming import closed_statements, decode_partial_code

_RUN_CODE_TOOL_NAME = 'run_code'

# Upper bound on early launches per streamed `run_code` part. Wrong-branch conditionals and
# rewritten lines make some speculations garbage; the cap bounds how much garbage one part can
# start. Adoption is unaffected once a call is in flight.
_MAX_SPECULATIONS_PER_PART = 32


@dataclass(kw_only=True)
class SpeculationStats:
    """Aggregate speculation counters, shared across the runs of one `CodeMode` instance."""

    launched: int = 0
    """Calls started early from partial code."""

    adopted: int = 0
    """Early calls whose result the executing snippet actually claimed."""

    evicted: int = 0
    """Early calls cancelled or discarded without being claimed."""


@dataclass(frozen=True, kw_only=True)
class _SpecOutcome:
    """What a speculative task settles to. Exactly one of `error`/the value fields is meaningful."""

    content: Any = None
    """The plain tool return value, before sandbox serialization."""

    serialized: Any = None
    """The sandbox-ready form of `content` (the value the snippet receives)."""

    metadata: Any = None
    """`ToolReturn.metadata` when the tool returned a `ToolReturn`."""

    error: BaseException | None = None
    """The exception the cold path would have raised at the sandbox call site."""

    denied_message: str | None = None
    """Set when a handler denied the call, so adoption can record `outcome='denied'`."""


@dataclass(kw_only=True)
class SpeculativeCall:
    """One in-flight early launch, waiting to be claimed by the executing snippet."""

    sandbox_name: str
    original_name: str
    kwargs: dict[str, Any]
    task: asyncio.Task[_SpecOutcome]
    launch_id: str
    started_at: float
    settled_at: float | None = None
    settle_emitted: bool = False

    def settled_state(self) -> Literal['pending', 'ready', 'failed']:
        """Where this launch currently stands, for settle and eviction events."""
        if not self.task.done() or self.task.cancelled():
            return 'pending'
        if self.task.exception() is not None:  # pragma: no cover - the runner settles, never raises
            return 'failed'
        return 'failed' if self.task.result().error is not None else 'ready'

    def elapsed_ms(self) -> float:
        """Wall-clock from launch until settled, or until now while still running.

        The done callback recording `settled_at` was added before any awaiter, so readers that
        await the task first always see it set; the fallback covers in-flight queries only.
        """
        end = self.settled_at if self.settled_at is not None else time.perf_counter()  # pragma: no branch
        return (end - self.started_at) * 1000


@dataclass
class _StepIngredients:
    """Everything the launcher needs to run a nested call, stashed by the toolset each step."""

    wrapped: AbstractToolset[Any]
    wrapped_tools: dict[str, ToolsetTool[Any]]
    sanitized_to_original: dict[str, str]
    eligible: frozenset[str]
    """Sandbox (possibly sanitized) names that may launch early this step."""

    serialize: Callable[[Any], Any]
    """Serializer matching the cold path's sandbox return serialization."""


@dataclass
class _PartWatch:
    """Accumulated state for one streamed `run_code` tool call part."""

    tool_call_id: str
    args_text: str = ''
    args_dict: dict[str, Any] | None = None
    consumed_statements: int = 0
    scanned_newlines: int = -1
    """Newline count of the code at the last full scan.

    Statements only close on line boundaries, so the AST work (parse loop plus call
    extraction) is skipped for the many deltas that arrive within a line -- they still
    produce a code-update event, just without reparsing.
    """

    demanded: dict[str, int] = field(default_factory=dict[str, int])
    """Occurrences of each `(function, arguments)` key this part's closed statements hold.

    Launches are deduplicated against calls already in flight (typically a failed
    attempt's surviving launches), so a retry claims instead of relaunching. Multiplicity
    within the part stays exact: the Nth occurrence launches once N exceeds the in-flight
    count.
    """
    launched: int = 0
    calls: dict[str, deque[SpeculativeCall]] = field(default_factory=dict[str, deque[SpeculativeCall]])
    """FIFO per canonical key: the k-th identical dispatch claims the k-th launch, so results
    of a nondeterministic tool called twice with the same arguments are never collapsed."""


def _canonical_key(sandbox_name: str, kwargs: dict[str, Any]) -> str:
    """Claim identity for one concrete call: both launch and claim hash through here."""
    return json.dumps([sandbox_name, kwargs], sort_keys=True, default=repr)


_DECLARED_SAFETY_KEYS = ('read_only', 'idempotent')
_MCP_SAFETY_HINTS = ('readOnlyHint', 'idempotentHint')


def _declares_speculation_safety(tool_def: ToolDefinition) -> bool:
    """Whether a tool's own definition presents evidence that early execution is safe.

    Two channels: first-party tools set `metadata={'read_only': True}` (or
    `'idempotent'`) on the `Tool`, and MCP servers publish `readOnlyHint` or
    `idempotentHint` tool annotations, which arrive under `metadata['annotations']`.
    Hints are the server's claim, not a proof; `speculate='declared'` extends the same
    trust to them that an explicit allowlist places in the user.

    The key vocabulary tracks pydantic-ai's tool behavior annotations
    (pydantic/pydantic-ai#6344, catalogued in pydantic/pydantic-ai#7955): plain metadata
    is the v1 contract, and if those names become first-class `ToolDefinition` fields the
    same words will already be in use here.
    """
    metadata = tool_def.metadata or {}
    if any(metadata.get(key) is True for key in _DECLARED_SAFETY_KEYS):
        return True
    raw: Any = metadata.get('annotations')
    if not isinstance(raw, dict):
        return False
    return any(hint in raw and raw[hint] is True for hint in _MCP_SAFETY_HINTS)


_SKIP_CONTAINERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _iter_calls(node: ast.AST) -> list[ast.Call]:
    """Collect `ast.Call` nodes in execution position, skipping bodies that don't run yet.

    `def`/`class`/`lambda` bodies execute only when later invoked, usually with non-literal
    arguments, so calls inside them are excluded. Conditional and loop bodies are included:
    launching a pure call from a branch that never runs wastes the call but stays correct,
    since unclaimed launches are evicted.
    """
    found: list[ast.Call] = []
    if isinstance(node, ast.Call):
        found.append(node)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SKIP_CONTAINERS):
            continue
        found.extend(_iter_calls(child))
    return found


@dataclass
class _ExtractedCall:
    """One speculatable call read off the AST, with the launching statement's line span."""

    sandbox_name: str
    kwargs: dict[str, Any]
    line_start: int
    line_end: int


def _literal_calls(statements: Sequence[ast.stmt], eligible: frozenset[str]) -> list[_ExtractedCall]:
    """Extract eligible sandbox calls whose arguments are entirely keyword literals.

    Positional arguments are never speculated: the sandbox rejects them at execution time, so an
    early launch would run a call the real snippet cannot claim.
    """
    out: list[_ExtractedCall] = []
    for statement in statements:
        if isinstance(statement, _SKIP_CONTAINERS):
            # A top-level `def`/`class` statement only defines; its body (and even its
            # decorators, conservatively) runs later, if ever.
            continue
        for call in _iter_calls(statement):
            func = call.func
            if not isinstance(func, ast.Name) or func.id not in eligible:
                continue
            kwargs = _literal_kwargs(call, func.id)
            if kwargs is None:
                continue
            out.append(
                _ExtractedCall(
                    sandbox_name=func.id,
                    kwargs=kwargs,
                    line_start=statement.lineno,
                    line_end=statement.end_lineno or statement.lineno,
                )
            )
    return out


def _literal_kwargs(call: ast.Call, name: str) -> dict[str, Any] | None:
    """Return the call's arguments as literal keyword values, or None if any are not.

    Positional arguments are never speculated: the sandbox rejects them at execution time,
    so an early launch would run a call the real snippet cannot claim.
    """
    if not isinstance(call.func, ast.Name) or call.func.id != name or call.args:
        return None
    kwargs: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            return None
    return kwargs


def _close_paren(code: str, start: int) -> int | None:
    """Return the index just past the paren that closes `code[start]`, or None if still open.

    A small scanner rather than a parse: the enclosing statement is usually incomplete, so
    only the call expression itself can be balanced. Tracks nesting across all bracket kinds
    and skips string literals (with escapes); the extracted span is verified by `ast.parse`
    afterwards, so the scanner only has to find a plausible end, not validate syntax.
    """
    depth = 0
    quote: str | None = None
    i = start
    while i < len(code):
        ch = code[i]
        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in '"\'':
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _text_literal_calls(code: str, eligible: frozenset[str]) -> list[_ExtractedCall]:
    """Extract complete eligible calls straight from streamed text, closed statement or not.

    This is what makes launches fire as soon as a call finishes streaming: `_literal_calls`
    only sees statements the line-conservative scanner has closed, which for a call inside a
    compound statement (an `if`/`else` arm) can trail the call's own text by many lines. Here
    a call is launchable once its closing paren has streamed, wherever its statement stands.

    Text-level extraction cannot see context, so a call spelled inside a string literal, a
    comment, or a `def` body can launch too. Those launches waste a pure call and are evicted
    at commit; the `def`/attribute lookbehind filters the two cheap-to-catch shapes.
    """
    out: list[_ExtractedCall] = []
    for name in eligible:
        for match in re.finditer(rf'\b{re.escape(name)}\s*\(', code):
            before = code[: match.start()]
            if before.rstrip().endswith('.') or re.search(r'\bdef\s*$', before):
                continue
            end = _close_paren(code, match.end() - 1)
            if end is None:
                continue
            try:
                expression = ast.parse(code[match.start() : end], mode='eval')
            except SyntaxError:
                continue
            if not isinstance(expression.body, ast.Call):
                continue  # pragma: no cover - a `name(...)` span that parses is always a Call
            kwargs = _literal_kwargs(expression.body, name)
            if kwargs is None:
                continue
            out.append(
                _ExtractedCall(
                    sandbox_name=name,
                    kwargs=kwargs,
                    line_start=before.count('\n') + 1,
                    line_end=code[:end].count('\n') + 1,
                )
            )
    return out


@dataclass
class SpeculationState:
    """Per-run speculation store shared between the `CodeMode` capability and its toolset.

    The capability's `wrap_run_event_stream` feeds streamed `run_code` argument deltas in;
    the toolset's dispatch path claims matching in-flight calls out. The capability creates one
    instance per run in `for_run`, and `CodeModeToolset` carries it by reference through its
    `for_run`/`for_run_step` copies.
    """

    allowlist: frozenset[str] | Literal['declared']
    """Original tool names the user declared side-effect free, or `'declared'` to trust
    the tools' own definitions (see `_declares_speculation_safety`)."""

    stats: SpeculationStats

    pending_events: list[CapabilityEvent] = field(default_factory=list[CapabilityEvent], init=False)
    """Execution-side events awaiting a capability hook context.

    Capability events may only be emitted from a capability hook or a capability-contributed
    toolset the framework can attribute; the dispatch path inside `run_code` is neither, so
    claim/miss/eviction events buffer here and `CodeMode` flushes them from its
    `after_tool_execute` / `on_tool_execute_error` hooks -- which also matches the UX: outcomes
    reveal when the snippet finishes executing.
    """

    _step: _StepIngredients | None = field(default=None, init=False)
    _parts: dict[str, _PartWatch] = field(default_factory=dict[str, _PartWatch], init=False)
    _index_to_part: dict[int, str] = field(default_factory=dict[int, str], init=False)

    def stash_step(
        self,
        *,
        wrapped: AbstractToolset[Any],
        wrapped_tools: dict[str, ToolsetTool[Any]],
        sanitized_to_original: dict[str, str],
        callable_defs: dict[str, ToolDefinition],
        serialize: Callable[[Any], Any],
    ) -> None:
        """Record this step's dispatch ingredients; called by `CodeModeToolset.get_tools`.

        Eligibility is resolved here: allowlisted names, minus `sequential` tools, whose
        rendering as `def` gives execution an ordering contract that an early launch would break.
        """
        if self.allowlist == 'declared':
            eligible = frozenset(
                name
                for name, tool_def in callable_defs.items()
                if not tool_def.sequential and _declares_speculation_safety(tool_def)
            )
        else:
            eligible = frozenset(
                name
                for name, tool_def in callable_defs.items()
                if not tool_def.sequential and sanitized_to_original.get(name, name) in self.allowlist
            )
        self._step = _StepIngredients(
            wrapped=wrapped,
            wrapped_tools=wrapped_tools,
            sanitized_to_original=sanitized_to_original,
            eligible=eligible,
            serialize=serialize,
        )

    # -- streaming side ---------------------------------------------------------------------

    async def observe(self, event: AgentStreamEvent, ctx: RunContext[Any]) -> list[CapabilityEvent]:
        """Feed one stream event; launches tasks for any newly speculatable calls.

        Returns the speculation events this stream event produced, for the caller to yield
        directly into the wrapped event stream. Stream-phase events are yielded rather than
        emitted through `ctx.emit_event` because the run's event buffer only drains live during
        tool execution; during a model request it flushes at node end, which would delay every
        update until the snippet finished streaming.
        """
        produced: list[CapabilityEvent] = []
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, ToolCallPart) and part.tool_name == _RUN_CODE_TOOL_NAME:
                watch = _PartWatch(tool_call_id=part.tool_call_id)
                if isinstance(part.args, str):
                    watch.args_text = part.args
                elif isinstance(part.args, dict):
                    watch.args_dict = dict(part.args)
                self._parts[part.tool_call_id] = watch
                self._index_to_part[event.index] = part.tool_call_id
                await self._scan(watch, ctx, produced)
            else:
                self._index_to_part.pop(event.index, None)
        elif isinstance(event, PartDeltaEvent):
            part_id = self._index_to_part.get(event.index)
            delta = event.delta
            if part_id is None or not isinstance(delta, ToolCallPartDelta):
                self._collect_settles(produced)
                return produced
            watch = self._parts[part_id]
            if isinstance(delta.args_delta, str):
                watch.args_text += delta.args_delta
            elif isinstance(delta.args_delta, dict):
                watch.args_dict = {**(watch.args_dict or {}), **delta.args_delta}
            await self._scan(watch, ctx, produced)
        elif isinstance(event, PartEndEvent):
            part_id = self._index_to_part.get(event.index)
            if part_id is not None:
                # The arguments are complete: every statement is closed now, including the
                # trailing ones the line-conservative scanner held back (streamed code rarely
                # ends with a newline, so without this a snippet's last statements never
                # launch and their dispatches go cold).
                await self._scan(self._parts[part_id], ctx, produced, final=True)
        self._collect_settles(produced)
        return produced

    async def _scan(
        self, watch: _PartWatch, ctx: RunContext[Any], produced: list[CapabilityEvent], *, final: bool = False
    ) -> None:
        step = self._step
        if step is None or not step.eligible:
            return
        if watch.args_dict is not None:
            code = watch.args_dict.get('code')
            if not isinstance(code, str):
                return
        else:
            maybe_code = decode_partial_code(watch.args_text)
            if maybe_code is None:
                return
            code = maybe_code
        newlines = code.count('\n')
        if not final and newlines == watch.scanned_newlines:
            # No line boundary since the last scan: nothing can have closed. Report the
            # grown code for live rendering and skip the parse work.
            produced.append(
                SpeculativeCodeUpdateEvent(
                    tool_call_id=watch.tool_call_id,
                    code=code,
                    closed_statements=watch.consumed_statements,
                )
            )
            return
        watch.scanned_newlines = newlines
        if final:
            try:
                body = ast.parse(code).body
            except SyntaxError:
                return
            watch.consumed_statements = len(body)
        else:
            _, watch.consumed_statements = closed_statements(code, watch.consumed_statements)
        produced.append(
            SpeculativeCodeUpdateEvent(
                tool_call_id=watch.tool_call_id,
                code=code,
                closed_statements=watch.consumed_statements,
            )
        )
        # Launch from the raw text, not the closed statements: a call is ready the moment
        # its closing paren streams, even while its enclosing statement (an `if` arm, a
        # `with` body) is still being generated. Rescans recount every occurrence in the
        # grown prefix, so `demanded` is reconciled to the count rather than incremented.
        seen: dict[str, int] = {}
        for extracted in _text_literal_calls(code, step.eligible):
            key = _canonical_key(extracted.sandbox_name, extracted.kwargs)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= watch.demanded.get(key, 0):
                continue
            if watch.launched >= _MAX_SPECULATIONS_PER_PART:
                return
            watch.demanded[key] = seen[key]
            in_flight = sum(len(w.calls.get(key, ())) for w in self._parts.values())
            if watch.demanded[key] <= in_flight:
                # Already covered, usually by a failed attempt's surviving launch; the
                # snippet claims it at execution. If another part claims it first, the
                # execution prefetch launches the deficit.
                continue
            self._launch(watch, step, extracted, ctx, produced)

    def _launch(
        self,
        watch: _PartWatch,
        step: _StepIngredients,
        extracted: _ExtractedCall,
        ctx: RunContext[Any],
        produced: list[CapabilityEvent],
        phase: Literal['streaming', 'execution'] = 'streaming',
    ) -> None:
        sandbox_name = extracted.sandbox_name
        original_name = step.sanitized_to_original.get(sandbox_name, sandbox_name)
        parent_manager = ctx.tool_manager
        tool_manager = ToolManager(
            toolset=step.wrapped,
            root_capability=parent_manager.root_capability if parent_manager is not None else None,
            ctx=ctx,
            tools=step.wrapped_tools,
        )
        watch.launched += 1
        self.stats.launched += 1
        launch_id = f'{watch.tool_call_id}__spec_{watch.launched}'
        task = asyncio.ensure_future(
            _run_speculative(tool_manager, original_name, launch_id, extracted.kwargs, step.serialize)
        )
        call = SpeculativeCall(
            sandbox_name=sandbox_name,
            original_name=original_name,
            kwargs=extracted.kwargs,
            task=task,
            launch_id=launch_id,
            started_at=time.perf_counter(),
        )
        task.add_done_callback(lambda _task, _call=call: setattr(_call, 'settled_at', time.perf_counter()))
        self.calls_for(watch, sandbox_name, extracted.kwargs).append(call)
        produced.append(
            SpeculativeCallLaunchedEvent(
                tool_call_id=watch.tool_call_id,
                launch_id=launch_id,
                sandbox_function=sandbox_name,
                wrapped_tool_name=original_name,
                arguments=extracted.kwargs,
                line_start=extracted.line_start,
                line_end=extracted.line_end,
                phase=phase,
            )
        )

    def _collect_settles(self, produced: list[CapabilityEvent]) -> None:
        """Report launches that finished since the watcher last saw stream traffic."""
        for watch in self._parts.values():
            for queue in watch.calls.values():
                for call in queue:
                    if call.settle_emitted or not call.task.done():
                        continue
                    call.settle_emitted = True
                    state = call.settled_state()
                    produced.append(
                        SpeculativeCallSettledEvent(
                            tool_call_id=watch.tool_call_id,
                            launch_id=call.launch_id,
                            outcome='failed' if state == 'failed' else 'ready',
                            elapsed_ms=call.elapsed_ms(),
                        )
                    )

    def calls_for(self, watch: _PartWatch, sandbox_name: str, kwargs: dict[str, Any]) -> deque[SpeculativeCall]:
        return watch.calls.setdefault(_canonical_key(sandbox_name, kwargs), deque())

    # -- execution side ---------------------------------------------------------------------

    def prelaunch_for_execution(self, parent_tool_call_id: str, code: str, ctx: RunContext[Any]) -> None:
        """Launch every literal eligible call the snippet holds, before Monty takes a step.

        Sequential `await`s execute one statement at a time, so a cold eligible call blocks
        every statement after it. At execution start the code is complete: launching the calls
        the stream watcher never saw (single-chunk argument deltas, provider quirks) means the
        snippet's awaits collect from tasks that are all already running, and wall time
        approaches the longest call instead of the sum. Only the deficit against launches
        already in flight is started, so FIFO multiplicity stays exact.
        """
        step = self._step
        if step is None or not step.eligible:
            return
        try:
            body = ast.parse(code).body
        except SyntaxError:
            return
        extracted_calls = list(_literal_calls(body, step.eligible))
        if not extracted_calls:
            return
        watch = self._parts.get(parent_tool_call_id)
        if watch is None:
            watch = _PartWatch(tool_call_id=parent_tool_call_id)
            self._parts[parent_tool_call_id] = watch
        demanded: dict[str, int] = {}
        for extracted in extracted_calls:
            key = _canonical_key(extracted.sandbox_name, extracted.kwargs)
            demanded[key] = demanded.get(key, 0) + 1
            in_flight = sum(len(w.calls.get(key, ())) for w in self._parts.values())
            if demanded[key] <= in_flight:
                continue
            if watch.launched >= _MAX_SPECULATIONS_PER_PART:
                return
            self._launch(watch, step, extracted, ctx, self.pending_events, phase='execution')

    def eligible(self, sandbox_name: str) -> bool:
        """Whether this sandbox function may speculate this step; drives miss reporting."""
        return self._step is not None and sandbox_name in self._step.eligible

    def claim(self, parent_tool_call_id: str, sandbox_name: str, kwargs: dict[str, Any]) -> SpeculativeCall | None:
        """Pop the oldest in-flight launch matching this dispatch, if any.

        Prefers the watch recorded under this part's id, then falls back to any other watch
        holding an exact `(function, arguments)` match: some providers re-key a tool call
        between the streamed part and its executed form, which would otherwise turn every
        launch into a miss, and the allowlist's purity promise makes an identical launch from
        another part interchangeable.
        """
        key = _canonical_key(sandbox_name, kwargs)
        primary = self._parts.get(parent_tool_call_id)
        watches = [primary] if primary is not None else []
        watches.extend(watch for part_id, watch in self._parts.items() if part_id != parent_tool_call_id)
        for watch in watches:
            queue = watch.calls.get(key)
            if queue:
                return queue.popleft()
        return None

    async def flush_events(self, ctx: RunContext[Any]) -> None:
        """Emit buffered execution-side events; called from `CodeMode`'s hook dispatches."""
        events, self.pending_events = self.pending_events, []
        for event in events:
            await emit_best_effort(ctx, event)

    def part_summary(self, parent_tool_call_id: str) -> dict[str, Any] | None:
        """Summarize one executed part's buffered outcomes, for the tool return's metadata.

        Must be read after `evict_part` and before the hook flush drains the buffer: that is
        the only window where the part's claims, misses, and evictions are all pending.
        Under composed eager execution, claims dispatched from pumped fragments carry the
        fragment's derived id (`{part}~e{n}`), so those count toward the part too.
        """

        def belongs(event_id: str | None) -> bool:
            return event_id == parent_tool_call_id or (event_id or '').startswith(f'{parent_tool_call_id}~e')

        hits = [
            event
            for event in self.pending_events
            if isinstance(event, SpeculativeCallClaimedEvent) and belongs(event.tool_call_id)
        ]
        misses = sum(
            1
            for event in self.pending_events
            if isinstance(event, SpeculativeCallMissedEvent) and belongs(event.tool_call_id)
        )
        wasted = sum(
            1
            for event in self.pending_events
            if isinstance(event, SpeculativeCallEvictedEvent) and belongs(event.tool_call_id)
        )
        if not hits and not misses and not wasted:
            return None
        return {
            'hits': len(hits),
            'hidden_ms': round(sum(event.elapsed_ms for event in hits), 3),
            'misses': misses,
            'wasted': wasted,
        }

    async def evict_part(self, parent_tool_call_id: str) -> None:
        """Drop unclaimed launches for one executed `run_code` part.

        Called when the snippet finishes successfully: whatever was not claimed --
        wrong-branch conditionals, rewritten lines -- is garbage for this part. Failed
        snippets keep their launches: syntax and type errors fail before any dispatch, and
        the retry claims the surviving launches under its fresh tool call id.
        """
        watch = self._parts.pop(parent_tool_call_id, None)
        if watch is not None:
            await self._cancel_watch(watch)

    async def close(self, ctx: RunContext[Any]) -> None:
        """Run-end cleanup: cancel every launch no snippet ever claimed, then report it."""
        parts, self._parts = self._parts, {}
        self._index_to_part.clear()
        for watch in parts.values():
            await self._cancel_watch(watch)
        await self.flush_events(ctx)

    async def _cancel_watch(self, watch: _PartWatch) -> None:
        evicted: list[tuple[SpeculativeCall, Literal['pending', 'ready', 'failed']]] = []
        tasks: list[asyncio.Task[_SpecOutcome]] = []
        for queue in watch.calls.values():
            for call in queue:
                # Capture where the launch stood before cancellation rewrites it.
                evicted.append((call, call.settled_state()))
                call.task.cancel()
                tasks.append(call.task)
                self.stats.evicted += 1
        if tasks:
            # Await the cancellations so dispatched work has fully unwound before the run moves
            # on, mirroring `MontyExecutor.run`'s cleanup. Outcomes are deliberately discarded.
            await asyncio.gather(*tasks, return_exceptions=True)
        for call, state in evicted:
            self.pending_events.append(
                SpeculativeCallEvictedEvent(
                    tool_call_id=watch.tool_call_id,
                    launch_id=call.launch_id,
                    wrapped_tool_name=call.original_name,
                    state=state,
                )
            )


async def _run_speculative(
    tool_manager: ToolManager[Any],
    original_name: str,
    provisional_id: str,
    kwargs: dict[str, Any],
    serialize: Callable[[Any], Any],
) -> _SpecOutcome:
    """Run one early launch through the same nested-manager path as a cold sandbox call.

    Mirrors the essential behavior of the cold dispatch in `CodeModeToolset.call_tool`
    (`run_tool_call`), settling into a `_SpecOutcome` instead of recording message parts: the
    provisional tool call id is replaced at adoption, when the executing snippet's own id and
    call counter exist. Failures settle rather than raise so an unclaimed failed launch never
    surfaces as an unretrieved task exception.
    """
    from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UserError
    from pydantic_ai.messages import ToolReturn
    from pydantic_ai.tools import ToolDenied

    call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=provisional_id)
    try:
        result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
    except (CallDeferred, ApprovalRequired) as e:
        return _SpecOutcome(
            error=UserError(
                f'Tool {original_name!r} raised {type(e).__name__} inside code mode, '
                'but no `HandleDeferredToolCalls` capability resolved it. Add a handler '
                'capability on the agent so deferred and approval-required calls can '
                'be resolved inline.'
            )
        )
    except Exception as e:
        return _SpecOutcome(error=e)

    if isinstance(result, ToolDenied):
        return _SpecOutcome(
            error=RuntimeError(f'Tool {original_name!r} call denied: {result.message}'),
            denied_message=result.message,
        )

    metadata: Any = None
    if isinstance(result, ToolReturn):
        metadata = result.metadata
        result = result.return_value
    return _SpecOutcome(content=result, serialized=serialize(result), metadata=metadata)

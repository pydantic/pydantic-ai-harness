"""Capability events emitted by `CodeMode`'s speculative execution.

The speculation lifecycle is otherwise invisible: launches happen while the model is still
streaming, and adoption happens inside `run_code`'s dispatch. These events surface every
transition on the run's event stream so UIs can render it live (the streamed code with its
closed/provisional boundary, per-call launch spans with in-flight timers, and hit/miss/evicted
outcomes at execution) and so other capabilities can subscribe with
[`on_event`][pydantic_ai.capabilities.on_event].

All events carry the `run_code` part's tool call id in the inherited
[`tool_call_id`][pydantic_ai.messages.CapabilityEvent.tool_call_id] field, so one streamed
snippet's events correlate across the stream and execution phases. Launch-scoped events share a
`launch_id` unique within the run.

Consumers should treat the stream as best-effort ordered but complete: a launch always produces
exactly one of claimed or evicted, and settles arrive between launch and that terminal event
when stream traffic allows (settled state is otherwise folded into the terminal event's fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent
from pydantic_ai.tools import RunContext

CODE_MODE_EVENTS = 'code_mode'
"""Namespace for `CodeMode` capability events."""


async def emit_best_effort(ctx: RunContext[Any], event: CapabilityEvent) -> None:
    """Emit a speculation event, tolerating contexts without a live event stream.

    Speculation events are observability, never load-bearing. A `RunContext` outside a running
    agent (direct toolset use, post-run teardown) has no stream buffer and
    [`emit_event`][pydantic_ai.tools.RunContext.emit_event] raises `UserError`; the event is
    dropped rather than failing the work it describes.
    """
    try:
        await ctx.emit_event(event)
    except UserError:
        return


@dataclass(kw_only=True)
class SpeculativeCodeUpdateEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """The decoded `run_code` snippet grew.

    Emitted per argument delta while a `run_code` tool call streams, carrying the full decoded
    code prefix, so consumers render live code without re-implementing partial-JSON decoding.
    `closed_statements` marks the boundary between statements that are provably complete
    (speculation-eligible; highlightable) and the still-growing tail (provisional; render dim).
    Mirrors the model's delta cadence -- consumers wanting fewer repaints can debounce.
    """

    code: str
    """The decoded code prefix streamed so far."""

    closed_statements: int
    """Top-level statements at the front of `code` that are provably complete."""


@dataclass(kw_only=True)
class SpeculativeCallLaunchedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """A sandbox tool call started executing while its snippet was still streaming."""

    launch_id: str
    """Identifier shared by this launch's later settle and claimed/evicted events."""

    sandbox_function: str
    """The function name as written in the snippet (possibly sanitized)."""

    wrapped_tool_name: str
    """The wrapped tool actually dispatched."""

    arguments: dict[str, Any]
    """The literal keyword arguments the call was launched with."""

    line_start: int
    """1-based first line of the launching statement within the snippet."""

    line_end: int
    """1-based last line of the launching statement within the snippet."""

    phase: Literal['streaming', 'execution'] = 'streaming'
    """When the launch happened: `streaming` overlaps the model's own generation;
    `execution` is the pre-Monty prefetch that parallelizes the snippet's sequential awaits."""


@dataclass(kw_only=True)
class SpeculativeCallSettledEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """A launched call finished (successfully or not) while the stream was still flowing.

    Emitted from the stream watcher when it next observes an event after the launch's task
    completes, so it can trail completion by a few deltas. A launch that settles after the
    stream ends produces no settled event; its terminal claimed/evicted event carries the
    settled state instead.
    """

    launch_id: str
    """Identifier from this launch's `SpeculativeCallLaunchedEvent`."""

    outcome: Literal['ready', 'failed']
    """Whether the early run produced a result or an error (delivered at claim, like a cold call's)."""

    elapsed_ms: float
    """Wall-clock from launch to completion."""


@dataclass(kw_only=True)
class SpeculativeCallClaimedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """The executing snippet dispatched a call and adopted an in-flight launch: a hit."""

    launch_id: str
    """Identifier from this launch's `SpeculativeCallLaunchedEvent`."""

    nested_tool_call_id: str
    """The adopted call's id in message history (`{part_id}__{n}`), distinct from the launch id."""

    wrapped_tool_name: str
    ready_at_claim: bool
    """Whether the early run had already finished when the snippet asked for it."""

    elapsed_ms: float
    """Wall-clock from launch until the result was available to the claimant."""


@dataclass(kw_only=True)
class SpeculativeCallMissedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """The executing snippet dispatched a speculation-eligible call that no launch matched.

    Only emitted for calls whose function was eligible to speculate this step; dispatches of
    never-eligible tools are ordinary calls, not misses. A miss is always safe -- the call runs
    cold -- but a systematic miss pattern (for example, every call missing on one provider)
    signals broken claim keying rather than bad luck.
    """

    sandbox_function: str
    wrapped_tool_name: str
    nested_tool_call_id: str


@dataclass(kw_only=True)
class SpeculativeCallEvictedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """A launch was discarded without being claimed: wasted work.

    Emitted when the snippet finishes without dispatching a matching call (untaken branch,
    rewritten plan, snippet error) or at run end for parts that never executed.
    """

    launch_id: str
    """Identifier from this launch's `SpeculativeCallLaunchedEvent`."""

    wrapped_tool_name: str
    state: Literal['pending', 'ready', 'failed']
    """Where the launch was when discarded: still running, or settled with a result or error."""


@dataclass(kw_only=True)
class EagerPrefixCommittedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """An eagerly executed prefix was adopted by the snippet's `run_code` dispatch.

    Emitted once per eager part whose pump fed at least one statement, after the tail
    executed successfully. The generation overlap the prefix bought is `executed_ms`
    minus `waited_ms`: what the pump ran versus what the dispatch still had to wait.
    """

    statements: int
    """Fragments the pump executed before the dispatch committed the tail."""

    executed_ms: float
    """Wall-clock the pump spent running the prefix statements."""

    waited_ms: float
    """How long the dispatch waited for the pump to drain before feeding the tail."""

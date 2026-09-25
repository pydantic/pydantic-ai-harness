"""Shared Monty execution loop for code-execution capabilities.

Drives a Monty REPL via the snapshot API (`feed_start`/`resume`), dispatching external
function calls back to a host-supplied async callback.

Two capabilities build on this:

- `code_mode`: the dispatch callback runs the agent's own tools.
- `dynamic_workflow`: the dispatch callback runs sub-agents.

The snapshot API (rather than `feed_run`) is used deliberately: it exposes each suspension
to this host-controlled loop, which owns sequential barriers, dispatch cancellation, and
trace context. The loop drives Monty's async bindings only: `AsyncMonty` for local worker
subprocesses and `AsyncMontyWebsocket` for remote workers, which hand it the same snapshot
types. Under Temporal, this loop runs workflow-side and replays; each Monty call then goes
through a blocking portal (see `call_monty`), while nested durable-wrapped tools cross their
configured activity boundaries.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Container, Coroutine
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal
from opentelemetry import context as otel_context
from pydantic_ai import RunContext
from typing_extensions import TypeVarTuple, Unpack

try:
    from pydantic_monty import (
        AsyncFunctionSnapshot,
        AsyncFutureSnapshot,
        AsyncNameLookupSnapshot,
        AsyncSnapshot,
        CollectString,
        ExternalException,
        ExternalFuture,
        ExternalReturnValue,
        ExternalSettledResult,
        MontyComplete,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for code-execution capabilities. Install it with: '
        'pip install "pydantic-ai-harness[code-mode]" or "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

# Dispatch callback: given the sandbox function name and keyword arguments,
# perform the host-side work (tool call or sub-agent run) and return the result.
DispatchFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]

_T = TypeVar('_T')
_Args = TypeVarTuple('_Args')


@runtime_checkable
class _TemporalDurability(Protocol):
    """The part of Temporal's public durability capability the Monty loop needs."""

    in_durable_context: bool


def in_temporal_workflow(ctx: RunContext[object]) -> bool:
    """Whether this tool call runs in a Temporal workflow, without importing its optional extra."""
    return any(
        any(base.__module__.startswith('pydantic_ai.durable_exec.temporal') for base in type(capability).__mro__)
        and isinstance(capability, _TemporalDurability)
        and capability.in_durable_context
        for capability in ctx.capabilities.values()
    )


# Running Monty inside a Temporal workflow
# ----------------------------------------
# Monty's bindings are async and complete each awaited call from Monty's own I/O thread by waking
# the event loop the `await` started on (`loop.call_soon_threadsafe`). A Temporal workflow's event
# loop cannot be woken that way, so the call would never complete. Inside a workflow, the four
# helpers below therefore route every Monty call through an `anyio` blocking portal: a helper
# thread running a normal asyncio loop. The workflow thread blocks until the sandbox suspends or
# completes, exactly as it did with Monty's former sync bindings, and control is back in the
# workflow between calls, where nested tools run as activities. Outside a workflow the portal is
# `None` and every helper is a plain `await`.


def open_monty_portal(stack: AsyncExitStack, *, in_temporal_workflow: bool) -> BlockingPortal | None:
    """Open the portal Monty calls need inside a Temporal workflow; `stack` closes it."""
    if not in_temporal_workflow:
        return None
    return stack.enter_context(start_blocking_portal())


async def call_monty(
    portal: BlockingPortal | None, fn: Callable[[Unpack[_Args]], Awaitable[_T]], *args: Unpack[_Args]
) -> _T:
    """Await one call into Monty's async bindings, `fn(*args)`, through `portal` when there is one."""
    if portal is None:
        return await fn(*args)
    return portal.call(fn, *args)


async def enter_monty(
    stack: AsyncExitStack, resource: AbstractAsyncContextManager[_T], portal: BlockingPortal | None
) -> _T:
    """Enter a Monty pool or session on `stack`, through `portal` when there is one."""
    if portal is None:
        return await stack.enter_async_context(resource)
    return stack.enter_context(portal.wrap_async_context_manager(resource))


async def release_monty(stack: AsyncExitStack) -> None:
    """Exit the Monty resources on `stack`, even while the run is being cancelled.

    Cancellation is delivered again at every suspension point while an enclosing cancel scope
    stays cancelled, which would abandon the session or pool exit half way and leak the worker.
    The exit waits for a snippet that is still running, so it is bounded by `max_duration_secs`
    (and, for a remote worker, the transport's per-turn deadline); Monty offers no way to
    interrupt a running feed.
    """
    with anyio.CancelScope(shield=True):
        await stack.aclose()


@dataclass
class PendingCall:
    """A dispatched call and the suspension context under which it executes."""

    call: asyncio.Task[Any] | Coroutine[Any, Any, Any]
    context: otel_context.Context


def is_sandbox_panic(exc: BaseException) -> bool:
    """Whether `exc` is a Rust-side sandbox panic surfacing through pyo3.

    pyo3 raises `pyo3_runtime.PanicException`, a `BaseException` (not `Exception`) subclass
    from a module that cannot be imported, so it is matched by name. A panic can surface from
    monty's host-side bindings, so callers should convert it to a retry rather than let it
    tear down the whole agent run.
    """
    return type(exc).__name__ == 'PanicException'


class PrintCapture:
    """Collects bounded print output from a Monty REPL."""

    def __init__(self) -> None:
        self.callback = CollectString()

    @property
    def joined(self) -> str:
        return self.callback.output

    def prepend_to(self, error_message: str) -> str:
        """Prefix captured stdout to an error message, so the model sees what printed before the error."""
        printed = self.joined.rstrip('\n')
        if not printed:
            return error_message
        return f'[stdout before error]\n{printed}\n[/stdout before error]\n{error_message}'


@dataclass
class MontyExecutor:
    """Drives a Monty REPL to completion, dispatching external calls to a host callback.

    Single-use: it accumulates per-run state in `_pending`/`_pre_resolved`, so construct a
    fresh executor for each `run` rather than reusing or sharing one across concurrent runs.

    External calls are handled by execution mode:

    - **Parallel** (`async def`): deferred via `resume({'future': ...})` and eagerly
      scheduled as `asyncio.Task`s. Resolved at `FutureSnapshot` via `asyncio.gather`.
    - **Per-call sequential** (`def`, name in `sequential_names`): resolved inline at
      `FunctionSnapshot`. Any pending parallel tasks are awaited first (barrier).
    - **Global sequential** (when selected by the run context): all calls deferred but
      stored as bare coroutines and awaited one-at-a-time to prevent interleaving.
    """

    dispatch: DispatchFn
    valid_names: Container[str]
    sequential_names: set[str] = field(default_factory=set[str])
    global_sequential: bool = False
    # Set inside a Temporal workflow; see `open_monty_portal`.
    portal: BlockingPortal | None = None

    # Parallel calls deferred but not yet resolved, keyed by Monty call id.
    _pending: dict[int, PendingCall] = field(default_factory=dict[int, PendingCall], init=False)
    # Parallel results awaited early at a sequential barrier, before their FutureSnapshot is reached.
    _pre_resolved: dict[int, ExternalSettledResult] = field(
        default_factory=dict[int, ExternalSettledResult], init=False
    )

    async def run(self, feed_start: Callable[[], Awaitable[AsyncSnapshot]]) -> MontyComplete:
        """Drive the REPL from `feed_start` (a bound `AsyncMontySession.feed_start`) until it completes."""
        try:
            state = await call_monty(self.portal, feed_start)
            while not isinstance(state, MontyComplete):
                if isinstance(state, AsyncNameLookupSnapshot):
                    # Leave the name undefined so the sandbox raises `NameError`.
                    state = await call_monty(self.portal, state.resume)
                elif isinstance(state, AsyncFunctionSnapshot):
                    state = await self._handle_function(state)
                else:
                    state = await self._resolve_futures(state)
        finally:
            cancelled: list[asyncio.Task[Any]] = []
            for pending in self._pending.values():
                call = pending.call
                if isinstance(call, asyncio.Task):
                    call.cancel()
                    cancelled.append(call)
                else:
                    call.close()
            if cancelled:
                # `cancel()` only schedules a `CancelledError` at each task's next suspension
                # point; await them so dispatched work (e.g. sub-agent runs mutating shared
                # usage) has fully unwound before this returns. `return_exceptions=True` keeps
                # one task's teardown error from masking the original exception, and the
                # results are deliberately discarded. Shielded: run cancellation can land here
                # with an enclosing anyio scope already cancelled, and that scope re-cancels
                # its tasks on every event-loop cycle -- each delivery either aborts this await
                # outright (abandoning the tasks mid-unwind) or is forwarded through the
                # `gather` into every task, breaking any await their cleanup performs. The
                # shield holds for anyio-scope cancellation; a raw second `Task.cancel()` can
                # still pierce it.
                with anyio.CancelScope(shield=True):
                    await asyncio.gather(*cancelled, return_exceptions=True)
        return state

    async def _handle_function(self, snapshot: AsyncFunctionSnapshot) -> AsyncSnapshot:
        """Dispatch (or defer) a single external function call."""
        if snapshot.is_os_function:
            # OS calls (env, clock, filesystem) are answered from the feed's mounts and the
            # `os=` handler captured at `feed_start`, falling back to monty's unhandled default.
            return await call_monty(self.portal, snapshot.resume_auto)

        name = snapshot.function_name
        if name not in self.valid_names:
            return await self._raise_in_sandbox(snapshot, NameError(f'Unknown function: {name}'))

        if snapshot.args:
            return await self._raise_in_sandbox(
                snapshot, TypeError(f'{name}() does not accept positional arguments; use keyword arguments')
            )

        if name in self.sequential_names:
            # Rendered as `def` (sync), so the sandbox code doesn't `await` the result --
            # resolve inline. Await pending parallel tasks first (barrier) for ordering.
            # The dispatch coroutine is created only after the barrier: it is not in
            # `_pending`, so if it existed while the barrier awaits and we were cancelled
            # there, `run`'s cleanup would never close it.
            for cid in list(self._pending):
                self._pre_resolved[cid] = await _await_external(self._pending.pop(cid))
            try:
                call = self._dispatch(snapshot, parallel=False)
            except Exception as exc:
                return await self._raise_in_sandbox(snapshot, exc)
            # The wrapped outcome (`{'return_value': ...}` / `{'exception': ...}`) is already
            # exactly the payload `resume` expects.
            return await call_monty(self.portal, snapshot.resume, await _await_external(call))

        # Deferred execution -- resolved later at FutureSnapshot.
        try:
            call = self._dispatch(snapshot, parallel=not self.global_sequential)
        except Exception as exc:
            # `dispatch` refused the call before building its coroutine (e.g. an exhausted
            # per-snippet budget). Deliver the error at the sandbox call site, the same way a
            # failure raised inside the coroutine is delivered, rather than letting it abort the
            # feed: calls that already completed keep the results the host recorded for them, and
            # the snippet can still return them. Nothing was scheduled, so there is no task to
            # clean up and no further work is admitted.
            return await self._raise_in_sandbox(snapshot, exc)
        self._pending[snapshot.call_id] = call
        return await call_monty(self.portal, snapshot.resume, ExternalFuture(future=...))

    async def _raise_in_sandbox(self, snapshot: AsyncFunctionSnapshot, exc: Exception) -> AsyncSnapshot:
        """Resume the suspended call by raising `exc` at its sandbox call site."""
        return await call_monty(self.portal, snapshot.resume, ExternalException(exception=exc))

    def _dispatch(self, snapshot: AsyncFunctionSnapshot, *, parallel: bool) -> PendingCall:
        # Compatibility with Monty before https://github.com/pydantic/monty/pull/885.
        trace_context: Callable[[], otel_context.Context] = getattr(snapshot, 'trace_context', otel_context.get_current)
        context = trace_context()
        token = otel_context.attach(context)
        try:
            call = self.dispatch(snapshot.function_name, snapshot.kwargs)
            # Tasks inherit the active context; bare coroutines need it restored when awaited.
            return PendingCall(asyncio.ensure_future(call) if parallel else call, context)
        finally:
            otel_context.detach(token)

    async def _resolve_futures(self, snapshot: AsyncFutureSnapshot) -> AsyncSnapshot:
        """Resolve the deferred calls a `FutureSnapshot` is waiting on."""
        pending_ids = snapshot.pending_call_ids
        results: dict[int, ExternalSettledResult] = {}
        for cid in pending_ids:
            if cid in self._pre_resolved:
                results[cid] = self._pre_resolved.pop(cid)
            elif self.global_sequential:
                results[cid] = await _await_external(self._pending.pop(cid))

        # Gather any remaining parallel tasks concurrently. They stay in `_pending` until
        # gather returns, so the cleanup in `run` can still cancel them if this is cancelled.
        gather_ids = [cid for cid in pending_ids if cid not in results]
        if gather_ids:
            settled = await asyncio.gather(*(self._pending[cid].call for cid in gather_ids), return_exceptions=True)
            for cid, outcome in zip(gather_ids, settled):
                del self._pending[cid]
                results[cid] = _wrap_gathered(outcome)

        return await call_monty(self.portal, snapshot.resume, results)


async def _await_external(call: PendingCall) -> ExternalReturnValue | ExternalException:
    """Await a single deferred call and wrap its outcome for Monty."""
    token = otel_context.attach(call.context)
    try:
        result = await call.call
    except Exception as exc:
        return ExternalException(exception=exc)
    finally:
        otel_context.detach(token)
    return ExternalReturnValue(return_value=result)


def _wrap_gathered(outcome: Any) -> ExternalReturnValue | ExternalException:
    """Wrap an `asyncio.gather(return_exceptions=True)` outcome for Monty."""
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):  # pragma: no cover
        raise outcome
    return ExternalReturnValue(return_value=outcome)

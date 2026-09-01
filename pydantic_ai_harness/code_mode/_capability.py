"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter, ValidationError
from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, NativeToolSearchReturnPart, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector
from typing_extensions import TypedDict

from pydantic_ai_harness.code_mode._eager import EagerState
from pydantic_ai_harness.code_mode._speculation import SpeculationState, SpeculationStats
from pydantic_ai_harness.code_mode._toolset import (
    CodeModeMount,
    CodeModeOS,
    CodeModeResourceLimits,
    CodeModeToolset,
    _in_durable_execution,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult


_DISCOVERY_ANNOUNCEMENT_PREFIX = (
    'New functions are now available inside `run_code`. Their signatures have been '
    'added to the available-functions catalog in the system prompt'
)


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that exposes selected tools as callables inside a `run_code` sandbox.

    By default (`tools='all'`) every eligible regular tool the agent has is wrapped
    behind a single `run_code` tool -- the model writes Python that calls them as
    functions instead of issuing tool calls directly. Framework control tools,
    undiscovered deferred tools, native fallbacks, and other code-execution tools
    remain native.

    Pass a list of tool names or a callable predicate to `tools` to split the
    toolset: matching tools become callables inside the sandbox, and the rest
    stay visible to the model as normal tool calls.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import CodeMode

    # Sandbox all tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode()])

    # Sandbox only specific tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode(tools=['search', 'fetch'])])
    ```

    By default, sandboxed code cannot touch the host -- no filesystem, environment
    variables, or clock. Two parameters open it up:

    - `mount` shares specific host directories: reach for it when the agent reads or
      writes real files.
    - `os_access` routes the sandbox's OS calls to a handler you provide: reach for it
      when the agent needs environment variables, the clock, or filesystem behavior you
      control.

    `mount` exposes selected host directories. The built-in `OSAccess` has an
    isolated filesystem and environment but uses the host clock by default; custom
    OS handlers can expose other host resources.

    ```python
    from pydantic_monty import MountDir

    agent = Agent('openai:gpt-5', capabilities=[CodeMode(mount=MountDir(virtual_path='/work', host_path='/tmp/agent-work'))])
    ```
    """

    tools: ToolSelector[AgentDepsT] = field(default='all')
    """Which wrapped tools should be sandboxed inside `run_code`.

    - `'all'` (default): every eligible regular tool the agent has is sandboxed.
    - `Sequence[str]`: only tools whose names are listed are sandboxed.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: tools where the
      callable returns `True` are sandboxed; the rest stay as native tool calls.
    """

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    _: KW_ONLY

    max_tool_calls: int = 100
    """Maximum nested tool calls dispatched by one `run_code` invocation.

    Budget is reserved before each call is scheduled, so a snippet cannot allocate host tasks
    beyond this many. Calls past the budget are refused at the sandbox call site.
    """

    os_access: CodeModeOS | None = None
    """Give sandboxed code environment variables, the clock, and file I/O through a handler you provide; unset, they are unavailable."""

    mount: CodeModeMount | None = None
    """Host directories to expose to sandboxed `pathlib` code; each mount's `mode` controls whether writes reach the host."""

    resource_limits: CodeModeResourceLimits | Literal['unlimited'] | None = None
    """Sandbox execution limits, applied per Monty session.

    `None` applies a 30-second execution and 256 MiB heap backstop. The guarantee is per snippet:
    no single `run_code` snippet runs longer than `max_duration_secs`. It is not a run-wide budget,
    since consecutive calls share one session allowance and any reset of the session (`restart:
    true`, a crash, a type error, a host-side failure) starts a fresh one. `'unlimited'` removes
    both caps.
    """

    speculate: Sequence[str] | Literal['declared'] | None = None
    """Tool names that may start executing while the model is still streaming `run_code` code.

    Experimental. When set, the streamed `code` argument is parsed as it arrives, and calls to
    these tools whose arguments are all keyword literals launch immediately; when the completed
    snippet executes, matching dispatches claim the in-flight results instead of starting cold.
    This overlaps tool latency with model generation (speculative programmatic tool calling,
    <https://alexzhang13.github.io/blog/2026/spec-ptc/>).

    Name only tools without observable side effects: a speculated call can run for a branch the
    snippet never takes, and its early launch is otherwise indistinguishable from the normal
    call only when re-running or discarding it is harmless. Tool hooks fire at launch time.
    Unclaimed launches are cancelled when the snippet finishes. Enabling this puts runs in
    streaming mode, and has no effect under durable execution.

    Pass `'declared'` instead of a list to trust the tools' own definitions as evidence:
    first-party tools marked `Tool(..., metadata={'read_only': True})` (or
    `'idempotent'`), and MCP tools whose server publishes the `readOnlyHint` or
    `idempotentHint` annotation.
    """

    eager: bool = False
    """Execute streamed `run_code` statements in the live REPL as they close.

    Experimental. The lighter streaming tier: no calls are predicted, so nothing can miss or
    be wasted -- each top-level statement runs, in program order, as soon as it has fully
    streamed, and the `run_code` dispatch only executes the remainder. Side effects land
    before the tool call is committed and are not rolled back; a statement that fails leaves
    the session exactly as a failed snippet does today (assignments before the failing line
    persist) and surfaces the error as the `run_code` result. Composes with `speculate`:
    eager execution advances the program frontier while speculation launches eligible
    calls beyond it (branch arms, calls behind a blocking statement), and the eager
    feeds' dispatches claim those launches. Enabling this puts runs in streaming mode,
    and has no effect under durable execution.

    Two consequences of running before the call completes:

    - Statements execute before the completed `run_code` call reaches `before_tool_execute`
      hooks, so a guard capability that would block, rewrite, or defer `run_code` for
      approval is applied only to the dispatch, after the streamed prefix already ran. Do
      not enable `eager` on runs that gate `run_code` behind such a guard.
    - A `restart: true` call re-executes the full snippet on a fresh session; the watcher
      stops feeding as soon as `restart` appears in the streamed arguments, but statements
      fed before the key streams have already run once, so their side effects repeat.

    Requires the asyncio event loop; on other async backends (Trio) the watcher stays
    inactive and `run_code` executes normally at dispatch.
    """

    speculation_stats: SpeculationStats = field(default_factory=SpeculationStats, init=False, repr=False)
    """Aggregate launch/adopt/evict counters across this instance's runs; observability for the POC."""

    _speculation_state: SpeculationState | None = field(default=None, init=False, repr=False)

    _eager_state: EagerState | None = field(default=None, init=False, repr=False)

    dynamic_catalog: bool = False
    """Keep the `run_code` tool definition cache-stable as the sandboxed toolset grows.

    By default the signatures of all sandboxed tools are rendered into `run_code`'s
    description, which lives in the prompt-cache-keyed tool-definitions block. When the
    toolset changes mid-run -- e.g. [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]
    reveals a new tool that then gets folded into `run_code` -- the description changes and
    busts the prefix cache from that point on.

    Set `dynamic_catalog=True` to instead:

    - keep only the static base prose (sandbox restrictions, return-value contract) in
      `run_code.description`, so the tool-definitions block stays byte-stable across
      discoveries;
    - move the "available functions" catalog (TypedDict definitions + signatures) into
      agent instructions as a dynamic
      [`InstructionPart`][pydantic_ai.messages.InstructionPart], which providers with
      static/dynamic instruction splitting (Anthropic, Bedrock) place after the cache
      breakpoint;
    - announce newly-discovered tools via a short
      [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] enqueued through
      [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue], so the model knows the
      new functions are callable without rewriting the cached description.

    This pays off when paired with [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]: the
    tool-definitions cache survives discoveries at the cost of a larger (but
    cache-friendly) system prompt. With a fixed toolset and no `ToolSearch`, the default
    keeps the system prompt shorter and is the better choice.
    """

    _announced_tools: set[str] = field(default_factory=set[str], init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject malformed configuration at construction."""
        if isinstance(self.speculate, str) and self.speculate != 'declared':
            raise UserError(
                f"`speculate` accepts a list of tool names or the string 'declared', "
                f'not {self.speculate!r}. To allowlist one tool, pass a one-element list.'
            )

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CodeMode[AgentDepsT]:
        """Return a fresh instance so concurrent runs don't share mutable per-run state."""
        if not self.dynamic_catalog and self.speculate is None and not self.eager:
            return self
        clone = replace(self)
        # `replace` re-runs `__init__`, resetting `init=False` fields: `_announced_tools` starts
        # fresh (intended), and the stats object is rebound so callers holding this instance
        # observe counters accumulated by its per-run clones.
        clone.speculation_stats = self.speculation_stats
        if self.speculate is not None:
            if isinstance(self.speculate, str):
                allowlist: frozenset[str] | Literal['declared'] = 'declared'
            else:
                allowlist = frozenset(self.speculate)
            clone._speculation_state = SpeculationState(allowlist=allowlist, stats=self.speculation_stats)
        if self.eager:
            clone._eager_state = EagerState()
        return clone

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        wrapper = CodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            max_tool_calls=self.max_tool_calls,
            resource_limits=self.resource_limits,
            dynamic_catalog=self.dynamic_catalog,
            os_access=self.os_access,
            mount=self.mount,
            speculation=self._speculation_state,
            eager=self._eager_state,
        )
        if self._eager_state is not None:
            self._eager_state.bind(wrapper)
        return wrapper

    @property
    def has_wrap_run_event_stream(self) -> bool:
        """Report the stream hook only when a streaming tier is enabled.

        The base class detects a class-level override, which would put every `CodeMode` user in
        streaming mode; gating on the instance keeps plain `CodeMode` runs non-streaming.
        """
        return self.speculate is not None or self.eager

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Feed streamed `run_code` argument deltas to whichever streaming tier is active.

        Wrapped events pass through unmodified. The eager watcher acts purely by side
        effect, enqueueing closed statements for the REPL pump. The speculation watcher
        launches eligible calls and produces events, which are yielded directly into the
        stream right behind the event that produced them; yielding (rather than
        `ctx.emit_event`) is what makes them live, at the cost of bypassing `@on_event`
        listener dispatch, which happens upstream of capability wrappers. Inactive under
        durable execution, where overlapping non-deterministic work with the stream has no
        place in a replayed workflow.
        """
        eager = self._eager_state
        speculation = self._speculation_state
        if _in_durable_execution(ctx):
            eager = None
            speculation = None
        if eager is not None or speculation is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # Not on asyncio (Trio via AnyIO): both tiers schedule asyncio tasks, so the
                # watchers stay inactive and `run_code` executes normally at dispatch.
                eager = None
                speculation = None
        run_id = next((rid for rid, cap in ctx.capabilities.items() if cap is self), None)
        try:
            async for event in stream:
                yield event
                if eager is not None:
                    await eager.observe(event, ctx)
                if speculation is not None:
                    for spec_event in await speculation.observe(event, ctx):
                        yield replace(spec_event, capability_id=run_id) if run_id is not None else spec_event
        finally:
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()

    async def after_run(self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        """Cancel speculative launches and stop any statement pump before the run returns."""
        if self._speculation_state is not None:
            await self._speculation_state.close(ctx)
        if self._eager_state is not None:
            await self._eager_state.close()
        return result

    async def on_run_error(self, ctx: RunContext[AgentDepsT], *, error: BaseException) -> AgentRunResult[Any]:
        """Cancel speculative launches and stop any statement pump, then let the error propagate."""
        if self._speculation_state is not None:
            await self._speculation_state.close(ctx)
        if self._eager_state is not None:
            await self._eager_state.close()
        raise error

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Flush buffered speculation and eager events, and announce newly-discovered tools.

        Speculation claim/miss/eviction events and the eager prefix-commit event buffer
        during `run_code` execution because
        capability-event attribution requires a capability hook context; this dispatch is the
        first one after the snippet finishes. The discovery announcement is only active with
        `dynamic_catalog=True`; the native-search path is handled by
        [`after_model_request`][pydantic_ai_harness.CodeMode.after_model_request] instead
        (server-side search emits a `NativeToolSearchReturnPart` rather than a regular tool
        execute result).
        """
        if self._speculation_state is not None:
            await self._speculation_state.flush_events(ctx)
        if self._eager_state is not None:
            await self._eager_state.flush_events(ctx)
        if self.dynamic_catalog and tool_def.tool_kind == 'tool-search':
            self._announce_newly_discovered(ctx, _extract_discovered_names(result))
        return result

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Flush speculation events buffered by a snippet whose failure skipped `after_tool_execute`.

        A snippet that fails into a `ModelRetry` produces no `after_tool_execute` dispatch, so
        its buffered claim/miss/eviction events would otherwise wait for run end, after the
        stream has closed. The retry's next model request is the first hook context with a live
        stream; flushing here puts the events on it.
        """
        if self._speculation_state is not None:
            await self._speculation_state.flush_events(ctx)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Announce newly-discovered tools from a native (server-side) tool-search return.

        Only active with `dynamic_catalog=True`.
        """
        if self.dynamic_catalog:
            for part in response.parts:
                if isinstance(part, NativeToolSearchReturnPart):
                    self._announce_newly_discovered(ctx, _extract_discovered_names(part.content))
        return response

    def _announce_newly_discovered(self, ctx: RunContext[AgentDepsT], names: Sequence[str]) -> None:
        """Enqueue a system-prompt announcement for any names we haven't already announced."""
        fresh = [n for n in names if n not in self._announced_tools]
        if not fresh:
            return
        self._announced_tools.update(fresh)
        listing = ', '.join(f'`{name}`' for name in fresh)
        # Enqueue a `SystemPromptPart` so the announcement is framed as system-level context.
        # Mid-conversation `SystemPromptPart`s are rendered inline (not hoisted to the top-level
        # system prompt) on all providers since pydantic/pydantic-ai#5509, so this is cache-safe.
        ctx.enqueue(SystemPromptPart(content=f'{_DISCOVERY_ANNOUNCEMENT_PREFIX}: {listing}.'))


class _DiscoveredCatalog(TypedDict):
    """Lenient view of a tool-search return: just the entry list, items left unvalidated."""

    discovered_tools: list[object]


class _DiscoveredEntry(TypedDict):
    """Lenient view of one discovered-tool entry: only the name we announce."""

    name: str


_CATALOG_ADAPTER = TypeAdapter(_DiscoveredCatalog)
_ENTRY_ADAPTER = TypeAdapter(_DiscoveredEntry)


def _extract_discovered_names(content: object) -> list[str]:
    """Read newly-discovered tool names from a tool-search return content.

    Carried on both the local `ToolSearchReturnPart` and the native
    `NativeToolSearchReturnPart`. Validated leniently: a malformed catalog yields `[]` and a
    malformed entry is skipped, since the announcement is a courtesy nudge, not load-bearing
    logic.
    """
    try:
        catalog = _CATALOG_ADAPTER.validate_python(content)
    except ValidationError:
        return []
    names: list[str] = []
    for entry in catalog['discovered_tools']:
        try:
            names.append(_ENTRY_ADAPTER.validate_python(entry)['name'])
        except ValidationError:
            continue
    return names

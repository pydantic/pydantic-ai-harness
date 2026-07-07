"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError
from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.exceptions import StopRun
from pydantic_ai.messages import ModelResponse, NativeToolSearchReturnPart, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector
from pydantic_graph import End
from typing_extensions import TypedDict

from pydantic_ai_harness.code_mode._toolset import (
    _FINAL_OUTPUT_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    _RUN_CODE_TOOL_NAME,  # pyright: ignore[reportPrivateUsage]
    CodeModeMount,
    CodeModeOS,
    CodeModeToolset,
    _FinalOutput,  # pyright: ignore[reportPrivateUsage]  -- shared within the code_mode package
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext


_DISCOVERY_ANNOUNCEMENT_PREFIX = (
    'New functions are now available inside `run_code`. Their signatures have been '
    'added to the available-functions catalog in the system prompt'
)


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that exposes selected tools as callables inside a `run_code` sandbox.

    By default (`tools='all'`) every tool the agent has is wrapped behind a single
    `run_code` tool -- the model writes Python that calls them as functions instead
    of issuing tool calls directly.

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

    Both expose the real host to model-written code, so grant only what the task needs.

    ```python
    from pydantic_monty import MountDir

    agent = Agent('openai:gpt-5', capabilities=[CodeMode(mount=MountDir('/work', '/tmp/agent-work'))])
    ```
    """

    tools: ToolSelector[AgentDepsT] = field(default='all')
    """Which wrapped tools should be sandboxed inside `run_code`.

    - `'all'` (default): every tool the agent has is sandboxed.
    - `Sequence[str]`: only tools whose names are listed are sandboxed.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: tools where the
      callable returns `True` are sandboxed; the rest stay as native tool calls.
    """

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    _: KW_ONLY

    os_access: CodeModeOS | None = None
    """Give sandboxed code environment variables, the clock, and file I/O through a handler you provide; unset, they are unavailable."""

    mount: CodeModeMount | None = None
    """Host directories to expose to sandboxed `pathlib` code; each mount's `mode` controls whether writes reach the host."""

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

    allow_final_output: bool = False
    """Let sandboxed `run_code` scripts commit the agent's final output directly.

    When `True`, a `final_output(value)` function is exposed inside the sandbox. Calling it
    records `value` and, once the `run_code` call returns, `CodeMode` ends the run with that
    value as the output -- no extra model turn -- while the `run_code` tool return stays in
    message history. The value is passed through the agent's output validators but is **not**
    coerced to the declared output type, so it must already match it.

    Off by default: most code-mode agents should not be able to finish the run from a script.
    A schema-matching `run_code` return is never committed implicitly; committing is always an
    explicit `final_output(...)` call.
    """

    _announced_tools: set[str] = field(default_factory=set[str], init=False, repr=False)

    # The value a `run_code` script committed via `final_output()` this run, or None. Recorded in
    # `after_tool_execute` and consumed in `after_node_run` to end the run. A fresh instance per
    # run (see `for_run`) keeps concurrent runs from sharing it.
    _final_output_candidate: _FinalOutput | None = field(default=None, init=False, repr=False)

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CodeMode[AgentDepsT]:
        """Return a fresh instance so concurrent runs don't share per-run state.

        Needed when `dynamic_catalog` tracks `_announced_tools` or `allow_final_output` tracks
        `_final_output_candidate`; otherwise the shared instance is safe to reuse.
        """
        if not (self.dynamic_catalog or self.allow_final_output):
            return self
        return replace(self)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        return CodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            dynamic_catalog=self.dynamic_catalog,
            allow_final_output=self.allow_final_output,
            os_access=self.os_access,
            mount=self.mount,
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Announce discovered tools and record a `run_code` final-output commit.

        Two concerns share this hook:

        - With `dynamic_catalog=True`, announce newly-discovered tools from a local
          `search_tools` return. The native-search path is handled by
          [`after_model_request`][pydantic_ai_harness.CodeMode.after_model_request] instead
          (server-side search emits a `NativeToolSearchReturnPart` rather than a regular tool
          execute result).
        - With `allow_final_output=True`, read back a value the `run_code` script committed via
          `final_output()`. The value rides on the `ToolReturn` metadata; recording it here lets
          [`after_node_run`][pydantic_ai_harness.CodeMode.after_node_run] end the run once the
          `run_code` return is in message history. The result is returned unchanged.
        """
        if self.dynamic_catalog and tool_def.tool_kind == 'tool-search':
            self._announce_newly_discovered(ctx, _extract_discovered_names(result))
        if self.allow_final_output and call.tool_name == _RUN_CODE_TOOL_NAME:
            # `run_code` always returns a `ToolReturn`; the committed value (if any) rides on its
            # metadata as a `_FinalOutput`. Reading `.metadata` off `result` (typed `Any`) avoids
            # narrowing `result` itself, keeping the pass-through `return result` fully typed.
            marker: object = result.metadata.get(_FINAL_OUTPUT_METADATA_KEY)
            if isinstance(marker, _FinalOutput):
                self._final_output_candidate = marker
        return result

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """End the run with a value committed via `final_output()`, once tool returns are recorded.

        Only active with `allow_final_output=True`. When a `run_code` script committed a value
        this run and the node didn't already end the run, raise
        [`StopRun`][pydantic_ai.exceptions.StopRun]: Pydantic AI runs the value through the
        agent's output validators, preserves the pending `run_code` tool return in message
        history, and finishes without another model request. Otherwise the result is unchanged.
        """
        if self._final_output_candidate is not None and not isinstance(result, End):
            raise StopRun(self._final_output_candidate.value)
        return result

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

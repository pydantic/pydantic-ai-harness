"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import TypeAdapter, ValidationError
from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, NativeToolSearchReturnPart, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector
from typing_extensions import NotRequired, TypedDict

from pydantic_ai_harness.code_mode._toolset import (
    CodeModeMount,
    CodeModeOS,
    CodeModeResourceLimits,
    CodeModeToolset,
    MountDir,
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext


_DISCOVERY_ANNOUNCEMENT_PREFIX = (
    'New functions are now available inside `run_code`. Their signatures have been '
    'added to the available-functions catalog in the system prompt'
)


class CodeModeMountSpec(TypedDict):
    """One `mount` entry as an agent spec expresses it: `MountDir`'s keyword arguments.

    `MountDir` is a compiled class with no JSON representation of its own, so specs
    describe mounts with this shape and `CodeMode.from_spec` constructs the real
    `MountDir` instances. Omitted keys keep `MountDir`'s own defaults.
    """

    host_path: str
    virtual_path: str
    mode: NotRequired[Literal['read-only', 'read-write', 'overlay']]
    write_bytes_limit: NotRequired[int | None]
    memory_usage_limit: NotRequired[int]


_MOUNT_SPEC_ADAPTER: TypeAdapter[CodeModeMountSpec | list[CodeModeMountSpec]] = TypeAdapter(
    CodeModeMountSpec | list[CodeModeMountSpec]
)


def _mount_from_spec(mount: CodeModeMountSpec | Sequence[CodeModeMountSpec] | None) -> CodeModeMount | None:
    """Build `MountDir` instances from spec mount entries, validating the entry shape first.

    Validation happens here rather than in `MountDir` so a misspelled key fails with the
    entry that carries it instead of an opaque constructor error.
    """
    if mount is None:
        return None
    entries = cast(Sequence[object], mount if isinstance(mount, Sequence) else [mount])
    allowed_keys = CodeModeMountSpec.__annotations__.keys()
    for entry in entries:
        if isinstance(entry, dict):
            entry_dict = cast(dict[str, object], entry)
            unknown_keys = entry_dict.keys() - allowed_keys
            if unknown_keys:
                raise ValueError(f'Unknown mount spec key(s): {sorted(unknown_keys)}')
    validated = _MOUNT_SPEC_ADAPTER.validate_python(mount)
    if isinstance(validated, list):
        return [MountDir(**entry) for entry in validated]
    return MountDir(**validated)


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

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CodeMode[AgentDepsT]:
        """Return a fresh instance so concurrent runs don't share `_announced_tools`."""
        if not self.dynamic_catalog:
            return self
        return replace(self)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        return CodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            max_tool_calls=self.max_tool_calls,
            resource_limits=self.resource_limits,
            dynamic_catalog=self.dynamic_catalog,
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
        """Announce newly-discovered tools from a local `search_tools` return.

        Only active with `dynamic_catalog=True`. The native-search path is handled by
        [`after_model_request`][pydantic_ai_harness.CodeMode.after_model_request] instead
        (server-side search emits a `NativeToolSearchReturnPart` rather than a regular tool
        execute result).
        """
        if self.dynamic_catalog and tool_def.tool_kind == 'tool-search':
            self._announce_newly_discovered(ctx, _extract_discovered_names(result))
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

    @classmethod
    def from_spec(
        cls,
        tools: Literal['all'] | Sequence[str] | dict[str, Any] = 'all',
        max_retries: int = 3,
        *,
        max_tool_calls: int = 100,
        mount: CodeModeMountSpec | Sequence[CodeModeMountSpec] | None = None,
        resource_limits: CodeModeResourceLimits | Literal['unlimited'] | None = None,
        dynamic_catalog: bool = False,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        **unsupported: Any,
    ) -> CodeMode[Any]:
        """Build from an agent spec, covering the fields a spec can express.

        Every parameter is named because this signature is what core reads to generate the
        spec's JSON schema, and a field whose type has no JSON representation would
        otherwise erase the whole `CodeMode` entry from that schema.

        `mount` arrives as `CodeModeMountSpec` mappings and becomes `MountDir` instances.
        `os_access` takes a live OS implementation or a callback, which no spec can carry,
        so a spec naming it is rejected rather than dropped. The callable form of `tools`
        is likewise construction-only; the `'all'`, name-list, and metadata-match forms
        all serialize.
        """
        if 'os_access' in unsupported:
            raise UserError(
                'CodeMode cannot be built from a spec with `os_access`: it takes a live OS '
                'implementation or a callback. Construct the capability in code to use it.'
            )
        if unsupported:
            raise UserError(f'CodeMode has no spec field(s) {sorted(unsupported)}.')
        return cls(
            tools=tools,
            max_retries=max_retries,
            max_tool_calls=max_tool_calls,
            mount=_mount_from_spec(mount),
            resource_limits=resource_limits,
            dynamic_catalog=dynamic_catalog,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

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

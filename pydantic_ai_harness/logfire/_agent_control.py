"""Back an agent's Agent Control configuration with one Logfire variable.

The contract itself -- what an `AgentConfig` holds, how leniently it validates, and what a published
value does to a request -- lives in [`logfire.agent_control`][], shared by every framework adapter and
by the Logfire UI. This module is the Pydantic AI half of it: the capability wiring, the toolset that
carries managed definitions to the model, the bridge between the contract's string block ids and
[`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id], and the write-backs a Logfire
managed variable needs.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import logfire
from logfire.agent_control import (
    AGENT_CONFIG_JSON_SCHEMA,
    AGENT_VARIABLE_PREFIX,
    MAX_TIMEOUT_SECONDS,
    AgentConfig,
    Block,
    OnUnmatched,
    ToolDef,
    UnappliedEntry,
    apply_instructions,
    apply_settings,
    apply_tool_definitions,
    build_baseline,
    is_representable_timeout,
)
from logfire.variables import Variable
from logfire.variables.abstract import NoOpVariableProvider
from pydantic_ai import AbstractToolset, RunContext, TemplateStr, ToolDefinition, WrapperToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestContext, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.logfire._managed_variable import (
    ManagedVariableCapability,
    _in_durable_context,  # pyright: ignore[reportPrivateUsage]
    _warn_durable_write_skipped,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from logfire.variables import ResolvedVariable
    from pydantic_ai.agent.abstract import AbstractAgent, AgentModelSettings
    from pydantic_ai.capabilities import AgentModel, ModelSelection
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.models import ModelSelectionContext
    from pydantic_ai.run import AgentRunResult

_AGENT_CONTROL_ID = 'agent-control'
"""The capability ID an `AgentControl` carries, so two of them read as one managed config."""

# Where a renamed tool records the code-side name it routes back to, on its own definition's
# `metadata`, so `call_tool` reads it off the very tool the model was handed. Recomputing the mapping
# from a fresh listing instead answers about whatever occupies that advertised name *now*: a dynamic
# toolset that put a different tool behind it in between supplies its code-side name, and the call
# then runs the tool the model chose while a name-based policy -- `ApprovalRequiredToolset` above all
# -- is handed the other one's name and authorizes that instead. It is why the contract's `routes`
# mapping is not what this adapter routes on. The key is removed again on the way through, so nothing
# downstream sees it.
_ROUTES_TO_METADATA_KEY = '__agent_control_routes_to__'

# Drop warnings already emitted in this process, keyed by the message itself. See `_warn_dropped`.
_warned_drops: set[str] = set()

# Destinations handed to a baseline publisher in this process. Marking before the thread starts
# prevents concurrent first requests from scheduling duplicate work; a failure is not retried by
# every later run.
_baseline_publish_attempted: set[tuple[logfire.Logfire, str]] = set()
_baseline_publish_lock = threading.Lock()


def _warn_dropped(message: str) -> None:
    """Surface a dropped managed value once per process.

    A drop means Logfire shows one thing and the agent does another, which has to be visible. But the
    config is resolved on every single run, so warning per drop would bury that signal under its own
    repetition. Deduplicating on the message rather than on the field costs nothing in clarity -- each
    message names its subject and the offending value -- and still lets a *different* unrecognized
    value surface later, which a per-field guard would swallow. A concurrent first run can at worst
    duplicate the warning, which is not worth a lock.

    The contract's own parser deduplicates the same way, in its own set: the two never emit the same
    message, so one message still reaches the user once however it was produced.
    """
    if message in _warned_drops:
        return
    _warned_drops.add(message)
    warnings.warn(message)


def _report_unmatched(policy: OnUnmatched, message: str) -> None:
    """Apply `AgentControl.on_unmatched` to one published entry that reached nothing.

    Called where the entry would have been applied -- the settings hook, the instruction hook, the
    toolset listing -- and not from validation, for two reasons. Validation runs inside Logfire's
    resolution, which turns any exception into a fallback to the code-defined agent, so an `'error'`
    raised there would be swallowed and un-manage the whole config instead of stopping the run. And
    the same validation builds the code baseline, where a key this SDK has no field for is the
    agent's own `extra_headers` rather than anything anyone published.

    The message is the same under every policy so a warning someone chose to tolerate reads like
    the error they would have gotten by not tolerating it.
    """
    if policy == 'error':
        raise UserError(message)
    if policy == 'warn':
        _warn_dropped(message)


def _report_unapplied(policy: OnUnmatched, entries: Sequence[UnappliedEntry]) -> None:
    """Report what the contract's apply helpers did not apply, as the error Pydantic AI refuses runs with.

    The helpers take a policy of their own and raise `ValueError` for `'error'`. They are called with
    `'ignore'` and hand back the entries precisely so an adapter can surface them its own way, which
    here means [`UserError`][pydantic_ai.exceptions.UserError]: a published config this deployment
    cannot apply is a misconfiguration, and that is the exception Pydantic AI raises for one.
    """
    for entry in entries:
        _report_unmatched(policy, entry.message)


def _reset_baseline_publish_guard() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear baseline publishing process state. Intended for tests only."""
    with _baseline_publish_lock:
        _baseline_publish_attempted.clear()


def _spawn_baseline_publish(variable: Variable[Any], example: str) -> None:
    """Move the provider read and targeted write off the model request's thread."""
    threading.Thread(target=_publish_baseline, args=(variable, example), daemon=True).start()


def _publish_baseline(variable: Variable[Any], example: str) -> None:
    """Update only the `example` on the provider's current complete variable definition.

    Read-modify-write, because the provider offers nothing narrower: `update_variable` PUTs the whole
    definition to `/v1/variables/{name}/` and takes no revision or `If-Match` input, so an edit saved
    in the Logfire UI between the read and the write is lost. The window is one HTTP round trip, the
    publish runs at most once per process per variable, and it returns early when `example` already
    matches -- but the race is real and cannot be closed from this side. Narrowing it needs a partial
    write or a conditional one on the platform API: https://github.com/pydantic/pydantic-ai-harness/issues/565
    """
    provider = variable.logfire_instance.config.get_variable_provider()
    try:
        config = provider.get_variable_config(variable.name)
        if config is None:
            if isinstance(provider, NoOpVariableProvider):
                return
            raise LookupError(f'variable {variable.name!r} was not found')
        if config.example == example:
            return
        provider.update_variable(variable.name, config.model_copy(update={'example': example}))
    except Exception as exc:
        variable.logfire_instance.warn(
            'Failed to publish the code baseline for Logfire managed variable {variable_name}',
            variable_name=variable.name,
            _exc_info=True,
        )
        warnings.warn(f'Failed to publish the code baseline for Logfire managed variable {variable.name!r}: {exc}')


def _toolset_key(toolset: AbstractToolset[Any]) -> str:
    """The string a tool's toolset is reported and matched as: its `id`, or its label when it has none.

    One rule for both directions, so the value the baseline shows in the editor is exactly the value
    an override can be narrowed by.
    """
    return toolset.id or toolset.label


def _instruction_key(part: InstructionPart) -> str | None:
    """The string key a managed config addresses this part by, or `None` if nothing addresses it.

    Pydantic AI issues a structured [`InstructionId`][pydantic_ai.messages.InstructionId]; a managed
    config arrives as JSON and names parts by the string that id renders to. Bridging the two once
    here keeps every caller comparing keys of the same kind.

    The contract keeps ids free-form and reserves only `'agent'` as the cross-framework name for the
    prompt as written; the rest of the namespace belongs to each implementation, and these are
    Pydantic AI's: `'agent'` for the agent's own literal instructions, `'toolset:<id>'` and
    `'capability:<id>'` for what a toolset or capability contributes, and `'agent:<declared name>'` /
    `'capability:<id>:<declared name>'` for a single declared block. Blocks Pydantic AI cannot key --
    a callable passed to `Agent(instructions=...)`, a toolset with no `id` of its own -- cannot be
    addressed at all.
    """
    return str(part.id) if part.id is not None else None


def _blocks(parts: Sequence[InstructionPart]) -> list[Block]:
    """The assembled instruction parts as the contract's neutral blocks."""
    return [Block(text=part.content, id=_instruction_key(part), dynamic=part.dynamic) for part in parts]


ConfigProvider = Callable[[], 'AgentConfig | None']
ToolsObserver = Callable[[list['ToolsetTool[Any]']], None]


def _advertised_tool(
    tool: ToolsetTool[AgentDepsT], applied: ToolDef, *, code_name: str, toolset: AbstractToolset[AgentDepsT]
) -> ToolsetTool[AgentDepsT]:
    """The tool as the model will be shown it, remembering the name a call routes back to.

    Only the LLM-facing fields move: a renamed tool keeps its implementation, its argument validation,
    and the parameter names and types it was defined with.
    """
    tool_def: ToolDefinition = tool.tool_def
    changes: dict[str, Any] = {}
    if applied.name != code_name:
        changes['name'] = applied.name
        changes['metadata'] = {**(tool_def.metadata or {}), _ROUTES_TO_METADATA_KEY: code_name}
    if applied.description != tool_def.description:
        changes['description'] = applied.description
    if applied.parameters_json_schema is not tool_def.parameters_json_schema:
        changes['parameters_json_schema'] = applied.parameters_json_schema
    if not changes:
        return tool
    # `replace` preserves concrete `ToolDefinition` subclasses and fields added by the framework.
    return replace(tool, toolset=toolset, tool_def=replace(tool_def, **changes))


@dataclass
class _ToolDefinitionOverridesToolset(WrapperToolset[AgentDepsT]):
    """Overlay advertised definitions while routing calls to their code-side tool names."""

    get_config: ConfigProvider = field(repr=False, compare=False)
    observe_code_tools: ToolsObserver = field(repr=False, compare=False)
    on_unmatched: OnUnmatched = field(default='warn', kw_only=True)

    def _effective_tools(
        self, config: AgentConfig, tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Advertise what the managed config says, keyed by the name the model will call.

        Which override wins, what a colliding rename costs, and which patch reached nothing are all
        the contract's to decide; this hands it the tools this listing advertises and applies what
        comes back. Pydantic AI advertises every tool into one flat namespace, which is the
        `collision_scope='global'` the contract defaults to.

        Reported on every listing, because tool availability is dynamic -- a toolset can advertise
        different tools from one step to the next -- and a report from any one listing is a report
        about that listing.
        """
        applied = apply_tool_definitions(
            [
                ToolDef(
                    name=name,
                    description=tool.tool_def.description,
                    parameters_json_schema=tool.tool_def.parameters_json_schema,
                    toolset=_toolset_key(tool.toolset),
                )
                for name, tool in tools.items()
            ],
            config,
            on_unmatched='ignore',
        )
        advertised = {
            applied_def.name: _advertised_tool(tool, applied_def, code_name=code_name, toolset=self)
            for (code_name, tool), applied_def in zip(tools.items(), applied.tools)
        }
        _report_unapplied(self.on_unmatched, applied.unapplied)
        return advertised

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return tools with managed definitions and collision-safe advertised names."""
        tools = await super().get_tools(ctx)
        self.observe_code_tools(list(tools.values()))
        config = self.get_config()
        if config is None or not config.tool_definitions:
            return tools
        return self._effective_tools(config, tools)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Translate a renamed model call back to its original implementation and context name."""
        metadata = tool.tool_def.metadata or {}
        original_name = metadata.get(_ROUTES_TO_METADATA_KEY)
        if isinstance(original_name, str) and original_name != name:
            ctx = replace(ctx, tool_name=original_name)
            routed_metadata = {key: value for key, value in metadata.items() if key != _ROUTES_TO_METADATA_KEY}
            tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name, metadata=routed_metadata or None))
            return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)


@dataclass
class AgentControl(ManagedVariableCapability[AgentDepsT, AgentConfig]):
    """Manage an agent's config through one `agent__<name>` Logfire variable.

    The variable holds an [`AgentConfig`][logfire.agent_control.AgentConfig], the contract every
    Agent Control SDK shares. Each present section -- `instructions`, `model`, `settings`, or
    `tool_definitions` -- is managed from Logfire, while an absent section keeps the code-defined
    behavior. Removing a section in Logfire deliberately reverts that section to code.

    Instructions are the one section that **composes with** the agent instead of patching it, so it
    works in two ways, and which one an entry uses is the difference between a prompt that reads well
    and one sent to the model twice.

    An entry with **no `id` adds** a block. Adding is also the one way the same text can reach the
    model twice: seeding a config from an agent's observed prompt while that text stays in
    `Agent(instructions=...)` sends every block of it twice over.

    An entry **with an `id` swaps out** the block Pydantic AI assembled under that key -- replacing its
    text, or dropping it with `instructions=None`. This is how a managed config reaches text no
    capability owns: the agent's own literal, a toolset's, an MCP server's, one `@agent.instructions`
    function out of several. See [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] for
    which blocks have a key at all.

    Both are applied in `before_model_request`, after every contribution has been assembled, so an
    override addresses what the model was about to be sent and an added block lands at the end of the
    run of static blocks -- last in the prompt as written, and still inside the prefix a provider can
    cache. An override leaves its block's position and its `dynamic` flag alone, so no override moves
    that cache boundary either. An added block is static for the same reason, unless `render_template`
    makes it a per-run rendering.

    When `name` is omitted, the variable name is derived from the agent's telemetry name using the
    same normalization as the Logfire UI, which is lossy: `checkout-assistant` and `checkout_assistant`
    both resolve `agent__checkout_assistant`, so two agents that differ only in punctuation -- in one
    service or across several in the same project -- share one managed config. Pass an explicit `name`
    to keep them apart. The variable is resolved once per run, and its label and version baggage
    remains active for the whole run.

    The managed `model` is sourced during model selection, so it slots in with the right precedence:
    a call-site `run(model=...)` beats it -- one run of one process, deliberately overriding what is
    published -- and it beats everything else, the agent's constructor model and any other
    capability's alike. A fully model-less `Agent(None, ...)` -- named or nameless -- can therefore be
    driven entirely from managed config. Managed settings work the same way, patching over the
    agent's and every other capability's; settings passed to `run()` still win.
    Static or callable targeting inputs participate in model selection, and the resulting variable
    resolution is reused for the rest of the run so the selected model and applied config cannot
    diverge. An unknown managed model warns and keeps the code model.

    Tool overrides change only the definitions shown to the model. Renames route back to the original
    implementation, collisions retain the original name, and an override the contract can't validate
    is dropped with a warning while its siblings still apply. An override names its tool by code-side
    `name`, narrowed to one toolset's tool of that name when it sets `toolset` too. Parameter names,
    types, requiredness, validation, and implementation stay code-owned.

    A published entry that reaches nothing -- an instruction `id` no assembled block carries (or only
    a dynamic one carries), a tool override no advertised tool matches, a rename another tool already
    answers to, a parameter patch the tool has no parameter for, a settings key the contract has no
    field for -- is governed by `on_unmatched`. The default warns once per process rather than
    raising, because one config is applied across deployments that need not all install the same
    toolsets, and a toolset can advertise different tools from one step to the next; an entry that
    reaches nothing here may be exactly right somewhere else. `'error'` is for the deployment that
    would rather stop than run with part of its published config silently unapplied.

    Missing, invalid, or unreachable remote values degrade to the code-defined agent through Logfire's
    resolution fallback, which is why a value the contract doesn't recognize degrades the narrowest
    unit that contains it -- one setting, one tool override, one section -- rather than the whole
    config. If the provider does not know the variable, auto-create is attempted once per process in
    the background, storing the contract's stored JSON schema on the variable and logging the creation
    to Logfire.

    Its `example` is an `AgentConfig`-shaped snapshot of the code-side agent taken from whichever model
    request happens to come first in the process. The Logfire UI presents that snapshot as the code
    baseline to diff managed values against, so it is worth knowing what it really is: for instructions
    or a toolset that vary with `deps`, run input, or the step within a run, it is one point-in-time
    sample rather than a description of the agent. An agent that never reaches a model request never
    auto-creates at all. Its `instructions` are the code-defined blocks, listed separately with the
    `id` that addresses each block and a `dynamic` flag. This lets the UI offer an override per block
    instead of one copy-the-whole-prompt button that would produce exactly the duplication described
    above.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import AgentControl

    logfire.configure()
    agent = Agent(
        'anthropic:claude-fable-5-1',
        name='checkout_assistant',
        instructions='You are a checkout assistant.',
        capabilities=[AgentControl(label='production')],
    )
    result = agent.run_sync('Refund my last order.')
    ```

    Runtime `{{...}}` placeholders pass through unless `render_template=True`. During a run,
    `logfire.managed.applied_sections` lists the present sections. The `model` section is reported
    when present even if a call-site model outranked it for that run.
    """

    name: str | Variable[AgentConfig] | None = None
    """Bare variable name, pre-built variable, or `None` to derive it from the agent name.

    A nameless capability derives its variable (and can source the model) from the agent's `name`,
    which must then be one the agent was given explicitly.
    """
    render_template: bool = False
    """Render `{{...}}` placeholders in *added* instruction text against run dependencies when enabled.

    An added block is then a rendering of this run's `deps` rather than fixed text, so it is
    contributed as a dynamic block -- the same way Pydantic AI treats any other
    [`TemplateStr`][pydantic_ai.TemplateStr] -- and sorts after the static prompt rather than inside
    the prefix a provider caches.

    An entry that addresses an existing block by `id` is applied to the assembled request and is never
    templated: it replaces a block with exactly the text that was published.
    """
    publish_baseline: bool = True
    """Publish the code-side agent snapshot to the variable's `example` when it changes.

    Enabled by default because `example` is documentation for the Logfire editor and is never resolved
    or applied to a run. A failed or stale publish therefore cannot change agent behavior. Disable it
    when the variables token is intentionally read-only or code must not update variable metadata.
    """
    on_unmatched: OnUnmatched = field(default='warn', kw_only=True)
    """What to do with a published entry that reaches nothing in this deployment.

    That is an instruction `id` no assembled block carries, or that only a dynamic block carries; a
    tool override whose `name` (and `toolset`, when set) matches no tool a toolset advertises; a
    rename another advertised tool already answers to; a parameter patch naming a parameter the tool
    does not have; and a `settings` key this version of the contract has no field for. Each is a place
    where Logfire shows one thing and the agent does another.

    - `'warn'` (the default) emits a `UserWarning` once per process per message, at the point the entry
      would have been applied.
    - `'error'` raises [`UserError`][pydantic_ai.exceptions.UserError] with the same message at that
      point, failing the run.
    - `'ignore'` applies nothing and says nothing.

    Warning rather than raising is the default because tool availability is dynamic: one config is
    applied across deployments that need not all install the same toolsets, and a toolset can
    advertise different tools from one request to the next, so an entry that reaches nothing now is
    not necessarily wrong. Once per process rather than once per run for the same reason every other
    drop is: the config is resolved on every run, and the signal has to survive its own repetition.
    """

    # Override the inherited default ID: a stable id is what tells Pydantic AI an agent has one
    # managed config and that two of it are one configuration stated twice, rather than two anonymous
    # capabilities that collide. Declared in the class body rather than assigned in `__post_init__`,
    # which is where the framework reads it from.
    id: str | None = field(default=_AGENT_CONTROL_ID, kw_only=True)

    _auto_create_in_wrap_run: ClassVar[bool] = False
    _selection_resolved: ContextVar[ResolvedVariable[AgentConfig] | None] = field(init=False, repr=False)
    _code_model: ContextVar[str | None] = field(init=False, repr=False)
    _code_settings: ContextVar[Mapping[str, Any] | None] = field(init=False, repr=False)
    """The settings in force before this capability's patch, snapshotted for the baseline.

    Held as a plain mapping rather than `ModelSettings`: `RunContext.model_settings` is whatever the
    session is running with, which is `RealtimeModelSettings` in a realtime session, and the only
    thing done with it is handing it to the contract's baseline builder, which takes any mapping.
    Narrowing to one of the two would either cast away a real case or drop a realtime agent's
    baseline.
    """
    _code_tools: ContextVar[list[ToolsetTool[Any]] | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._selection_resolved = ContextVar('agent_control_selection_resolved', default=None)
        self._code_model = ContextVar('agent_control_code_model', default=None)
        self._code_settings = ContextVar('agent_control_code_settings', default=None)
        self._code_tools = ContextVar('agent_control_code_tools', default=None)
        # The empty config is the only code-side default there is. The agent itself is what a
        # published value is layered onto, so a second place to say the same thing would only be
        # somewhere for the two to disagree.
        self._setup_variable(
            self.name,
            prefix=AGENT_VARIABLE_PREFIX,
            value_type=AgentConfig,
            default=AgentConfig(),
            json_schema=AGENT_CONFIG_JSON_SCHEMA,
        )

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> AbstractCapability[AgentDepsT]:
        """Bind alongside the companion that carries the overriding half of this capability.

        See [`_AgentControlOverrides`][] for why the two halves cannot be one capability. The pair is
        assembled here rather than at construction so `AgentControl` stays the leaf a user builds,
        passes around, and finds with
        [`find_capability`][pydantic_ai.capabilities.find_capability] -- flattening a
        `CombinedCapability` keeps its children as siblings, so a container in this position would
        take the user's capability out of the tree entirely.
        """
        if self._deferred is not None and not agent.name:
            raise UserError(
                "`AgentControl` without an explicit `name` reads the agent's `name`, and this agent has none. "
                'Pydantic AI would infer one from the variable the agent is assigned to, so renaming that '
                'variable would silently point the agent at a different managed config -- which is why it is '
                'refused rather than inferred. Give the agent a `name=...`, or pass an explicit `name` to '
                '`AgentControl`.'
            )
        return CombinedCapability([self, _AgentControlOverrides(self)])

    def _resolve_for_selection(
        self, variable: Variable[AgentConfig], ctx: ModelSelectionContext[AgentDepsT]
    ) -> ResolvedVariable[AgentConfig]:
        """Resolve with the same callable targeting inputs that the authoritative run will reuse."""
        targeting_key = self.targeting_key(ctx) if callable(self.targeting_key) else self.targeting_key  # pyright: ignore[reportArgumentType]
        attributes = self.attributes(ctx) if callable(self.attributes) else self.attributes  # pyright: ignore[reportArgumentType]
        return variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)

    def _model_selector(self) -> Callable[[ModelSelectionContext[AgentDepsT]], ModelSelection]:
        """Build the per-run selector a nameless `get_model` returns.

        Pydantic AI evaluates a selector once per new request step, but the managed model is a
        run-stable config value, so the selector memoizes its first choice and every later step
        reuses it -- one resolve per run, matching the static (named) path rather than re-reading the
        variable each step. A fresh selector (and fresh memo) is built per `get_model` call, i.e. per
        run, so nothing leaks across runs. On the first evaluation it derives the backing variable
        from `ctx.agent` (the same derivation `wrap_run` performs), reads the managed model, and
        falls back to the model Pydantic AI already selected (`ctx.model`) when none is managed --
        raising only when there is no model at all (a nameless, model-less agent with nothing
        published yet), so the misconfiguration surfaces clearly instead of downstream.
        """
        selected: list[ModelSelection] = []

        def select(ctx: ModelSelectionContext[AgentDepsT]) -> ModelSelection:
            if not selected:
                code_model = ctx.model
                self._code_model.set(f'{code_model.system}:{code_model.model_name}' if code_model is not None else None)
                resolved = self._resolve_for_selection(self._ensure_variable_for_agent(ctx.agent), ctx)
                model = resolved.value.model
                if model is not None:
                    try:
                        infer_model(model)
                    except (UserError, ValueError) as error:
                        _warn_dropped(
                            f'Managed agent config selects unknown model {model!r} ({error}); keeping the code model.'
                        )
                        model = None
                if model is not None:
                    selected.append(model)
                elif ctx.model is not None:
                    selected.append(ctx.model)
                else:
                    raise UserError(
                        'A nameless `AgentControl` on a model-less agent has no model to run: the agent '
                        'defines no model and none is published in Logfire yet. Give the agent a model, '
                        'pass one to `run(model=...)`, or publish a `model` in the managed config.'
                    )
                # Handed off only once selection has succeeded. `wrap_run` is what clears this, and a
                # selection that raises never reaches it -- so setting it earlier would leave this run's
                # instructions, settings and tool overrides in the context for whatever runs next.
                self._selection_resolved.set(resolved)
            return selected[0]

        return select

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the agent toolset with managed LLM-facing definition overlays."""
        return _ToolDefinitionOverridesToolset(
            wrapped=toolset,
            get_config=self._current_config,
            observe_code_tools=self._observe_code_tools,
            on_unmatched=self.on_unmatched,
        )

    def _observe_code_tools(self, tools: list[ToolsetTool[Any]]) -> None:
        self._code_tools.set(tools)

    def _current_config(self) -> AgentConfig | None:
        """The active run's managed config, or `None` outside a resolved run."""
        resolved = self.resolved
        return None if resolved is None else resolved.value

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Add applied-section baggage inside the base's once-per-run resolution context."""
        resolved = self._selection_resolved.get() or self._resolve(ctx)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                sections = ','.join(
                    name
                    for name in ('instructions', 'model', 'settings', 'tool_definitions')
                    if getattr(resolved.value, name) is not None
                )
                if sections:
                    with logfire.set_baggage(**{'logfire.managed.applied_sections': sections}):
                        return await handler()
                return await handler()
            finally:
                self._resolved.reset(token)
                self._selection_resolved.set(None)

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Capture the code-side baseline from the first assembled request.

        Runs on this outermost half deliberately: the snapshot has to describe the agent *before*
        managed values reach it, and the overriding half applies the managed instructions from the
        innermost position, after this. The request is left untouched here.
        """
        self._publish_request_baseline(ctx, request_context)
        return request_context

    def _managed_settings(self, config: AgentConfig) -> ModelSettings:
        """The managed settings patch, in Pydantic AI's flat `ModelSettings` shape.

        Every field name in the section is already a `ModelSettings` key -- that is what makes them
        canonical -- so the contract's patch is the patch, and Pydantic AI merges it with the
        precedence `_AgentControlOverrides` exists to give it.

        The two kinds of published key the contract cannot lower are reported here rather than by
        `apply_settings`, which reports them under a policy of its own that raises `ValueError`.
        Re-stating them is what keeps `'error'` raising the `UserError` Pydantic AI refuses a run
        with.
        """
        settings = config.settings
        if settings is None:
            return ModelSettings()
        for name in settings.unrecognized:
            _report_unmatched(
                self.on_unmatched,
                f'Managed agent config sets {name!r}, which this version of the SDK has no model setting '
                'for; that key is not applied.',
            )
        timeout = settings.timeout
        if timeout is not None and not is_representable_timeout(timeout):
            _report_unmatched(
                self.on_unmatched,
                f'Managed agent config sets a request timeout of {timeout!r} seconds, which is not a budget a '
                f'request can be given -- it has to be finite, not negative, and no larger than '
                f'{MAX_TIMEOUT_SECONDS} seconds; that key is not applied.',
            )
        # A `dict[str, Any]` of canonical keys is a `ModelSettings` by construction, which is a
        # `TypedDict` and therefore not something `isinstance` can narrow to.
        return cast(ModelSettings, apply_settings(config, on_unmatched='ignore'))

    def _apply_instructions(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Apply the managed `instructions` section to the blocks this request assembled.

        What an entry does -- add a block, replace one, drop one -- and which entries reached nothing
        are the contract's to decide. This maps the assembled parts onto the blocks it takes, and maps
        what comes back onto parts again: a block that passed through keeps the very part it came from,
        a replaced one keeps its part's `id`, `name`, and `dynamic` flag with the published text swapped
        in, and an added one becomes a new part. Pydantic AI sorts static blocks ahead of dynamic ones
        so a provider can cache the stable prefix, so carrying the flag over is what keeps an override
        from moving that boundary on every request.

        A dynamic block is deliberately not addressable, and the contract refuses it rather than
        applying it. A function that returns a constant is flagged dynamic all the same, which is the
        right call rather than a gap to close: `dynamic` is what decides which side of the provider's
        cache breakpoint a block sits on, and nothing about a function's shape says whether its text
        came from the run. An author who wants fixed text addressable says so on the part instead of on
        the function -- `InstructionPart(content=..., name='style')` is static by default and is
        accepted anywhere instructions are, a toolset's `get_instructions()` included.

        The new parameters are assigned onto the given context rather than returned on a
        `dataclasses.replace` copy of it, which would look tidier and be wrong: `ModelRequestContext`
        declares `model_id` and `streaming` as `init=False`, the agent graph sets both immediately
        before calling this hook, and `replace()` re-initializes them to `None`/`False`. Losing them
        costs a streamed run its streaming flag and a durable-execution worker the selection token it
        re-resolves an aliased model from. Reported upstream.
        """
        config = self._current_config()
        if config is None or not config.instructions:
            return request_context
        parameters = request_context.model_request_parameters
        parts = list(parameters.instruction_parts or [])
        blocks = _blocks(parts)
        applied = apply_instructions(blocks, config, on_unmatched='ignore')
        # A block that passed through untouched is the object it went in as, so identity is what says
        # which part it came from. A replaced one is a copy carrying the same `id`, and only a
        # non-dynamic part can be replaced, so consuming those in order matches them up even when an
        # agent assembles several blocks under one key.
        by_block = {id(block): part for block, part in zip(blocks, parts)}
        replaceable: dict[str, list[InstructionPart]] = {}
        for block, part in zip(blocks, parts):
            if block.id is not None and not block.dynamic:
                replaceable.setdefault(block.id, []).append(part)
        applied_parts: list[InstructionPart] = []
        for block in applied.blocks:
            part = by_block.get(id(block))
            if part is not None:
                applied_parts.append(part)
            elif block.id is not None:
                applied_parts.append(replace(replaceable[block.id].pop(0), content=block.text))
            else:
                applied_parts.append(self._added_part(block.text, ctx))
        _report_unapplied(self.on_unmatched, applied.unapplied)
        if applied_parts != parts:
            request_context.model_request_parameters = replace(parameters, instruction_parts=applied_parts)
        return request_context

    def _added_part(self, text: str, ctx: RunContext[AgentDepsT]) -> InstructionPart:
        """One added block as a part, rendered against `deps` when `render_template` is set."""
        if not self.render_template:
            return InstructionPart(content=text)
        return InstructionPart(content=TemplateStr[AgentDepsT](text).render(ctx.deps), dynamic=True)

    def _publish_request_baseline(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> None:
        """Publish the code-side `AgentConfig` baseline at the first eligible model request.

        Model, settings, and tool definitions are captured at their override sites before managed
        behavior replaces them, and the instructions are the ones assembled before the overriding half
        applies anything. The result is what the agent would do with `AgentControl` removed, without
        reverse-engineering an already modified request.

        Instructions are snapshotted per block, straight off
        [`instruction_parts`][pydantic_ai.models.ModelRequestParameters.instruction_parts], keeping each
        block's `id` and `dynamic` flag. That is the whole reason the UI can offer an override at all:
        the joined prompt telemetry records has no seams in it, so a baseline built from that could only
        ever be copied wholesale -- which, since managed instructions *add*, is how you get the agent's
        own text sent to the model twice with a frozen `Today is <date>` in the middle of it.

        Dynamic instructions and dynamic toolsets make the snapshot a sample of one request. Only the
        first request in a process is eligible, so request-to-request variation does not turn into
        writes from live traffic. A new process publishes a changed deployed baseline.

        An `example` is a description of the code, not a value to apply -- nothing resolves it -- which
        is what lets it use these same fields to say *what exists* rather than *what to change*.
        """
        resolved = self.resolved
        if resolved is None or not self.publish_baseline:
            return
        # The agent's own variable, not a shared attribute: a nameless capability backs one variable
        # per agent, so publishing has to name the one this run resolved.
        variable = self._ensure_variable(ctx)
        if _in_durable_context(ctx):
            _warn_durable_write_skipped(variable)
            return
        # Nothing about describing the agent raises: code-side text the contract cannot hold -- a
        # block past the length bound, a setting it has no word for -- is left out of the baseline and
        # warned about, so an agent whose own prompt is too big to publish keeps making requests.
        serialized = json.dumps(self._code_baseline(request_context).model_dump(exclude_none=True), indent=2)
        key = (variable.logfire_instance, variable.name)
        with _baseline_publish_lock:
            if key in _baseline_publish_attempted:
                return
            _baseline_publish_attempted.add(key)
        if self._should_auto_create_for(variable, resolved):
            self._maybe_auto_create(variable, example=serialized, ctx=ctx)
            return
        _spawn_baseline_publish(variable, serialized)

    def _code_baseline(self, request_context: ModelRequestContext) -> AgentConfig:
        """The agent as written: what it would do with `AgentControl` removed.

        Instructions come straight off the assembled parts, which at this point carry nothing managed:
        this half runs before the overriding one, which is where every managed block is applied. Model,
        settings, and tool definitions were captured at their override sites, before managed behavior
        replaced them.

        What a baseline may say about each of them is the contract's rule rather than this adapter's:
        a dynamic block contributes its seam and never its rendered text, every top-level parameter is
        listed so an undocumented one can be described from Logfire, and the run's settings are reduced
        to the canonical keys -- which is what keeps `extra_headers` and `extra_body`, where
        authorization headers and signed bodies live, out of a variable every project member can read.
        """
        return build_baseline(
            instructions=_blocks(request_context.model_request_parameters.instruction_parts or []),
            model=self._code_model.get(),
            settings=self._code_settings.get(),
            tools=[
                ToolDef(
                    name=tool.tool_def.name,
                    description=tool.tool_def.description,
                    parameters_json_schema=tool.tool_def.parameters_json_schema,
                    toolset=_toolset_key(tool.toolset),
                )
                for tool in self._code_tools.get() or []
            ],
        )


@dataclass
class _AgentControlOverrides(AbstractCapability[AgentDepsT]):
    """The half of [`AgentControl`][pydantic_ai_harness.logfire.AgentControl] that has to win.

    A capability's [`CapabilityOrdering`][pydantic_ai.capabilities.CapabilityOrdering] position
    answers two questions at once: how its hooks nest around the run, and where its contributions
    land when Pydantic AI merges them. `AgentControl` needs opposite answers. Its resolution's label
    and version baggage has to envelop the whole run including the run span, which means `outermost`
    -- and `outermost` contributions are merged *first*, so every other capability's model and
    settings would be merged over the published ones. A managed value is an override, so the
    contributions that have to beat other capabilities live here instead, pinned `innermost` where
    they are merged last and win whatever order the capabilities were registered in.

    That is why this is a separate capability rather than a few more methods on `AgentControl`, and
    why it is *not* what a user constructs: `AgentControl` stays the leaf they build and find, and
    [`for_agent`][pydantic_ai_harness.logfire.AgentControl.for_agent] stitches this alongside it. The
    two share one object's state by reference, so there is still one variable, one resolution per
    run, and one thing to configure. Decoupling ordering from precedence in the framework itself is
    tracked in [#7420](https://github.com/pydantic/pydantic-ai/issues/7420); until then this split is
    what gives a published value the precedence the Logfire UI promises.
    """

    control: AgentControl[AgentDepsT] = field(repr=False, compare=False)
    """The capability whose state this contributes from; never a copy, so one resolution serves both."""

    def get_ordering(self) -> CapabilityOrdering:
        """Merge last, so a published model or settings beats every other capability's."""
        return CapabilityOrdering(position='innermost')

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Contribute the lowered managed settings patch for each model request.

        A published key the contract has no field for is reported here, where the patch is applied,
        rather than from validation; see `_report_unmatched` for why.
        """

        def model_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            self.control._code_settings.set(dict(ctx.model_settings or {}))  # pyright: ignore[reportPrivateUsage]
            config = self.control._current_config()  # pyright: ignore[reportPrivateUsage]
            if config is None:
                return ModelSettings()
            return self.control._managed_settings(config)  # pyright: ignore[reportPrivateUsage]

        return model_settings

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        """Source the managed model with the right precedence (`run(model=...)` > managed > everything else).

        The selector records the lower-precedence code model, resolves static or callable targeting
        inputs, and saves that exact resolution for `wrap_run`. This keeps model selection consistent
        with the config and telemetry used by the run even when a targeting callable is stateful. The
        selector does not enter the resolution as a context manager; `wrap_run` remains the owner of
        its baggage. A fully model-less agent can be driven from Logfire, and a call-site
        `run(model=...)` still wins. An unknown managed model warns and falls back to the code model.
        """
        return self.control._model_selector()  # pyright: ignore[reportPrivateUsage]

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Apply the managed instruction blocks to the request the model is about to be sent.

        Runs from the innermost position for the same reason the model and settings are contributed
        from here: an override another capability's hook could overwrite afterwards is not an
        override. Being last also means every contribution has been assembled, which is what lets it
        reach text no capability owns -- the agent's own literal, a toolset's, an MCP server's -- and
        what lets an added block land at the end of the assembled static prompt.
        """
        return self.control._apply_instructions(ctx, request_context)  # pyright: ignore[reportPrivateUsage]

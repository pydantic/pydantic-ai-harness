"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai._utils import replace_no_init  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.agent import Agent, AgentRunResult, EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness._workspace import workspace_attached
from pydantic_ai_harness.subagents._disk import (
    AgentOverride,
    DiskDefinition,
    host_folders,
    load_host_definitions,
    load_workspace_definitions,
)
from pydantic_ai_harness.subagents._effort import clamp_effort
from pydantic_ai_harness.subagents._models import ModelOption, as_option, model_label, validate_restriction
from pydantic_ai_harness.subagents._toolset import SubAgent, SubAgentToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

ToolResolver = Callable[[str], 'Sequence[AgentToolset[object]] | None']
"""Maps one tool name from a disk definition's `tools` list to the toolsets that
provide it, or `None` when the name is unknown (the loader warns and skips it)."""


def _option_line(key: str, option: ModelOption) -> str:
    """One model-menu line for the prompt listing: the key, its model, its hint."""
    label = f'- {key} ({model_label(option.model)})'
    return f'{label}: {option.description}' if option.description else label


_MERGEABLE_FIELDS = frozenset({'agents', 'models'})
"""The only fields a merge composes: the roster, and the model options that roster may pick from.

An allow-list rather than a list of exceptions. `SubAgents` has fifteen public fields, and all but
these two say *how* the delegates run rather than *who* they are -- so merging them applies one
harness's policy to the other's sub-agents. Enumerating those instead would mean a field added
later merges silently by default, which is the wrong way round for a decision nobody made.
"""


@dataclass
class SubAgents(AbstractCapability[AgentDepsT]):
    """Let an agent delegate self-contained tasks to named sub-agents.

    Exposes a single `delegate_task(agent_name, task)` tool. Each delegation
    runs the chosen sub-agent in a fresh, isolated run (it never sees the parent
    conversation), and the available sub-agents are listed in the system prompt
    as a static, cache-stable instruction.

    Sub-agents are passed as a sequence of `SubAgent` entries, each pairing an
    agent with its per-delegate run controls (a `usage_limits` budget, a
    wall-clock `timeout_seconds`, a per-run `max_calls` budget, an `on_failure`
    steering message, and optional `name`/`description` overrides). A delegate's
    name is its `SubAgent.name`, or the agent's own `name` when unset; two
    explicitly-passed delegates resolving to the same name is an error.

    Delegations run on the sub-agent's own model unless a `models` menu is
    configured, in which case `delegate_task` also takes a `model` argument naming
    one of the menu's keys, so the parent routes each task to the model that fits
    it. A `SubAgent` can restrict which keys it accepts (`SubAgent.models`).

    Sub-agents are also loaded from disk by default: each markdown agent definition
    under `.agents/agents/` in the run's workspace and `~/.agents/agents/` on the host
    (or the `.claude/` equivalent) becomes a delegate, built with the parent's model.
    The workspace folder is read at the start of every run, through `ctx.workspace`. Disk delegates get no tools
    by default (`inherit_tools` is `False`); set `inherit_tools=True` to expose the
    parent's tools, or pass a `tool_resolver` to map their frontmatter tool names.
    Disk delegates coexist with explicitly-passed ones; explicitly-passed agents take
    precedence, then the project folder, then the home folder. A disk delegate whose
    name is already taken is skipped with a warning. Configure or disable this with
    `agent_folders`; see also `agent_overrides` and `tool_resolver`.

    The parent's `deps` are forwarded to each sub-agent (sub-agents therefore
    share the parent's `AgentDepsT`), and by default the parent's `usage` is
    shared so usage limits apply across the whole agent tree. Optionally, the
    parent's tools can be inherited (`inherit_tools`), extra capabilities can be
    applied to every sub-agent run (`shared_capabilities`), and sub-agent events
    can be streamed to a handler (`event_stream_handler`).

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.subagents import SubAgent, SubAgents

    researcher = Agent('anthropic:claude-sonnet-4-6', name='researcher', description='Researches topics')
    writer = Agent('anthropic:claude-sonnet-4-6', name='writer', description='Writes prose')

    orchestrator = Agent(
        'anthropic:claude-opus-4-7',
        capabilities=[SubAgents(agents=[SubAgent(researcher), SubAgent(writer)])],
    )
    ```
    """

    agents: Sequence[SubAgent[AgentDepsT]] = ()
    """The sub-agents to expose, each a `SubAgent` pairing an agent with its
    per-delegate run controls. See `SubAgent`. These take precedence over any
    disk-loaded agents of the same name."""

    models: Mapping[str, Model | KnownModelName | str | ModelOption] = field(
        default_factory=dict[str, 'Model | KnownModelName | str | ModelOption']
    )
    """A menu of models the parent can route an individual delegation to, keyed by
    the name the parent uses to pick one. Off by default: with no menu the delegate
    tool has no `model` argument and every delegation runs the way it always did.

    Each value is a model reference, or a `ModelOption` carrying a routing hint and
    its own `ModelSettings` (so one key can mean "same model, more thinking"). The
    keys and their descriptions are listed in the system prompt, so name them for
    the job -- `'fast'`, `'deep'` -- rather than for the vendor. `SubAgent.models`
    restricts which of them a given delegate accepts.

    ```python
    from pydantic_ai_harness.subagents import SubAgents

    SubAgents(models={'fast': 'anthropic:claude-haiku-4-5', 'deep': 'anthropic:claude-opus-4-7'})
    ```
    """

    agent_folders: str | Sequence[Path] | None = 'agents'
    """Where to load markdown agent definitions from, in addition to `agents`.
    Defaults to the conventional layout, so constructing the capability auto-loads
    a repo's agent files with no extra configuration.

    - a folder-name `str` (the default `'agents'` is the conventional layout): for
      the project root then the home root, load from `<root>/.agents/<name>/`,
      falling back to `<root>/.claude/<name>/` when `<root>/.agents/` is absent. The
      project root is the run workspace's working directory, read through
      `ctx.workspace` at the start of each run (skipped when the run has no
      workspace); the home root is the host's, read once at construction.
    - a sequence of paths: load from exactly those host folders, in order, once at
      construction.
    - `None`: disable disk loading entirely (only `agents` are exposed).

    Missing folders are skipped. Within a folder every `*.md` file is a candidate."""

    agent_overrides: Mapping[str, AgentOverride] = field(default_factory=dict[str, AgentOverride])
    """Per-disk-agent overrides keyed by the agent's name. An entry can set the
    agent's `model` (otherwise the parent's model is inherited) and its `effort`
    (otherwise the minimum floor). Has no effect on explicitly-passed `agents`."""

    tool_resolver: ToolResolver | None = None
    """Optional override for how a disk agent gets its tools. When set, each tool
    name in a definition's `tools`/`allowed-tools` frontmatter is passed to this
    resolver and the returned toolsets are attached to that agent; an unknown name
    (resolver returns `None`) is skipped with a warning. When unset, the
    frontmatter tool list is ignored and disk agents inherit the parent's tools
    via `inherit_tools` (set `inherit_tools=True` to expose them)."""

    forward_usage: bool = True
    """If `True`, the parent run's `usage` is shared with each sub-agent run, so
    token usage aggregates and usage limits apply across the whole agent tree."""

    inherit_tools: bool = False
    """If `True`, the parent agent's tools are exposed to each sub-agent run (the
    delegate tool itself is filtered out, so sub-agents can't recurse into
    further delegation). Off by default to avoid silently widening sub-agent access."""

    shared_capabilities: Sequence[AgentCapability[AgentDepsT]] = ()
    """Capabilities applied to every sub-agent run, in addition to whatever each
    sub-agent already has."""

    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    """If set, this handler is passed to each sub-agent run, so the sub-agent's
    model-streaming and tool events surface to the caller. The handler receives
    the sub-agent's own `RunContext` and event stream."""

    tool_name: str = 'delegate_task'
    """Name of the delegate tool exposed to the model."""

    id: str | None = field(default='sub_agents', kw_only=True)
    """One-off: an agent exposes a single delegate tool, so the id is fixed.

    `tool_name` is one name, so two `SubAgents` capabilities register the same tool and collide.
    Declaring the id here is what makes two of them merge instead, unioning their rosters -- which
    is what lets a packaged harness that delegates compose with another that does the same.

    Keyword-only on the field rather than through a `KW_ONLY` marker: a marker applies to every
    field after it, which would take `tool_retries` and `contain_errors` off the positional
    contract they already have.
    """

    tool_retries: int | None = 2
    """Retries for the delegate tool -- how many extra attempts it gets after a
    sub-agent error before the parent run aborts. A sub-agent failure (e.g. it
    exhausts its own output retries) surfaces to the parent as a tool retry it
    can react to by re-delegating with a corrected task. The retry counter
    resets after any successful delegation, so this bounds consecutive failures,
    not total ones. Defaults to `2` (pydantic-ai's per-tool default is `1`) so a
    repeated flaky sub-agent does not abort the parent run on its first repeat;
    set `None` to inherit the parent agent's default tool retries instead."""

    contain_errors: bool = False
    """Default for `SubAgent.contain_errors`: whether an unexpected sub-agent crash
    is caught and returned to the parent as a bounded `ModelRetry` instead of
    aborting the parent run. Off by default, so a crash propagates. Any `SubAgent`
    can override this per delegate. See `SubAgent.contain_errors` for the
    containment contract and what always propagates regardless."""

    _by_name: dict[str, SubAgent[AgentDepsT]] = field(
        default_factory=dict[str, 'SubAgent[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Sub-agents keyed by resolved name, built in `__post_init__` (and rebuilt per run in
    `before_run` once the workspace's definitions are read) and passed to the toolset.
    Insertion order matches `agents` for a stable prompt listing."""

    _host_definitions: list[DiskDefinition] = field(
        default_factory=list[DiskDefinition], init=False, repr=False, compare=False
    )
    """Definitions read from host folders in `__post_init__`, in precedence order."""

    _built: dict[DiskDefinition, SubAgent[AgentDepsT]] = field(
        default_factory=dict[DiskDefinition, 'SubAgent[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Disk delegates built so far, shared with every per-run copy. A definition is built once, so
    runs over unchanged files reuse the same agents and `tool_resolver` is not asked again."""

    _run_toolset: SubAgentToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """This run's delegate toolset, on a per-run copy only. Built once per run, so every step of the
    run sees the same toolset instance."""

    _per_run: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether this instance is a per-run copy made by `for_run` to read the workspace's definitions."""

    _menu: dict[str, ModelOption] = field(default_factory=dict[str, ModelOption], init=False, repr=False, compare=False)
    """`models` normalized to `ModelOption` entries, built in `__post_init__`.
    Insertion order matches `models` for a stable prompt listing and enum."""

    _call_counts: dict[str, dict[str, int]] = field(
        default_factory=dict[str, 'dict[str, int]'], init=False, repr=False, compare=False
    )
    """Run-scoped delegation counts (run_id -> name -> count), shared with the
    toolset and cleared per run in `wrap_run`. Backs `SubAgent.max_calls`."""

    def __post_init__(self) -> None:
        if self.agent_folders is not None:
            self._host_definitions = load_host_definitions(host_folders(self.agent_folders, Path.home()))
        self._build_roster(self._disk_agents(self._host_definitions))

    def _disk_agents(self, definitions: Sequence[DiskDefinition]) -> list[SubAgent[AgentDepsT]]:
        """The delegates for `definitions`, built on first sight and reused after that."""
        result: list[SubAgent[AgentDepsT]] = []
        for definition in definitions:
            sub_agent = self._built.get(definition)
            if sub_agent is None:
                sub_agent = self._built[definition] = self._build_disk_agent(definition)
            result.append(sub_agent)
        return result

    def _build_roster(self, disk_agents: list[SubAgent[AgentDepsT]]) -> None:
        by_name: dict[str, SubAgent[AgentDepsT]] = {}
        for sub_agent in self.agents:
            name = sub_agent.resolved_name
            if name is None:
                raise ValueError('Sub-agent has no name: give its `Agent` a `name`, or set `SubAgent(name=...)`.')
            if name in by_name:
                raise ValueError(
                    f'Duplicate sub-agent name {name!r}. Each sub-agent needs a distinct name; '
                    f'set `SubAgent(name=...)` to disambiguate.'
                )
            by_name[name] = sub_agent
        # Disk agents are lower precedence than explicit ones and than earlier
        # folders, so a name already taken is shadowed (a warning, not an error --
        # overriding a home agent from the project, or a disk agent from code, is
        # the intended path).
        for sub_agent in disk_agents:
            name = sub_agent.resolved_name
            if name is None:  # pragma: no cover - disk agents always get a name (frontmatter or stem)
                continue
            if name in by_name:
                warnings.warn(
                    f'Disk sub-agent {name!r} is shadowed by a higher-precedence definition; skipping it.',
                    stacklevel=2,
                )
                continue
            by_name[name] = sub_agent
        self._by_name = by_name
        self._menu = {key: as_option(value) for key, value in self.models.items()}
        for name, sub_agent in by_name.items():
            validate_restriction(name, sub_agent.models, self._menu)

    def _build_disk_agent(self, definition: DiskDefinition) -> SubAgent[AgentDepsT]:
        """Build one disk-defined sub-agent: parent model + floored effort, tools resolved or inherited.

        The agent is constructed with `deps_type=object` so the parent's deps (of
        any type) flow through unused at delegation; this also lets a disk
        `SubAgent[object]` sit in the parent's `SubAgent[AgentDepsT]` roster.
        """
        name, parsed = definition.name, definition.parsed
        override = self.agent_overrides.get(name)
        model = override.model if override is not None else None
        effort = override.effort if override is not None else None
        toolsets = self._resolve_disk_tools(parsed.tools) if self.tool_resolver is not None else None
        agent = Agent(
            model,
            deps_type=object,
            name=name,
            description=parsed.description,
            instructions=parsed.body or None,
            model_settings=ModelSettings(thinking=clamp_effort(effort)),
            toolsets=toolsets,
        )
        return SubAgent(agent)

    def _resolve_disk_tools(self, tool_names: Sequence[str]) -> list[AgentToolset[object]]:
        """Map a definition's tool names to toolsets via `tool_resolver`, warning on unknown names."""
        resolver = self.tool_resolver
        if resolver is None:  # pragma: no cover - only called when tool_resolver is set
            return []
        toolsets: list[AgentToolset[object]] = []
        for tool_name in tool_names:
            resolved = resolver(tool_name)
            if resolved is None:
                warnings.warn(f'Unknown tool {tool_name!r} in disk sub-agent definition; skipping it.', stacklevel=2)
                continue
            toolsets.extend(resolved)
        return toolsets

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> SubAgents[AgentDepsT]:
        """A per-run copy when the project folder is read from the workspace; otherwise `self`.

        The copy starts from the host roster; `before_run` adds the workspace's definitions. Its
        instructions and toolset are read after that, so they list the delegates this run has.
        """
        if not isinstance(self.agent_folders, str):
            return self
        run = replace_no_init(self)
        run._per_run = True
        run._run_toolset = run._make_toolset()
        return run

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Read the project folder's definitions through `ctx.workspace` and rebuild this run's roster."""
        if not self._per_run or not isinstance(self.agent_folders, str) or not workspace_attached(ctx.workspace):
            return
        project = await load_workspace_definitions(ctx.workspace, self.agent_folders)
        if not project:
            return
        # A home definition identical to a project one is the same file seen twice (the workspace
        # root is the home root), so it is dropped rather than warned about as shadowed.
        home = [definition for definition in self._host_definitions if definition not in project]
        self._build_roster(self._disk_agents([*project, *home]))
        self._run_toolset = self._make_toolset()

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the parent agent, then drop this run's delegation counts so they don't accumulate."""
        try:
            return await handler()
        finally:
            self._call_counts.pop(ctx.run_id or '', None)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Cache-stable listing of the available sub-agents and models.

        A per-run copy returns it as a function, rendered after `before_run` has read the
        workspace's definitions; it is the same text on every step of the run.
        """
        if self._per_run:
            return lambda _ctx: self._render_instructions()
        return self._render_instructions()

    def _render_instructions(self) -> str | None:
        if not self._by_name:
            return None
        lines: list[str] = []
        for name, sub_agent in self._by_name.items():
            description = sub_agent.description or sub_agent.agent.description
            restriction = f' (models: {", ".join(sub_agent.models)})' if sub_agent.models else ''
            lines.append(f'- {name}: {description}{restriction}' if description else f'- {name}{restriction}')
        listing = '\n'.join(lines)
        instructions = (
            f'You can delegate self-contained tasks to these sub-agents using the `{self.tool_name}` '
            f'tool. Each runs in its own fresh context and does not see this conversation, so pass '
            f'everything it needs.\n\nAvailable sub-agents:\n{listing}'
        )
        if not self._menu:
            return instructions
        options = '\n'.join(_option_line(key, option) for key, option in self._menu.items())
        return (
            f'{instructions}\n\nPass one of these keys as `model` to run a sub-agent on it, matching the '
            f"option to how hard the task is. Omit `model` to use the sub-agent's default. A sub-agent "
            f'listed with its own `(models: ...)` accepts only those.\n\nAvailable models:\n{options}'
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the delegate tool, or `None` when no sub-agents are configured.

        A per-run copy returns a function yielding the toolset `before_run` settled on, the same
        instance for every step of the run.
        """
        if self._per_run:
            return lambda _ctx: self._run_toolset
        return self._make_toolset()

    def _make_toolset(self) -> SubAgentToolset[AgentDepsT] | None:
        if not self._by_name:
            return None
        return SubAgentToolset(
            agents=self._by_name,
            forward_usage=self.forward_usage,
            inherit_tools=self.inherit_tools,
            shared_capabilities=self.shared_capabilities,
            event_stream_handler=self.event_stream_handler,
            tool_name=self.tool_name,
            tool_retries=self.tool_retries,
            contain_errors=self.contain_errors,
            call_counts=self._call_counts,
            models=self._menu,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable -- the capability holds live `Agent` instances."""
        return None

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Compose the rosters, and require everything else to already agree.

        Two packaged harnesses on one agent each bring their delegates, and composing them is what
        the shared `id` is for. Only `agents` and `models` are composed. Every other field decides
        how the delegates *run* -- what capabilities they are handed, whether they see the parent's
        tools, what the delegate tool is called, where delegates are loaded from -- so merging it
        would apply one harness's policy to the other's sub-agents, which neither author asked for.
        Those must agree, and say so when they do not.

        The roster is rebuilt from the definitions the inputs already loaded rather than by
        re-running `__post_init__`: that rereads `agent_folders` and re-invokes `tool_resolver`, so
        a merge could answer differently than either input did.
        """
        first = capabilities[0]
        assert isinstance(first, cls)
        merged_agents = list(first.agents)
        merged_models = dict(first.models)
        for other in capabilities[1:]:
            assert isinstance(other, cls)
            for field_info in dataclasses.fields(first):
                name = field_info.name
                if name in _MERGEABLE_FIELDS or not field_info.compare or name == 'id':
                    continue
                mine, theirs = getattr(first, name), getattr(other, name)
                if mine != theirs:
                    raise UserError(
                        f'Capability id {first.id!r} is used by multiple SubAgents capabilities that disagree '
                        f'on {name!r} ({mine!r} and {theirs!r}). Only the roster is composed; everything else '
                        "decides how the delegates run, so merging it would apply one set of delegates' "
                        f'configuration to the other. Give them distinct `id`s to keep both, or make {name!r} '
                        'agree.'
                    )
            merged_agents.extend(other.agents)
            merged_models.update(other.models)

        merged = replace_no_init(first, agents=merged_agents, models=merged_models)
        merged._build_roster(first._disk_agents(first._host_definitions))
        if merged._per_run:
            merged._run_toolset = merged._make_toolset()
        return merged

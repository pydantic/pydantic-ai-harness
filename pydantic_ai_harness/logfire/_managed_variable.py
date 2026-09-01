"""Shared base for capabilities backed by a Logfire managed variable.

`ManagedPrompt` and `AgentControl` both resolve a Logfire
[managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) once per
run and keep its baggage active for the whole run. This base owns that shared plumbing -- the
targeting inputs, the per-run resolution context variable, `get_ordering`, and `wrap_run` -- so each
capability only declares its own variable and exposes the resolved value through its own surface.
"""

from __future__ import annotations

import re
import threading
import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import logfire
from logfire.variables import Variable, VariableAlreadyExistsError
from logfire.variables.abstract import NoOpVariableProvider
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from logfire import Logfire
    from logfire.variables import ResolvedVariable, VariableConfig
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.run import AgentRunResult

# `AgentDepsT` carries a PEP 696 default, so `ValueT` needs one too to follow it in the type
# parameter list. Subclasses always bind it explicitly, so the default is never actually used.
ValueT = TypeVar('ValueT', default=object)


def resolution_reason(resolved: ResolvedVariable[Any]) -> str | None:
    """The reason a variable resolved the way it did (`'unrecognized_variable'`, `'code_default'`, ...).

    Reads the public `reason` attribute where the logfire SDK exposes it, falling back to the private
    `_reason` on older versions that predate it. Kept in one place so callers (and the demo) don't
    each reimplement the compatibility shim.
    """
    reason = getattr(resolved, 'reason', None)
    if reason is not None:
        return reason
    return getattr(resolved, '_reason', None)


# Variables we have already attempted to auto-create in this process, guarded by a lock. The key is
# the destination -- the Logfire instance the variable would be created in -- as well as its name, so
# a process serving more than one Logfire project creates the variable in each of them rather than
# letting the first one it touched stand in for all of them. The contract is one attempt per process
# per key: we mark a key when spawning the creation thread (not on success), so a failed create --
# e.g. a read-only token -- does not retry on every run.
_auto_create_attempted: set[tuple[Logfire, str]] = set()
_auto_create_lock = threading.Lock()
_durable_write_warnings: set[tuple[Logfire, str]] = set()


def _auto_create_key(variable: Variable[Any]) -> tuple[Logfire, str]:
    """The once-per-process guard key for a variable: where it would be created, and under what name."""
    return (variable.logfire_instance, variable.name)


# Resolution reasons that mean "the value fell back to the code default because the provider had no
# value for it". logfire >= 4.37 collapses the "provider doesn't recognize this variable" case into
# the general `'code_default'` reason; older SDKs surface `'unrecognized_variable'` directly. We
# accept both and then confirm the actual cause against the provider (see `_maybe_auto_create_for`).
_CODE_DEFAULT_REASONS = frozenset({'code_default', 'unrecognized_variable'})


def _reset_auto_create_guard() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear the once-per-process auto-create guard. Intended for tests only."""
    with _auto_create_lock:
        _auto_create_attempted.clear()


def _normalize_agent_name(name: str) -> str:
    """Normalize a telemetry agent name exactly as the Logfire managed-agents UI does.

    The rule trims and lowercases the name, replaces hyphens and every other non-`[a-z0-9_]`
    character with `_`, collapses runs of underscores, and strips underscores from both ends. The
    SDK and UI must apply the same rule so they land on the same variable for a given agent name.

    The rule is lossy, so distinct agents can normalize onto one variable: `checkout-assistant`,
    `Checkout Assistant`, and `checkout_assistant` all become `agent__checkout_assistant` and share
    a single managed config, including across services in the same Logfire project. Give the
    capability an explicit `name` to keep two such agents apart.
    """
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9_]', '_', name.strip().lower().replace('-', '_'))).strip('_')


def _spawn_create(variable: Variable[Any], config: VariableConfig) -> None:
    """Run the (blocking, sync-HTTP) creation off the run's thread so it never blocks or fails it.

    Isolated as a module-level function so tests can monkeypatch it to run `_create_variable`
    inline for determinism.
    """
    threading.Thread(target=_create_variable, args=(variable, config), daemon=True).start()


def _in_durable_context(ctx: RunContext[Any]) -> bool:
    """Use the active durability capability's engine-specific context detection."""
    root = ctx.root_capability
    if root is None:
        return False
    return any(getattr(capability, 'in_durable_context', False) is True for capability in leaf_capabilities(root))


def _warn_durable_write_skipped(variable: Variable[Any]) -> None:
    """Explain a skipped workflow-side write once per destination."""
    key = _auto_create_key(variable)
    if key in _durable_write_warnings:
        return
    _durable_write_warnings.add(key)
    warnings.warn(
        f'Skipping the write-back for Logfire managed variable {variable.name!r} inside a durable workflow '
        'because background threads and remote writes are not replay-safe there. Reading managed config '
        'still works. Create the variable in the Logfire UI, or run the SDK outside a workflow once.'
    )


def _create_variable(variable: Variable[Any], config: VariableConfig) -> None:
    """Create the variable in Logfire from its code default, JSON schema, and description.

    The outcome is reported through the Logfire instance the variable belongs to: creation writes a
    persistent, teammate-visible object into the user's project, so it belongs in the place they are
    already looking rather than in a `warnings.warn` raised on a daemon thread that no one sees. The
    failure keeps the warning as well, for local development that exports nowhere.

    Best-effort: an already-existing variable (a race with another process or the UI) is fine, and
    any other failure is surfaced rather than crashing the background thread.
    """
    provider = variable.logfire_instance.config.get_variable_provider()
    # Duck-type the write path: a provider without it (or without persistence) can't be created into.
    create = getattr(provider, 'create_variable', None)
    if not callable(create):  # pragma: no cover
        return
    try:
        create(config)
    except VariableAlreadyExistsError:
        # The variable already exists server-side (another process or the UI created it first).
        pass
    except Exception as exc:
        variable.logfire_instance.warn(
            'Failed to auto-create Logfire managed variable {variable_name}',
            variable_name=variable.name,
            _exc_info=True,
        )
        warnings.warn(f'Failed to auto-create Logfire managed variable {variable.name!r}: {exc}')
    else:
        variable.logfire_instance.info(
            'Created Logfire managed variable {variable_name} from the code default; '
            'set a value in Logfire to manage this agent from there',
            variable_name=variable.name,
        )


@dataclass(frozen=True)
class _DeferredVariable(Generic[ValueT]):
    """The inputs to build the backing variable lazily.

    Remembered by [`ManagedVariableCapability._setup_variable`][] when the capability's `name` was
    omitted, so the backing variable can be constructed as `<prefix><agent name>` from the running
    agent's own `name` on first run-time use (there is no agent at construction time on this
    pydantic-ai version).
    """

    prefix: str
    value_type: type[ValueT]
    default: ValueT


@dataclass
class ManagedVariableCapability(AbstractCapability[AgentDepsT], Generic[AgentDepsT, ValueT]):
    """Base for capabilities that resolve a Logfire managed variable once per run.

    Subclasses call `_setup_variable` from their `__post_init__` with their prefix, value type, and
    default. When an explicit `name` (or a pre-built [`Variable`][logfire.variables.Variable]) is
    given, the backing variable is built eagerly; when `name` is omitted, its construction is
    deferred to the first run-time use, where it is derived from the running agent's own `name` (see
    `_ensure_variable`). Either way, the capability exposes the active run's resolution through its
    own surface (instructions, a toolset wrapper, ...).
    """

    label: str | None = field(default=None, kw_only=True)
    """Explicit targeting label on the Logfire managed variable to resolve (e.g. `'production'`).
    When `None`, the targeting rules on the managed variable select the label."""

    targeting_key: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, kw_only=True)
    """Stable key that seeds Logfire's deterministic rollout assignment -- the same key always
    lands in the same percentage bucket. Accepts a static value or a callable that derives it from
    the [`RunContext`][pydantic_ai.tools.RunContext]. When `None`, Logfire falls back to its own
    targeting context and then the active trace id."""

    attributes: Mapping[str, Any] | Callable[[RunContext[AgentDepsT]], Mapping[str, Any] | None] | None = field(
        default=None, kw_only=True
    )
    """Attributes for condition-based targeting rules, or a callable that derives them
    from the [`RunContext`][pydantic_ai.tools.RunContext]."""

    logfire_instance: Logfire | None = field(default=None, kw_only=True)
    """Logfire instance to resolve the variable on. When `None`, the global default instance is
    used. Ignored when the capability is given a pre-built `Variable`."""

    auto_create: bool = field(default=True, kw_only=True)
    """Whether to create the variable in Logfire the first time it is used but doesn't exist there yet.

    When the variable is unknown to the configured Logfire provider, it is created in the background
    with the code default as its value (plus the payload's JSON schema and description), so the
    Logfire UI becomes the editing surface without a manual create-in-UI step. Until someone
    configures a label there, resolution keeps falling back to the code default. Creation happens
    off the run's thread and never blocks or fails the run; it is attempted at most once per process
    per variable and Logfire instance. Set to `False` to opt out.

    Because the variable is persistent and visible to everyone with access to the Logfire project,
    both outcomes are reported: a successful creation is logged to the same Logfire instance, and a
    failure is logged there and raised as a `UserWarning`. Creation is triggered by a run that
    actually resolves the variable, so an agent that is never run creates nothing."""

    _variable: Variable[ValueT] = field(init=False, repr=False, compare=False)
    """The managed variable backing this capability. Assigned eagerly in `_setup_variable` for an
    explicit name/`Variable`; for a nameless capability it is left unset until `_ensure_variable`
    builds it from the agent's `name` on first run-time use (use `_built_variable` to read it safely
    before then)."""

    _deferred: _DeferredVariable[ValueT] | None = field(init=False, default=None, repr=False, compare=False)
    """The inputs to build `_variable` lazily, set only when `name` was omitted; `None` otherwise."""

    _variables_by_agent: dict[str, Variable[ValueT]] = field(
        init=False, default_factory=dict[str, 'Variable[ValueT]'], repr=False, compare=False
    )
    """Variables derived from an agent's `name`, keyed by the normalized name they were derived from.

    Only used in the deferred (nameless) case. Keyed rather than singular because one capability
    instance can back more than one agent -- `SubAgents.shared_capabilities` passes the same object
    to every sub-agent -- and each of them must read its own `<prefix><agent name>`."""

    _json_schema: dict[str, Any] | None = field(init=False, default=None, repr=False, compare=False)
    """The JSON schema auto-create stores on the variable, when the capability maintains its own.

    `None` keeps the Pydantic-derived schema [`Variable.to_config`][logfire.variables.Variable.to_config]
    produces, which is right for a payload whose schema is trivially stable (`ManagedPrompt`'s
    `{'type': 'string'}`). A capability whose stored schema is a contract shared with the Logfire UI
    passes its own through `_setup_variable` instead."""

    _build_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False, compare=False)
    """Guards the lazy build of `_variable` so concurrent first runs don't each construct one."""

    _resolved: ContextVar[ResolvedVariable[ValueT] | None] = field(init=False, repr=False, compare=False)
    """Per-run resolution, isolated across concurrent runs via the context variable."""

    _auto_create_in_wrap_run: ClassVar[bool] = True
    """Whether the base run wrapper triggers auto-create; subclasses with richer config set `False`."""

    def _new_resolved(self) -> ContextVar[ResolvedVariable[ValueT] | None]:
        """A fresh per-run resolution context variable; `None` means nothing is resolved yet."""
        return ContextVar('managed_variable_resolved', default=None)

    def _setup_variable(
        self,
        name: str | Variable[ValueT] | None,
        *,
        prefix: str,
        value_type: type[ValueT],
        default: ValueT,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        """Wire up the backing variable from `name`, deferring construction when `name` was omitted.

        Called from each subclass's `__post_init__`. A str `name` builds `<prefix><name>` eagerly; a
        pre-built [`Variable`][logfire.variables.Variable] is used as-is (with full `get_model`
        support); `None` records the build inputs so the variable is derived from the running agent's
        own `name` on first run-time use (this pydantic-ai version has no construction-time agent hook).

        `json_schema` overrides the schema auto-create stores on the variable; see `_json_schema`.
        """
        self._json_schema = json_schema
        self._resolved = self._new_resolved()
        if isinstance(name, str):
            self._variable = self._build_managed_variable(name, prefix=prefix, value_type=value_type, default=default)
        elif name is not None:
            self._warn_logfire_instance_ignored('name')
            self._variable = name
        else:
            # Nameless: leave `_variable` unset (there is no agent yet) and remember the build inputs,
            # so `_ensure_variable` can derive `<prefix><agent name>` on the first run.
            self._deferred = _DeferredVariable(prefix=prefix, value_type=value_type, default=default)

    @property
    def _built_variable(self) -> Variable[ValueT] | None:
        """The backing variable if it has been built yet, else `None`.

        For a nameless capability `_variable` is unset until the first run builds it, so read it
        through this rather than touching `_variable` directly outside a run.
        """
        return getattr(self, '_variable', None)

    def _ensure_variable(self, ctx: RunContext[AgentDepsT]) -> Variable[ValueT]:
        """Return the backing variable, building it from the running agent's `name` on first use.

        Thin wrapper over [`_ensure_variable_for_agent`][] reading the agent off a `RunContext`. Model
        selection (which has a `ModelSelectionContext`, not a `RunContext`) calls the agent-based
        method directly, so both entry points build the same variable and share the cache.
        """
        return self._ensure_variable_for_agent(ctx.agent)

    def _ensure_variable_for_agent(self, agent: AbstractAgent[AgentDepsT, Any] | None) -> Variable[ValueT]:
        """Return the backing variable, building it from the agent's `name` on first use.

        For an eagerly-built variable (an explicit `name` or `Variable` was given) this just returns
        it. For a nameless capability it derives `<prefix><agent name>` from `agent.name` and caches
        the result *per normalized agent name*: one capability instance can serve more than one agent
        -- `SubAgents.shared_capabilities` hands the same object to every sub-agent -- and a single
        cache would make every later agent read whichever agent happened to run first. Takes the agent
        directly (rather than a `RunContext`) so both the run-time hooks and model selection -- which
        sees a narrower `ModelSelectionContext` before any `RunContext` exists -- derive the same
        variable. Raises [`UserError`][pydantic_ai.exceptions.UserError] when there is no agent name
        to derive from, and when the name has nothing an identifier can be made of.
        """
        variable = self._built_variable
        if variable is not None:
            return variable
        deferred = self._deferred
        assert deferred is not None  # `_variable` is unset only in the nameless (deferred) case.
        agent_name = agent.name if agent is not None else None
        if not agent_name:
            raise UserError(
                'A managed capability without an explicit `name` derives its backing variable from '
                "the agent's `name`, but this agent has none. Give the agent a `name=...`, or pass an "
                'explicit `name` to the capability.'
            )
        normalized_name = _normalize_agent_name(agent_name)
        if not normalized_name:
            # `_normalize_agent_name` is lossy by design, but an empty result is not a collision
            # between two agents that read alike -- it is every such agent landing on the bare prefix,
            # which names no agent at all. Refuse it the way a missing name is refused.
            raise UserError(
                'A managed capability without an explicit `name` derives its backing variable from '
                f"the agent's `name`, but {agent_name!r} has no letters, digits, or underscores to "
                'derive one from. Give the agent a `name=...` that has some, or pass an explicit '
                '`name` to the capability.'
            )
        variable = self._variables_by_agent.get(normalized_name)
        if variable is not None:
            return variable
        with self._build_lock:
            # A concurrent first run may have built this agent's variable while we waited for the lock.
            variable = self._variables_by_agent.get(normalized_name)
            if variable is not None:
                return variable
            variable = self._build_managed_variable(
                normalized_name,
                prefix=deferred.prefix,
                value_type=deferred.value_type,
                default=deferred.default,
            )
            self._variables_by_agent[normalized_name] = variable
            return variable

    def _warn_logfire_instance_ignored(self, field_name: str) -> None:
        if self.logfire_instance is not None:
            warnings.warn(
                f'`logfire_instance` is ignored when `{field_name}` is a `Variable`; '
                'the variable already carries its own Logfire instance.',
                # 1=warn, 2=_warn_logfire_instance_ignored, 3=__post_init__,
                # 4=dataclass-generated __init__, 5=user's `ManagedTool(...)` call.
                stacklevel=4,
            )

    def _build_managed_variable(
        self, name: str, *, prefix: str, value_type: type[ValueT], default: ValueT
    ) -> Variable[ValueT]:
        """Declare the backing variable as `<prefix><name>`, normalizing and validating the name."""
        # Strip the prefix if the user accidentally passed it so we can still apply
        # hyphen-to-underscore normalization, then re-add the prefix below.
        if name.startswith(prefix):
            warnings.warn(
                f'The {prefix!r} prefix is added automatically; pass the bare name rather than {name!r}.',
                # Same chain as `_warn_logfire_instance_ignored`: helper → __post_init__ → dataclass __init__ → user.
                stacklevel=4,
            )
            name = name[len(prefix) :]

        variable_name = f'{prefix}{name.replace("-", "_")}'
        if not variable_name.isidentifier():
            raise ValueError(
                f'Name {name!r} produces an invalid variable name {variable_name!r}; '
                'names may only contain letters, digits, hyphens, and underscores.'
            )

        # Construct the variable directly (rather than via `logfire.var`) so redeclaring the
        # same name is idempotent: `logfire.var` registers in a per-instance registry and raises
        # on a duplicate name, which would break sharing one variable across agents.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        return Variable(variable_name, type=value_type, default=default, logfire_instance=instance)

    @property
    def resolved(self) -> ResolvedVariable[ValueT] | None:
        """The resolution for the active run, or `None` outside a run.

        Exposes the full [`ResolvedVariable`][logfire.variables.ResolvedVariable] (`value`, `label`,
        `version`, `reason`, ...) so callers can inspect which version is in play.
        """
        return self._resolved.get()

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the resolution's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

    def _auto_create_config(self, variable: Variable[ValueT], *, example: str | None) -> VariableConfig:
        """The config auto-create writes: the variable's own, with the capability's schema and example.

        [`Variable.to_config`][logfire.variables.Variable.to_config] derives `json_schema` from the
        payload's Pydantic type adapter, which tracks both the model's Python types and the Pydantic
        version that generated them. A capability that maintains its own stored schema (`_json_schema`)
        substitutes it here, so the persisted contract does not change shape underneath the Logfire UI
        when either side is upgraded.
        """
        updates: dict[str, Any] = {}
        if self._json_schema is not None:
            updates['json_schema'] = self._json_schema
        if example is not None:
            updates['example'] = example
        return variable.to_config().model_copy(update=updates)

    def _maybe_auto_create(
        self, variable: Variable[ValueT], *, example: str | None = None, ctx: RunContext[Any] | None = None
    ) -> None:
        """Kick off background creation of the backing variable, at most once per process per key."""
        if ctx is not None and _in_durable_context(ctx):
            _warn_durable_write_skipped(variable)
            return
        key = _auto_create_key(variable)
        with _auto_create_lock:
            if key in _auto_create_attempted:
                return
            # Mark before spawning: one attempt per process, so a failed create doesn't retry.
            _auto_create_attempted.add(key)
        _spawn_create(variable, self._auto_create_config(variable, example=example))

    def _should_auto_create_for(self, variable: Variable[ValueT], resolved: ResolvedVariable[ValueT]) -> bool:
        """Whether a configured provider does not recognize this variable and creation is still eligible.

        Auto-create is for exactly one case: a provider is configured but has no entry for this name,
        so resolution fell back to the code default. logfire >= 4.37 reports that as `'code_default'`
        (older SDKs as `'unrecognized_variable'`), but `'code_default'` also covers "no provider
        configured" and "known variable with no targeted value" -- neither of which should create
        anything. The once-per-process guard is only peeked at here; `_maybe_auto_create` re-checks
        and marks it under the same lock, making concurrent callers race-safe.

        Takes the variable the caller resolved rather than reading it off the capability: a nameless
        capability backs one variable per agent, so there is no single `_variable` to read.

        The provider probe is the one remote call on the run thread. Auto-create is documented as
        best-effort -- an unreachable provider degrades to the code default and never fails a run --
        so any error it raises means "cannot tell whether this exists", which is answered `False`.
        """
        if not self.auto_create or resolution_reason(resolved) not in _CODE_DEFAULT_REASONS:
            return False
        with _auto_create_lock:
            if _auto_create_key(variable) in _auto_create_attempted:
                return False
        provider = variable.logfire_instance.config.get_variable_provider()
        if isinstance(provider, NoOpVariableProvider):
            return False
        try:
            return provider.get_variable_config(variable.name) is None
        except Exception:
            return False

    def _maybe_auto_create_for(
        self, variable: Variable[ValueT], resolved: ResolvedVariable[ValueT], *, ctx: RunContext[Any] | None = None
    ) -> None:
        """Trigger background auto-create when a configured provider doesn't recognize the variable yet.

        Auto-create is for exactly one case: a provider is configured but has no entry for this name,
        so resolution fell back to the code default. logfire >= 4.37 reports that as `'code_default'`
        (older SDKs as `'unrecognized_variable'`), but `'code_default'` also covers "no provider
        configured" and "known variable with no targeted value" -- neither of which should create
        anything. So we confirm against the provider itself: it must be a real (non-`NoOp`) provider
        that has no config for this name. A `resolved`/`context_override` value isn't a candidate at
        all, and is filtered out by the reason check up front.
        """
        if self._should_auto_create_for(variable, resolved):
            self._maybe_auto_create(variable, ctx=ctx)

    def _resolve(self, ctx: RunContext[AgentDepsT]) -> ResolvedVariable[ValueT]:
        """Resolve the backing variable for this run using the capability's targeting inputs.

        Shared by `wrap_run` (the base's per-run resolution point) and subclasses that must resolve
        earlier -- e.g. in `for_run`, where the resolved value drives what the run is assembled from.
        Builds the backing variable from the agent's `name` first when the capability is nameless.
        """
        variable = self._ensure_variable(ctx)

        if callable(self.targeting_key):
            targeting_key = self.targeting_key(ctx)
        else:
            targeting_key = self.targeting_key

        if callable(self.attributes):
            attributes = self.attributes(ctx)
        else:
            attributes = self.attributes

        return variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Resolve the variable once and keep its baggage active for the duration of the run."""
        resolved = self._resolve(ctx)
        if self._auto_create_in_wrap_run:
            self._maybe_auto_create_for(self._ensure_variable(ctx), resolved, ctx=ctx)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                return await handler()
            finally:
                self._resolved.reset(token)

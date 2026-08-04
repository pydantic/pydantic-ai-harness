"""The `Scheduling` capability for model-managed scheduled runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.scheduling._runner import scheduled_run_var
from pydantic_ai_harness.scheduling._store import InMemoryScheduleStore, ScheduleStore, SqliteScheduleStore
from pydantic_ai_harness.scheduling._toolset import SchedulingToolset

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentInstructions

_DEFAULT_GUIDANCE = (
    'Use `create_schedule`, `list_schedules`, `get_schedule`, `update_schedule`, `pause_schedule`, '
    '`resume_schedule`, `delete_schedule`, and `run_schedule_now` to manage scheduled work. Schedule strings accept '
    '`every <N><m|h|d>`, `in <N><m|h|d>`, ISO 8601 datetimes, or five-field cron expressions. Schedules only '
    'execute while the application runs a `ScheduleRunner`.'
)


@dataclass
class Scheduling(AbstractCapability[AgentDepsT]):
    """Model-managed schedules backed by a shared `ScheduleStore`.

    The default store is a fresh `InMemoryScheduleStore` owned by this capability
    instance. Pass an explicit store when schedules must persist across process
    restarts or be shared with separately constructed runners.
    """

    store: ScheduleStore | None = None
    """Storage backend. `None` creates a fresh in-memory store for this instance."""

    timezone: str = 'UTC'
    """Default IANA timezone for schedules created by the tools."""

    guidance: str | None = None
    """Static scheduling guidance for the system prompt.

    Three states make opting out explicit:

    - `None` uses the built-in guidance.
    - `''` provides no guidance.
    - Any other string replaces the built-in guidance verbatim.

    Treating `None` as an opt-out would leave no explicit way to request the
    default and could turn a configuration value resolving to `None` into an
    unintended omission.
    """

    def __post_init__(self) -> None:
        """Validate configuration and create the instance-owned default store."""
        try:
            ZoneInfo(self.timezone)
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Unknown IANA timezone: {self.timezone!r}') from exc
        if self.store is None:
            self.store = InMemoryScheduleStore()

    @property
    def resolved_store(self) -> ScheduleStore:
        """Return the store owned or provided by this capability."""
        if self.store is None:  # pragma: no cover - `__post_init__` resolves this field
            raise RuntimeError('Scheduling store was not initialized.')
        return self.store

    def get_toolset(self) -> SchedulingToolset[AgentDepsT]:
        """Provide the recursion-guarded `scheduling` toolset."""
        return SchedulingToolset(self)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Provide scheduling guidance outside scheduled runs."""

        def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            del ctx
            if scheduled_run_var.get() is not None:
                return None
            if self.guidance is not None:
                return self.guidance or None
            return _DEFAULT_GUIDANCE

        return instructions

    @classmethod
    def from_spec(
        cls,
        *,
        backend: Literal['memory', 'sqlite'] = 'memory',
        database: str = '.agent-schedules.db',
        timezone: str = 'UTC',
        guidance: str | None = None,
    ) -> Scheduling[AgentDepsT]:
        """Construct `Scheduling` from serializable options."""
        if backend != 'sqlite' and database != '.agent-schedules.db':
            raise ValueError('database is only valid with backend="sqlite"')
        if backend == 'memory':
            store: ScheduleStore | None = None
        elif backend == 'sqlite':
            store = SqliteScheduleStore(database)
        else:
            # Spec values arrive from YAML or JSON before the `Literal` can protect this branch.
            raise ValueError(f"Unknown scheduling backend {backend!r}; expected 'memory' or 'sqlite'.")
        return cls(store=store, timezone=timezone, guidance=guidance)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the agent-spec serialization name."""
        return 'Scheduling'

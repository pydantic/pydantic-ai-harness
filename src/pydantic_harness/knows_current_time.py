"""KnowsCurrentTime capability: injects the current date/time into the system prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset


def _resolve_tz(tz: str) -> timezone | ZoneInfo:
    """Resolve a timezone string to a ``tzinfo`` object."""
    if tz == 'UTC':
        return timezone.utc
    return ZoneInfo(tz)


def _format_datetime(dt: datetime, fmt: str) -> str:
    """Format a datetime with both the strftime format and a human-readable suffix.

    Returns a string like::

        The current date and time is: 2026-04-02T20:30:00Z (Wednesday, April 2, 2026)
    """
    formatted = dt.strftime(fmt)
    human = f'{dt.strftime("%A")}, {dt.strftime("%B")} {dt.day}, {dt.year}'
    return f'The current date and time is: {formatted} ({human})'


@dataclass
class KnowsCurrentTime(AbstractCapability[AgentDepsT]):
    """Injects the current date and time into the system prompt.

    The simplest possible capability: ``get_instructions()`` returns a dynamic
    callable that produces a formatted datetime string on each model request.

    Optionally also registers a ``get_current_time`` tool so the agent can
    re-check the time mid-conversation.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness import KnowsCurrentTime

        agent = Agent('openai:gpt-4o', capabilities=[KnowsCurrentTime()])
    """

    tz: str = 'UTC'
    """IANA timezone name (e.g. ``'UTC'``, ``'America/New_York'``)."""

    format: str = '%Y-%m-%dT%H:%M:%SZ'
    """``strftime`` format string for the datetime."""

    include_tool: bool = False
    """Whether to also register a ``get_current_time`` tool."""

    _tzinfo: timezone | ZoneInfo = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve and validate the timezone string.

        Raises:
            ValueError: If ``tz`` is not a valid IANA timezone name.
        """
        try:
            self._tzinfo = _resolve_tz(self.tz)
        except (KeyError, ModuleNotFoundError) as exc:
            raise ValueError(
                f'{self.tz!r} is not a valid IANA timezone name. Examples: "UTC", "America/New_York", "Europe/London".'
            ) from exc

    def _now(self) -> datetime:
        return datetime.now(tz=self._tzinfo)

    def _formatted_now(self) -> str:
        return _format_datetime(self._now(), self.format)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return a dynamic callable that produces the current datetime on each request."""

        def _instructions() -> str:
            return self._formatted_now()

        return _instructions

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Optionally return a toolset with a ``get_current_time`` tool."""
        if not self.include_tool:
            return None

        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        @toolset.tool_plain
        def get_current_time() -> str:  # pyright: ignore[reportUnusedFunction]
            """Get the current date and time."""
            return self._formatted_now()

        return toolset

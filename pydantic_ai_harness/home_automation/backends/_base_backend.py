from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ServiceArgument:
    name: str
    value_type: str
    required: bool = False
    description: str | None = None
    enum: tuple[str, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None


@dataclass
class Service:
    domain: str
    name: str
    args: tuple[ServiceArgument, ...] = ()


@dataclass
class Entity:
    entity_id: str
    domain: str
    name: str | None
    current_state: str


@dataclass
class EntityState:
    entity_id: str
    last_updated: str
    state: str


def _empty_entity_states() -> list[EntityState]:
    return []


@dataclass
class ServiceCallResult:
    changed_states: list[EntityState] = field(default_factory=_empty_entity_states)
    service_response: dict[str, Any] | None = None
    verified_state: EntityState | None = None


class HomeBackend(Protocol):
    """Backend contract for home automation service discovery."""

    async def list_services(self, domain: str | None = None) -> list[Service]:  # pragma: no cover
        """Return the services that can be called, optionally filtered by domain."""
        ...

    async def list_entities(self, domain: str | None = None) -> list[Entity]:  # pragma: no cover
        """Return known entities, optionally filtered by domain."""
        ...

    async def get_state(self, entity_id: str) -> EntityState:  # pragma: no cover
        """Return the current state for a single entity."""
        ...

    async def list_states(self, domain: str | None = None) -> list[EntityState]:  # pragma: no cover
        """Return current states, optionally filtered by domain."""
        ...

    async def call_service(
        self,
        domain: str,
        entity_id: str,
        service_name: str,
        *,
        want_response: bool = False,
        **data: Any,
    ) -> ServiceCallResult:  # pragma: no cover
        """Call a service for one entity and return the observed outcome."""
        ...

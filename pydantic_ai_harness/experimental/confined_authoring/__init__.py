"""Confined self-authoring: let an agent author its own typed, sandboxed, persistent tools."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.confined_authoring._capability import ConfinedAuthoring
from pydantic_ai_harness.experimental.confined_authoring._slots import (
    AuthoredSlot,
    InjectedFunction,
    SlotKind,
    SlotParameter,
    SlotStatus,
    SlotValueType,
)
from pydantic_ai_harness.experimental.confined_authoring._store import SlotStore
from pydantic_ai_harness.experimental.confined_authoring._toolset import ConfinedAuthoringToolset
from pydantic_ai_harness.experimental.confined_authoring._validate import (
    SlotValidationError,
    validate_tool_slot,
)

warn_experimental('confined_authoring')

__all__ = [
    'AuthoredSlot',
    'ConfinedAuthoring',
    'ConfinedAuthoringToolset',
    'InjectedFunction',
    'SlotKind',
    'SlotParameter',
    'SlotStatus',
    'SlotStore',
    'SlotValidationError',
    'SlotValueType',
    'validate_tool_slot',
]

"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables."""

from pydantic_ai_harness.logfire._agent_control import (
    AGENT_CONFIG_JSON_SCHEMA,
    AgentConfig,
    AgentConfigSettings,
    AgentControl,
    InstructionBlock,
    ToolDefinitionOverride,
)
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_variable import resolution_reason

__all__ = [
    'AGENT_CONFIG_JSON_SCHEMA',
    'AgentControl',
    'AgentConfig',
    'AgentConfigSettings',
    'InstructionBlock',
    'ManagedPrompt',
    'ToolDefinitionOverride',
    'resolution_reason',
]

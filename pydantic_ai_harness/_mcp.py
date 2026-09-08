"""Helpers shared by capabilities that wrap hosted MCP servers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_ai.tools import ToolDefinition


def is_read_only(tool_def: ToolDefinition) -> bool:
    """Whether the server marked the tool `readOnlyHint`; an unannotated tool counts as a write.

    `MCPToolset` copies each tool's MCP annotations into `tool_def.metadata['annotations']`.
    """
    metadata: dict[str, Any] = tool_def.metadata or {}
    annotations = metadata.get('annotations')
    return isinstance(annotations, Mapping) and annotations.get('readOnlyHint') is True  # pyright: ignore[reportUnknownMemberType]

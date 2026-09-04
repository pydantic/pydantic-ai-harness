"""Linear capability: issues, projects, comments, and related workspace data for agents.

Requires Pydantic AI's `mcp` optional group: `uv add "pydantic-ai-slim[mcp]"`.
"""

from pydantic_ai_harness.linear._capability import (
    LINEAR_MCP_URL,
    LINEAR_READ_ONLY_MCP_URL,
    Linear,
)

__all__ = ['LINEAR_MCP_URL', 'LINEAR_READ_ONLY_MCP_URL', 'Linear']

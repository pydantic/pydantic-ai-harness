"""Logfire's hosted MCP server as a capability.

Requires the `logfire-mcp` extra: `uv add "pydantic-ai-harness[logfire-mcp]"`.
"""

from pydantic_ai_harness.logfire_mcp._capability import LogfireMCP

__all__ = ('LogfireMCP',)

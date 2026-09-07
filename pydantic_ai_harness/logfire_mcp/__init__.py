"""Logfire's hosted MCP server as a capability.

Requires the `logfire-mcp` extra: `uv add "pydantic-ai-harness[logfire-mcp]"`.
"""

from pydantic_ai_harness.logfire_mcp._capability import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP

__all__ = ['LOGFIRE_EU_MCP_URL', 'LOGFIRE_US_MCP_URL', 'LogfireMCP']

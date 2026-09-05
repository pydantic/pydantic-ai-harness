"""Project-scoped access to the hosted Pydantic Logfire MCP server.

Requires the `logfire-mcp` extra: `uv add "pydantic-ai-harness[logfire-mcp]"`.
"""

from pydantic_ai_harness.logfire_mcp._capability import LogfireMCP, LogfireRegion

__all__ = ['LogfireMCP', 'LogfireRegion']

"""Atlassian Rovo MCP integration for Jira and selected related products.

Install the optional dependencies with `uv add "pydantic-ai-harness[atlassian]"`.
"""

from pydantic_ai_harness.atlassian._capability import Atlassian
from pydantic_ai_harness.atlassian._toolset import (
    AtlassianAccess,
    AtlassianProduct,
    AtlassianToolset,
)

__all__ = [
    'Atlassian',
    'AtlassianAccess',
    'AtlassianProduct',
    'AtlassianToolset',
]

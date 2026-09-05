"""Stripe capability for account-scoped API access over Stripe's hosted MCP server.

Install it with `uv add "pydantic-ai-harness[stripe]"`.
"""

from pydantic_ai_harness.stripe._capability import Stripe

__all__ = ['Stripe']

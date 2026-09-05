"""Supabase capability for the official hosted MCP server.

Install with `uv add "pydantic-ai-harness[supabase]"`.
"""

from pydantic_ai_harness.supabase._capability import Supabase, SupabaseFeature

__all__ = ['Supabase', 'SupabaseFeature']

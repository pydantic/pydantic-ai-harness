"""Agent harness for composable, reusable AI agent capabilities, built on PydanticAI.

Usage:
    from pydantic_harness import Memory, Skills, Guardrails, ...
"""

# Each capability module is imported and re-exported here.
# Capabilities are listed alphabetically.

from pydantic_harness.repo_context_injection import RepoContextInjection

__all__: list[str] = [
    'RepoContextInjection',
]

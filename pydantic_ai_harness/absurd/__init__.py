"""Expose the `pydantic-ai-absurd` durability capability through Harness."""

from importlib.util import find_spec

if find_spec('pydantic_ai_absurd') is None:
    raise ImportError(
        'pydantic-ai-absurd is required for AbsurdDurability. Install it with: '
        'pip install "pydantic-ai-harness[absurd]"'
    )

from pydantic_ai_absurd import AbsurdDurability, AbsurdParallelExecutionMode

__all__ = ['AbsurdDurability', 'AbsurdParallelExecutionMode']

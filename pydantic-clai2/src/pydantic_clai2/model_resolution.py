"""Resolve CLAI's credential-backed providers for either interface."""

import asyncio

from pydantic_ai.models import Model

from . import openrouter, vllm
from .auth import CodexAuth


async def resolve_model(name: str, *, auth: CodexAuth) -> Model | str:
    """Retain saved credentials and custom endpoints instead of using core's string inference."""
    if name.startswith('openrouter:'):
        return await asyncio.to_thread(openrouter.model, name)
    if name.startswith('vllm:'):
        return await asyncio.to_thread(vllm.model, name)
    return auth.model(name) if name.startswith('openai-codex:') else name

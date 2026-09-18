"""Fly.io Sprites sandbox capability for Pydantic AI agents.

`SpriteSandbox` is the supported entry point. `SpriteSandboxSession` exposes
lower-level lifecycle, command, and file access for applications that need to
share a caller-owned Sprite across runs.
"""

from pydantic_ai_harness.sprites._capability import SpriteSandbox
from pydantic_ai_harness.sprites._session import (
    SpriteSandboxAuthError,
    SpriteSandboxError,
    SpriteSandboxExecResult,
    SpriteSandboxOwnershipError,
    SpriteSandboxSession,
    SpriteSandboxTerminalError,
    SpriteSandboxUnavailableError,
)

__all__ = [
    'SpriteSandbox',
    'SpriteSandboxAuthError',
    'SpriteSandboxError',
    'SpriteSandboxExecResult',
    'SpriteSandboxOwnershipError',
    'SpriteSandboxSession',
    'SpriteSandboxTerminalError',
    'SpriteSandboxUnavailableError',
]

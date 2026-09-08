"""Fly.io Sprites integration for core sandboxes."""

from ._backend import SpriteSandboxAuthError, SpriteSandboxBackend, SpriteSandboxError, SpriteSandboxUnavailableError
from ._capability import SpriteSandbox

__all__ = (
    'SpriteSandbox',
    'SpriteSandboxAuthError',
    'SpriteSandboxBackend',
    'SpriteSandboxError',
    'SpriteSandboxUnavailableError',
)

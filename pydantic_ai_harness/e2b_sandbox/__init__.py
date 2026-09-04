"""E2B sandbox capability: gives agents an isolated cloud computer to work in.

`E2BSandbox` is the supported entry point; build an agent with it and add tools
that consume `ctx.sandbox`.
`E2BSandboxBackend` is the E2B implementation of Pydantic AI's sandbox backend protocol,
public for applications that want to create or attach to a sandbox themselves and pass it to
a run as `sandbox=`.
"""

from pydantic_ai_harness.e2b_sandbox._backend import (
    E2BSandboxAuthError,
    E2BSandboxBackend,
    E2BSandboxError,
    E2BSandboxUnavailableError,
)
from pydantic_ai_harness.e2b_sandbox._capability import E2BSandbox

__all__ = [
    'E2BSandbox',
    'E2BSandboxAuthError',
    'E2BSandboxBackend',
    'E2BSandboxError',
    'E2BSandboxUnavailableError',
]

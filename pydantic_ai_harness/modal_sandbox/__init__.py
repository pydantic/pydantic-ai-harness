"""Modal sandbox capability: gives agents an isolated cloud sandbox to work in.

`ModalSandbox` is the supported entry point; build an agent with it and add tools
that consume `ctx.sandbox`.
`ModalSandboxBackend` is the Modal implementation of Pydantic AI's sandbox backend protocol,
public for applications that want to create or attach to a sandbox themselves and pass it to
a run as `sandbox=`.
"""

from pydantic_ai_harness.modal_sandbox._backend import (
    ModalSandboxAuthError,
    ModalSandboxBackend,
    ModalSandboxError,
    ModalSandboxUnavailableError,
)
from pydantic_ai_harness.modal_sandbox._capability import ModalSandbox

__all__ = [
    'ModalSandbox',
    'ModalSandboxAuthError',
    'ModalSandboxBackend',
    'ModalSandboxError',
    'ModalSandboxUnavailableError',
]

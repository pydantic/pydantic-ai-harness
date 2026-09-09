"""E2B sandbox capability: gives agents an isolated cloud computer to work in.

`E2BWorkspace` is the supported entry point; build an agent with it and add tools
that consume `ctx.workspace`.
`E2BWorkspaceBackend` is the E2B implementation of Pydantic AI's workspace backend protocol,
public for applications that want to create or attach to a sandbox themselves and pass it to
a run as `workspace=`.
"""

from pydantic_ai_harness.e2b_workspace._backend import E2BWorkspaceBackend
from pydantic_ai_harness.e2b_workspace._capability import E2BWorkspace

__all__ = [
    'E2BWorkspace',
    'E2BWorkspaceBackend',
]

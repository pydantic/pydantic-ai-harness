"""Runnable agent instance for the `Coder` harness."""

import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness.coder._capability import Coder

coder_agent = Agent[object](
    name='coder',
    capabilities=[
        # Commands inherit nothing from this process; `PATH` and `HOME` let them find the user's tools.
        LocalWorkspace[object]('.', env={name: os.environ[name] for name in ('PATH', 'HOME') if name in os.environ}),
        Coder[object](),
    ],
)
"""Model-less coding agent for CLIs that load `module:variable` targets."""

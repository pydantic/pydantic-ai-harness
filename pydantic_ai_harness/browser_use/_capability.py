"""Browser Use capability via the Browser Use CLI"""
# ruff: noqa: D415

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.browser_use._toolset import SKILL_HEADER, BrowserUseToolset

_BROWSER_INSTRUCTIONS = (
    'Both the browser and your Python variables persist between `browser_exec` calls, so you can '
    'collect results over several calls and use them later. If a call times out, the Python '
    'session restarts while the browser survives, so re-derive what you need from the page. '
    'Batching a whole step (navigate, wait, extract, act) into one call is still faster than one '
    'call per action.'
)


@dataclass
class BrowserUse(AbstractCapability[AgentDepsT]):
    """Web browsing for agents powered by the [Browser Use](https://browser-use.com) CLI.

    Adds a `browser_exec` tool: model-written Python runs against a persistent
    browser session -- local Chrome over CDP, a headless Chrome, or a Browser Use
    cloud browser. Local browsing needs no account or API key.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.browser_use import BrowserUse

    agent = Agent('openai:gpt-5.5', capabilities=[BrowserUse()])
    ```

    Needs the `browser-use` CLI (`uv tool install browser-use`); falls back to
    `uvx browser-use` when `uvx` is available. LLM provider API keys are
    scrubbed from the CLI subprocess automatically.
    """

    browser: Literal['local', 'headless', 'cloud'] = 'local'
    """'local' = the Chrome already running here, logins included; 'headless' = a
    throwaway Chrome launched per run (`BH_CHROME_PATH` picks the binary);
    'cloud' = a Browser Use cloud browser, provisioned and stopped per run
    (needs auth, bills while running)."""

    scope: Literal['run', 'agent'] = 'run'
    """'run' = fresh session per run, safe concurrently; 'agent' = one session
    shared across runs inside `async with agent:` (chat loops)"""

    command: str = 'browser-use'
    """Name or path of the CLI binary"""

    default_timeout: float = 300.0
    """Seconds per call when the model passes no `timeout_seconds`"""

    progress: Callable[[str], None] | None = None
    """Sink for live narration of browser steps, e.g. `print`"""

    guidance: str | None = None
    """Replace the default system-prompt guidance; `''` disables it"""

    def __post_init__(self) -> None:
        """Reject out-of-range configuration"""
        if self.default_timeout <= 0:
            raise ValueError(f'default_timeout must be positive, got {self.default_timeout}')

    def get_toolset(self) -> BrowserUseToolset[AgentDepsT]:
        """Build the toolset that provides the `browser_exec` tool"""
        return BrowserUseToolset[AgentDepsT](
            command=self.command,
            default_timeout=self.default_timeout,
            browser=self.browser,
            scope=self.scope,
            progress=self.progress,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT]:
        """Contribute the capability's guidance plus the CLI's own skill documentation"""

        async def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            parts: list[str] = []
            if self.guidance is None:
                parts.append(_BROWSER_INSTRUCTIONS)
            elif self.guidance:
                parts.append(self.guidance)
            skill = await self.get_toolset().cli_skill_text()
            if skill is not None:
                parts.append(f'{SKILL_HEADER.strip()}\n\n{skill}')
            return '\n\n'.join(parts) or None

        return instructions

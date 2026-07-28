"""Browser Use capability via the Browser Use CLI"""
# ruff: noqa: D415

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.browser_use._progress import Detail
from pydantic_ai_harness.browser_use._toolset import SESSION_RE, SKILL_HEADER, BrowserUseToolset

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
    `uvx browser-use` when `uvx` is available.
    """

    browser: Literal['local', 'headless', 'cloud'] = 'local'
    """'local' = the Chrome already running here, logins included; 'headless' = a
    throwaway Chrome launched per run; 'cloud' = a Browser Use cloud browser
    (needs auth, bills while running). Ignored when `cdp_url` is set."""

    cloud_session: str | None = None
    """Attach to a cloud browser you manage, instead of one per run"""

    cloud_timeout_minutes: int = 60
    """Server-side lifetime cap on a per-run cloud browser (1-240); the billing
    backstop when the process dies before cleanup"""

    chrome_path: str | None = None
    """Chrome or Chromium executable to launch when `browser='headless'`"""

    cdp_url: str | None = None
    """Connect to an existing browser over CDP instead of finding one"""

    command: str = 'browser-use'
    """Name or path of the CLI binary"""

    cwd: str | Path = '.'
    """Working directory the CLI runs in"""

    default_timeout: float = 300.0
    """Seconds per call when the model passes no `timeout_seconds`"""

    max_timeout: float = 1800.0
    """Cap on a model-supplied `timeout_seconds`"""

    max_output_chars: int = 50_000
    """Output cap per call, keeping the tail"""

    workspace: str | Path | None = None
    """CLI agent-workspace dir (`BH_AGENT_WORKSPACE`); persists across calls"""

    env: Mapping[str, str] | None = None
    """Explicit environment for the CLI subprocess, replacing inheritance"""

    denied_env_patterns: Sequence[str] = field(default_factory=list[str])
    """Glob patterns for environment variable names to strip before spawning"""

    fallback_to_uvx: bool = True
    """Run `uvx browser-use` when the CLI is not on PATH"""

    persist_variables: bool = True
    """One persistent Python session per run: variables survive between calls, a
    timeout restarts the session (the browser survives). `False` = stateless."""

    scope: Literal['run', 'agent'] = 'run'
    """'run' = fresh session per run, safe concurrently; 'agent' = one session
    shared across runs inside `async with agent:` (chat loops)"""

    progress: Callable[[str], None] | None = None
    """Sink for live narration of browser calls, e.g. `print`"""

    progress_detail: Detail = 'steps'
    """'steps' = one label per call plus errors; 'code' = also code and output"""

    guidance: str | None = None
    """Replace the default system-prompt guidance; `''` disables it"""

    def __post_init__(self) -> None:
        """Reject out-of-range configuration"""
        if self.default_timeout <= 0:
            raise ValueError(f'default_timeout must be positive, got {self.default_timeout}')
        if self.max_timeout < self.default_timeout:
            raise ValueError(f'max_timeout ({self.max_timeout}) must be >= default_timeout ({self.default_timeout})')
        if self.max_output_chars <= 0:
            raise ValueError(f'max_output_chars must be positive, got {self.max_output_chars}')
        if not 1 <= self.cloud_timeout_minutes <= 240:
            raise ValueError(f'cloud_timeout_minutes must be between 1 and 240, got {self.cloud_timeout_minutes}')
        if self.cloud_session is not None and SESSION_RE.match(self.cloud_session) is None:
            raise ValueError(f'invalid cloud_session {self.cloud_session!r}: use 1-64 chars from [A-Za-z0-9_-]')

    def get_toolset(self) -> BrowserUseToolset[AgentDepsT]:
        """Build the toolset that provides the `browser_exec` tool"""
        return BrowserUseToolset[AgentDepsT](
            command=self.command,
            cwd=Path(self.cwd),
            default_timeout=self.default_timeout,
            max_timeout=self.max_timeout,
            max_output_chars=self.max_output_chars,
            workspace=Path(self.workspace) if self.workspace is not None else None,
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
            fallback_to_uvx=self.fallback_to_uvx,
            persist_variables=self.persist_variables,
            browser=self.browser,
            cloud_session=self.cloud_session,
            cloud_timeout_minutes=self.cloud_timeout_minutes,
            cdp_url=self.cdp_url,
            chrome_path=self.chrome_path,
            scope=self.scope,
            progress=self.progress,
            progress_detail=self.progress_detail,
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

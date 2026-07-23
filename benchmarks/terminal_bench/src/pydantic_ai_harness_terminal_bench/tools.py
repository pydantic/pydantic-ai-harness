"""The bash tool: the agent's single hand on the environment.

Terminal-Bench runs each task in a container. Harbor exposes that container
through `environment.exec(...)`. This module turns that one async call into the
agent's `bash` tool, so the reference agent stays substrate-agnostic: it depends
on a `CommandExecutor` callable, not on Harbor. The Harbor adapter supplies an
executor backed by `environment.exec`; the tests supply a scripted fake.

Two tools total is deliberate. Anthropic's launch scaffold and mini-swe-agent
both show a bash tool (optionally plus a file editor) is enough; Terminus 2's
tmux-keystroke design is the thing to avoid, not copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class CommandResult:
    """The outcome of one command in the environment."""

    output: str
    """Combined stdout and stderr, in that order."""

    exit_code: int
    """Process exit code. Non-zero is surfaced to the model, not raised."""


class CommandExecutor(Protocol):
    """Runs one shell command in the task environment and returns its result.

    The Harbor adapter implements this over `environment.exec`; tests implement
    it with a recorded, scripted fake. Keeping the agent behind this protocol is
    what lets the seam be tested without Docker.
    """

    async def __call__(self, command: str, *, timeout_sec: int | None = None) -> CommandResult:
        """Execute `command`, waiting at most `timeout_sec` when provided."""
        ...


@dataclass
class TerminalBenchDeps:
    """Run-scoped dependencies the `bash` tool reaches through `RunContext`."""

    execute: CommandExecutor
    """The environment-backed command executor."""

    default_timeout_sec: int = 120
    """Per-command timeout when the model does not ask for one."""

    max_output_chars: int = 16_000
    """Head/tail budget for a single command's output before truncation."""


def _truncate(text: str, limit: int) -> str:
    """Keep a head and tail slice of `text`, marking what was dropped.

    A single command (a chatty build, `cat` of a large file) can produce more
    output than is useful to feed back. Keeping both ends preserves the start of
    a log and its final error, which is where the signal usually is.
    """
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    dropped = len(text) - limit
    return f'{text[:head]}\n[... {dropped} characters truncated ...]\n{text[-tail:]}'


def format_result(result: CommandResult, *, max_output_chars: int) -> str:
    """Render a `CommandResult` as the string the model sees.

    Non-zero exit codes are appended as a line rather than raised, so the model
    can read the error and try again -- the recovery loop stays in the model,
    which is the point of a minimal scaffold.
    """
    body = _truncate(result.output, max_output_chars).rstrip()
    if result.exit_code != 0:
        suffix = f'[exit code {result.exit_code}]'
        return f'{body}\n{suffix}' if body else suffix
    return body or '(no output)'


async def bash(ctx: RunContext[TerminalBenchDeps], command: str) -> str:
    """Run a shell command in the task environment and return its output.

    Each call is a fresh shell: `cd` does not persist across calls, so change
    directory within a single command (`cd path && ...`) or use absolute paths.
    Combine stdout and stderr are returned together, followed by the exit code
    when it is non-zero.
    """
    deps = ctx.deps
    result = await deps.execute(command, timeout_sec=deps.default_timeout_sec)
    return format_result(result, max_output_chars=deps.max_output_chars)


def build_bash_toolset() -> FunctionToolset[TerminalBenchDeps]:
    """Build the single-tool toolset the reference agent runs on."""
    toolset = FunctionToolset[TerminalBenchDeps]()
    toolset.add_function(bash)
    return toolset

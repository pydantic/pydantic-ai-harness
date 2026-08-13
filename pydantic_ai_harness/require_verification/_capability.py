"""Verification evidence tracking and completion guard."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, ValidatedToolArgs
from pydantic_ai.messages import SystemPromptPart, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_graph import End

VerificationKind = Literal['test', 'lint', 'typecheck', 'build']

_VERIFICATION_PATTERNS: tuple[tuple[VerificationKind, re.Pattern[str]], ...] = (
    (
        'test',
        re.compile(
            r'\b(?:pytest|unittest|tox|nox|go\s+test|cargo\s+test|dotnet\s+test)\b'
            r'|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b'
            r'|\b(?:mvn|gradle|gradlew|make)\s+(?:\S+\s+)*test\w*\b',
            re.IGNORECASE,
        ),
    ),
    (
        'lint',
        re.compile(
            r'\b(?:ruff\s+check|flake8|pylint|eslint|golangci-lint|cargo\s+clippy)\b'
            r'|\b(?:biome|make)\s+(?:\S+\s+)*lint\w*\b',
            re.IGNORECASE,
        ),
    ),
    (
        'typecheck',
        re.compile(r'\b(?:pyright|basedpyright|mypy|tsc\b[^\r\n;&|]*--noEmit|make\s+typecheck)\b', re.IGNORECASE),
    ),
    (
        'build',
        re.compile(
            r'\b(?:python\s+-m\s+build|uv\s+build|cargo\s+(?:build|check)|go\s+build|dotnet\s+build)\b'
            r'|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?build\b'
            r'|\b(?:mvn|gradle|gradlew|make)\s+(?:\S+\s+)*(?:build|package|verify)\b',
            re.IGNORECASE,
        ),
    ),
)

_MUTATING_COMMAND = re.compile(
    r'\b(?:apply_patch|black|ruff\s+format|cargo\s+fmt|git\s+apply)\b'
    r'|\bgofmt\b[^\r\n;&|]*\s-w(?:\s|$)'
    r'|\b(?:sed|perl)\b[^\r\n;&|]*\s-(?:\S*?i\S*?|\S*?p\S*?i\S*?)(?:\s|$)'
    r'|(?:--fix|--write)(?:\s|=|$)',
    re.IGNORECASE,
)

_VERIFICATION_COMMAND_PREFIX = re.compile(
    r'^\s*(?:env\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:command\s+)?'
    r'(?:(?:uv|poetry|pipenv)\s+run\s+)?'
    r'(?:(?:\S*[/\\])?(?:python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?\s+-m\s+)?'
    r'(?:\S*[/\\])?'
    r'(?:pytest|py\.test|unittest|tox|nox|go|cargo|dotnet|npm|pnpm|yarn|bun|mvn|gradle|gradlew|'
    r'make|ruff|flake8|pylint|eslint|golangci-lint|biome|pyright|basedpyright|mypy|tsc)'
    r'(?:\.exe)?\b',
    re.IGNORECASE,
)

_UV_BUILD_PREFIX = re.compile(r'^\s*(?:\S*[/\\])?uv(?:\.exe)?\s+build\b', re.IGNORECASE)

_UNSAFE_VERIFICATION_SHELL_FORM = re.compile(r'[;&|`\r\n]|\$\(')

_NUDGE = (
    'You edited code, but there is no fresh passing verification evidence after the latest edit. '
    'Run the relevant tests, lint, type checking, or build now. If verification cannot run, explain '
    'the concrete blocker and do not claim that checks passed.'
)


@dataclass(frozen=True)
class _Evidence:
    kind: VerificationKind | str
    command: str
    passed: bool
    generation: int


def _verification_kind(command: str) -> VerificationKind | None:
    if _UNSAFE_VERIFICATION_SHELL_FORM.search(command) or not (
        _VERIFICATION_COMMAND_PREFIX.search(command) or _UV_BUILD_PREFIX.search(command)
    ):
        return None
    return next((kind for kind, pattern in _VERIFICATION_PATTERNS if pattern.search(command)), None)


def _command_passed(result: Any) -> bool:
    if not isinstance(result, str):
        return True
    lowered = result.lower()
    return '[exit code:' not in lowered and '[command timed out' not in lowered


@dataclass
class RequireVerification(AbstractCapability[AgentDepsT]):
    """Redirect a bounded number of unverified completion attempts after code edits.

    Successful file-edit tools make earlier evidence stale. Recognized verification
    commands then record passing or failing evidence against the current edit generation.
    If the model tries to finish without a fresh pass, the capability enqueues a bounded
    verification nudge and gives the model another turn.
    """

    max_attempts: int = 2
    """Maximum number of completion redirects in one run."""

    _: KW_ONLY

    mutating_tools: Sequence[str] = ('write_file', 'edit_file')
    """Successful tools that make prior verification evidence stale."""

    verification_tools: Sequence[str] = ()
    """Custom tools whose successful execution counts as verification."""

    shell_tools: Sequence[str] = ('run_command',)
    """Shell tools whose `command` argument is inspected for edits and verification."""

    exempt_path_patterns: Sequence[str] = ('*.md', '*.mdx', '*.rst')
    """Edited paths matching these patterns do not require verification."""

    _generation: int = field(default=0, init=False, repr=False)
    _has_code_edits: bool = field(default=False, init=False, repr=False)
    _attempts: int = field(default=0, init=False, repr=False)
    _evidence: list[_Evidence] = field(default_factory=list[_Evidence], init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError('max_attempts must be at least 0')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RequireVerification[AgentDepsT]:
        """Return isolated evidence state for each run."""
        return replace(self)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Update the edit generation and evidence ledger after successful tools."""
        tool_name = tool_def.name
        if tool_name in self.mutating_tools and not self._is_exempt_path(args.get('path')):
            self._mark_code_edit()

        if tool_name in self.shell_tools:
            command = args.get('command')
            if isinstance(command, str):
                if _MUTATING_COMMAND.search(command):
                    self._mark_code_edit()
                if kind := _verification_kind(command):
                    self._evidence.append(
                        _Evidence(
                            kind=kind,
                            command=command,
                            passed=_command_passed(result),
                            generation=self._generation,
                        )
                    )
        elif tool_name in self.verification_tools:
            self._evidence.append(
                _Evidence(kind=tool_name, command=tool_name, passed=True, generation=self._generation)
            )
        return result

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Redirect an unverified completion through the pending-message queue."""
        if (
            isinstance(result, End)
            and self._has_code_edits
            and not self._has_fresh_pass()
            and self._attempts < self.max_attempts
        ):
            self._attempts += 1
            ctx.enqueue(SystemPromptPart(content=self._nudge()), priority='when_idle')
        return result

    def _mark_code_edit(self) -> None:
        self._generation += 1
        self._has_code_edits = True

    def _is_exempt_path(self, path: object) -> bool:
        if not isinstance(path, str):
            return False
        normalized = PurePosixPath(path.replace('\\', '/'))
        return any(normalized.match(pattern) for pattern in self.exempt_path_patterns)

    def _has_fresh_pass(self) -> bool:
        return any(item.passed and item.generation == self._generation for item in self._evidence)

    def _nudge(self) -> str:
        fresh_failures = [item for item in self._evidence if item.generation == self._generation and not item.passed]
        if not fresh_failures:
            return _NUDGE
        latest = fresh_failures[-1]
        return f'{_NUDGE} The latest recognized {latest.kind} command failed: `{latest.command}`.'

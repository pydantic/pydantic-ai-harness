"""Harbor `BaseAgent` adapter: run the reference agent against a task container.

The reference agent runs host-side. Its `bash` tool calls back into the task
container through `environment.exec`, so nothing needs to be installed in the
container (unlike a `BaseInstalledAgent`). This is the thin seam the whole
package exists to prove: instruction in as the prompt, `environment.exec` in as
the bash executor, token usage out onto the context.

Contrast with VStorm's pydantic-deep Harbor adapter (MIT), which is a
`BaseInstalledAgent`: it pip-installs its whole CLI into each container and
shells out to it. That fits a shipping product with its own binary. For a
minimal reference agent, the in-process `BaseAgent` is smaller and keeps the
agent substrate-agnostic and unit-testable without Docker. The model-name
mapping in `config.convert_model_name` is adapted from their adapter.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness_terminal_bench.agent import build_agent
from pydantic_ai_harness_terminal_bench.config import (
    DEFAULT_COMPACTION_TARGET_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_LIMIT,
    DEFAULT_TOOL_CALLS_LIMIT,
    DEFAULT_TOOL_TIMEOUT_SEC,
    DEFAULT_TOTAL_TOKENS_LIMIT,
    build_usage_limits,
    convert_model_name,
)
from pydantic_ai_harness_terminal_bench.tools import CommandResult, TerminalBenchDeps

_AGENT_NAME = 'pydantic-ai-harness'


def _package_version() -> str:
    try:
        return _pkg_version('pydantic-ai-harness-terminal-bench')
    except PackageNotFoundError:  # pragma: no cover -- only in an uninstalled checkout
        return 'unknown'


class PydanticAITerminalBenchAgent(BaseAgent):
    """Runs the Pydantic AI + harness reference agent on a Harbor task."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        *args: Any,
        tool_timeout_sec: int = DEFAULT_TOOL_TIMEOUT_SEC,
        workdir: str | None = None,
        compaction_target_tokens: int = DEFAULT_COMPACTION_TARGET_TOKENS,
        summarizer_model: str | None = None,
        request_limit: int = DEFAULT_REQUEST_LIMIT,
        tool_calls_limit: int = DEFAULT_TOOL_CALLS_LIMIT,
        total_tokens_limit: int = DEFAULT_TOTAL_TOKENS_LIMIT,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tool_timeout_sec = tool_timeout_sec
        self._workdir = workdir
        self._compaction_target_tokens = compaction_target_tokens
        self._summarizer_model = summarizer_model
        self._request_limit = request_limit
        self._tool_calls_limit = tool_calls_limit
        self._total_tokens_limit = total_tokens_limit

    @staticmethod
    def name() -> str:
        return _AGENT_NAME

    def version(self) -> str | None:
        return _package_version()

    async def setup(self, environment: BaseEnvironment) -> None:
        """No container-side setup: the agent runs host-side over `exec`."""

    def resolve_model(self) -> str | Model:
        """The Pydantic AI model this run uses.

        Overridden by the CI smoke agent to inject a scripted `FunctionModel`,
        which is how the Docker seam is exercised with no API keys.
        """
        if self.model_name is None:
            return DEFAULT_MODEL
        return convert_model_name(self.model_name)

    def build_agent(self) -> Agent[TerminalBenchDeps, str]:
        """Construct the reference agent for this run."""
        model = self.resolve_model()
        return build_agent(
            model=model,
            summarizer_model=self._summarizer_model,
            compaction_target_tokens=self._compaction_target_tokens,
        )

    def _make_deps(self, environment: BaseEnvironment) -> TerminalBenchDeps:
        async def execute(command: str, *, timeout_sec: int | None = None) -> CommandResult:
            result = await environment.exec(
                command,
                cwd=self._workdir,
                timeout_sec=timeout_sec or self._tool_timeout_sec,
            )
            output = _join_streams(result.stdout, result.stderr)
            return CommandResult(output=output, exit_code=result.return_code)

        return TerminalBenchDeps(execute=execute, default_timeout_sec=self._tool_timeout_sec)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        agent = self.build_agent()
        deps = self._make_deps(environment)
        result = None
        try:
            result = await agent.run(
                instruction,
                deps=deps,
                usage_limits=build_usage_limits(
                    request_limit=self._request_limit,
                    tool_calls_limit=self._tool_calls_limit,
                    total_tokens_limit=self._total_tokens_limit,
                ),
            )
        finally:
            self._populate_context(context, result)

    def _populate_context(self, context: AgentContext, result: Any) -> None:
        """Copy token usage onto the Harbor context.

        Called from a `finally` so partial usage is recorded even if the run
        raised (a usage-limit trip, a provider error). `result` is None when the
        run raised before returning.
        """
        if result is None:
            return
        usage = result.usage
        context.n_input_tokens = usage.input_tokens
        context.n_output_tokens = usage.output_tokens
        context.n_cache_tokens = usage.cache_read_tokens
        context.metadata = {
            'requests': usage.requests,
            'tool_calls': usage.tool_calls,
        }


def _join_streams(stdout: str | None, stderr: str | None) -> str:
    """Combine stdout and stderr into one block, in that order."""
    parts = [stream for stream in (stdout, stderr) if stream]
    return '\n'.join(parts)

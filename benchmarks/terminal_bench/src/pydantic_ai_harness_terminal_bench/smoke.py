"""A keyless, scripted agent that exercises the Harbor + Docker + adapter seam.

The point is to prove the plumbing end-to-end in CI without any API keys: a
`FunctionModel` issues a couple of deterministic `bash` commands and then stops.
It does not "solve" anything by reasoning -- but the commands it runs do satisfy
the bundled `tasks/smoke-hello` task, so a green trial also proves that a write
from a host-side tool call actually lands in the container and persists.

Run it in CI (see `.github/workflows/benchmark-smoke.yml`):

    harbor run -p benchmarks/terminal_bench/tasks/smoke-hello \\
        -a pydantic_ai_harness_terminal_bench.smoke:ScriptedSmokeAgent
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness_terminal_bench.harbor_agent import PydanticAITerminalBenchAgent

SMOKE_MARKER_PATH = '/app/smoke_done.txt'
"""File the scripted run writes; the smoke-hello verifier checks it."""

SMOKE_TEXT = 'hello'
"""Contents written to the marker file."""


def _count_tool_returns(messages: list[ModelMessage]) -> int:
    """How many tool results the model has already seen.

    Drives the scripted state machine: the count is the step index, so the model
    is a pure function of history and needs no mutable state.
    """
    total = 0
    for message in messages:
        if isinstance(message, ModelRequest):
            total += sum(1 for part in message.parts if isinstance(part, ToolReturnPart))
    return total


async def _scripted_turn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Write a file, read it back to verify, then finish."""
    step = _count_tool_returns(messages)
    if step == 0:
        command = f'echo {SMOKE_TEXT} > {SMOKE_MARKER_PATH}'
        return ModelResponse(parts=[ToolCallPart('bash', {'command': command})])
    if step == 1:
        return ModelResponse(parts=[ToolCallPart('bash', {'command': f'cat {SMOKE_MARKER_PATH}'})])
    return ModelResponse(parts=[TextPart(f'Wrote {SMOKE_TEXT!r} to {SMOKE_MARKER_PATH} and verified it.')])


def scripted_model() -> FunctionModel:
    """The deterministic, keyless model that drives the smoke run."""
    return FunctionModel(_scripted_turn)


class ScriptedSmokeAgent(PydanticAITerminalBenchAgent):
    """Reference agent with its model replaced by the scripted smoke model.

    Everything else -- the adapter, the bash tool, `environment.exec`, context
    population -- is the real path, so a passing trial validates the seam.
    """

    def resolve_model(self) -> str | Model:
        return scripted_model()

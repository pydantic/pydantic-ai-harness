"""`TrimRetryHistory` -- compact repeated output-validation retry context."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryPromptPart
from pydantic_ai.tools import RunContext

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
    from pydantic_ai.models import ModelRequestContext


@dataclass
class TrimRetryHistory(AbstractCapability[AgentDepsT]):
    """Keep only the newest output-validation retry pair in a retry request.

    Output validation retries append the rejected response and validation feedback to the
    conversation. When validation fails more than once, earlier rejected response/feedback
    pairs add context without helping the model correct its latest answer. This capability
    presents the model handler with a request-local list containing the original task and the
    latest pair only, without changing the persisted run history.

    Tool retries are unaffected because their retry prompts name the tool that failed.
    """

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Trim repeated output-validation retry pairs only for the model call."""
        messages = request_context.messages
        first_pair = _first_output_retry_pair(messages, _output_tool_names(request_context))
        if first_pair is None or first_pair == len(messages) - 2:
            return await handler(request_context)

        compacted = [*messages[:first_pair], *messages[-2:]]
        return await handler(replace(request_context, messages=compacted))


def _first_output_retry_pair(messages: list[ModelMessage], output_tool_names: set[str]) -> int | None:
    """Find the first pair in a trailing sequence of output-validation retries."""
    if (
        len(messages) < 2
        or not isinstance(messages[-2], ModelResponse)
        or not _is_output_retry_request(messages[-1], output_tool_names)
    ):
        return None

    first_pair = len(messages) - 2
    while first_pair >= 2:
        previous_pair = first_pair - 2
        if not isinstance(messages[previous_pair], ModelResponse) or not _is_output_retry_request(
            messages[previous_pair + 1], output_tool_names
        ):
            break
        first_pair = previous_pair
    return first_pair


def _output_tool_names(request_context: ModelRequestContext) -> set[str]:
    """Return the output-tool names registered for this model request."""
    return {tool.name for tool in request_context.model_request_parameters.output_tools}


def _is_output_retry_request(message: ModelMessage, output_tool_names: set[str]) -> bool:
    """Return whether a request contains only output-validation retry feedback."""
    return (
        isinstance(message, ModelRequest)
        and bool(message.parts)
        and all(
            isinstance(part, RetryPromptPart) and (part.tool_name is None or part.tool_name in output_tool_names)
            for part in message.parts
        )
    )

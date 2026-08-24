"""Regression tests for streamed `SummarizingCompaction` summary requests."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.compaction import SummarizingCompaction


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('first response')]),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('second response')]),
    ]


def _outer_model() -> FunctionModel:
    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _summary_text(messages: list[ModelMessage]) -> str:
    summary_message = messages[0]
    assert isinstance(summary_message, ModelRequest)
    summary_part = summary_message.parts[0]
    assert isinstance(summary_part, SystemPromptPart)
    return summary_part.content


class TestSummarizingCompactionStreaming:
    @pytest.mark.anyio
    async def test_stream_only_model(self) -> None:
        calls = 0

        async def request_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            nonlocal calls
            calls += 1
            yield 'stream-only summary'

        summary_model = FunctionModel(stream_function=request_stream)
        agent = Agent(
            _outer_model(),
            capabilities=[
                SummarizingCompaction(
                    model=summary_model,
                    max_messages=3,
                    keep_messages=1,
                    preserve_first_user_message=False,
                    incremental=False,
                    stream_summary=True,
                )
            ],
        )

        result = await agent.run('continue', message_history=_history())

        assert result.output == 'done'
        assert calls == 1
        assert 'stream-only summary' in _summary_text(result.all_messages())

    @pytest.mark.anyio
    async def test_non_streaming_by_default(self) -> None:
        calls = 0

        def request(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart('ordinary summary')])

        summary_model = FunctionModel(request)
        agent = Agent(
            _outer_model(),
            capabilities=[
                SummarizingCompaction(
                    model=summary_model,
                    max_messages=3,
                    keep_messages=1,
                    preserve_first_user_message=False,
                    incremental=False,
                )
            ],
        )

        result = await agent.run('continue', message_history=_history())

        assert result.output == 'done'
        assert calls == 1
        assert 'ordinary summary' in _summary_text(result.all_messages())

    def test_stream_summary_is_keyword_only(self) -> None:
        parameter = inspect.signature(SummarizingCompaction).parameters['stream_summary']
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

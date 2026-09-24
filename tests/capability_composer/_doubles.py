"""Test doubles for `CapabilityComposer`.

Jev is stood in for by a `FunctionModel` that answers the composer's output tool and reports
`provider_details['confidence']` the way Pydantic AI's `TypeSafeModel` does.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass, field

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import AgentStreamEvent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, Tool
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.capability_composer import CapabilitiesComposedEvent, CapabilityComposer, ComposableCapability
from pydantic_ai_harness.subagents import ModelOption


@dataclass
class Notes(AbstractCapability[object]):
    """A catalog capability with one tool, named from its argument so catalog arguments are observable."""

    prefix: str = 'notes'

    def get_toolset(self) -> FunctionToolset[object]:
        def take_note() -> str:
            return 'noted'

        return FunctionToolset([Tool(take_note, name=f'{self.prefix}_take')])


@dataclass
class Clock(AbstractCapability[object]):
    """A second catalog capability."""

    def get_toolset(self) -> FunctionToolset[object]:
        def now() -> str:
            return 'noon'

        return FunctionToolset([Tool(now, name='clock_now')])


CATALOG = {
    'notes': ComposableCapability(description='Keep notes', capability=Notes, arguments={'prefix': 'memo'}),
    'clock': ComposableCapability(description='Tell the time', capability=Clock),
}


@dataclass
class Jev:
    """A picker model with a fixed answer, recording the prompts and output schemas it was sent."""

    model: str = 'strong'
    thinking: str = 'high'
    capabilities: Sequence[str] = ('notes',)
    confidence: dict[str, float] | None = None
    prompts: list[str] = field(default_factory=list[str])
    schemas: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [prompt] = [p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
        assert isinstance(prompt, str)
        self.prompts.append(prompt)
        [tool] = info.output_tools
        self.schemas.append(tool.parameters_json_schema)
        args = {'model': self.model, 'thinking': self.thinking, 'capabilities': list(self.capabilities)}
        confidence = self.confidence if self.confidence is not None else {'model': 0.9, 'thinking': 0.8}
        return ModelResponse(parts=[ToolCallPart(tool.name, args)], provider_details={'confidence': confidence})

    @property
    def model_(self) -> FunctionModel:
        return FunctionModel(self.respond, model_name='jev-test')


@dataclass
class Seen:
    """What a menu model was asked: the latest prompt, the tools offered, and the request details."""

    prompt: str
    tools: list[str]
    instructions: str | None
    output_tools: list[str]


def menu_model(name: str, seen: list[Seen]) -> FunctionModel:
    """A menu model that answers with its name, or fills the agent's output tool when it has one."""

    def record(messages: list[ModelMessage], info: AgentInfo) -> None:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        [prompt] = [p.content for p in request.parts if isinstance(p, UserPromptPart)]
        seen.append(
            Seen(
                prompt=str(prompt),
                tools=sorted(tool.name for tool in info.function_tools),
                instructions=info.instructions,
                output_tools=[tool.name for tool in info.output_tools],
            )
        )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        record(messages, info)
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 7})], model_name=name)
        return ModelResponse(parts=[TextPart(content=f'{name} answered')], model_name=name)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        record(messages, info)
        yield f'{name} answered'

    return FunctionModel(respond, stream_function=stream, model_name=name)


def main_model(calls: list[str]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('main')
        return ModelResponse(parts=[TextPart(content='main answered')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('main')
        yield 'main answered'

    return FunctionModel(respond, stream_function=stream)


def composer(jev: Jev, seen: list[Seen], **kwargs: object) -> CapabilityComposer[object]:
    return CapabilityComposer(
        models={
            'fast': ModelOption(menu_model('fast', seen), description='Quick answers'),
            'strong': ModelOption(menu_model('strong', seen), description='Hard problems'),
        },
        catalog=CATALOG,
        picker_model=jev.model_,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


@dataclass
class Events:
    """An event stream handler that keeps the composer's events."""

    composed: list[CapabilitiesComposedEvent] = field(default_factory=list[CapabilitiesComposedEvent])

    async def __call__(self, ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if isinstance(event, CapabilitiesComposedEvent):
                self.composed.append(event)


def recording_tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def compose_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    [span] = [s for s in exporter.get_finished_spans() if s.name == 'capability_composer compose']
    return span

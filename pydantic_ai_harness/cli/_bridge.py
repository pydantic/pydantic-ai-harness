"""`CliBridge`: renders the run's event stream to the terminal with Termflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, TextIO

from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.tools import AgentDepsT, RunContext

try:
    from termflow import Parser, Renderer  # pyright: ignore[reportMissingTypeStubs]
    from termflow.ansi import DIM_OFF, DIM_ON, RESET, fg_color, truncate_ansi  # pyright: ignore[reportMissingTypeStubs]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'termflow-md is required for CliBridge and needs Python 3.11 or newer. '
        'Install it with: pip install "pydantic-ai-harness[cli]"'
    ) from _import_error

from pydantic_ai_harness.cli._config import Config, Theme


class _Stream(Protocol):
    """A part being rendered as its deltas arrive."""

    def feed(self, text: str) -> None: ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover


class _DimStream:
    """Writes thinking text dimmed and as-is; it is not Markdown."""

    def __init__(self, output: TextIO) -> None:
        self._output = output
        self._tail = '\n'

    def feed(self, text: str) -> None:
        self._output.write(f'{DIM_ON}{text}{DIM_OFF}')
        self._output.flush()
        self._tail = text[-1:] or self._tail

    def close(self) -> None:
        if self._tail != '\n':
            self._output.write('\n')
            self._output.flush()


class _MarkdownStream:
    """Feeds streamed Markdown to Termflow one complete line at a time.

    Termflow parses whole lines, so deltas are buffered until a newline; `close` renders the
    tail and closes any open block.
    """

    def __init__(self, renderer: Renderer) -> None:
        self._parser = Parser()
        self._renderer = renderer
        self._pending = ''

    def feed(self, text: str) -> None:
        *lines, self._pending = (self._pending + text).split('\n')
        for line in lines:
            self._renderer.render_all(self._parser.parse_line(line))

    def close(self) -> None:
        if self._pending:
            self._renderer.render_all(self._parser.parse_line(self._pending))
        self._renderer.render_all(self._parser.finalize())


@dataclass(kw_only=True)
class CliBridge(AbstractCapability[AgentDepsT]):
    """Render the run's event stream to a terminal.

    Text parts stream through Termflow as Markdown; each tool call and its result print as one
    line bounded to the terminal width; thinking parts print dimmed when `config.show_thinking`
    is set. Wire it last in `capabilities=[...]` so it observes every other capability's events.
    Capability events are not rendered yet.
    """

    output: TextIO | None = None
    """Where rendered output goes. `sys.stdout` when `None`, resolved at the start of each run."""
    width: int | None = None
    """Terminal width in columns. Detected from the terminal when `None`, falling back to 80."""
    config: Config | None = None
    """The theme and render policy. Read from `Config.default_path()` at the start of each run when `None`."""

    _renderer: Renderer = field(init=False, repr=False, compare=False)
    _streams: dict[int, _Stream] = field(default_factory=dict[int, _Stream], init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        theme = Theme() if self.config is None else self.config.theme
        self._renderer = Renderer(
            output=self.output, width=self.width, style=theme.render_style(), highlighter=theme.highlighter()
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CliBridge[AgentDepsT]:
        """A fresh bridge per run: the config file is read now, and open text streams never leak between runs."""
        return replace(self, config=Config.load() if self.config is None else self.config)

    @on_event(PartStartEvent)
    async def _on_part_start(self, ctx: RunContext[AgentDepsT], event: PartStartEvent) -> None:
        stream: _Stream
        if isinstance(event.part, TextPart):
            stream = _MarkdownStream(self._renderer)
        elif isinstance(event.part, ThinkingPart) and self.config is not None and self.config.show_thinking:
            stream = _DimStream(self._renderer.output)
        else:
            return
        self._streams[event.index] = stream
        stream.feed(event.part.content)

    @on_event(PartDeltaEvent)
    async def _on_part_delta(self, ctx: RunContext[AgentDepsT], event: PartDeltaEvent) -> None:
        stream = self._streams.get(event.index)
        if (
            stream is not None
            and isinstance(event.delta, TextPartDelta | ThinkingPartDelta)
            and event.delta.content_delta
        ):
            stream.feed(event.delta.content_delta)

    @on_event(PartEndEvent)
    async def _on_part_end(self, ctx: RunContext[AgentDepsT], event: PartEndEvent) -> None:
        stream = self._streams.pop(event.index, None)
        if stream is not None:
            stream.close()

    @on_event(FunctionToolCallEvent)
    async def _on_tool_call(self, ctx: RunContext[AgentDepsT], event: FunctionToolCallEvent) -> None:
        self._line('>', event.part.tool_name, event.part.args_as_json_str())

    @on_event(FunctionToolResultEvent)
    async def _on_tool_result(self, ctx: RunContext[AgentDepsT], event: FunctionToolResultEvent) -> None:
        part = event.part
        if isinstance(part, RetryPromptPart):
            self._line('!', part.tool_name or 'retry', part.model_response())
        else:
            self._line('<', part.tool_name, part.model_response_str())

    def _line(self, marker: str, tool_name: str, detail: str) -> None:
        """One line per tool event: first line of `detail`, cut to the terminal width."""
        first, _, rest = detail.partition('\n')
        if rest:
            more = rest.count('\n') + 1
            first = f'{first} (+{more} lines)'
        head = f'{fg_color(self._renderer.style.symbol)}{marker} {tool_name}{RESET} '
        self._renderer.output.write(truncate_ansi(f'{head}{DIM_ON}{first}{DIM_OFF}', self._renderer.width) + '\n')
        self._renderer.output.flush()

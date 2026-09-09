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

from pydantic_ai_harness.cli._approve import Approver, CliDeps, DeclineAll
from pydantic_ai_harness.cli._config import Config, Theme
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileOperation, FilesSearchedEvent
from pydantic_ai_harness.shell import (
    ShellCommandEndEvent,
    ShellCommandRequestEvent,
    ShellCommandStartEvent,
    ShellOutputLineEvent,
)

NO_APPROVER = DeclineAll(reason='nobody can approve this; give the run `CliDeps` or the bridge an `approver`')

_FILE_VERBS: dict[FileOperation, str] = {'write': 'write', 'edit': 'edit', 'create_directory': 'create directory'}


def _diff_color(line: str) -> str | None:
    """The color a unified diff line renders in, or `None` for context lines."""
    if line.startswith(('+++', '---')):
        return None
    if line.startswith('+'):
        return 'green'
    if line.startswith('-'):
        return 'red'
    if line.startswith('@@'):
        return 'cyan'
    return None


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

    Shell commands render from `shell.*` events: the command as it starts, each output line, and
    an exit summary that replaces the generic result line. File changes render from `file_system.*`
    events: the proposed diff, and a match count that replaces a search's result line. A
    `ShellCommandRequestEvent` or `FileChangeRequestEvent` is put to the `approver` first and
    cancelled with the approver's reason when declined.
    """

    output: TextIO | None = None
    """Where rendered output goes. `sys.stdout` when `None`, resolved at the start of each run."""
    width: int | None = None
    """Terminal width in columns. Detected from the terminal when `None`, falling back to 80."""
    config: Config | None = None
    """The theme and render policy. Read from `Config.default_path()` at the start of each run when `None`."""
    approver: Approver | None = None
    """Who answers decision events. When `None`, the run's `CliDeps.approver`; without either, every request is declined."""

    _renderer: Renderer = field(init=False, repr=False, compare=False)
    _streams: dict[int, _Stream] = field(default_factory=dict[int, _Stream], init=False, repr=False, compare=False)
    _summarised: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Tool call IDs whose result a capability event already rendered."""

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
        elif part.tool_call_id not in self._summarised:
            self._line('<', part.tool_name, part.model_response_str())

    @on_event(ShellCommandRequestEvent)
    async def _on_shell_request(self, ctx: RunContext[AgentDepsT], event: ShellCommandRequestEvent) -> None:
        verdict = await self._approver(ctx)(event, description=f'run {event.command}')
        if not verdict.allowed:
            event.cancel(verdict.reason)

    @on_event(FileChangeRequestEvent)
    async def _on_file_change_request(self, ctx: RunContext[AgentDepsT], event: FileChangeRequestEvent) -> None:
        """Show the proposed diff, then ask; the user decides on what they can see."""
        for line in event.diff.splitlines():
            color = _diff_color(line)
            self._write(line if color is None else f'{fg_color(color)}{line}{RESET}')
        if event.truncated:
            self._write(f'{DIM_ON}(diff truncated){DIM_OFF}')
        verdict = await self._approver(ctx)(event, description=f'{_FILE_VERBS[event.operation]} {event.path}')
        if not verdict.allowed:
            event.cancel(verdict.reason)

    @on_event(FilesSearchedEvent)
    async def _on_files_searched(self, ctx: RunContext[AgentDepsT], event: FilesSearchedEvent) -> None:
        if event.tool_call_id is not None:
            self._summarised.add(event.tool_call_id)
        noun = 'match' if event.match_count == 1 else 'matches'
        note = ', truncated' if event.truncated else ''
        self._write(f'{DIM_ON}{event.match_count} {noun} for {event.pattern!r} in {event.path}{note}{DIM_OFF}')

    def _approver(self, ctx: RunContext[AgentDepsT]) -> Approver:
        if self.approver is not None:
            return self.approver
        return ctx.deps.approver if isinstance(ctx.deps, CliDeps) else NO_APPROVER

    @on_event(ShellCommandStartEvent)
    async def _on_shell_start(self, ctx: RunContext[AgentDepsT], event: ShellCommandStartEvent) -> None:
        self._write(f'{fg_color(self._renderer.style.symbol)}$ {event.command}{RESET}')

    @on_event(ShellOutputLineEvent)
    async def _on_shell_line(self, ctx: RunContext[AgentDepsT], event: ShellOutputLineEvent) -> None:
        self._write(event.line if event.stream == 'stdout' else f'{DIM_ON}{event.line}{DIM_OFF}')

    @on_event(ShellCommandEndEvent)
    async def _on_shell_end(self, ctx: RunContext[AgentDepsT], event: ShellCommandEndEvent) -> None:
        if event.tool_call_id is not None:
            self._summarised.add(event.tool_call_id)
        if event.timed_out:
            outcome = 'timed out'
        elif event.exit_code is None:
            outcome = 'stopped'
        else:
            outcome = f'exit {event.exit_code}'
        note = ', output truncated' if event.truncated else ''
        self._write(f'{DIM_ON}{outcome} ({event.duration_seconds:.1f}s{note}){DIM_OFF}')

    def _line(self, marker: str, tool_name: str, detail: str) -> None:
        """One line per tool event: first line of `detail`, cut to the terminal width."""
        first, _, rest = detail.partition('\n')
        if rest:
            more = rest.count('\n') + 1
            first = f'{first} (+{more} lines)'
        head = f'{fg_color(self._renderer.style.symbol)}{marker} {tool_name}{RESET} '
        self._write(truncate_ansi(f'{head}{DIM_ON}{first}{DIM_OFF}', self._renderer.width))

    def _write(self, line: str) -> None:
        self._renderer.output.write(line + '\n')
        self._renderer.output.flush()

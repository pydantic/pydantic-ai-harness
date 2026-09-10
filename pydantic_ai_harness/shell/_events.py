"""Events emitted by the shell capability.

Every command gets a `command_id` at start so a subscriber can correlate its
lines and its end with its start when several commands run at once. For a
background command the same id is the handle `check_command` and
`stop_command` take, so the model and the subscriber name the process alike.

Output in events is bounded: a line is cut at `MAX_EVENT_LINE_CHARS` and an
end event keeps the tail of each stream up to `max_output_chars`, with a
`truncated` flag. `command` and `cwd` are carried as is, and the number of
line events is not bounded; a host that persists or forwards events applies
its own budget.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import CapabilityEvent

SHELL_EVENTS = 'shell'

MAX_EVENT_LINE_CHARS = 256
"""Characters kept per `ShellOutputLineEvent.line` before it is cut."""

OutputStream = Literal['stdout', 'stderr']


@dataclass(kw_only=True)
class ShellCommandRequestEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='command_request', dispatch='immediate'):
    """A command is about to run; listeners may cancel or rewrite it first.

    A cancelled command returns `cancel_reason` to the model as the tool
    result instead of running. A cancel is final: `cancelled` is read-only,
    so a later listener cannot lift an earlier veto, which is what lets a
    cancel from any listener beat every rewrite. A rewritten command runs in
    place of the original and the model is told it was rewritten and why; the
    rewrite is subject to the same allow and deny policy as the original.
    Decisions go through `cancel()` and `rewrite()` only: assigning fields
    such as `command` directly has no effect on what runs.
    """

    command: str
    cwd: str
    timeout: float | None
    """Seconds the command may run, or `None` for a background command."""
    background: bool
    _cancelled: bool = field(default=False, init=False)
    _rewritten: str | None = field(default=None, init=False)
    cancel_reason: str | None = None
    rewrite_reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Whether any listener vetoed the command. Once set, it stays set."""
        return self._cancelled

    def cancel(self, reason: str | None = None) -> None:
        """Stop the command from running. A cancel is final; later listeners cannot lift it."""
        self._cancelled = True
        self.cancel_reason = reason

    def rewrite(self, command: str, *, reason: str) -> None:
        """Replace the command that will run; `reason` is all the model sees of it.

        A rewrite is final, like a cancel: a later listener's direct assignment
        to `command` steers nothing.
        """
        self.command = command
        self._rewritten = command
        self.rewrite_reason = reason


@dataclass(kw_only=True)
class ShellCommandStartEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='command_start', dispatch='immediate'):
    """A command process was spawned.

    Listeners run as the tool spawns the process, not when the stream consumer
    reaches the event: a raising listener must end the run from inside the
    tool, which kills the process group instead of leaving it running.
    """

    command_id: str
    command: str
    cwd: str
    timeout: float | None
    background: bool
    pid: int


@dataclass(kw_only=True)
class ShellOutputLineEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='output_line'):
    """A foreground command wrote one line to stdout or stderr.

    Background commands write to files the model reads through
    `check_command`, so they emit no line events.
    """

    command_id: str
    stream: OutputStream
    line: str
    truncated: bool


@dataclass(kw_only=True)
class ShellCommandEndEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='command_end'):
    """A command finished, timed out, or was stopped.

    A process killed by a signal reports a negative `exit_code` (`-15` for
    SIGTERM). A background command ends when `check_command` first sees it
    exited or when `stop_command` kills it. A run cancelled mid-command ends
    without this event.
    """

    command_id: str
    command: str
    background: bool
    exit_code: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str
    truncated: bool

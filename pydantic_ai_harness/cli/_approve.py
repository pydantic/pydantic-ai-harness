"""Who answers a capability's request to do something: the user at the terminal, or a policy.

`CliBridge` renders a decision event (a `ShellCommandRequestEvent`, later a file change) and asks
the run's `Approver` whether to let it proceed. The approver is a run-time dependency carried in
`CliDeps` so a policy engine can replace the terminal prompt without touching the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TextIO

from pydantic_ai.messages import CapabilityEvent
from termflow.ansi import DIM_OFF, DIM_ON, RESET, fg_color  # pyright: ignore[reportMissingTypeStubs]

YES = frozenset({'y', 'yes'})


class Answers(Protocol):
    """A source of one-line answers; `Lines.ask` is the terminal's."""

    async def ask(self) -> str | None: ...  # pragma: no cover


@dataclass(kw_only=True, frozen=True)
class Verdict:
    """An approver's answer. `reason` reaches the model when the request is declined."""

    allowed: bool
    reason: str | None = None


ALLOWED = Verdict(allowed=True)
DECLINED_BY_USER = Verdict(allowed=False, reason='declined by the user')


class Approver(Protocol):
    """Decide whether the action a decision event describes may proceed."""

    async def __call__(self, event: CapabilityEvent, *, description: str) -> Verdict:
        """`description` is the one-line summary of the action the bridge would show."""
        ...  # pragma: no cover


async def allow_all(event: CapabilityEvent, *, description: str) -> Verdict:
    """Yolo mode: every request proceeds without asking."""
    return ALLOWED


@dataclass(kw_only=True)
class DeclineAll:
    """Refuse every request with `reason`, for hosts that cannot ask anyone."""

    reason: str

    async def __call__(self, event: CapabilityEvent, *, description: str) -> Verdict:
        return Verdict(allowed=False, reason=self.reason)


@dataclass(kw_only=True)
class TerminalApprover:
    """Ask the user on the terminal; anything but `y` or `yes` declines."""

    answers: Answers
    output: TextIO
    color: str = 'yellow'

    async def __call__(self, event: CapabilityEvent, *, description: str) -> Verdict:
        self.output.write(f'{fg_color(self.color)}? {description}{RESET} {DIM_ON}[y/N]{DIM_OFF} ')
        self.output.flush()
        answer = await self.answers.ask()
        if answer is not None and answer.strip().lower() in YES:
            return ALLOWED
        return DECLINED_BY_USER


@dataclass(kw_only=True)
class CliDeps:
    """Run-time dependencies the CLI hands to each run for `CliBridge` to use."""

    approver: Approver

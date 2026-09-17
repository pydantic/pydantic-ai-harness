"""`/compact` and automatic compaction, wired to harness's `SummarizingCompaction`.

The summary itself is harness's job (`compact_now`, `SummarizingCompaction.with_focus`). This
module decides *when* to ask for one and tells the user what happened.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic_ai.models import Model
from pydantic_ai_harness.compaction import (
    SummarizingCompaction,
    compact_now,
    estimate_context_tokens,
    resolve_context_window,
)
from rich.console import Console

from . import theme
from ._session import Session
from .status import Status

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


def context_window(model: str, *, override: int | None) -> int | None:
    """The window auto-compaction measures against: the user's saved value, else the catalog's."""
    return override if override is not None else resolve_context_window(model)


@dataclass(kw_only=True)
class Compactor(Generic[DepsT, OutputT]):
    """Replace the retained history with a summary, on request or when the window fills up."""

    session: Session[DepsT, OutputT]
    status: Status
    console: Console
    fallback_model: Callable[[], Model | str | None]
    """The agent's own model, used when the session has not chosen one."""
    _noted: set[str] = field(default_factory=set[str], init=False)

    async def command(self, args: list[str]) -> str:
        """`/compact [focus...]`: summarise everything now."""
        if not self.session.messages:
            return 'Nothing to compact: the conversation is empty.'
        return await self._compact(focus=' '.join(args) or None)

    def prepare(self, model: str, *, window: int | None, compact_at: float) -> None:
        """Record the budget for the status row and warn once per model when it cannot be known."""
        self.status.context_window = window
        self.status.compact_at = compact_at
        if compact_at > 0 and window is None and model not in self._noted:
            self._noted.add(model)
            self.console.print(
                f'Context window for {model} is unknown, so automatic compaction is off. '
                'Set context_window in /model settings to enable it.',
                style=theme.MUTED,
                markup=False,
            )

    async def auto(self) -> None:
        """Compact before the turn when the last response crossed `compact_at`; say so first.

        A failed summary is reported and the turn goes ahead uncompacted, so a summariser
        outage does not lock the user out of the conversation.
        """
        window, tokens = self.status.context_window, self.status.context_tokens
        if window is None or tokens is None or not self.status.over_limit:
            return
        self.console.print(
            f'Context at {tokens / window:.0%} of {window:,} tokens; compacting before this turn.',
            style=theme.INFO,
        )
        try:
            self.console.print(await self._compact(focus=None), markup=False)
        except Exception as exc:  # noqa: BLE001 -- the turn still runs; the provider has the last word on size.
            self.console.print(f'Compaction failed: {type(exc).__name__}: {exc}', style=theme.ERROR, markup=False)
        self.console.print()

    async def _compact(self, *, focus: str | None) -> str:
        model = await self.session.resolved_model() or self.fallback_model()
        if model is None:
            raise ValueError('Choose a model first: /set model <Tab>')
        before = self.session.messages
        strategy: SummarizingCompaction[None] = SummarizingCompaction(
            max_messages=1, keep_messages=0, preserve_first_user_message=False
        )
        after = await compact_now(strategy, before, model=model, focus=focus)
        self.session.replace_messages(after)
        used = estimate_context_tokens(before)
        remaining = estimate_context_tokens(after)
        self.status.context_tokens = remaining
        return f'Compacted {len(before)} messages into a summary; about {max(used - remaining, 0):,} tokens saved.'

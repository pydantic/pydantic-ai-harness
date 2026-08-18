"""`TruncatingCompaction` -- retain a recent token-bounded history tail."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_ai._run_context import AgentDepsT

from pydantic_ai_harness.compaction._sliding_window_compaction import SlidingWindowCompaction


class TruncatingCompaction(SlidingWindowCompaction[AgentDepsT]):
    """Drop old messages while retaining a tool-pair-safe recent token tail.

    Unlike `SlidingWindowCompaction`, this strategy has no separate trigger:
    every request is compacted to `keep_tokens` when needed. This also makes it
    a deterministic zero-LLM final tier for `TieredCompaction`.
    """

    def __init__(
        self,
        keep_tokens: int,
        *,
        tokenizer: Callable[[str], int] | None = None,
        preserve_first_user_message: bool = True,
        receipts: bool = False,
    ) -> None:
        super().__init__(
            max_tokens=1,
            keep_tokens=keep_tokens,
            tokenizer=tokenizer,
            preserve_first_user_message=preserve_first_user_message,
            receipts=receipts,
        )

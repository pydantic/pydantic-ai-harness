"""Compaction capabilities: keep an agent's conversation history within the context window.

Each capability lives in its own module; shared utilities (token estimation, the
`CompactionStrategy` protocol, tool-pair-safe cutoffs, in-place clearing) live in `_shared`.
"""

from pydantic_ai_harness.compaction._clamp_oversized_messages import ClampOversizedMessages
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._context_usage import ContextUsage, ContextUsageMonitor
from pydantic_ai_harness.compaction._context_window import (
    DEFAULT_CONTEXT_WINDOW,
    resolve_context_window,
)
from pydantic_ai_harness.compaction._deduplicate_file_reads import DeduplicateFileReads
from pydantic_ai_harness.compaction._limit_warner import LimitWarner, WarningKind
from pydantic_ai_harness.compaction._manual import SupportsFocus, compact_now
from pydantic_ai_harness.compaction._shared import CompactionStrategy, estimate_token_count
from pydantic_ai_harness.compaction._sliding_window import SlidingWindow
from pydantic_ai_harness.compaction._summarizing_compaction import SummarizingCompaction
from pydantic_ai_harness.compaction._tiered_compaction import TieredCompaction

__all__ = [
    'DEFAULT_CONTEXT_WINDOW',
    'ClampOversizedMessages',
    'ClearToolResults',
    'CompactionStrategy',
    'ContextUsage',
    'ContextUsageMonitor',
    'DeduplicateFileReads',
    'LimitWarner',
    'SlidingWindow',
    'SummarizingCompaction',
    'SupportsFocus',
    'TieredCompaction',
    'WarningKind',
    'compact_now',
    'estimate_token_count',
    'resolve_context_window',
]

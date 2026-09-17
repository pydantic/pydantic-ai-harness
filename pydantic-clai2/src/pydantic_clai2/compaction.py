"""The built-in `compaction` plugin: harness's `SummarizingCompaction`, `/compact`, and a context gauge."""

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext
from pydantic_ai_harness.compaction import (
    ContextUsageEvent,
    ReportContextUsage,
    SummarizingCompaction,
    compact_now,
    estimate_token_count,
)

from .commands import Command
from .plugins import PluginHost


class CompactionSettings(BaseModel):
    """What `/plugins add compaction pydantic_clai2.compaction '{...}'` may override."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    max_fraction: float = Field(
        default=0.8,
        gt=0,
        le=1,
        allow_inf_nan=False,
        description='Summarise older messages once the history fills this fraction of the context window.',
    )
    keep_messages: int = Field(default=20, ge=0, description='Most recent messages kept verbatim after a summary.')
    context_window: int | None = Field(
        default=None,
        gt=0,
        description='Context window in tokens, when the catalog is wrong or silent. Unset resolves it from the model.',
    )


def activate(host: PluginHost[None]) -> None:
    """Bind the strategy per run, gauge usage before each request, and offer `/compact [focus]`.

    Typed for `None` deps because `compact_now` runs the strategy on a context with no deps;
    the strategy never reads them, so the plugin works with any agent.
    """
    config = host.settings(CompactionSettings)
    strategy: SummarizingCompaction[None] = SummarizingCompaction(
        max_fraction=config.max_fraction,
        keep_messages=config.keep_messages,
        context_window=config.context_window,
    )
    host.add(strategy)
    host.add(ReportContextUsage(context_window=config.context_window))

    @host.on(ContextUsageEvent)
    async def gauge(ctx: RunContext[None], event: ContextUsageEvent) -> None:
        host.status.context_alert = event.fraction > config.max_fraction

    async def compact(args: list[str]) -> str:
        before = host.conversation.messages
        if not before:
            return 'Nothing to compact: the conversation is empty.'
        model = await host.conversation.resolved_model()
        if model is None:
            raise ValueError('Choose a model first: /set model <Tab>')
        after = await compact_now(strategy, before, model=model, focus=' '.join(args) or None)
        if after is before:
            return f'Nothing to compact: the last {config.keep_messages} messages are always kept.'
        host.conversation.replace_messages(after)
        host.status.context_alert = False
        saved = estimate_token_count(before) - estimate_token_count(after)
        return f'Compacted {len(before)} messages down to {len(after)}; about {max(saved, 0):,} tokens saved.'

    host.commands.register(
        Command(
            name='compact',
            description='Summarise the conversation so far; add words to say what the summary must keep',
            handler=compact,
        )
    )

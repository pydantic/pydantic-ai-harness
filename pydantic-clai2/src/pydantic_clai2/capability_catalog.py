"""Opt-in harness capabilities, declared without importing their optional dependencies."""

from collections.abc import Sequence

from .config import PluginSettings
from .settings_store import SettingsStore

# Coder, AskUser, RepoContext and Slack already have shell-integrated built-in entries.
_FACTORIES = (
    ('advisor', 'advisor:Advisor'),
    ('aws_lambda', 'aws_lambda:AWSLambdaDurability'),
    ('background_tools', 'background_tools:BackgroundTools'),
    ('browser_use', 'browser_use:BrowserUse'),
    ('capability_creation', 'capability_creation:CapabilityCreation'),
    ('code_mode', 'code_mode:CodeMode'),
    ('clamp_oversized_messages', 'compaction:ClampOversizedMessages'),
    ('clear_tool_results', 'compaction:ClearToolResults'),
    ('deduplicate_file_reads', 'compaction:DeduplicateFileReads'),
    ('fallback_compaction', 'compaction:FallbackCompaction'),
    ('report_context_usage', 'compaction:ReportContextUsage'),
    ('sliding_window_compaction', 'compaction:SlidingWindowCompaction'),
    ('summarizing_compaction', 'compaction:SummarizingCompaction'),
    ('tiered_compaction', 'compaction:TieredCompaction'),
    ('warn_near_limits', 'compaction:WarnNearLimits'),
    ('conversation_search', 'conversation_search:ConversationSearch'),
    ('day_ai', 'day_ai:DayAI'),
    ('dynamic_workflow', 'dynamic_workflow:DynamicWorkflow'),
    ('exa_agent', 'exa:ExaAgent'),
    ('exa_search', 'exa:ExaSearch'),
    ('filesystem', 'filesystem:FileSystem'),
    ('github', 'github:GitHub'),
    ('google_workspace', 'google_workspace:GoogleWorkspace'),
    ('grain', 'grain:Grain'),
    ('input_guardrail', 'guardrails:InputGuardrail'),
    ('output_guardrail', 'guardrails:OutputGuardrail'),
    ('tool_guardrail', 'guardrails:ToolGuardrail'),
    ('linear', 'linear:Linear'),
    ('localstack', 'localstack:LocalStack'),
    ('managed_prompt', 'logfire:ManagedPrompt'),
    ('logfire_mcp', 'logfire_mcp:LogfireMCP'),
    ('macroscope', 'macroscope:Macroscope'),
    ('memory', 'memory:Memory'),
    ('modal_sandbox', 'modal_sandbox:ModalSandbox'),
    ('notion', 'notion:Notion'),
    ('ordinal', 'ordinal:Ordinal'),
    ('planning', 'planning:Planning'),
    ('playwright', 'playwright:PlaywrightBrowser'),
    ('posthog', 'posthog:PostHog'),
    ('prompt_injection_defender', 'prompt_injection_defender:PromptInjectionDefender'),
    ('pydantic_ai_docs', 'pydantic_ai_docs:PydanticAIDocs'),
    ('pylon', 'pylon:Pylon'),
    ('repair_tool_arguments', 'repair_tool_arguments:RepairToolArguments'),
    ('researcher', 'researcher:Researcher'),
    ('shell', 'shell:Shell'),
    ('skills', 'skills:Skills'),
    ('spend_limits', 'spend:SpendLimits'),
    ('stackone', 'stackone:StackOne'),
    ('step_persistence', 'step_persistence:StepPersistence'),
    ('subagents', 'subagents:SubAgents'),
    ('system_reminders', 'system_reminders:SystemReminders'),
    ('tool_output_limits', 'tool_output_limits:ToolOutputLimits'),
    ('trajectory_judge', 'trajectory_judge:TrajectoryJudge'),
    ('warn_on_cache_busts', 'warn_on_cache_busts:WarnOnCacheBusts'),
    ('you_search', 'youdotcom:YouSearch'),
    ('you_research', 'youdotcom:YouResearch'),
)

HARNESS_PLUGINS: tuple[PluginSettings, ...] = tuple(
    PluginSettings(id=name, factory=f'pydantic_ai_harness.{factory}', enabled=False) for name, factory in _FACTORIES
)
"""Disabled built-ins. The loader imports a capability only when the user enables it."""

_PROMOTED = {'slack': 'pydantic_ai_harness.slack:Slack'}
"""Former catalog rows that are now shell-integrated built-ins: the factory each id had in the catalog."""


def adopt_promoted(store: SettingsStore, builtin: Sequence[PluginSettings]) -> None:
    """Point a saved copy of a promoted catalog row at its built-in, keeping whether the user enabled it.

    Toggling a catalog row saved the whole declaration, and a saved declaration outranks the built-in, so
    without this the old raw capability would keep loading. A declaration with the user's own settings is theirs.
    """
    shipped = {plugin.id: plugin for plugin in builtin}
    for saved in store.plugins():
        factory = _PROMOTED.get(saved.id)
        if factory is None or saved.id not in shipped:
            continue
        if saved == PluginSettings(id=saved.id, factory=factory, enabled=saved.enabled):
            store.save_plugin(shipped[saved.id].model_copy(update={'enabled': saved.enabled}))

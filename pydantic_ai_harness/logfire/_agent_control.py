"""Back an agent's Agent Control configuration with one Logfire variable."""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import logfire
from logfire.variables import Variable
from logfire.variables.abstract import NoOpVariableProvider
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator
from pydantic_ai import AbstractToolset, RunContext, TemplateStr, ToolDefinition, WrapperToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestContext, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.logfire._managed_variable import (
    ManagedVariableCapability,
    _in_durable_context,  # pyright: ignore[reportPrivateUsage]
    _warn_durable_write_skipped,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from logfire.variables import ResolvedVariable
    from pydantic_ai.agent.abstract import AbstractAgent, AgentModelSettings
    from pydantic_ai.capabilities import AgentModel, ModelSelection
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.models import ModelSelectionContext
    from pydantic_ai.run import AgentRunResult

_AGENT_VARIABLE_PREFIX = 'agent__'

# 64 KiB of text is already roughly 16K tokens for typical English prose. It accommodates substantial
# instructions while preventing one managed entry from adding megabytes to every model request.
_MAX_MODEL_FACING_TEXT_LENGTH = 65_536

# Settings kept out of the published baseline. `AgentConfigSettings` allows extra keys deliberately --
# a provider-specific setting like `openai_reasoning_effort` has to survive -- and these two come
# through that same door. Pydantic AI forwards them to the provider request, which is exactly where
# authorization headers and signed bodies live, and a baseline is published to a Logfire variable
# every project member can read. They still apply to the run; they just don't describe the agent to
# anyone else. This is a denylist rather than an allowlist because the extras that are *not*
# credentials are the whole point of `extra='allow'`.
_SETTINGS_WITHHELD_FROM_BASELINE = frozenset({'extra_headers', 'extra_body'})

# Where a renamed tool records the code-side name it routes back to, on its own definition's
# `metadata`, so `call_tool` reads it off the very tool the model was handed. Recomputing the mapping
# from a fresh listing instead answers about whatever occupies that advertised name *now*: a dynamic
# toolset that put a different tool behind it in between supplies its code-side name, and the call
# then runs the tool the model chose while a name-based policy -- `ApprovalRequiredToolset` above all
# -- is handed the other one's name and authorizes that instead. The key is removed again on the way
# through, so nothing downstream sees it.
_ROUTES_TO_METADATA_KEY = '__agent_control_routes_to__'

_EntryT = TypeVar('_EntryT')

# Drop warnings already emitted in this process, keyed by the message itself. See `_warn_dropped`.
_warned_drops: set[str] = set()

# Destinations handed to a baseline publisher in this process. Marking before the thread starts
# prevents concurrent first requests from scheduling duplicate work; a failure is not retried by
# every later run.
_baseline_publish_attempted: set[tuple[logfire.Logfire, str]] = set()
_baseline_publish_lock = threading.Lock()


def _warn_dropped(message: str) -> None:
    """Surface a dropped managed value once per process.

    A drop means Logfire shows one thing and the agent does another, which has to be visible. But the
    config is resolved on every single run, so warning per drop would bury that signal under its own
    repetition. Deduplicating on the message rather than on the field costs nothing in clarity -- each
    message names its subject and the offending value -- and still lets a *different* unrecognized
    value surface later, which a per-field guard would swallow. A concurrent first run can at worst
    duplicate the warning, which is not worth a lock.
    """
    if message in _warned_drops:
        return
    _warned_drops.add(message)
    warnings.warn(message)


def _reset_baseline_publish_guard() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear baseline publishing process state. Intended for tests only."""
    with _baseline_publish_lock:
        _baseline_publish_attempted.clear()


def _spawn_baseline_publish(variable: Variable[Any], example: str) -> None:
    """Move the provider read and targeted write off the model request's thread."""
    threading.Thread(target=_publish_baseline, args=(variable, example), daemon=True).start()


def _publish_baseline(variable: Variable[Any], example: str) -> None:
    """Update only the `example` on the provider's current complete variable definition.

    Read-modify-write, because the provider offers nothing narrower: `update_variable` PUTs the whole
    definition to `/v1/variables/{name}/` and takes no revision or `If-Match` input, so an edit saved
    in the Logfire UI between the read and the write is lost. The window is one HTTP round trip, the
    publish runs at most once per process per variable, and it returns early when `example` already
    matches -- but the race is real and cannot be closed from this side. Narrowing it needs a partial
    write or a conditional one on the platform API: https://github.com/pydantic/pydantic-ai-harness/issues/565
    """
    provider = variable.logfire_instance.config.get_variable_provider()
    try:
        config = provider.get_variable_config(variable.name)
        if config is None:
            if isinstance(provider, NoOpVariableProvider):
                return
            raise LookupError(f'variable {variable.name!r} was not found')
        if config.example == example:
            return
        provider.update_variable(variable.name, config.model_copy(update={'example': example}))
    except Exception as exc:
        variable.logfire_instance.warn(
            'Failed to publish the code baseline for Logfire managed variable {variable_name}',
            variable_name=variable.name,
            _exc_info=True,
        )
        warnings.warn(f'Failed to publish the code baseline for Logfire managed variable {variable.name!r}: {exc}')


class AgentConfigSettings(BaseModel):
    """Canonical model settings managed as one section of an `AgentConfig`.

    Every field name matches a key in [`ModelSettings`][pydantic_ai.settings.ModelSettings], so the
    payload lowers without translation. Unset fields keep their code-defined values. Extra fields
    are allowed so canonical settings introduced by a newer Logfire UI can flow through older SDKs.
    A field whose *value* this SDK doesn't recognize -- an effort level or service tier a newer
    Pydantic AI accepts -- is dropped with a warning so that setting alone keeps its code-defined
    value, rather than failing the enclosing `AgentConfig`. The model itself is the sibling `model`
    field on `AgentConfig`.
    """

    model_config = ConfigDict(extra='allow')

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    parallel_tool_calls: bool | None = None
    timeout: float | None = None
    stop_sequences: list[str] | None = None
    thinking: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = None
    service_tier: Literal['auto', 'default', 'flex', 'priority'] | None = None
    provider_options: dict[str, dict[str, Any]] | None = None
    """Per-provider escape hatch.

    `provider_options[provider][key]` lowers to the flat `<provider>_<key>` model setting. Provider
    options are applied after canonical fields, so a provider-specific value wins on collision.
    """

    @model_validator(mode='before')
    @classmethod
    def _drop_unrecognized_values(cls, data: Any) -> Any:
        """Drop a versioned field whose value this SDK doesn't recognize, keeping the rest of the patch.

        Tolerating unknown *keys* is only half of forward compatibility: a newer Logfire UI (or a newer
        Pydantic AI adding an effort level or service tier) writes a value that the stored schema and
        the Logfire backend both accept, and an older SDK would then fail the whole `AgentConfig` over
        it and revert instructions, model, and every tool override to code along with it. Dropping the
        one field the SDK can't act on leaves the value it doesn't understand out of the lowered
        settings, so that setting -- and only that setting -- keeps its code-defined behavior.

        Unrecognized values are rejected, not passed through: the field's public type is the guarantee
        `_lower_settings` and its consumers hold, and forwarding an unknown effort level to the
        provider would trade a version-skew problem for a request-time one. Keys the SDK has never
        heard of are a different case and stay untouched, so `extra='allow'` still carries them.
        """
        if not isinstance(data, dict):
            return data
        values = cast(dict[str, Any], data)
        dropped: dict[str, Any] = {}
        for name, adapter in _VERSIONED_SETTINGS.items():
            if name in values:
                try:
                    adapter.validate_python(values[name])
                except ValidationError:
                    dropped[name] = values[name]
        if not dropped:
            return values
        for name, value in dropped.items():
            _warn_dropped(
                f'Managed agent config sets {name!r} to {value!r}, which this version of the SDK does not '
                f'recognize; ignoring that setting and keeping the rest of the managed config.'
            )
        return {name: value for name, value in values.items() if name not in dropped}

    @model_validator(mode='before')
    @classmethod
    def _drop_malformed_values(cls, data: Any) -> Any:
        """Drop malformed setting values independently so valid siblings still apply."""
        if not isinstance(data, dict):
            return data
        values = cast(dict[str, Any], data)
        kept: dict[str, Any] = {}
        for name, value in values.items():
            field_info = cls.model_fields.get(name)
            if field_info is None:
                kept[name] = value
                continue
            if _enumerates_values(field_info.annotation):
                # The version-skew validator owns enumerated fields so its warning distinguishes a
                # newer value from structurally malformed input.
                kept[name] = value
                continue
            try:
                TypeAdapter(field_info.annotation).validate_python(value)
            except ValidationError:
                _warn_dropped(
                    f'Managed agent config setting {name!r} has invalid value {value!r}; ignoring that setting '
                    'and keeping the rest of the managed config.'
                )
            else:
                kept[name] = value
        return kept


def _enumerates_values(annotation: object) -> bool:
    """Whether an annotation spells out the values it accepts, i.e. names a `Literal` anywhere."""
    return get_origin(annotation) is Literal or any(_enumerates_values(arg) for arg in get_args(annotation))


_VERSIONED_SETTINGS: dict[str, TypeAdapter[Any]] = {
    name: TypeAdapter(field_info.annotation)
    for name, field_info in AgentConfigSettings.model_fields.items()
    if _enumerates_values(field_info.annotation)
}
"""Per-field validators for the settings whose accepted values grow from release to release.

A `Literal` in a field's annotation is exactly the signal that the field enumerates what *this*
release knows about, so the set of version-skew-prone fields (`thinking`, `service_tier`) and the
values each one accepts are both read off the annotations rather than restated here, where the two
would drift apart. Fields without a `Literal` -- `provider_options` above all, the deliberate
free-form escape hatch -- are left alone: a wrong type there is malformed rather than merely newer,
and the stored schema already rejects it at write time.
"""


def _lower_settings(value: AgentConfigSettings) -> ModelSettings:
    """Lower an `AgentConfigSettings` value to Pydantic AI's flat `ModelSettings` shape.

    Only explicitly set, non-`None` canonical fields are emitted. `provider_options` entries are
    flattened to `<provider>_<key>` afterwards, giving them precedence over canonical fields that
    lower to the same key.
    """
    settings: dict[str, Any] = value.model_dump(exclude_none=True, exclude={'provider_options'})
    if value.provider_options:
        for provider, options in value.provider_options.items():
            for key, option in options.items():
                settings[f'{provider}_{key}'] = option
    return cast(ModelSettings, settings)


NonEmptyStr: TypeAlias = Annotated[str, Field(min_length=1)]
"""A managed string that has to say something.

`''` is never a meaningful managed value -- not "no model", not "no instructions", just a value someone
left half-filled -- and `None` already means "leave this to code". Rejecting it keeps the two apart at
every level: the stored JSON schema won't accept the write, and a value that reaches an older SDK
anyway degrades that one field instead of the whole config.
"""

InstructionText: TypeAlias = Annotated[str, Field(min_length=1, max_length=_MAX_MODEL_FACING_TEXT_LENGTH)]
"""A non-empty instruction block bounded before it can become recurring model input."""


class InstructionBlock(BaseModel):
    """One entry in `AgentConfig.instructions`: a block to add, or a patch on a block the agent assembles.

    Instructions are the one section that composes rather than replaces, so an entry has to say *which*
    text it means. An entry with no `id` adds a block; an entry with an `id` addresses the instruction
    blocks Pydantic AI has already assembled under that key.

    | Entry | Effect |
    |---|---|
    | `'text'` (a bare string in the list) | Adds a block. Shorthand for `InstructionBlock(instructions='text')`. |
    | `InstructionBlock(instructions='text')` | Adds a block. |
    | `InstructionBlock(id=key, instructions='text')` | Replaces the text of every block keyed `key`. |
    | `InstructionBlock(id=key)` | Drops every block keyed `key`. |

    An `id` nothing matches is inert, exactly like an unknown tool name: one config is applied across
    deployments that need not all install the same toolsets. Two entries naming the same `id` keep the
    first, with a warning.
    """

    id: str | None = None
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] to address, or `None` to add a block.

    Pydantic AI keys each block it can name, and those keys are what this field takes: `'agent'` for the
    agent's own literal instructions, `'toolset:<id>'` and `'capability:<id>'` for everything a toolset
    or capability contributes, and `'agent:<declared id>'` / `'capability:<id>:<declared id>'` for a
    single declared block. Blocks Pydantic AI cannot key -- a callable passed to `Agent(instructions=...)`,
    a toolset with no `id` of its own -- cannot be addressed at all.
    """
    instructions: InstructionText | None = None
    """The block's text, or `None` to drop the addressed block.

    `None` is how a block is disabled, which is why `''` is rejected rather than taken as a quiet way to
    blank one: an entry meaning "send nothing here" and an entry someone left half-filled should not look
    identical. An entry with neither `id` nor `instructions` says nothing and is dropped with a warning.
    """
    dynamic: bool | None = None
    """Whether the addressed block is recomputed per request. Informational; ignored when a value is applied.

    Written into a variable's `example` by the baseline snapshot, where it earns its keep: it is how the
    Logfire UI can warn that replacing a computed block -- today's date, the signed-in user -- pins
    whatever it happened to evaluate to when the snapshot was taken.
    """


class ToolDefinitionOverride(BaseModel):
    """A patch over a tool's LLM-facing definition.

    Overrides change only what the model is shown. Schema structure, validation, and execution
    remain code-defined. `name` looks the tool up by its original code-side name. An entry that
    doesn't validate is dropped from `AgentConfig.tool_definitions` with a warning, leaving its
    siblings in place; two entries naming the same tool keep the first, with a warning.
    """

    name: str = Field(min_length=1)
    """The tool's original code-side name, which is what this entry patches.

    A tool no toolset advertises is inert rather than an error, for the same reason an unmatched
    instruction `id` is.
    """
    new_name: str | None = Field(default=None, min_length=1)
    """Replacement name shown to the model; `None` keeps the original.

    A renamed call still routes back to the original code implementation.
    """
    description: str | None = None
    """Replacement description shown to the model; `None` keeps the code-defined description."""
    parameter_descriptions: dict[str, str] | None = None
    """Replacement description text for named top-level parameters.

    Names, types, and requiredness stay exactly as defined in code, so argument validation is
    unaffected. Unknown parameter names are ignored.
    """


def _entry_errors(error: ValidationError, *, whole: str) -> str:
    """Render a rejected entry's failure -- offending field, value, and reason -- for one warning.

    `whole` names the entry itself, for a failure that isn't about any one field (something that isn't an
    object at all): the two sections share this renderer, so neither may be labelled with the other's noun.
    """
    return '; '.join(
        f'{".".join(str(part) for part in details["loc"]) or whole}={details["input"]!r} ({details["msg"]})'
        for details in error.errors()
    )


def _first_by_key(entries: list[tuple[str, _EntryT]], subject: str) -> dict[str, _EntryT]:
    """Index entries by key, keeping the first of any duplicates with a warning.

    Both list sections address things by key, so both can be written with the same key twice -- by a
    hand-edited value, or by a UI bug. Keeping the first matches how a colliding rename is resolved in
    `_ToolDefinitionOverridesToolset`: the run stays predictable and the ignored entry is named, rather
    than the last writer silently winning depending on how the JSON happened to be ordered.
    """
    indexed: dict[str, _EntryT] = {}
    for key, entry in entries:
        if key in indexed:
            _warn_dropped(
                f'Managed agent config names {subject} {key!r} more than once; keeping the first entry '
                f'and ignoring the rest.'
            )
            continue
        indexed[key] = entry
    return indexed


def _instruction_blocks(config: AgentConfig) -> list[InstructionBlock]:
    """The `instructions` section as blocks, whichever of its two shapes was written."""
    instructions = config.instructions
    if instructions is None:
        return []
    if isinstance(instructions, str):
        return [InstructionBlock(instructions=instructions)]
    return [InstructionBlock(instructions=entry) if isinstance(entry, str) else entry for entry in instructions]


def _added_instructions(config: AgentConfig) -> str | None:
    """The text of every entry that adds a block, joined in list order.

    Added blocks are contributed through `get_instructions` as one string rather than injected into the
    assembled parts, so they land exactly where a capability's instructions have always landed -- after
    the agent's own static text, before dynamic toolset text -- and the list form changes nothing about
    ordering or prompt-cache boundaries for a config that only adds text.
    """
    # Validation already rejects an entry with neither an `id` nor text, so the `is not None` here is
    # narrowing `instructions` for the join rather than a case that can turn up.
    added = [
        block.instructions
        for block in _instruction_blocks(config)
        if block.id is None and block.instructions is not None
    ]
    return '\n\n'.join(added) or None


def _instruction_key(part: InstructionPart) -> str | None:
    """The string key a managed config addresses this part by, or `None` if nothing addresses it.

    Pydantic AI issues a structured [`InstructionId`][pydantic_ai.messages.InstructionId]; a managed
    config arrives as JSON and names parts by the string that id renders to. Bridging the two once
    here keeps every caller comparing keys of the same kind.
    """
    return str(part.id) if part.id is not None else None


def _instruction_overrides(config: AgentConfig) -> dict[str, str | None]:
    """Replacement text per addressed [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id], `None` to drop it."""
    return _first_by_key(
        [(block.id, block.instructions) for block in _instruction_blocks(config) if block.id is not None],
        'instruction id',
    )


def _tool_overrides(config: AgentConfig) -> dict[str, ToolDefinitionOverride]:
    """Tool definition overrides indexed by the code-side tool name each one patches."""
    return _first_by_key([(override.name, override) for override in config.tool_definitions or []], 'tool')


class AgentConfig(BaseModel):
    """The schema contract shared with the Logfire Agent Control UI.

    Every managed value is a patch on the code-defined agent. A key present in the value is managed
    from Logfire; an absent key keeps code-defined behavior. Removing a key in Logfire is therefore
    a deliberate revert to code.

    Nothing the SDK fails to understand costs more than the part that contains it. Extra keys retain
    Pydantic's default `ignore` behavior, and a *value* it cannot make sense of -- an effort level or
    service tier a newer Pydantic AI accepts, a tool override that doesn't validate -- drops only its
    own setting or its own override, with a warning. Both rules exist for the same reason: a value
    that fails validation falls back through Logfire's resolution to the code-defined agent *in its
    entirety*, so without them one unfamiliar key or enum value would silently un-manage the
    instructions, the model, and every tool override alongside it.
    """

    model_config = ConfigDict(protected_namespaces=())

    instructions: InstructionText | list[InstructionText | InstructionBlock] | None = None
    """Instruction blocks to add to -- or swap out of -- the ones the agent assembles in code.

    A capability contributes instructions, it cannot take them over: Pydantic AI appends every
    contribution to the agent's own. Blocks with no `id` are therefore *added*, which is why a bare
    string means one added block and text that also lives in `Agent(instructions=...)` reaches the model
    twice. The code-side home for a managed base prompt is `AgentControl.instructions` (or
    `AgentControl.default`), which this value supersedes rather than adds to; see
    [`AgentControl`][pydantic_ai_harness.logfire.AgentControl].

    Blocks *with* an `id` reach what no capability owns -- the agent's own literal, a toolset's, an MCP
    server's -- by addressing the assembled
    [`instruction_parts`][pydantic_ai.models.ModelRequestParameters.instruction_parts] directly; see
    [`InstructionBlock`][pydantic_ai_harness.logfire.InstructionBlock].

    A bare string is exactly `[InstructionBlock(instructions='text')]` and is kept as written rather
    than rewritten into the list form, so a published value stays the shape its author chose and
    successive versions stay readable as a diff. An entry that doesn't validate is dropped with a
    warning, leaving its siblings and the rest of the config alone.

    `{{...}}` runtime placeholders are rendered against `deps` only when `AgentControl.render_template`
    is set.
    """
    model: NonEmptyStr | None = None
    """A Pydantic AI model string such as `'openai:gpt-5'`; `None` keeps the code model.

    Non-empty for a blunt reason: `''` is not "no model", it is a model named `''`, and Pydantic AI
    rejects it with `Unknown model:` on every request the agent makes. Publishing one would take the
    agent down, and the resolution fallback cannot catch it because the config itself is perfectly valid.
    """
    settings: AgentConfigSettings | None = None
    """Canonical model settings patch; see `AgentConfigSettings`."""
    tool_definitions: list[ToolDefinitionOverride] | None = None
    """LLM-facing overlays, each naming the tool it patches; see `ToolDefinitionOverride`."""

    @field_validator('instructions', mode='before')
    @classmethod
    def _drop_invalid_instructions(cls, data: Any) -> Any:
        """Drop an instruction entry that doesn't validate, keeping its siblings and the rest of the config.

        An entry is the natural unit of degradation here, the same way a tool override is: each one adds
        or addresses exactly one block, so an entry this SDK can't make sense of -- an empty string where
        text was meant, an entry that says neither what nor where, something that is neither a string nor
        an object -- can be left out while every other block still applies. Without this, one bad entry
        would fail the whole `AgentConfig` and revert the model, the settings, and every tool override to
        code alongside it.

        A bare string is left alone for the field itself to validate: it is one block by definition, so
        there is no sibling to save by rescuing it, and preserving the shape keeps a published value
        looking the way its author wrote it. Entries in a list are returned already validated so the
        field doesn't validate them a second time.
        """
        if isinstance(data, str):
            if 0 < len(data) <= _MAX_MODEL_FACING_TEXT_LENGTH:
                return data
            if len(data) > _MAX_MODEL_FACING_TEXT_LENGTH:
                _warn_dropped(
                    f'Managed instructions section contains {len(data)} characters, exceeding the '
                    f'{_MAX_MODEL_FACING_TEXT_LENGTH}-character limit; ignoring that section and keeping the rest '
                    'of the managed config.'
                )
                return None
            _warn_dropped(
                "Managed instructions section is invalid -- instructions=''; ignoring that section and keeping "
                'the rest of the managed config.'
            )
            return None
        if not isinstance(data, list):
            if data is not None:
                _warn_dropped(
                    f'Managed instructions section has invalid container {data!r}; ignoring that section and '
                    'keeping the rest of the managed config.'
                )
                return None
            return data
        blocks: list[InstructionBlock] = []
        # The bound is on the text this section adds to every model request, so it has to be the total
        # across entries, not just each one: a section written as one string is capped, and the same
        # text written as ten entries has to be capped too or the limit means nothing.
        remaining = _MAX_MODEL_FACING_TEXT_LENGTH
        for entry in cast(list[Any], data):
            text: object | None = entry if isinstance(entry, str) else None
            if isinstance(entry, dict):
                # Runtime narrowing cannot recover a dictionary's generic parameters from untyped JSON.
                typed_entry = cast(dict[str, object], entry)
                text = typed_entry.get('instructions')
            if isinstance(text, str) and len(text) > remaining:
                _warn_dropped(
                    f'Managed instruction entry contains {len(text)} characters, which does not fit in the '
                    f'{remaining} remaining of the {_MAX_MODEL_FACING_TEXT_LENGTH}-character limit across all '
                    'entries; ignoring that entry and keeping the rest of the managed config.'
                )
                continue
            if isinstance(text, str):
                remaining -= len(text)
            try:
                block = InstructionBlock.model_validate({'instructions': entry} if isinstance(entry, str) else entry)
            except ValidationError as error:
                _warn_dropped(
                    f'Managed instruction entry {entry!r} is invalid -- {_entry_errors(error, whole="entry")}; '
                    f'ignoring that entry and keeping the rest of the managed config.'
                )
                continue
            if block.id is None and block.instructions is None:
                _warn_dropped(
                    f'Managed instruction entry {entry!r} has neither an `id` to address nor text to add; '
                    f'ignoring that entry and keeping the rest of the managed config.'
                )
                continue
            blocks.append(block)
        return blocks

    @field_validator('tool_definitions', mode='before')
    @classmethod
    def _drop_invalid_overrides(cls, data: Any) -> Any:
        """Drop an override entry that doesn't validate, keeping its siblings and the rest of the config.

        A tool override is the natural unit here: each entry patches exactly one tool, so an entry the
        SDK can't validate -- a missing or empty `name`, a field carrying a shape it doesn't know,
        something that isn't an object at all -- can be left out while every other tool keeps its managed
        definition. Without this, one bad entry would fail the whole `AgentConfig` and revert the
        agent's instructions, model, and settings to code as well. (Merely *unknown* keys inside an
        entry are ignored by `ToolDefinitionOverride` itself and cost nothing.)

        Entries are returned already validated so the field doesn't validate them a second time.
        """
        if not isinstance(data, list):
            if data is not None:
                _warn_dropped(
                    f'Managed tool definitions section has invalid container {data!r}; ignoring that section and '
                    'keeping the rest of the managed config.'
                )
                return None
            return data
        overrides: list[ToolDefinitionOverride] = []
        for entry in cast(list[Any], data):
            try:
                overrides.append(ToolDefinitionOverride.model_validate(entry))
            except ValidationError as error:
                _warn_dropped(
                    f'Managed tool definition override {entry!r} is invalid -- {_entry_errors(error, whole="override")}; '
                    f'ignoring that override and keeping the rest of the managed config.'
                )
        return overrides

    @field_validator('settings', mode='before')
    @classmethod
    def _drop_invalid_settings_container(cls, data: Any) -> Any:
        if isinstance(data, AgentConfigSettings):
            return data
        if data is None or isinstance(data, dict):
            return None if data is None else cast(dict[str, Any], data)
        _warn_dropped(
            f'Managed settings section has invalid container {data!r}; ignoring that section and keeping the rest '
            'of the managed config.'
        )
        return None


AGENT_CONFIG_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'instructions': {
            'description': (
                'Instruction blocks added to the ones the agent assembles in code, not a replacement for '
                'them. A bare string is one added block. An entry with an `id` swaps out the block the '
                'agent already sends under that key instead of adding one.'
            ),
            'anyOf': [
                {'type': 'string', 'minLength': 1, 'maxLength': _MAX_MODEL_FACING_TEXT_LENGTH},
                {
                    'type': 'array',
                    'items': {
                        'anyOf': [
                            {'type': 'string', 'minLength': 1, 'maxLength': _MAX_MODEL_FACING_TEXT_LENGTH},
                            {
                                'type': 'object',
                                'properties': {
                                    'id': {
                                        'type': 'string',
                                        'minLength': 1,
                                        'description': (
                                            "The instruction block to address: 'agent', 'toolset:<id>', "
                                            "'capability:<id>', or one of those plus ':<declared id>'. "
                                            'Omit to add a block instead. A block the agent recomputes '
                                            'per request cannot be addressed: replacing it would pin one '
                                            'rendering and dropping it would remove the computation, so '
                                            'either is ignored.'
                                        ),
                                    },
                                    'instructions': {
                                        'anyOf': [
                                            {
                                                'type': 'string',
                                                'minLength': 1,
                                                'maxLength': _MAX_MODEL_FACING_TEXT_LENGTH,
                                            },
                                            {'type': 'null'},
                                        ],
                                        'description': 'The text to send, or null to drop the addressed block.',
                                    },
                                    'dynamic': {
                                        'type': 'boolean',
                                        'description': (
                                            'Whether the block is recomputed per request. Set on the '
                                            'code-side baseline and ignored on input -- but a block '
                                            'marked true is not addressable, so an editor should offer '
                                            'no override for it.'
                                        ),
                                    },
                                },
                            },
                        ]
                    },
                },
            ],
        },
        'model': {
            'type': 'string',
            'minLength': 1,
            'description': "A Pydantic AI model string, such as 'openai:gpt-5'.",
        },
        'settings': {
            'type': 'object',
            'description': (
                'Model settings patch. Keys match `pydantic_ai.settings.ModelSettings`; '
                'a key this schema does not name is still allowed.'
            ),
            'properties': {
                'max_tokens': {'type': 'integer'},
                'temperature': {'type': 'number'},
                'top_p': {'type': 'number'},
                'top_k': {'type': 'integer'},
                'seed': {'type': 'integer'},
                'presence_penalty': {'type': 'number'},
                'frequency_penalty': {'type': 'number'},
                'parallel_tool_calls': {'type': 'boolean'},
                'timeout': {'type': 'number'},
                'stop_sequences': {'type': 'array', 'items': {'type': 'string'}},
                'thinking': {
                    'anyOf': [{'type': 'boolean'}, {'type': 'string'}],
                    'description': "Enabled/disabled, or an effort level: 'minimal', 'low', 'medium', 'high', 'xhigh'.",
                },
                'service_tier': {
                    'type': 'string',
                    'description': "Provider service tier: 'auto', 'default', 'flex', or 'priority'.",
                },
                'provider_options': {
                    'type': 'object',
                    'description': (
                        'Per-provider settings: `provider_options.<provider>.<key>` lowers to the '
                        '`<provider>_<key>` model setting.'
                    ),
                    'additionalProperties': {'type': 'object'},
                },
            },
        },
        'tool_definitions': {
            'type': 'array',
            'description': (
                'LLM-facing overlays, each naming the tool it patches by its code-side name. Parameter '
                'names, types, requiredness, validation, and implementation stay code-defined.'
            ),
            'items': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'name': {
                        'type': 'string',
                        'minLength': 1,
                        'description': "The tool's code-side name, which is what this entry patches.",
                    },
                    'new_name': {
                        'type': 'string',
                        'minLength': 1,
                        'description': 'Name shown to the model; a call to it routes back to the original tool.',
                    },
                    'description': {'type': 'string'},
                    'parameter_descriptions': {
                        'type': 'object',
                        'description': 'Replacement description text per top-level parameter name.',
                        'additionalProperties': {'type': 'string'},
                    },
                },
            },
        },
    },
}
"""The stored JSON schema for an `agent__<name>` variable, shared with the Logfire Agent Control UI.

The Logfire UI holds a copy of this in `app/project/managed-agents/agent-config.ts`, and whichever
side creates the variable first is the one whose schema is persisted. The schema is not cosmetic: the
Logfire backend validates every new version of the value against it, so anything this schema rejects
cannot be written at all.

It is maintained by hand rather than taken from `AgentConfig.model_json_schema()` because the two
artifacts answer different questions. Pydantic's output describes *this* release's model on *this*
Pydantic version -- `$defs`, `anyOf [T, null]` wrappers, `default: null`, and `title` noise -- while
the stored schema is a long-lived contract between a Logfire project and every SDK version that will
ever write to it.

That is also why it is permissive at every level: `AgentConfig` ignores extra keys precisely so a key
written by a newer UI does not fail validation and revert the whole config to code, and an
`additionalProperties: false` anywhere in the stored schema would defeat that by rejecting the key at
write time instead. For the same reason the fields whose accepted values grow over releases
(`thinking`, `service_tier`) are typed rather than enumerated, and the known `settings` keys are named
for the editor's benefit while unnamed ones stay writable.

The constraints it does keep are structural rather than versioned, and each one closes a hole a
permissive schema would otherwise leave open. Every `minLength: 1` says the same thing as
[`NonEmptyStr`][pydantic_ai_harness.logfire._agent_control.NonEmptyStr]: `''` is a half-filled field,
never a value, and `model: ''` in particular takes an agent down with `Unknown model:` on every
request. `tool_definitions` items require a `name` because an overlay that names no tool cannot be
applied to anything, so accepting it would only let the UI save a row that silently does nothing.
"""


OverridesProvider: TypeAlias = Callable[[], Mapping[str, ToolDefinitionOverride]]
ToolsObserver: TypeAlias = Callable[[list[ToolDefinition]], None]


def _with_parameter_descriptions(
    parameters_json_schema: dict[str, Any], parameter_descriptions: dict[str, str]
) -> dict[str, Any]:
    """Patch top-level parameter descriptions while preserving all schema structure."""
    if not isinstance(parameters_json_schema.get('properties'), dict):
        return parameters_json_schema
    properties: dict[str, Any] = parameters_json_schema['properties']
    new_properties: dict[str, Any] = {}
    changed = False
    for name, schema in properties.items():
        if name in parameter_descriptions and isinstance(schema, dict):
            param_schema: dict[str, Any] = properties[name]
            new_properties[name] = {**param_schema, 'description': parameter_descriptions[name]}
            changed = True
        else:
            new_properties[name] = schema
    if not changed:
        return parameters_json_schema
    return {**parameters_json_schema, 'properties': new_properties}


def _apply_override(tool_def: ToolDefinition, override: ToolDefinitionOverride) -> ToolDefinition:
    """Apply the LLM-facing parts of an override, returning the original definition for a no-op."""
    changes: dict[str, Any] = {}
    if override.new_name is not None and override.new_name != tool_def.name:
        changes['name'] = override.new_name
    if override.description is not None and override.description != tool_def.description:
        changes['description'] = override.description
    if override.parameter_descriptions:
        schema = _with_parameter_descriptions(tool_def.parameters_json_schema, override.parameter_descriptions)
        if schema is not tool_def.parameters_json_schema:
            changes['parameters_json_schema'] = schema
    # `replace` preserves concrete `ToolDefinition` subclasses and fields added by the framework.
    return replace(tool_def, **changes) if changes else tool_def


@dataclass
class _ToolDefinitionOverridesToolset(WrapperToolset[AgentDepsT]):
    """Overlay advertised definitions while routing calls to their code-side tool names."""

    get_overrides: OverridesProvider = field(repr=False, compare=False)
    observe_code_tools: ToolsObserver = field(repr=False, compare=False)

    def _effective_tools(
        self, tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> dict[str, tuple[str, ToolsetTool[AgentDepsT]]]:
        """Build the advertised-name to (code-side name, tool) mapping.

        A renamed tool carries the name it routes back to in its definition's metadata, so `call_tool`
        can read it off the very tool the model was handed. A colliding rename is dropped while its
        other patches remain, preserving every tool under a callable name.
        """
        overrides = self.get_overrides()
        result: dict[str, tuple[str, ToolsetTool[AgentDepsT]]] = {}
        for original_name, tool in tools.items():
            override = overrides.get(original_name)
            new_tool_def = tool.tool_def if override is None else _apply_override(tool.tool_def, override)
            new_name = new_tool_def.name
            if new_name != original_name and (new_name in tools or new_name in result):
                warnings.warn(
                    f'Managed tool definition override renames {original_name!r} to {new_name!r}, '
                    f'which is already advertised by another tool; keeping the original name {original_name!r}.',
                    stacklevel=2,
                )
                new_tool_def = replace(new_tool_def, name=original_name)
                new_name = original_name
            if new_name != original_name:
                new_tool_def = replace(
                    new_tool_def, metadata={**(new_tool_def.metadata or {}), _ROUTES_TO_METADATA_KEY: original_name}
                )
            result[new_name] = (
                original_name,
                tool if new_tool_def is tool.tool_def else replace(tool, toolset=self, tool_def=new_tool_def),
            )
        return result

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return tools with managed definitions and collision-safe advertised names."""
        tools = await super().get_tools(ctx)
        self.observe_code_tools([tool.tool_def for tool in tools.values()])
        if not self.get_overrides():
            return tools
        return {name: tool for name, (_, tool) in self._effective_tools(tools).items()}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Translate a renamed model call back to its original implementation and context name."""
        metadata = tool.tool_def.metadata or {}
        original_name = metadata.get(_ROUTES_TO_METADATA_KEY)
        if isinstance(original_name, str) and original_name != name:
            ctx = replace(ctx, tool_name=original_name)
            routed_metadata = {key: value for key, value in metadata.items() if key != _ROUTES_TO_METADATA_KEY}
            tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name, metadata=routed_metadata or None))
            return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)


@dataclass
class AgentControl(ManagedVariableCapability[AgentDepsT, AgentConfig]):
    """Manage an agent's config through one `agent__<name>` Logfire variable.

    The variable holds an `AgentConfig`. Each present section -- `instructions`, `model`, `settings`,
    or `tool_definitions` -- is managed from Logfire, while an absent section keeps the code-defined
    behavior. Removing a section in Logfire deliberately reverts that section to code.

    Instructions are the one section that **composes with** the agent instead of patching it, so it
    works in two ways, and which one an entry uses is the difference between a prompt that reads well
    and one sent to the model twice.

    An entry with **no `id` adds** a block. A capability can only ever contribute instructions -- it
    cannot take them over -- so this is what a bare `instructions` string has always done, and it is
    also why the agent's own text is not something a managed value can edit away:

    - `Agent(instructions=...)` is never managed. Anything published in Logfire is *added* to it, so
      text kept there cannot be edited or removed by adding more. Seeding a managed config from an
      agent's observed system prompt while the same text stays on the agent sends it to the model twice.
    - `AgentControl.instructions` (shorthand for `default=AgentConfig(instructions=...)`) is the
      code-side base prompt. `get_instructions` contributes the published value *or* the default and
      never both, so publishing supersedes this text instead of duplicating it -- which is what makes
      it, not the agent, the place for a base prompt you intend to manage.
    - The published `instructions` in Logfire takes over from that default the moment it is set.

    An entry **with an `id` swaps out** the block Pydantic AI assembled under that key -- replacing its
    text, or dropping it with `instructions=None`. This is how a managed config reaches text no
    capability owns: the agent's own literal, a toolset's, an MCP server's, one `@agent.instructions`
    function out of several. It is applied in `before_model_request`, after every contribution has been
    assembled, so an override addresses what the model was actually about to be sent; see
    [`InstructionBlock`][pydantic_ai_harness.logfire.InstructionBlock] for the keys and
    [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] for which blocks have one at all.

    Added blocks compose the way a capability's instructions always have. Pydantic AI groups static
    instruction text ahead of dynamic text so providers can cache the stable prefix, and keeps source
    order within each group, which puts them after the agent's own literal and `@agent.instructions`
    text and before dynamic toolset instructions. An override leaves its block's position and its
    `dynamic` flag alone, so no override moves the prompt-cache boundary.

    When `name` is omitted, the variable name is derived from the agent's telemetry name using the
    same normalization as the Logfire UI, which is lossy: `checkout-assistant` and `checkout_assistant`
    both resolve `agent__checkout_assistant`, so two agents that differ only in punctuation -- in one
    service or across several in the same project -- share one managed config. Pass an explicit `name`
    to keep them apart. The variable is resolved once per run, and its label and version baggage
    remains active for the whole run.

    The managed `model` is sourced during model selection, so it slots in with the right precedence:
    a call-site `run(model=...)` beats it -- one run of one process, deliberately overriding what is
    published -- and it beats everything else, the agent's constructor model and any other
    capability's alike. A fully model-less `Agent(None, ...)` -- named or nameless -- can therefore be
    driven entirely from managed config. Managed settings work the same way, patching over the
    agent's and every other capability's; settings passed to `run()` still win.
    Static or callable targeting inputs participate in model selection, and the resulting variable
    resolution is reused for the rest of the run so the selected model and applied config cannot
    diverge. An unknown managed model warns and keeps the code model.

    Tool overrides change only the definitions shown to the model. Renames route back to the original
    implementation, collisions retain the original name with a warning, unknown tool keys are inert,
    and an override the SDK can't validate is dropped with a warning while its siblings still apply.
    Parameter names, types, requiredness, validation, and implementation stay code-owned.

    Missing, invalid, or unreachable remote values degrade to the code-defined agent through Logfire's
    resolution fallback, which is why a value this SDK doesn't recognize degrades the narrowest unit
    that contains it -- one setting, one tool override -- rather than the whole config; see
    [`AgentConfig`][pydantic_ai_harness.logfire.AgentConfig]. If the provider does not know the
    variable, auto-create is attempted once per
    process in the background, storing
    [`AGENT_CONFIG_JSON_SCHEMA`][pydantic_ai_harness.logfire.AGENT_CONFIG_JSON_SCHEMA] as the
    variable's schema and logging the creation to Logfire.

    Its `example` is an `AgentConfig`-shaped snapshot of the code-side agent taken from whichever model
    request happens to come first in the process. The Logfire UI presents that snapshot as the code
    baseline to diff managed values against, so it is worth knowing what it really is: for instructions
    or a toolset that vary with `deps`, run input, or the step within a run, it is one point-in-time
    sample rather than a description of the agent. An agent that never reaches a model request never
    auto-creates at all. Its `instructions` are the code-defined blocks, excluding this capability's
    own managed contribution by id, listed separately with the `id` that addresses each block and a
    `dynamic` flag. This lets the UI offer an override per block instead of one
    copy-the-whole-prompt button that would produce exactly the duplication described above.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import AgentControl

    logfire.configure()
    agent = Agent(
        'openai:gpt-5',
        name='checkout_assistant',
        capabilities=[AgentControl(instructions='You are a checkout assistant.', label='production')],
    )
    result = agent.run_sync('Refund my last order.')
    ```

    Runtime `{{...}}` placeholders pass through unless `render_template=True`. During a run,
    `logfire.managed.applied_sections` lists the present sections. The `model` section is reported
    when present even if a call-site model outranked it for that run.
    """

    name: str | Variable[AgentConfig] | None = None
    """Bare variable name, pre-built variable, or `None` to derive it from the agent name.

    A nameless capability derives its variable (and can source the model) from the agent's `name`
    the first time it's needed in a run; the agent must then have a `name`.
    """
    default: AgentConfig | None = None
    """Code-side fallback config; omitted sections preserve the corresponding agent behavior.

    Mutually exclusive with the `instructions` shorthand below.
    """
    instructions: InstructionText | list[InstructionText | InstructionBlock] | None = None
    """Code-side base prompt, exactly equivalent to `default=AgentConfig(instructions=...)`.

    The base prompt belongs here rather than on `Agent(instructions=...)`, which a published config
    can only add to. Mutually exclusive with `default`: an agent that also needs code-side defaults
    for `model`, `settings`, or `tool_definitions` carries its instructions on that `AgentConfig`,
    and passing both raises [`UserError`][pydantic_ai.exceptions.UserError] rather than picking one.

    The other sections have no such shorthand because they have no such trap: a managed `model`,
    `settings`, or `tool_definitions` supersedes the agent's own, so `Agent(model=...)`,
    `Agent(model_settings=...)`, and a tool's own docstring remain the natural code-side homes.
    """
    render_template: bool = False
    """Render `{{...}}` placeholders in *added* instruction text against run dependencies when enabled.

    An entry that addresses an existing block by `id` is applied to the assembled request and is never
    templated: it replaces a block with exactly the text that was published.
    """
    publish_baseline: bool = True
    """Publish the code-side agent snapshot to the variable's `example` when it changes.

    Enabled by default because `example` is documentation for the Logfire editor and is never resolved
    or applied to a run. A failed or stale publish therefore cannot change agent behavior. Disable it
    when the variables token is intentionally read-only or code must not update variable metadata.
    """

    _auto_create_in_wrap_run: ClassVar[bool] = False
    _selection_resolved: ContextVar[ResolvedVariable[AgentConfig] | None] = field(init=False, repr=False)
    _code_model: ContextVar[str | None] = field(init=False, repr=False)
    _code_settings: ContextVar[Mapping[str, Any] | None] = field(init=False, repr=False)
    """The settings in force before this capability's patch, snapshotted for the baseline.

    Held as a plain mapping rather than `ModelSettings`: `RunContext.model_settings` is whatever the
    session is running with, which is `RealtimeModelSettings` in a realtime session, and the only
    thing done with it is `AgentConfigSettings.model_validate`, which takes any mapping. Narrowing
    to one of the two would either cast away a real case or drop a realtime agent's baseline.
    """
    _code_tools: ContextVar[list[ToolDefinition] | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.id is None:
            # A stable id lets the baseline exclude this capability's own instruction contribution
            # without comparing text, which would confuse identical code and managed blocks.
            self.id = 'agent-control'
        self._selection_resolved = ContextVar('agent_control_selection_resolved', default=None)
        self._code_model = ContextVar('agent_control_code_model', default=None)
        self._code_settings = ContextVar('agent_control_code_settings', default=None)
        self._code_tools = ContextVar('agent_control_code_tools', default=None)
        if self.instructions is not None:
            if self.instructions == '':
                raise UserError(
                    "`AgentControl` was given '' as `instructions`, which has no text to contribute. Pass "
                    'instruction text, a list of blocks, or leave it out entirely to keep the code-defined instructions.'
                )
            if self.default is not None:
                raise UserError(
                    '`AgentControl` was given both `instructions` and `default`, which set the same value: '
                    '`instructions=...` is shorthand for `default=AgentConfig(instructions=...)`. Pass one or '
                    'the other, putting the base prompt on the `default` config when other sections need '
                    'code-side defaults too.'
                )
            # Normalized into `default` so the resolved value -- and every reader of it -- sees one
            # code-side config, and `get_instructions` keeps a single path through `resolved.value`.
            self.default = AgentConfig(instructions=self.instructions)
        self._setup_variable(
            self.name,
            prefix=_AGENT_VARIABLE_PREFIX,
            value_type=AgentConfig,
            default=self.default or AgentConfig(),
            json_schema=AGENT_CONFIG_JSON_SCHEMA,
        )

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """Contribute the managed instruction blocks that *add* text, optionally rendered against `deps`.

        Entries that address an existing block by `id` are not contributed here -- there is nothing to
        append for them -- and are applied to the assembled parts in
        [`before_model_request`][pydantic_ai_harness.logfire.AgentControl.before_model_request] instead.

        `resolved.value` is either the published config or this capability's `default`, never a merge
        of the two, so a code-side base prompt set through `instructions`/`default` is superseded by a
        published one rather than sent alongside it.
        """

        def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            resolved = self.resolved
            if resolved is None:
                return None
            value = _added_instructions(resolved.value)
            if value is None:
                return None
            return TemplateStr[AgentDepsT](value).render(ctx.deps) if self.render_template else value

        return instructions

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> AbstractCapability[AgentDepsT]:
        """Bind alongside the companion that carries the overriding half of this capability.

        See [`_AgentControlOverrides`][] for why the two halves cannot be one capability. The pair is
        assembled here rather than at construction so `AgentControl` stays the leaf a user builds,
        passes around, and finds with
        [`find_capability`][pydantic_ai.capabilities.find_capability] -- flattening a
        `CombinedCapability` keeps its children as siblings, so a container in this position would
        take the user's capability out of the tree entirely.
        """
        return CombinedCapability([self, _AgentControlOverrides(self)])

    def _resolve_for_selection(
        self, variable: Variable[AgentConfig], ctx: ModelSelectionContext[AgentDepsT]
    ) -> ResolvedVariable[AgentConfig]:
        """Resolve with the same callable targeting inputs that the authoritative run will reuse."""
        targeting_key = self.targeting_key(ctx) if callable(self.targeting_key) else self.targeting_key  # pyright: ignore[reportArgumentType]
        attributes = self.attributes(ctx) if callable(self.attributes) else self.attributes  # pyright: ignore[reportArgumentType]
        return variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)

    def _model_selector(self) -> Callable[[ModelSelectionContext[AgentDepsT]], ModelSelection]:
        """Build the per-run selector a nameless `get_model` returns.

        Pydantic AI evaluates a selector once per new request step, but the managed model is a
        run-stable config value, so the selector memoizes its first choice and every later step
        reuses it -- one resolve per run, matching the static (named) path rather than re-reading the
        variable each step. A fresh selector (and fresh memo) is built per `get_model` call, i.e. per
        run, so nothing leaks across runs. On the first evaluation it derives the backing variable
        from `ctx.agent` (the same derivation `wrap_run` performs), reads the managed model, and
        falls back to the model Pydantic AI already selected (`ctx.model`) when none is managed --
        raising only when there is no model at all (a nameless, model-less agent with nothing
        published yet), so the misconfiguration surfaces clearly instead of downstream.
        """
        selected: list[ModelSelection] = []

        def select(ctx: ModelSelectionContext[AgentDepsT]) -> ModelSelection:
            if not selected:
                code_model = ctx.model
                self._code_model.set(f'{code_model.system}:{code_model.model_name}' if code_model is not None else None)
                resolved = self._resolve_for_selection(self._ensure_variable_for_agent(ctx.agent), ctx)
                model = resolved.value.model
                if model is not None:
                    try:
                        infer_model(model)
                    except (UserError, ValueError) as error:
                        _warn_dropped(
                            f'Managed agent config selects unknown model {model!r} ({error}); keeping the code model.'
                        )
                        model = None
                if model is not None:
                    selected.append(model)
                elif ctx.model is not None:
                    selected.append(ctx.model)
                else:
                    raise UserError(
                        'A nameless `AgentControl` on a model-less agent has no model to run: the agent '
                        'defines no model and none is published in Logfire yet. Give the agent a model, '
                        'pass one to `run(model=...)`, or publish a `model` in the managed config.'
                    )
                # Handed off only once selection has succeeded. `wrap_run` is what clears this, and a
                # selection that raises never reaches it -- so setting it earlier would leave this run's
                # instructions, settings and tool overrides in the context for whatever runs next.
                self._selection_resolved.set(resolved)
            return selected[0]

        return select

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the agent toolset with managed LLM-facing definition overlays."""
        return _ToolDefinitionOverridesToolset(
            wrapped=toolset, get_overrides=self._current_overrides, observe_code_tools=self._observe_code_tools
        )

    def _observe_code_tools(self, tools: list[ToolDefinition]) -> None:
        self._code_tools.set(tools)

    def _current_overrides(self) -> Mapping[str, ToolDefinitionOverride]:
        """Return the active run's overrides by tool name, or an empty mapping outside a resolved run."""
        resolved = self.resolved
        return {} if resolved is None else _tool_overrides(resolved.value)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Add applied-section baggage inside the base's once-per-run resolution context."""
        resolved = self._selection_resolved.get() or self._resolve(ctx)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                sections = ','.join(
                    name
                    for name in ('instructions', 'model', 'settings', 'tool_definitions')
                    if getattr(resolved.value, name) is not None
                )
                if sections:
                    with logfire.set_baggage(**{'logfire.managed.applied_sections': sections}):
                        return await handler()
                return await handler()
            finally:
                self._resolved.reset(token)
                self._selection_resolved.set(None)

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Capture the code-side baseline from the first assembled request.

        Runs on this outermost half deliberately: the snapshot has to describe the agent *before*
        managed values reach it, and the overriding half applies the instruction overrides from the
        innermost position, after this. The request is left untouched here.
        """
        self._publish_request_baseline(ctx, request_context)
        return request_context

    def _apply_instruction_overrides(self, request_context: ModelRequestContext) -> ModelRequestContext:
        """Replace or drop the assembled instruction parts the managed config addresses by `id`.

        A dynamic block is not one of them. Its text is recomputed per request from whatever the run
        carries, so replacing it pins one rendering forever and dropping it removes the computation --
        either way the block stops doing the thing it was written to do, and nothing about the managed
        value says which. Overriding it is refused and warned about rather than applied. That is also
        why a dynamic block publishes only its seam to the baseline: the editor can show that the block
        is there and that it is not yours to change.

        A function that returns a constant is flagged dynamic all the same, and that is the right call
        rather than a gap to close: `dynamic` is what decides which side of the provider's cache
        breakpoint a block sits on, and nothing about a function's shape says whether its text came
        from the run. An author who wants fixed text addressable says so on the part instead of on the
        function -- `InstructionPart(content=..., name='style')` is static by default and is accepted
        anywhere instructions are, a toolset's `get_instructions()` included.

        `dynamic` is deliberately carried over untouched on the parts that *are* replaced. Pydantic AI
        sorts static blocks ahead of dynamic ones so a provider can cache the stable prefix, so
        re-flagging a replaced block would move the cache boundary for every request -- a silent cost
        regression in exchange for nothing.

        A part with no `id` is unaddressable by construction and passes through. An `id` that matches no
        part is inert.

        The new parameters are assigned onto the given context rather than returned on a
        `dataclasses.replace` copy of it, which would look tidier and be wrong: `ModelRequestContext`
        declares `model_id` and `streaming` as `init=False`, the agent graph sets both immediately
        before calling this hook, and `replace()` re-initializes them to `None`/`False`. Losing them
        costs a streamed run its streaming flag and a durable-execution worker the selection token it
        re-resolves an aliased model from. Reported upstream.
        """
        resolved = self.resolved
        if resolved is None:
            return request_context
        overrides = _instruction_overrides(resolved.value)
        parameters = request_context.model_request_parameters
        if not overrides or not parameters.instruction_parts:
            return request_context
        parts: list[InstructionPart] = []
        for part in parameters.instruction_parts:
            key = _instruction_key(part)
            if key is None or key not in overrides:
                parts.append(part)
                continue
            if part.dynamic:
                _warn_dropped(
                    f'Managed agent config addresses instruction block {key!r}, which the agent recomputes '
                    'per request; a managed value would pin or remove that computation, so it is ignored '
                    'and the block keeps what the code produces. An instruction function that returns '
                    'fixed text is flagged this way too: contribute it as an `InstructionPart` instead '
                    'of from a function to make it addressable.'
                )
                parts.append(part)
                continue
            replacement = overrides[key]
            if replacement is not None:
                parts.append(replace(part, content=replacement))
        request_context.model_request_parameters = replace(parameters, instruction_parts=parts)
        return request_context

    def _publish_request_baseline(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> None:
        """Publish the code-side `AgentConfig` baseline at the first eligible model request.

        Model, settings, and tool definitions are captured at their override sites before managed
        behavior replaces them. Instructions are filtered by this capability's stable id. The result
        is what the agent would do with `AgentControl` removed, without reverse-engineering an already
        modified request. Snapshot serialization is guarded with the background write so neither can
        affect the run.

        Instructions are snapshotted per block, straight off
        [`instruction_parts`][pydantic_ai.models.ModelRequestParameters.instruction_parts], keeping each
        block's `id` and `dynamic` flag. That is the whole reason the UI can offer an override at all:
        the joined prompt telemetry records has no seams in it, so a baseline built from that could only
        ever be copied wholesale -- which, since managed instructions *add*, is how you get the agent's
        own text sent to the model twice with a frozen `Today is <date>` in the middle of it.

        Dynamic instructions and dynamic toolsets make the snapshot a sample of one request. Only the
        first request in a process is eligible, so request-to-request variation does not turn into
        writes from live traffic. A new process publishes a changed deployed baseline.

        An `example` is a description of the code, not a value to apply -- nothing resolves it -- which
        is what lets it use these same fields to say *what exists* rather than *what to change*.
        """
        resolved = self.resolved
        if resolved is None or not self.publish_baseline:
            return
        # The agent's own variable, not a shared attribute: a nameless capability backs one variable
        # per agent, so publishing has to name the one this run resolved.
        variable = self._ensure_variable(ctx)
        if _in_durable_context(ctx):
            _warn_durable_write_skipped(variable)
            return
        # Built inside the `try`, alongside the serialization it feeds. `AgentConfig` says what to
        # *apply* as well as what exists, so its validation -- the length bound in particular -- also
        # runs over code-side text nobody asked to change. An agent whose own instructions exceed the
        # bound would otherwise raise straight out of `before_model_request` and take down every model
        # request, when what it has is a snapshot too big to publish, not a broken run.
        try:
            example = self._code_baseline(request_context)
            serialized = json.dumps(example.model_dump(exclude_none=True), indent=2)
        except Exception as error:
            warnings.warn(
                f'Failed to publish the code baseline for Logfire managed variable {variable.name!r}: {error}'
            )
            return
        key = (variable.logfire_instance, variable.name)
        with _baseline_publish_lock:
            if key in _baseline_publish_attempted:
                return
            _baseline_publish_attempted.add(key)
        if self._should_auto_create_for(variable, resolved):
            self._maybe_auto_create(variable, example=serialized, ctx=ctx)
            return
        _spawn_baseline_publish(variable, serialized)

    def _code_baseline(self, request_context: ModelRequestContext) -> AgentConfig:
        """The agent as written: what it would do with `AgentControl` removed.

        Instructions come straight off the assembled parts, minus this capability's own block, since
        the point is to describe what a managed value would be layered onto. Model, settings, and tool
        definitions were captured at their override sites, before managed behavior replaced them.

        Only ever called from inside the caller's `try`: everything here validates, and a snapshot
        that can't be built must not reach the run.
        """
        own_instruction_id = f'capability:{self.id}'
        instructions: list[InstructionText | InstructionBlock] = []
        for part in request_context.model_request_parameters.instruction_parts or []:
            key = _instruction_key(part)
            if not part.content.strip() or key == own_instruction_id:
                continue
            if not part.dynamic:
                instructions.append(InstructionBlock(id=key, instructions=part.content, dynamic=False))
                continue
            # A dynamic block contributes that it exists, not what it said once. Its text is this
            # request's rendering, built from whatever the run carried -- a tenant name, a user id, a
            # retrieved document -- and the baseline is published to a variable every project member
            # can read. The seam is what the editor needs anyway: enough to show the block and that it
            # is recomputed per request, which is why it is not something to override.
            if key is not None:
                instructions.append(InstructionBlock(id=key, dynamic=True))
        tool_definitions: list[ToolDefinitionOverride] = []
        for tool in self._code_tools.get() or []:
            descriptions: dict[str, str] = {}
            properties = tool.parameters_json_schema.get('properties')
            if isinstance(properties, dict):
                typed_properties: dict[str, Any] = tool.parameters_json_schema['properties']
                for name, schema in typed_properties.items():
                    if isinstance(schema, dict):
                        parameter_schema: dict[str, Any] = typed_properties[name]
                        description = parameter_schema.get('description')
                        if isinstance(description, str):
                            descriptions[name] = description
            tool_definitions.append(
                ToolDefinitionOverride(
                    name=tool.name, description=tool.description or None, parameter_descriptions=descriptions or None
                )
            )
        settings = {
            name: value
            for name, value in (self._code_settings.get() or {}).items()
            if name not in _SETTINGS_WITHHELD_FROM_BASELINE
        }
        return AgentConfig(
            instructions=instructions or None,
            model=self._code_model.get(),
            settings=AgentConfigSettings.model_validate(settings) if settings else None,
            tool_definitions=tool_definitions or None,
        )


@dataclass
class _AgentControlOverrides(AbstractCapability[AgentDepsT]):
    """The half of [`AgentControl`][pydantic_ai_harness.logfire.AgentControl] that has to win.

    A capability's [`CapabilityOrdering`][pydantic_ai.capabilities.CapabilityOrdering] position
    answers two questions at once: how its hooks nest around the run, and where its contributions
    land when Pydantic AI merges them. `AgentControl` needs opposite answers. Its resolution's label
    and version baggage has to envelop the whole run including the run span, which means `outermost`
    -- and `outermost` contributions are merged *first*, so every other capability's model and
    settings would be merged over the published ones. A managed value is an override, so the
    contributions that have to beat other capabilities live here instead, pinned `innermost` where
    they are merged last and win whatever order the capabilities were registered in.

    That is why this is a separate capability rather than a few more methods on `AgentControl`, and
    why it is *not* what a user constructs: `AgentControl` stays the leaf they build and find, and
    [`for_agent`][pydantic_ai_harness.logfire.AgentControl.for_agent] stitches this alongside it. The
    two share one object's state by reference, so there is still one variable, one resolution per
    run, and one thing to configure. Decoupling ordering from precedence in the framework itself is
    tracked in [#7420](https://github.com/pydantic/pydantic-ai/issues/7420); until then this split is
    what gives a published value the precedence the Logfire UI promises.
    """

    control: AgentControl[AgentDepsT] = field(repr=False, compare=False)
    """The capability whose state this contributes from; never a copy, so one resolution serves both."""

    def get_ordering(self) -> CapabilityOrdering:
        """Merge last, so a published model or settings beats every other capability's."""
        return CapabilityOrdering(position='innermost')

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Contribute the lowered managed settings patch for each model request."""

        def model_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            self.control._code_settings.set(dict(ctx.model_settings or {}))  # pyright: ignore[reportPrivateUsage]
            resolved = self.control.resolved
            if resolved is None or resolved.value.settings is None:
                return ModelSettings()
            return _lower_settings(resolved.value.settings)

        return model_settings

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        """Source the managed model with the right precedence (`run(model=...)` > managed > everything else).

        The selector records the lower-precedence code model, resolves static or callable targeting
        inputs, and saves that exact resolution for `wrap_run`. This keeps model selection consistent
        with the config and telemetry used by the run even when a targeting callable is stateful. The
        selector does not enter the resolution as a context manager; `wrap_run` remains the owner of
        its baggage. A fully model-less agent can be driven from Logfire, and a call-site
        `run(model=...)` still wins. An unknown managed model warns and falls back to the code model.
        """
        return self.control._model_selector()  # pyright: ignore[reportPrivateUsage]

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Swap out the instruction blocks the managed config addresses by `id`.

        Runs from the innermost position for the same reason the model and settings are contributed
        from here: an override that another capability's hook could overwrite afterwards is not an
        override. Being last also means every contribution has been assembled, which is what lets it
        reach text no capability owns -- the agent's own literal, a toolset's, an MCP server's.
        """
        return self.control._apply_instruction_overrides(request_context)  # pyright: ignore[reportPrivateUsage]

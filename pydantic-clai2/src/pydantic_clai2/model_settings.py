"""The per-model settings a user can edit, validated before they reach `ModelSettings`."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.settings import ModelSettings

from .custom_params import expand_params


class ModelSettingsForm(BaseModel):
    """Overrides for one model. Unset fields leave the provider default in place."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    max_tokens: int | None = Field(default=None, gt=0, description='Cap on tokens the model may generate per request.')
    temperature: float | None = Field(
        default=None, ge=0, le=2, allow_inf_nan=False, description='Sampling randomness; 0 is deterministic.'
    )
    top_p: float | None = Field(
        default=None, gt=0, le=1, allow_inf_nan=False, description='Nucleus sampling cutoff; lower is narrower.'
    )
    top_k: int | None = Field(default=None, gt=0, description='Sample from the top K tokens only (where supported).')
    seed: int | None = Field(default=None, description='Fixed seed for repeatable sampling (where supported).')
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False, description='Request timeout in seconds.')
    presence_penalty: float | None = Field(
        default=None, ge=-2, le=2, allow_inf_nan=False, description='Push the model toward new topics.'
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2, le=2, allow_inf_nan=False, description='Push the model away from repeating itself.'
    )
    parallel_tool_calls: bool | None = Field(default=None, description='Let the model call several tools at once.')
    thinking: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = Field(
        default=None, description='Extended thinking: on, off, or an effort level (where supported).'
    )
    service_tier: Literal['auto', 'default', 'flex', 'priority'] | None = Field(
        default=None, description='Provider service tier (OpenAI).'
    )

    openai_reasoning_effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] | None = Field(
        default=None, description='OpenAI reasoning effort. Overrides generic thinking.'
    )
    openai_reasoning_context: Literal['auto', 'current_turn', 'all_turns'] | None = Field(
        default=None, description='Responses reasoning retained across turns.'
    )
    openai_reasoning_mode: Literal['standard', 'pro'] | None = Field(
        default=None, description='Responses reasoning mode.'
    )
    openai_reasoning_summary: Literal['auto', 'concise', 'detailed'] | None = Field(
        default=None, description='Responses reasoning summary display.'
    )
    openai_text_verbosity: Literal['low', 'medium', 'high'] | None = Field(
        default=None, description='Responses answer verbosity.'
    )
    anthropic_thinking_mode: Literal['enabled', 'adaptive', 'disabled'] | None = Field(
        default=None, description='Claude thinking mode. Overrides generic thinking.'
    )
    anthropic_thinking_budget: int | None = Field(
        default=None, ge=1024, description='Classic thinking token budget, below max_tokens. Default: 10000.'
    )
    anthropic_effort: Literal['low', 'medium', 'high', 'xhigh', 'max'] | None = Field(
        default=None, description='Claude response effort.'
    )

    custom_params: dict[str, JsonValue] | None = Field(
        default=None, description='Custom request body parameters. Dotted keys nest; custom values win.'
    )

    def to_model_settings(self) -> ModelSettings | None:
        """What `agent.run(model_settings=...)` receives; `None` when nothing is set.

        Built field by field: `ModelSettings.timeout` admits `httpx.Timeout`, so Pydantic
        cannot validate the TypedDict as a whole.
        """
        settings = ModelSettings()
        if self.max_tokens is not None:
            settings['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            settings['temperature'] = self.temperature
        if self.top_p is not None:
            settings['top_p'] = self.top_p
        if self.top_k is not None:
            settings['top_k'] = self.top_k
        if self.seed is not None:
            settings['seed'] = self.seed
        if self.timeout is not None:
            settings['timeout'] = self.timeout
        if self.presence_penalty is not None:
            settings['presence_penalty'] = self.presence_penalty
        if self.frequency_penalty is not None:
            settings['frequency_penalty'] = self.frequency_penalty
        if self.parallel_tool_calls is not None:
            settings['parallel_tool_calls'] = self.parallel_tool_calls
        if self.thinking is not None:
            settings['thinking'] = self.thinking
        if self.service_tier is not None:
            settings['service_tier'] = self.service_tier
        settings.update(self._openai_settings())
        settings.update(self._anthropic_settings())
        if self.custom_params:
            settings['extra_body'] = expand_params(pairs=self.custom_params)
        return settings or None

    def _openai_settings(self) -> OpenAIResponsesModelSettings:
        openai = OpenAIResponsesModelSettings()
        if self.openai_reasoning_effort is not None:
            openai['openai_reasoning_effort'] = self.openai_reasoning_effort
        if self.openai_reasoning_context is not None:
            openai['openai_reasoning_context'] = self.openai_reasoning_context
        if self.openai_reasoning_mode is not None:
            openai['openai_reasoning_mode'] = self.openai_reasoning_mode
        if self.openai_reasoning_summary is not None:
            openai['openai_reasoning_summary'] = self.openai_reasoning_summary
        if self.openai_text_verbosity is not None:
            openai['openai_text_verbosity'] = self.openai_text_verbosity
        return openai

    def _anthropic_settings(self) -> AnthropicModelSettings:
        anthropic = AnthropicModelSettings()
        if self.anthropic_effort is not None:
            anthropic['anthropic_effort'] = self.anthropic_effort
        if self.anthropic_thinking_mode == 'enabled':
            if self.max_tokens is None:
                anthropic['max_tokens'] = (self.anthropic_thinking_budget or 10000) + 4096
            anthropic['anthropic_thinking'] = {
                'type': 'enabled',
                'budget_tokens': self.anthropic_thinking_budget or 10000,
            }
        elif self.anthropic_thinking_mode == 'adaptive':
            anthropic['anthropic_thinking'] = {'type': 'adaptive'}
        elif self.anthropic_thinking_mode == 'disabled':
            anthropic['anthropic_thinking'] = {'type': 'disabled'}
        return anthropic


def model_defaults(*, model: str) -> dict[str, JsonValue]:
    """CLAI defaults for GPT-6 and GPT-5.6 families, independent of provider."""
    name = model.partition(':')[2] if ':' in model else model
    name = name.rsplit('/', 1)[-1]
    if not re.match(r'^gpt-(?:6(?:\.\d+)?|5\.6)(?:$|[-:])', name):
        return {}
    return {
        'thinking': True,
        'service_tier': 'default',
        'openai_reasoning_effort': 'medium',
        'openai_reasoning_context': 'all_turns',
        'openai_reasoning_mode': 'standard',
        'openai_reasoning_summary': 'detailed',
        'openai_text_verbosity': 'low',
    }


def model_settings_from_json(values: dict[str, JsonValue], *, model: str = '') -> ModelSettingsForm:
    """Resolve saved overrides against family defaults, then validate."""
    return ModelSettingsForm.model_validate({**model_defaults(model=model), **values})

"""Secret masking capability that redacts secrets from tool outputs and model responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition

# --- Built-in pattern categories ---

_API_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    'openai_key': re.compile(r'sk-[A-Za-z0-9_-]{20,}'),
    'anthropic_key': re.compile(r'sk-ant-[A-Za-z0-9_-]{20,}'),
    'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
    'github_token': re.compile(r'gh[psorat]_[A-Za-z0-9_]{36,}'),
    'slack_token': re.compile(r'xox[bpas]-[A-Za-z0-9-]+'),
    'google_api_key': re.compile(r'AIza[A-Za-z0-9_-]{35}'),
    'azure_subscription_key': re.compile(r'(?i)Ocp-Apim-Subscription-Key\s*[:=]\s*[A-Fa-f0-9]{32}'),
    'stripe_secret_key': re.compile(r'sk_live_[A-Za-z0-9]{24,}'),
    'stripe_publishable_key': re.compile(r'pk_live_[A-Za-z0-9]{24,}'),
    'sendgrid_key': re.compile(r'SG\.[A-Za-z0-9_-]{22,}\.[A-Za-z0-9_-]{22,}'),
    'twilio_key': re.compile(r'SK[0-9a-fA-F]{32}'),
    'gcp_service_account_key': re.compile(r'"private_key"\s*:\s*"-----BEGIN (?:RSA )?PRIVATE KEY-----'),
    'generic_api_key': re.compile(
        r"""(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?key)\s*[:=]\s*['"]?[A-Za-z0-9_\-/+=]{16,}['"]?"""
    ),
}

_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    'bearer_token': re.compile(r'Bearer\s+[A-Za-z0-9_\-./+=]{20,}'),
    'jwt': re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_\-+=]+'),
}

_CONNECTION_STRING_PATTERNS: dict[str, re.Pattern[str]] = {
    'password_in_url': re.compile(r'://[^:/?#\s]+:[^@/?#\s]+@[^/?#\s]+'),
    'database_connection': re.compile(r'(?i)(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://[^\s]+'),
}

_PRIVATE_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

_ENV_FILE_PATTERNS: dict[str, re.Pattern[str]] = {
    'env_key_value': re.compile(r'(?m)^[A-Z][A-Z0-9_]+=.+$'),
}

_BUILTIN_CATEGORIES: dict[str, dict[str, re.Pattern[str]]] = {
    'api_keys': _API_KEY_PATTERNS,
    'tokens': _TOKEN_PATTERNS,
    'connection_strings': _CONNECTION_STRING_PATTERNS,
    'private_keys': _PRIVATE_KEY_PATTERNS,
    'env_file': _ENV_FILE_PATTERNS,
}

_ALL_BUILTIN_PATTERNS: dict[str, re.Pattern[str]] = {}
for _patterns in _BUILTIN_CATEGORIES.values():
    _ALL_BUILTIN_PATTERNS.update(_patterns)


def _mask_text(text: str, patterns: dict[str, re.Pattern[str]], replacement: str) -> str:
    """Apply all patterns to replace matched secrets in `text`."""
    for pattern in patterns.values():
        text = pattern.sub(replacement, text)
    return text


def _partial_mask_text(text: str, patterns: dict[str, re.Pattern[str]], visible_chars: int = 4) -> str:
    """Apply all patterns, keeping the first `visible_chars` characters and masking the rest."""
    for pattern in patterns.values():
        text = pattern.sub(
            lambda m: m.group()[:visible_chars] + '****' if len(m.group()) > visible_chars else '****', text
        )
    return text


def _mask_dict_values(
    d: dict[str, Any],
    patterns: dict[str, re.Pattern[str]],
    replacement: str,
    *,
    partial: bool = False,
    visible_chars: int = 4,
) -> dict[str, Any]:
    """Recursively scrub secret patterns from string values in a dict."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, str):
            if partial:
                result[key] = _partial_mask_text(value, patterns, visible_chars)
            else:
                result[key] = _mask_text(value, patterns, replacement)
        elif isinstance(value, dict):
            result[key] = _mask_dict_values(
                cast(dict[str, Any], value), patterns, replacement, partial=partial, visible_chars=visible_chars
            )
        elif isinstance(value, list):
            result[key] = _mask_list_values(
                cast('list[Any]', value), patterns, replacement, partial=partial, visible_chars=visible_chars
            )
        else:
            result[key] = value
    return result


def _mask_list_values(
    items: list[Any],
    patterns: dict[str, re.Pattern[str]],
    replacement: str,
    *,
    partial: bool = False,
    visible_chars: int = 4,
) -> list[Any]:
    """Recursively scrub secret patterns from string items in a list."""
    result: list[Any] = []
    for item in items:
        if isinstance(item, str):
            if partial:
                result.append(_partial_mask_text(item, patterns, visible_chars))
            else:
                result.append(_mask_text(item, patterns, replacement))
        elif isinstance(item, dict):
            result.append(
                _mask_dict_values(
                    cast(dict[str, Any], item), patterns, replacement, partial=partial, visible_chars=visible_chars
                )
            )
        elif isinstance(item, list):
            result.append(
                _mask_list_values(
                    cast('list[Any]', item), patterns, replacement, partial=partial, visible_chars=visible_chars
                )
            )
        else:
            result.append(item)
    return result


@dataclass
class SecretMasking(AbstractCapability[Any]):
    """Redacts secrets, API keys, and sensitive data from tool args, outputs, and model responses.

    Uses `before_tool_execute` to scrub secrets from tool arguments,
    `after_tool_execute` to scrub tool return values, and `after_model_request`
    to scrub model response text before they enter the conversation history.

    By default all built-in pattern categories are enabled: `api_keys`, `tokens`,
    `connection_strings`, `private_keys`, and `env_file`.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import SecretMasking

        agent = Agent('openai:gpt-5', capabilities=[SecretMasking()])
        ```
    """

    categories: list[str] | None = None
    """Built-in pattern categories to enable.

    Choose from `'api_keys'`, `'tokens'`, `'connection_strings'`, `'private_keys'`, `'env_file'`.
    When `None` (default), all categories are enabled.
    """

    custom_patterns: dict[str, str] | None = None
    """Additional regex patterns as `{name: pattern}` pairs.

    These are compiled once at init time and applied alongside the built-in patterns.
    """

    replacement: str = '[REDACTED]'
    """The string that replaces matched secrets."""

    partial_mask: bool = False
    """When True, keep the first 4 characters of matched secrets visible and mask the rest
    (e.g. ``sk-pr****`` instead of ``[REDACTED]``). The ``replacement`` field is ignored
    when partial masking is enabled."""

    _compiled: dict[str, re.Pattern[str]] = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile built-in and custom patterns."""
        if self.categories is not None:
            for category in self.categories:
                if category not in _BUILTIN_CATEGORIES:
                    raise ValueError(
                        f'Unknown secret pattern category {category!r}, expected one of {sorted(_BUILTIN_CATEGORIES)}'
                    )
                self._compiled.update(_BUILTIN_CATEGORIES[category])
        else:
            self._compiled.update(_ALL_BUILTIN_PATTERNS)

        if self.custom_patterns:
            for name, pattern in self.custom_patterns.items():
                self._compiled[name] = re.compile(pattern)

    def _apply_mask(self, text: str) -> str:
        """Mask secrets in ``text`` using full or partial masking depending on config."""
        if self.partial_mask:
            return _partial_mask_text(text, self._compiled)
        return _mask_text(text, self._compiled, self.replacement)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Scrub secrets from tool call argument values before the tool executes."""
        return _mask_dict_values(
            args,
            self._compiled,
            self.replacement,
            partial=self.partial_mask,
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Scrub secrets from tool return values."""
        if isinstance(result, str):
            return self._apply_mask(result)
        # For non-string results, convert to string to check, but only replace if secrets found.
        text = str(result)
        masked = self._apply_mask(text)
        if masked != text:
            return masked
        return result

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Scrub secrets from model response text parts."""
        for part in response.parts:
            if isinstance(part, TextPart):
                part.content = self._apply_mask(part.content)
        return response

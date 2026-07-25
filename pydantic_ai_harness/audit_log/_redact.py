"""Argument redaction and size bounding for audit records.

The default redactor strips values whose key names a secret (token, password,
api key, ...). Redaction is keyed on the argument name, so a custom `Redactor`
can implement any policy over `(key, value)` -- value-shape detection, an
allowlist, or a no-op that keeps everything.
"""

from __future__ import annotations

import json
from collections.abc import Callable

Redactor = Callable[[str, object], object]
"""Decides what an audited argument value becomes: `(arg_name, arg_value) -> value`.

Return the value unchanged to keep it, or a placeholder (e.g. `'***'`) to
redact it. Called once per top-level argument before the arguments are
serialized to JSON.
"""

_REDACTED = '***'

_SECRET_HINTS = frozenset(
    {
        'password',
        'passwd',
        'secret',
        'token',
        'api_key',
        'apikey',
        'access_key',
        'secret_key',
        'private_key',
        'client_secret',
        'authorization',
        'credential',
        'session',
        'cookie',
    }
)


def default_secret_redactor(key: str, value: object) -> object:
    """Redact values whose key names a secret; pass everything else through.

    Matches case-insensitively on a substring of the argument name, with `-`
    and `.` separators normalized to `_` before matching, so `API_KEY`,
    `x-api-key`, and `client_secret` are all caught. Replace this with a
    stricter or looser `Redactor` on the capability when the default set does
    not fit.
    """
    normalized = key.lower().replace('-', '_').replace('.', '_')
    if any(hint in normalized for hint in _SECRET_HINTS):
        return _REDACTED
    return value


def bound_text(text: str, max_chars: int) -> str:
    """Truncate `text` to at most `max_chars` characters.

    The bound keeps a single oversized argument or result from ballooning a
    record; the trailing marker signals the value was cut.
    """
    if len(text) <= max_chars:
        return text
    marker = '...[truncated]'
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)] + marker


def redact_arguments(arguments: dict[str, object], *, redactor: Redactor, max_chars: int) -> str:
    """Apply `redactor` to each argument, bound each value, and serialize to JSON.

    Each value is bounded to `max_chars` before the arguments dict is
    serialized, not the serialized document as a whole, so a truncated value
    can never land inside a JSON token -- the result is always valid JSON.
    Non-JSON-serializable values fall back to their string form (`default=str`),
    so an arbitrary tool argument never makes the record unserializable.
    """
    redacted = {key: _bound_value(redactor(key, value), max_chars) for key, value in arguments.items()}
    return json.dumps(redacted, default=str, ensure_ascii=False)


def _bound_value(value: object, max_chars: int) -> object:
    """Bound one redacted argument value to `max_chars`.

    Strings are truncated directly. `None` and numeric/boolean scalars pass
    through unbounded since serializing them can't balloon a record. Anything
    else (a nested dict/list, or an object that needs `default=str`) is
    measured by its own serialization and, only if that exceeds the bound,
    replaced by its truncated string form. Either way the value handed back
    is safe for the caller to serialize again: the original JSON-native value,
    or a plain string.
    """
    if isinstance(value, str):
        return bound_text(value, max_chars)
    if value is None or isinstance(value, (int, float)):
        return value
    serialized = json.dumps(value, default=str, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return value
    return bound_text(serialized, max_chars)

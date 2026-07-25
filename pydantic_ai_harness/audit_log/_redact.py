"""Argument redaction and size bounding for audit records.

Redaction policy belongs to the consumer, not this library: the default
`redactor` is a no-op that keeps every value unchanged. Pass your own
`Redactor` -- `(key, value) -> value` -- to strip whatever your deployment
considers a secret. A hint-list matcher, value-shape detection, and an
allowlist are all just a function of that shape; this module ships none of
them as a default.
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


def identity_redactor(_key: str, value: object) -> object:
    """`AuditLog`'s default `Redactor`: every value passes through unchanged.

    The library takes no position on what counts as a secret. Pass your own
    `redactor` on `AuditLog` for a policy that fits your deployment.
    """
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

    Strings are truncated directly. `None` and `bool` pass through unbounded;
    neither can grow large enough to balloon a record. Any other `int` or
    `float` -- notably an arbitrary-precision integer like `10**3000` -- has
    no such ceiling, so it is bounded by its own decimal text the same way a
    string is. Anything else (a nested dict/list, or an object that needs
    `default=str`) is measured by its own serialization and, only if that
    exceeds the bound, replaced by its truncated string form. Either way the
    value handed back is safe for the caller to serialize again: the original
    JSON-native value, or a plain string.
    """
    if isinstance(value, str):
        return bound_text(value, max_chars)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        text = str(value)
        return value if len(text) <= max_chars else bound_text(text, max_chars)
    serialized = json.dumps(value, default=str, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return value
    return bound_text(serialized, max_chars)

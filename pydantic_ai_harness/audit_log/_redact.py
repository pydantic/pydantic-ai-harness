"""Argument redaction for audit records.

Redaction policy belongs to the consumer, not this library: the default
`redactor` is a no-op that keeps every value unchanged. Pass your own
`Redactor` -- `(key, value) -> value` -- to strip whatever your deployment
considers a secret. A hint-list matcher, value-shape detection, and an
allowlist are all just a function of that shape; this module ships none of
them as a default.

Values are recorded faithfully: this module has no size cap of its own. A
`redactor` that also shortens or drops an oversized value is how a consumer
limits size, the same extension point used for secret-handling.
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


def redact_arguments(arguments: dict[str, object], *, redactor: Redactor) -> str:
    """Apply `redactor` to each argument and serialize the result to JSON.

    Values are recorded faithfully -- an audit trail that quietly truncated
    what a tool was called with would be less correct, not more useful.
    Non-JSON-serializable values fall back to their string form (`default=str`),
    so an arbitrary tool argument never makes the record unserializable. A
    consumer who wants to cap value size does so in their own `redactor`
    (e.g. truncate a string before returning it).

    The one value `default=str` cannot rescue is an integer beyond Python's
    int-to-str conversion digit limit (`sys.get_int_max_str_digits`, 4300 by
    default) anywhere in the argument tree: encoding a JSON number means
    converting the int to decimal text, which raises `ValueError` past that
    limit instead of returning text, and `default` is never consulted for a
    type `json.dumps` already knows how to encode. On that failure, each
    argument is re-serialized individually so the fault is isolated to the
    one unencodable value instead of losing every sibling argument.
    """
    redacted = {key: redactor(key, value) for key, value in arguments.items()}
    try:
        return json.dumps(redacted, default=str, ensure_ascii=False)
    except ValueError:
        return json.dumps({key: _json_safe(value) for key, value in redacted.items()}, default=str, ensure_ascii=False)


def _json_safe(value: object) -> object:
    """Return `value` unchanged, or a placeholder if `json.dumps` cannot encode it at all.

    This is a serialization guard, not a size policy: it only replaces a
    value `json.dumps` cannot represent by any means (see `redact_arguments`),
    never one it can represent, no matter how large.
    """
    try:
        json.dumps(value, default=str)
    except ValueError:
        return '<value exceeds JSON int-to-str conversion limit>'
    return value

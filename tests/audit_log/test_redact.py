"""Tests for argument redaction."""

from __future__ import annotations

import json

from pydantic_ai_harness.audit_log._redact import identity_redactor, redact_arguments


class _Stringifies:
    def __str__(self) -> str:
        return 'STR_FORM'


class TestIdentityRedactor:
    def test_keeps_every_value_unchanged(self):
        # This is `AuditLog`'s default `redactor`: the library ships no
        # secret-detection policy, so nothing is redacted unless the caller
        # supplies their own `redactor`.
        assert identity_redactor('api_key', 'sk-123') == 'sk-123'
        assert identity_redactor('password', 'hunter2') == 'hunter2'
        assert identity_redactor('anything', None) is None


class TestRedactArguments:
    def test_identity_redactor_keeps_everything_and_still_serializes(self):
        out = redact_arguments({'query': 'q', 'token': 'secret'}, redactor=identity_redactor)
        assert json.loads(out) == {'query': 'q', 'token': 'secret'}

    def test_custom_redactor_applies_its_own_policy(self):
        # There is no default secret policy, so a caller who wants redaction
        # supplies their own `redactor` -- this is the extension point.
        def redact_tokens(key: str, value: object) -> object:
            return '***' if 'token' in key else value

        out = redact_arguments({'query': 'q', 'token': 'secret'}, redactor=redact_tokens)
        assert json.loads(out) == {'query': 'q', 'token': '***'}

    def test_custom_redactor_is_how_a_consumer_now_limits_size(self):
        # `redact_arguments` has no size cap of its own -- a caller who wants
        # one truncates in their own `redactor`, the same extension point
        # used for secret-handling.
        def truncate_long_values(_key: str, value: object) -> object:
            return value[:10] if isinstance(value, str) and len(value) > 10 else value

        out = redact_arguments({'blob': 'y' * 500, 'short': 'ok'}, redactor=truncate_long_values)
        assert json.loads(out) == {'blob': 'yyyyyyyyyy', 'short': 'ok'}

    def test_non_serializable_value_falls_back_to_str(self):
        out = redact_arguments({'obj': _Stringifies()}, redactor=identity_redactor)
        assert json.loads(out) == {'obj': 'STR_FORM'}

    def test_oversized_value_is_recorded_faithfully_not_truncated(self):
        # No size cap: an oversized value is serialized in full. A consumer
        # who wants a bound applies one in their own `redactor` (see
        # `test_custom_redactor_is_how_a_consumer_now_limits_size`).
        out = redact_arguments({'blob': 'y' * 5000}, redactor=identity_redactor)
        assert json.loads(out) == {'blob': 'y' * 5000}

    def test_non_string_value_is_kept_as_native_json(self):
        # A nested value keeps its native JSON shape (not stringified),
        # however large -- there is no size threshold that stringifies it.
        out = redact_arguments({'ids': list(range(500))}, redactor=identity_redactor)
        parsed = json.loads(out)
        assert parsed == {'ids': list(range(500))}

    def test_huge_but_representable_integer_is_kept_native(self):
        # An arbitrary-precision integer like `10**3000` (3001 digits, under
        # Python's int-to-str conversion limit) is recorded as a native JSON
        # number, not stringified or truncated -- there is no size cap, and
        # no int/float exemption from one either, because there is no bound
        # at all to be exempt from.
        out = redact_arguments({'n': 10**3000}, redactor=identity_redactor)
        assert json.loads(out) == {'n': 10**3000}

    def test_huge_integer_beyond_str_conversion_limit_is_recorded_as_a_placeholder_instead_of_crashing(self):
        # Python 3.11+ caps int-to-str conversion at `sys.get_int_max_str_digits()`
        # decimal digits (4300 by default): `10**5000` (5001 digits) exceeds
        # it, so `json.dumps` cannot even attempt to render it as a JSON
        # number and raises `ValueError` partway through encoding -- not just
        # `str()`/`repr()`. This is a serialization guard, not a size policy:
        # the placeholder is only for the one value that cannot be encoded by
        # any means, while every sibling argument is recorded faithfully.
        out = redact_arguments({'n': 10**5000, 'q': 'normal'}, redactor=identity_redactor)
        parsed = json.loads(out)  # must not raise json.JSONDecodeError
        assert parsed['q'] == 'normal'
        assert isinstance(parsed['n'], str)
        assert 'exceeds' in parsed['n']

    def test_huge_integer_nested_in_a_container_argument_is_also_guarded(self):
        # The guard applies to the whole value, not just a bare top-level
        # int: an oversized int buried inside a nested dict is exactly as
        # unencodable to `json.dumps` as a bare one, so it gets the same
        # placeholder treatment instead of crashing record construction.
        out = redact_arguments({'payload': {'n': 10**5000}}, redactor=identity_redactor)
        parsed = json.loads(out)  # must not raise json.JSONDecodeError
        assert isinstance(parsed['payload'], str)
        assert 'exceeds' in parsed['payload']

    def test_numeric_values_are_kept_native(self):
        # An ordinary int/float keeps its native JSON numeric type instead of
        # being stringified.
        out = redact_arguments({'count': 42, 'ratio': 0.5}, redactor=identity_redactor)
        assert json.loads(out) == {'count': 42, 'ratio': 0.5}

    def test_bool_and_none_pass_through_unchanged(self):
        out = redact_arguments({'flag': True, 'missing': None}, redactor=identity_redactor)
        assert json.loads(out) == {'flag': True, 'missing': None}

    def test_non_ascii_values_are_kept_raw(self):
        # `ensure_ascii=False` keeps non-ASCII readable in the record instead of escaping it.
        out = redact_arguments({'name': 'café'}, redactor=identity_redactor)
        assert '"café"' in out
        assert 'caf\\u00e9' not in out
        assert json.loads(out) == {'name': 'café'}

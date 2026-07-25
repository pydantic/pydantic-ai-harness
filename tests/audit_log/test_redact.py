"""Tests for argument redaction and size bounding."""

from __future__ import annotations

import json

from pydantic_ai_harness.audit_log._redact import bound_text, identity_redactor, redact_arguments


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


class TestBoundText:
    def test_short_text_is_unchanged(self):
        assert bound_text('hello', 10) == 'hello'
        assert bound_text('hello', 5) == 'hello'

    def test_oversized_text_is_truncated_with_marker(self):
        out = bound_text('x' * 100, 20)
        assert out == 'x' * (20 - len('...[truncated]')) + '...[truncated]'
        assert len(out) == 20

    def test_at_exact_bound_keeps_full_text(self):
        # len(text) == max_chars (and room for the marker): kept whole, not truncated.
        text = 'x' * 20
        assert bound_text(text, 20) == text

    def test_at_marker_length_bound_truncates_to_content(self):
        # max_chars == marker length: hard-truncate to content, not emit a bare marker.
        marker_len = len('...[truncated]')
        assert bound_text('a' * 100, marker_len) == 'a' * marker_len

    def test_bound_below_marker_length_hard_truncates(self):
        # When the bound is shorter than the marker, there is no room for it.
        assert bound_text('abcdef', 4) == 'abcd'


class TestRedactArguments:
    def test_identity_redactor_keeps_everything_and_still_serializes(self):
        out = redact_arguments({'query': 'q', 'token': 'secret'}, redactor=identity_redactor, max_chars=2000)
        assert json.loads(out) == {'query': 'q', 'token': 'secret'}

    def test_custom_redactor_applies_its_own_policy(self):
        # There is no default secret policy, so a caller who wants redaction
        # supplies their own `redactor` -- this is the extension point.
        def redact_tokens(key: str, value: object) -> object:
            return '***' if 'token' in key else value

        out = redact_arguments({'query': 'q', 'token': 'secret'}, redactor=redact_tokens, max_chars=2000)
        assert json.loads(out) == {'query': 'q', 'token': '***'}

    def test_non_serializable_value_falls_back_to_str(self):
        out = redact_arguments({'obj': _Stringifies()}, redactor=identity_redactor, max_chars=2000)
        assert json.loads(out) == {'obj': 'STR_FORM'}

    def test_oversized_serialization_is_bounded(self):
        # The value is bounded to `max_chars`, not the serialized document as
        # a whole (see `test_oversized_value_still_yields_parseable_json`):
        # the document also carries the key and JSON punctuation, so it is
        # longer than `max_chars` once a value is actually cut.
        out = redact_arguments({'blob': 'y' * 500}, redactor=identity_redactor, max_chars=50)
        assert len(json.loads(out)['blob']) == 50

    def test_oversized_value_still_yields_parseable_json(self):
        # A value is bounded before it is serialized, not the serialized
        # document as a whole, so a cut can never land inside a quoted
        # string or otherwise corrupt the surrounding JSON.
        out = redact_arguments({'blob': 'y' * 500}, redactor=identity_redactor, max_chars=50)
        parsed = json.loads(out)  # must not raise json.JSONDecodeError
        assert parsed == {'blob': ('y' * (50 - len('...[truncated]'))) + '...[truncated]'}

    def test_oversized_non_string_value_is_bounded_as_a_string(self):
        # A large nested value (not a plain string) that exceeds the bound
        # once serialized is replaced by its own truncated serialization
        # instead of being embedded as a broken fragment of JSON structure.
        out = redact_arguments({'items': list(range(500))}, redactor=identity_redactor, max_chars=50)
        parsed = json.loads(out)  # must not raise json.JSONDecodeError
        assert isinstance(parsed['items'], str)
        assert parsed['items'].endswith('...[truncated]')
        assert len(parsed['items']) == 50

    def test_small_non_string_value_is_kept_as_native_json(self):
        # Values within the bound keep their native JSON shape (not
        # stringified), so ordinary nested arguments round-trip normally.
        out = redact_arguments({'ids': [1, 2, 3]}, redactor=identity_redactor, max_chars=2000)
        assert json.loads(out) == {'ids': [1, 2, 3]}

    def test_huge_integer_is_bounded_instead_of_ballooning(self):
        # `int`/`float` are not exempt from the size bound: an
        # arbitrary-precision integer like `10**3000` has no natural ceiling,
        # so it is truncated the same way an oversized string is instead of
        # ballooning the record unbounded.
        out = redact_arguments({'n': 10**3000}, redactor=identity_redactor, max_chars=50)
        parsed = json.loads(out)  # must not raise json.JSONDecodeError
        assert isinstance(parsed['n'], str)
        assert parsed['n'].endswith('...[truncated]')
        assert len(parsed['n']) == 50

    def test_small_numeric_values_are_kept_native(self):
        # An ordinary int/float well within the bound keeps its JSON-native
        # numeric type instead of being stringified -- the bound only fires
        # once the value's own text is actually oversized.
        out = redact_arguments({'count': 42, 'ratio': 0.5}, redactor=identity_redactor, max_chars=2000)
        assert json.loads(out) == {'count': 42, 'ratio': 0.5}

    def test_bool_and_none_pass_through_unbounded(self):
        # Unlike int/float, bool and None can never grow large enough to
        # balloon a record, so they keep their native JSON type even under an
        # absurdly small bound.
        out = redact_arguments({'flag': True, 'missing': None}, redactor=identity_redactor, max_chars=1)
        assert json.loads(out) == {'flag': True, 'missing': None}

    def test_non_ascii_values_are_kept_raw(self):
        # `ensure_ascii=False` keeps non-ASCII readable in the record instead of escaping it.
        out = redact_arguments({'name': 'café'}, redactor=identity_redactor, max_chars=2000)
        assert '"café"' in out
        assert 'caf\\u00e9' not in out
        assert json.loads(out) == {'name': 'café'}

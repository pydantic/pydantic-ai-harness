"""Tests for argument redaction and size bounding."""

from __future__ import annotations

import json

from pydantic_ai_harness.audit_log import default_secret_redactor
from pydantic_ai_harness.audit_log._redact import bound_text, redact_arguments


class _Stringifies:
    def __str__(self) -> str:
        return 'STR_FORM'


class TestDefaultSecretRedactor:
    def test_redacts_secret_named_keys(self):
        assert default_secret_redactor('api_key', 'sk-123') == '***'
        assert default_secret_redactor('Authorization', 'Bearer x') == '***'
        assert default_secret_redactor('db_password', 'hunter2') == '***'

    def test_passes_through_non_secret_keys(self):
        assert default_secret_redactor('query', 'select 1') == 'select 1'
        assert default_secret_redactor('limit', 10) == 10


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
    def test_redacts_and_serializes(self):
        out = redact_arguments({'query': 'q', 'token': 'secret'}, redactor=default_secret_redactor, max_chars=2000)
        assert json.loads(out) == {'query': 'q', 'token': '***'}

    def test_non_serializable_value_falls_back_to_str(self):
        out = redact_arguments({'obj': _Stringifies()}, redactor=default_secret_redactor, max_chars=2000)
        assert json.loads(out) == {'obj': 'STR_FORM'}

    def test_oversized_serialization_is_bounded(self):
        out = redact_arguments({'blob': 'y' * 500}, redactor=default_secret_redactor, max_chars=50)
        assert len(out) == 50

    def test_custom_redactor_overrides_default(self):
        out = redact_arguments({'query': 'q'}, redactor=lambda _k, _v: 'X', max_chars=2000)
        assert json.loads(out) == {'query': 'X'}

    def test_non_ascii_values_are_kept_raw(self):
        # `ensure_ascii=False` keeps non-ASCII readable in the record instead of escaping it.
        out = redact_arguments({'name': 'café'}, redactor=default_secret_redactor, max_chars=2000)
        assert '"café"' in out
        assert 'caf\\u00e9' not in out
        assert json.loads(out) == {'name': 'café'}

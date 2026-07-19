from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from pydantic_harness.secret_masking import (
    _ALL_BUILTIN_PATTERNS,
    _BUILTIN_CATEGORIES,
    SecretMasking,
    _mask_dict_values,
    _mask_text,
    _partial_mask_text,
)

# --- Unit tests for _mask_text ---


class TestMaskText:
    def test_no_match_returns_original(self):
        assert _mask_text('hello world', _ALL_BUILTIN_PATTERNS, '[REDACTED]') == 'hello world'

    def test_openai_key(self):
        text = 'key is sk-abc123def456ghi789jkl012mno'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk-abc123' not in result
        assert '[REDACTED]' in result

    def test_anthropic_key(self):
        text = 'sk-ant-api03-abcdefghijklmnopqrstuvwxyz'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk-ant-' not in result
        assert result == '[REDACTED]'

    def test_aws_access_key(self):
        text = 'AWS key: AKIAIOSFODNN7EXAMPLE'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'AKIA' not in result
        assert '[REDACTED]' in result

    def test_github_token(self):
        text = 'token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'ghp_' not in result

    def test_slack_token(self):
        text = 'xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'xoxb-' not in result

    def test_google_api_key(self):
        text = 'AIzaSyD-abcdefghijklmnopqrstuvwxyz01234'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'AIza' not in result

    def test_generic_api_key(self):
        text = 'api_key = "abcdef1234567890abcdef"'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'abcdef1234567890' not in result

    def test_bearer_token(self):
        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'Bearer eyJ' not in result

    def test_jwt(self):
        text = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'eyJ' not in result

    def test_password_in_url(self):
        text = 'postgresql://admin:s3cret_pass@db.example.com:5432/mydb'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 's3cret_pass' not in result

    def test_database_connection_string(self):
        text = 'mongodb+srv://user:pass@cluster.mongodb.net/db'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'user:pass' not in result

    def test_private_key(self):
        text = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK...'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN RSA PRIVATE KEY-----' not in result

    def test_ec_private_key(self):
        text = '-----BEGIN EC PRIVATE KEY-----\ndata...'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN EC PRIVATE KEY-----' not in result

    def test_openssh_private_key(self):
        text = '-----BEGIN OPENSSH PRIVATE KEY-----\ndata...'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN OPENSSH PRIVATE KEY-----' not in result

    def test_multiple_secrets_in_one_string(self):
        text = 'key=sk-abc123def456ghi789jkl012mno, token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk-abc123' not in result
        assert 'ghp_' not in result
        assert result.count('[REDACTED]') >= 2

    def test_custom_replacement(self):
        text = 'sk-abc123def456ghi789jkl012mno'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '***')
        assert result == '***'

    # --- New provider key patterns ---

    def test_azure_subscription_key(self):
        text = 'Ocp-Apim-Subscription-Key: abcdef1234567890abcdef1234567890'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'abcdef1234567890' not in result
        assert '[REDACTED]' in result

    def test_stripe_secret_key(self):
        text = 'sk_live_abcdefghijklmnopqrstuvwx'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk_live_' not in result
        assert result == '[REDACTED]'

    def test_stripe_publishable_key(self):
        text = 'pk_live_abcdefghijklmnopqrstuvwx'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'pk_live_' not in result
        assert result == '[REDACTED]'

    def test_sendgrid_key(self):
        text = 'SG.abcdefghijklmnopqrstuv.abcdefghijklmnopqrstuvwx'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'SG.' not in result
        assert result == '[REDACTED]'

    def test_twilio_key(self):
        text = 'SK0123456789abcdef0123456789abcdef'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'SK01234' not in result
        assert '[REDACTED]' in result

    def test_gcp_service_account_key(self):
        text = '"private_key": "-----BEGIN RSA PRIVATE KEY-----\\nMIIE..."'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN RSA PRIVATE KEY-----' not in result
        assert '[REDACTED]' in result

    # --- .env content detection ---

    def test_env_key_value_single_line(self):
        """DATABASE_URL isn't a sensitive *name*, but the credentials in its value still get
        caught by the connection_strings category, so the variable name can stay visible."""
        text = 'DATABASE_URL=postgres://user:pass@localhost/db'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'user:pass' not in result
        assert 'DATABASE_URL=' in result

    def test_env_key_value_multiline(self):
        text = 'API_KEY=some_secret_value\nDB_PASSWORD=hunter2\nDEBUG=true'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'some_secret_value' not in result
        assert 'hunter2' not in result

    def test_env_key_value_does_not_match_lowercase(self):
        """Lowercase variable names are not typical .env format and should not match."""
        patterns = _BUILTIN_CATEGORIES['env_file']
        text = 'lowercase_var=value'
        result = _mask_text(text, patterns, '[REDACTED]')
        assert result == text

    def test_env_key_value_ignores_non_sensitive_names(self):
        """Ordinary system env vars with no security-relevant name should survive untouched."""
        patterns = _BUILTIN_CATEGORIES['env_file']
        text = 'PATH=/usr/local/bin:/usr/bin\nHOME=/root\nDEBUG=true\nNODE_ENV=production'
        result = _mask_text(text, patterns, '[REDACTED]')
        assert result == text


# --- Unit tests for _partial_mask_text ---


class TestPartialMaskText:
    def test_partial_mask_openai_key(self):
        text = 'sk-abc123def456ghi789jkl012mno'
        result = _partial_mask_text(text, _ALL_BUILTIN_PATTERNS)
        assert result.startswith('sk-a')
        assert result.endswith('****')
        assert 'abc123def456' not in result

    def test_partial_mask_preserves_surrounding_text(self):
        text = 'key is sk-abc123def456ghi789jkl012mno here'
        result = _partial_mask_text(text, _ALL_BUILTIN_PATTERNS)
        assert result.startswith('key is sk-a')
        assert 'here' in result

    def test_partial_mask_short_match_becomes_stars(self):
        """When matched text is 4 chars or fewer, the whole thing becomes ****."""
        patterns = {'short': re.compile(r'AB')}
        result = _partial_mask_text('xABx', patterns)
        assert result == 'x****x'


# --- Unit tests for _mask_dict_values ---


class TestMaskDictValues:
    def test_masks_string_values(self):
        d = {'key': 'sk-abc123def456ghi789jkl012mno', 'name': 'safe'}
        result = _mask_dict_values(d, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert result['key'] == '[REDACTED]'
        assert result['name'] == 'safe'

    def test_masks_nested_dicts(self):
        d = {'outer': {'inner_key': 'sk-abc123def456ghi789jkl012mno'}}
        result = _mask_dict_values(d, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert result['outer']['inner_key'] == '[REDACTED]'

    def test_mask_list_values(self):
        d = {
            'headers': ['Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test', 'safe-header'],
            'configs': [{'token': 'sk-abc123def456ghi789jkl012mno'}],
        }
        result = _mask_dict_values(d, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'eyJ' not in result['headers'][0]
        assert result['headers'][1] == 'safe-header'
        assert result['configs'][0]['token'] == '[REDACTED]'

    def test_non_string_values_unchanged(self):
        d: dict[str, Any] = {'count': 42, 'flag': True, 'items': [1, 2]}
        result = _mask_dict_values(d, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert result == d

    def test_partial_mask_in_dict(self):
        d = {'token': 'sk-abc123def456ghi789jkl012mno'}
        result = _mask_dict_values(d, _ALL_BUILTIN_PATTERNS, '[REDACTED]', partial=True)
        assert result['token'].startswith('sk-a')
        assert result['token'].endswith('****')

    def test_empty_dict(self):
        assert _mask_dict_values({}, _ALL_BUILTIN_PATTERNS, '[REDACTED]') == {}


# --- Tests for SecretMasking dataclass construction ---


class TestSecretMaskingInit:
    def test_defaults(self):
        sm = SecretMasking()
        assert sm.categories is None
        assert sm.custom_patterns is None
        assert sm.replacement == '[REDACTED]'
        assert sm.partial_mask is False
        assert sm._compiled == _ALL_BUILTIN_PATTERNS

    def test_specific_categories(self):
        sm = SecretMasking(categories=['api_keys', 'tokens'])
        expected = {**_BUILTIN_CATEGORIES['api_keys'], **_BUILTIN_CATEGORIES['tokens']}
        assert sm._compiled == expected

    def test_single_category(self):
        sm = SecretMasking(categories=['private_keys'])
        assert sm._compiled == _BUILTIN_CATEGORIES['private_keys']

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown secret pattern category 'bogus'"):
            SecretMasking(categories=['bogus'])

    def test_custom_patterns(self):
        sm = SecretMasking(custom_patterns={'my_secret': r'SECRET-\d{6}'})
        assert 'my_secret' in sm._compiled
        assert sm._compiled['my_secret'].pattern == r'SECRET-\d{6}'

    def test_custom_patterns_with_categories(self):
        sm = SecretMasking(categories=['api_keys'], custom_patterns={'my_secret': r'SECRET-\d{6}'})
        assert 'openai_key' in sm._compiled
        assert 'my_secret' in sm._compiled
        assert 'bearer_token' not in sm._compiled

    def test_custom_replacement(self):
        sm = SecretMasking(replacement='<MASKED>')
        assert sm.replacement == '<MASKED>'

    def test_partial_mask_flag(self):
        sm = SecretMasking(partial_mask=True)
        assert sm.partial_mask is True

    def test_env_file_category(self):
        sm = SecretMasking(categories=['env_file'])
        assert 'env_key_value' in sm._compiled


# --- Tests for before_tool_execute ---


class TestBeforeToolExecute:
    @pytest.fixture()
    def capability(self) -> SecretMasking:
        return SecretMasking()

    @pytest.fixture()
    def ctx(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def call(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def tool_def(self) -> Any:
        return MagicMock()

    @pytest.mark.anyio()
    async def test_scrubs_secret_in_args(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        args = {'api_key': 'sk-abc123def456ghi789jkl012mno', 'query': 'hello'}
        result = await capability.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        assert result['api_key'] == '[REDACTED]'
        assert result['query'] == 'hello'

    @pytest.mark.anyio()
    async def test_scrubs_nested_dict_args(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        args: dict[str, Any] = {'config': {'token': 'sk-abc123def456ghi789jkl012mno'}, 'name': 'test'}
        result = await capability.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        assert result['config']['token'] == '[REDACTED]'
        assert result['name'] == 'test'

    @pytest.mark.anyio()
    async def test_no_secrets_unchanged(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        args = {'query': 'hello world', 'count': 5}
        result = await capability.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        assert result == args

    @pytest.mark.anyio()
    async def test_partial_mask_in_args(self, ctx: Any, call: Any, tool_def: Any):
        capability = SecretMasking(partial_mask=True)
        args = {'key': 'sk-abc123def456ghi789jkl012mno'}
        result = await capability.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        assert result['key'].startswith('sk-a')
        assert result['key'].endswith('****')

    @pytest.mark.anyio()
    async def test_empty_args(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.before_tool_execute(ctx, call=call, tool_def=tool_def, args={})
        assert result == {}


# --- Tests for after_tool_execute ---


class TestAfterToolExecute:
    @pytest.fixture()
    def capability(self) -> SecretMasking:
        return SecretMasking()

    @pytest.fixture()
    def ctx(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def call(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def tool_def(self) -> Any:
        return MagicMock()

    @pytest.mark.anyio()
    async def test_string_result_with_secret(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='key: sk-abc123def456ghi789jkl012mno'
        )
        assert isinstance(result, str)
        assert 'sk-abc123' not in result
        assert '[REDACTED]' in result

    @pytest.mark.anyio()
    async def test_string_result_without_secret(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.after_tool_execute(ctx, call=call, tool_def=tool_def, args={}, result='hello world')
        assert result == 'hello world'

    @pytest.mark.anyio()
    async def test_non_string_result_with_secret(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result=['key', 'sk-abc123def456ghi789jkl012mno']
        )
        assert isinstance(result, str)
        assert 'sk-abc123' not in result

    @pytest.mark.anyio()
    async def test_non_string_result_without_secret(
        self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any
    ):
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result={'status': 'ok'}
        )
        assert result == {'status': 'ok'}

    @pytest.mark.anyio()
    async def test_custom_replacement(self, ctx: Any, call: Any, tool_def: Any):
        capability = SecretMasking(replacement='<HIDDEN>')
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='sk-abc123def456ghi789jkl012mno'
        )
        assert result == '<HIDDEN>'

    @pytest.mark.anyio()
    async def test_custom_pattern(self, ctx: Any, call: Any, tool_def: Any):
        capability = SecretMasking(categories=[], custom_patterns={'internal': r'INT-[A-Z]{8}'})
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='secret: INT-ABCDEFGH'
        )
        assert 'INT-ABCDEFGH' not in result
        assert '[REDACTED]' in result

    @pytest.mark.anyio()
    async def test_partial_mask_in_tool_result(self, ctx: Any, call: Any, tool_def: Any):
        capability = SecretMasking(partial_mask=True)
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='sk-abc123def456ghi789jkl012mno'
        )
        assert isinstance(result, str)
        assert result.startswith('sk-a')
        assert result.endswith('****')


# --- Tests for after_model_request ---


class TestAfterModelRequest:
    @pytest.fixture()
    def capability(self) -> SecretMasking:
        return SecretMasking()

    @pytest.fixture()
    def ctx(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def request_context(self) -> Any:
        return MagicMock()

    def _make_response(self, *texts: str) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=t) for t in texts])

    @pytest.mark.anyio()
    async def test_scrubs_text_parts(self, capability: SecretMasking, ctx: Any, request_context: Any):
        response = self._make_response('Your key is sk-abc123def456ghi789jkl012mno')
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        assert isinstance(result.parts[0], TextPart)
        assert 'sk-abc123' not in result.parts[0].content
        assert '[REDACTED]' in result.parts[0].content

    @pytest.mark.anyio()
    async def test_clean_text_unchanged(self, capability: SecretMasking, ctx: Any, request_context: Any):
        response = self._make_response('No secrets here')
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].content == 'No secrets here'

    @pytest.mark.anyio()
    async def test_multiple_parts(self, capability: SecretMasking, ctx: Any, request_context: Any):
        response = self._make_response(
            'key: AKIAIOSFODNN7EXAMPLE',
            'clean text',
            'token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn',
        )
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        parts = result.parts
        assert isinstance(parts[0], TextPart)
        assert 'AKIA' not in parts[0].content
        assert isinstance(parts[1], TextPart)
        assert parts[1].content == 'clean text'
        assert isinstance(parts[2], TextPart)
        assert 'ghp_' not in parts[2].content

    @pytest.mark.anyio()
    async def test_non_text_parts_are_untouched(self, capability: SecretMasking, ctx: Any, request_context: Any):
        tool_call = ToolCallPart(tool_name='get_secret', args='{}')
        response = ModelResponse(parts=[tool_call])
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        assert result.parts[0] is tool_call

    @pytest.mark.anyio()
    async def test_partial_mask_in_model_response(self, ctx: Any, request_context: Any):
        capability = SecretMasking(partial_mask=True)
        response = ModelResponse(parts=[TextPart(content='key: sk-abc123def456ghi789jkl012mno')])
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        part = result.parts[0]
        assert isinstance(part, TextPart)
        assert part.content.startswith('key: sk-a')
        assert part.content.endswith('****')


# --- Test pattern categories ---


class TestPatternCategories:
    def test_all_categories_exist(self):
        assert set(_BUILTIN_CATEGORIES) == {
            'api_keys',
            'tokens',
            'connection_strings',
            'private_keys',
            'env_file',
        }

    def test_api_keys_category(self):
        patterns = _BUILTIN_CATEGORIES['api_keys']
        assert 'openai_key' in patterns
        assert 'anthropic_key' in patterns
        assert 'aws_access_key' in patterns
        assert 'github_token' in patterns
        assert 'slack_token' in patterns
        assert 'google_api_key' in patterns
        assert 'generic_api_key' in patterns
        assert 'azure_subscription_key' in patterns
        assert 'stripe_secret_key' in patterns
        assert 'stripe_publishable_key' in patterns
        assert 'sendgrid_key' in patterns
        assert 'twilio_key' in patterns
        assert 'gcp_service_account_key' in patterns

    def test_tokens_category(self):
        patterns = _BUILTIN_CATEGORIES['tokens']
        assert 'bearer_token' in patterns
        assert 'jwt' in patterns

    def test_connection_strings_category(self):
        patterns = _BUILTIN_CATEGORIES['connection_strings']
        assert 'password_in_url' in patterns
        assert 'database_connection' in patterns

    def test_private_keys_category(self):
        patterns = _BUILTIN_CATEGORIES['private_keys']
        assert 'private_key' in patterns

    def test_env_file_category(self):
        patterns = _BUILTIN_CATEGORIES['env_file']
        assert 'env_key_value' in patterns

    def test_all_builtin_is_union_of_categories(self):
        expected: dict[str, re.Pattern[str]] = {}
        for cat_patterns in _BUILTIN_CATEGORIES.values():
            expected.update(cat_patterns)
        assert _ALL_BUILTIN_PATTERNS == expected


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_categories_list_with_custom(self):
        sm = SecretMasking(categories=[], custom_patterns={'test': r'TEST-\d+'})
        # Only custom patterns, no builtins.
        assert 'test' in sm._compiled
        assert 'openai_key' not in sm._compiled

    def test_empty_categories_no_custom(self):
        sm = SecretMasking(categories=[])
        assert sm._compiled == {}

    @pytest.mark.anyio()
    async def test_empty_string_tool_result(self):
        sm = SecretMasking()
        ctx = MagicMock()
        result = await sm.after_tool_execute(ctx, call=MagicMock(), tool_def=MagicMock(), args={}, result='')
        assert result == ''

    @pytest.mark.anyio()
    async def test_none_tool_result(self):
        sm = SecretMasking()
        ctx = MagicMock()
        result = await sm.after_tool_execute(ctx, call=MagicMock(), tool_def=MagicMock(), args={}, result=None)
        assert result is None

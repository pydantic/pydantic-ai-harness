"""Tests for the ready-made detectors and for chaining several guards."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import GuardResult, InputGuard, OutputBlocked, OutputGuard
from pydantic_ai_harness.guardrails.detectors import (
    DEFAULT_PII_PATTERNS,
    DEFAULT_SECRET_PATTERNS,
    TextDetector,
    blocked_keywords,
    for_text,
    personal_data,
    redact_personal_data,
    redact_secrets,
    secrets,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


_OPENAI_KEY = 'sk-abcdefghijklmnopqrstuvwxyz01'


def _echo_prompt() -> FunctionModel:
    """A model that answers with the prompt it received, so a redaction is observable."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        return ModelResponse(parts=[TextPart(content=str(prompts[-1]))])

    return FunctionModel(respond)


class TestSecretRedaction:
    """Credentials are rewritten, not refused."""

    def test_a_key_is_replaced_and_named(self):
        result = redact_secrets(f'the key is {_OPENAI_KEY}')

        assert result.action == 'replace'
        assert result.replacement == 'the key is [redacted:openai_key]'

    def test_clean_text_is_left_alone(self):
        assert redact_secrets('nothing to see').action == 'allow'

    @pytest.mark.parametrize('name', sorted(DEFAULT_SECRET_PATTERNS))
    def test_every_default_pattern_matches_something(self, name: str):
        """A pattern that matches nothing is dead weight, and a typo in one is invisible."""
        samples = {
            'openai_key': _OPENAI_KEY,
            'anthropic_key': 'sk-ant-abcdefghijklmnopqrstuvwx',
            'aws_access_key': 'AKIAIOSFODNN7EXAMPLE',
            'github_token': 'ghp_abcdefghijklmnopqrstuvwxyz0123',
            'slack_token': 'xoxb-1234567890-abcdef',
            'stripe_key': 'sk_live_abcdefghijklmnopqrstuvwx',
            'google_api_key': 'AIza' + 'a' * 35,
            'jwt': 'eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4',
            'private_key': '-----BEGIN RSA PRIVATE KEY-----',
        }
        assert secrets(only=[name])(samples[name]).action == 'replace'

    def test_a_subset_leaves_the_rest_alone(self):
        detector = secrets(only=['aws_access_key'])

        assert detector(f'key {_OPENAI_KEY}').action == 'allow'
        assert detector('key AKIAIOSFODNN7EXAMPLE').action == 'replace'

    def test_an_unknown_pattern_name_is_refused(self):
        with pytest.raises(UserError, match='Unknown pattern'):
            secrets(only=['nope'])

    def test_a_custom_pattern_joins_the_defaults(self):
        detector = secrets(extra={'internal': r'INT-\d{4}'})

        assert detector('ticket INT-4321').replacement == 'ticket [redacted:internal]'

    def test_a_placeholder_without_the_name_is_used_verbatim(self):
        assert secrets(placeholder='***')(f'k {_OPENAI_KEY}').replacement == 'k ***'


class TestPersonalData:
    """Personal data is rewritten too, for the same reason."""

    @pytest.mark.parametrize(
        ('name', 'sample'),
        [
            ('email', 'write to a.b@example.com'),
            ('us_ssn', 'ssn 123-45-6789'),
            ('credit_card', 'card 4111 1111 1111 1111'),
            ('iban', 'iban GB29NWBK60161331926819'),
        ],
    )
    def test_each_default_pattern_matches(self, name: str, sample: str):
        assert name in DEFAULT_PII_PATTERNS
        assert personal_data(only=[name])(sample).action == 'replace'

    def test_ordinary_text_survives(self):
        assert redact_personal_data('the meeting is at 3pm on the 4th').action == 'allow'


class TestBlockedKeywords:
    """A keyword list refuses rather than rewrites."""

    def test_a_match_blocks_and_names_the_term(self):
        result = blocked_keywords(['classified'])('this is CLASSIFIED')

        assert result.action == 'block'
        assert result.message == "Blocked term: 'CLASSIFIED'."

    def test_case_sensitivity_is_opt_in(self):
        assert blocked_keywords(['SECRET'], case_sensitive=True)('secret').action == 'allow'
        assert blocked_keywords(['SECRET'], case_sensitive=True)('SECRET').action == 'block'

    def test_whole_words_stops_matching_inside_a_longer_word(self):
        detector = blocked_keywords(['class'], whole_words=True)

        assert detector('classification').action == 'allow'
        assert detector('the class').action == 'block'

    def test_a_keyword_with_punctuation_matches_itself(self):
        """Keywords are escaped, so regex characters are not a mini-language."""
        assert blocked_keywords(['a.b'])('axb').action == 'allow'
        assert blocked_keywords(['a.b'])('a.b').action == 'block'

    def test_a_custom_message_hides_the_list(self):
        result = blocked_keywords(['classified'], message='Not available.')('classified')

        assert result.message == 'Not available.'

    def test_no_keywords_is_refused(self):
        with pytest.raises(UserError, match='no keywords'):
            blocked_keywords([])


class TestForText:
    """A text detector meeting a value that is not text."""

    def test_text_reaches_the_detector(self):
        assert for_text(redact_secrets)(f'k {_OPENAI_KEY}').action == 'replace'

    def test_a_non_string_fails_loudly_by_default(self):
        with pytest.raises(UserError, match='cannot rewrite without changing'):
            for_text(redact_secrets)(42)

    def test_a_non_string_can_be_skipped_deliberately(self):
        assert for_text(redact_secrets, on_other='allow')(42).action == 'allow'


class TestGuardChain:
    """Several guards in one capability, run in order."""

    async def test_a_redaction_is_threaded_into_the_next_guard(self):
        """The second guard sees the cleaned text, which is what makes ordering useful."""
        seen: list[str] = []

        def record(prompt: str) -> GuardResult:
            seen.append(prompt)
            return GuardResult.allow()

        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuard(guard=[redact_secrets, record])],
        )
        result = await agent.run(f'my key is {_OPENAI_KEY}')

        assert seen == ['my key is [redacted:openai_key]']
        assert result.output == 'my key is [redacted:openai_key]'

    async def test_the_first_block_ends_the_chain(self):
        reached: list[str] = []

        def never(prompt: str) -> GuardResult:  # pragma: no cover - the point is that it is not reached
            reached.append(prompt)
            return GuardResult.allow()

        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuard(guard=[blocked_keywords(['classified']), never])],
        )
        result = await agent.run('this is classified')

        assert result.output == "Blocked term: 'classified'."
        assert reached == []

    async def test_a_chain_that_only_allows_changes_nothing(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuard(guard=[redact_secrets, blocked_keywords(['nope'])])],
        )

        assert await agent.run('nothing sensitive here') is not None

    async def test_an_output_chain_blocks_on_the_second_guard(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='here is the classified plan')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[OutputGuard(guard=[for_text(redact_secrets), for_text(blocked_keywords(['classified']))])],
        )

        with pytest.raises(OutputBlocked, match='Blocked term'):
            await agent.run('hi')

    async def test_an_output_chain_returns_the_accumulated_replacement(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=f'key {_OPENAI_KEY} for a.b@example.com')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[OutputGuard(guard=[for_text(redact_secrets), for_text(redact_personal_data)])],
        )
        result = await agent.run('hi')

        assert result.output == 'key [redacted:openai_key] for [redacted:email]'

    async def test_an_empty_chain_is_refused(self):
        """A guardrail that inspects nothing reads as configured and behaves as absent."""
        guards: list[TextDetector] = []
        agent = Agent(_echo_prompt(), deps_type=type(None), capabilities=[InputGuard(guard=guards)])

        with pytest.raises(UserError, match='empty sequence of guards'):
            await agent.run('hi')

    async def test_a_mid_chain_replacement_must_still_be_prompt_text(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuard(guard=[lambda prompt: GuardResult.replace(123), redact_secrets])],
        )

        with pytest.raises(UserError, match='guard at position 0 returned int'):
            await agent.run('hi')

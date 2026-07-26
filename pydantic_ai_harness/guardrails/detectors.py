"""Ready-made checks to plug into a guard.

These are plain functions returning a
[`GuardResult`][pydantic_ai_harness.GuardResult], not capabilities. That is the
difference that makes them compose: several go into one `InputGuard` or
`OutputGuard` and run as a chain, next to whatever you write yourself, sharing
one place in the capability list and one set of spans.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import InputGuard
from pydantic_ai_harness.guardrails.detectors import blocked_keywords, redact_secrets

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[InputGuard(guard=[redact_secrets, blocked_keywords(['internal-only'])])],
)
```

They read text, so they suit a prompt directly. A structured agent output is
not text, and replacing one with a scrubbed string would change its type, so
reach for [`for_text`][pydantic_ai_harness.guardrails.detectors.for_text] to
say what should happen there.

What these cannot do is worth stating. A regex finds a credential because
credentials have a shape; it does not find a prompt injection, which is
ordinary language, and it does not understand context, so a redactor will
sometimes take a string that only looks like a key. Treat them as one cheap
layer, not as the answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.guardrails._capability import GuardResult

TextDetector = Callable[[str], GuardResult]
"""A check over text. Plug one into `InputGuard`, or into `OutputGuard` via `for_text`."""

DEFAULT_SECRET_PATTERNS: Mapping[str, str] = {
    'openai_key': r'sk-[A-Za-z0-9]{20,}',
    'anthropic_key': r'sk-ant-[A-Za-z0-9-]{20,}',
    'aws_access_key': r'AKIA[0-9A-Z]{16}',
    'github_token': r'(?:ghp|gho|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}',
    'slack_token': r'xox[bporas]-[A-Za-z0-9-]{10,}',
    'stripe_key': r'sk_(?:live|test)_[A-Za-z0-9]{20,}',
    'google_api_key': r'AIza[A-Za-z0-9_-]{35}',
    'jwt': r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    # The whole PEM block, not its header: replacing the BEGIN line alone would
    # leave the key material sitting in the text. An unterminated block is taken
    # to the end of the input, since a partial key is not worth keeping either.
    'private_key': r'-----BEGIN[^-]*PRIVATE KEY-----[\s\S]*?(?:-----END[^-]*PRIVATE KEY-----|\Z)',
}
"""Credential shapes worth redacting. Distinctive prefixes, so false positives are rare."""

DEFAULT_PII_PATTERNS: Mapping[str, str] = {
    'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'us_ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b(?:\d{4}[ -]?){3}\d{4}\b',
    'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',
}
"""Personal data shapes. Narrower than the secret set, because these overlap ordinary text."""

_REDACTED = '[redacted:{name}]'


def _compile(
    patterns: Mapping[str, str], only: Iterable[str] | None, extra: Mapping[str, str] | None
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """The patterns a detector will run, as `(name, compiled)` pairs."""
    selected = dict(patterns) if only is None else {}
    if only is not None:
        for name in only:
            if name not in patterns:
                raise UserError(f'Unknown pattern {name!r}; available: {sorted(patterns)}.')
            selected[name] = patterns[name]
    if extra:
        selected.update(extra)
    return tuple((name, re.compile(pattern)) for name, pattern in selected.items())


def _redactor(compiled: tuple[tuple[str, re.Pattern[str]], ...], placeholder: str) -> TextDetector:
    """A detector that rewrites every match and allows text with none."""

    def detect(text: str) -> GuardResult:
        cleaned = text
        for name, pattern in compiled:
            # `replace` rather than `format`, so a placeholder containing braces
            # for any other reason is left alone instead of raising.
            cleaned = pattern.sub(placeholder.replace('{name}', name), cleaned)
        return GuardResult.replace(cleaned) if cleaned != text else GuardResult.allow()

    return detect


def secrets(
    *,
    only: Iterable[str] | None = None,
    extra: Mapping[str, str] | None = None,
    placeholder: str = _REDACTED,
) -> TextDetector:
    """Build a detector that rewrites credentials out of text.

    Redacting rather than refusing is the useful default here: an agent that
    quotes a key back has still done the work, and blocking the answer loses it
    while leaving the key in the message history either way.

    Args:
        only: Restrict to these names from `DEFAULT_SECRET_PATTERNS`.
        extra: Additional `name -> regex` patterns to include.
        placeholder: What each match is replaced with. `{name}` in it becomes
            the pattern that matched, so a redaction says what it removed.
    """
    return _redactor(_compile(DEFAULT_SECRET_PATTERNS, only, extra), placeholder)


def personal_data(
    *,
    only: Iterable[str] | None = None,
    extra: Mapping[str, str] | None = None,
    placeholder: str = _REDACTED,
) -> TextDetector:
    """Build a detector that rewrites personal data out of text.

    Redaction, not refusal. An email address in a prompt is usually the user
    telling the agent something it needs, so blocking the turn would break
    ordinary use; removing it before it reaches the model does not.

    Args and behaviour match [`secrets`][pydantic_ai_harness.guardrails.detectors.secrets],
    over `DEFAULT_PII_PATTERNS`.
    """
    return _redactor(_compile(DEFAULT_PII_PATTERNS, only, extra), placeholder)


redact_secrets = secrets()
"""`secrets()` with every default pattern, for the common case."""

redact_personal_data = personal_data()
"""`personal_data()` with every default pattern, for the common case."""


def blocked_keywords(
    keywords: Iterable[str],
    *,
    case_sensitive: bool = False,
    whole_words: bool = False,
    message: str | None = None,
) -> TextDetector:
    """Build a detector that refuses text containing any of `keywords`.

    Args:
        keywords: Literal strings to look for. They are escaped, so a keyword
            with regex punctuation matches itself.
        case_sensitive: Match case exactly. Off by default.
        whole_words: Require word boundaries, so `class` stops matching inside
            `classification`.
        message: The refusal text. The default names the keyword that matched,
            which is useful to an operator and visible to the model, so pass
            something neutral when the list itself is sensitive.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    terms = tuple(keywords)
    if any(not keyword for keyword in terms):
        raise UserError('blocked_keywords() was given an empty keyword, which would match every input.')
    compiled = tuple(
        # `(?<!\w)`/`(?!\w)` rather than `\b`: a boundary needs a word character on
        # one side, so `\bC\+\+\b` never matches `C++` and the keyword would be
        # silently inert.
        re.compile(rf'(?<!\w){re.escape(keyword)}(?!\w)' if whole_words else re.escape(keyword), flags)
        for keyword in terms
    )
    if not compiled:
        raise UserError('blocked_keywords() was given no keywords, so it would never match anything.')

    def detect(text: str) -> GuardResult:
        for pattern in compiled:
            match = pattern.search(text)
            if match:
                return GuardResult.block(message or f'Blocked term: {match.group()!r}.')
        return GuardResult.allow()

    return detect


def for_text(
    detector: TextDetector, *, on_other: Literal['raise', 'allow'] = 'raise'
) -> Callable[[object], GuardResult]:
    """Adapt a text detector to a guard over a value that may not be text.

    An `OutputGuard` receives the agent output unchanged, which for a structured
    output is a model instance rather than a string. Substituting a scrubbed
    string for it would change the output's type, so this refuses to guess:
    `'raise'` fails loudly, telling you to apply the detector to a field
    instead, and `'allow'` lets non-text through when you know the other output
    types carry nothing to check.
    """

    def guard(value: object) -> GuardResult:
        if isinstance(value, str):
            return detector(value)
        if on_other == 'allow':
            return GuardResult.allow()
        raise UserError(
            f'A text detector received {type(value).__name__}, which it cannot rewrite without changing the '
            "output's type. Apply it to a field of the output, or pass on_other='allow' to skip non-text."
        )

    return guard

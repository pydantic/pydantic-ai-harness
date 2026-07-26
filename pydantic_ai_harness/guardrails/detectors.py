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
ordinary language, and it does not understand context.

That last point has a concrete cost for `email`, which matches anything shaped
like an address because that is all an address is:

```python
redact_personal_data('git clone git@github.com:pydantic/pydantic-ai.git')
# -> replaces `git@github.com`, leaving a command that no longer runs
```

An input guard rewrites the prompt in place, so the model receives the broken
version. On an agent that handles code or paths, reach for
`personal_data(only=['us_ssn', 'credit_card', 'iban'])`, or apply the detector
to the output rather than the prompt. Treat all of these as one cheap layer,
not as the answer.
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
    # Vendor key bodies are base64url, whose alphabet includes `_`. A class that
    # stops at `_` redacts half a key and leaves the rest under a label saying it
    # is gone, which is worse than not matching at all.
    'anthropic_key': r'sk-ant-[A-Za-z0-9_-]{20,}',
    # Declared after the Anthropic shape it must not claim, and excluding it
    # explicitly so the label does not depend on declaration order.
    'openai_key': r'sk-(?!ant-)[A-Za-z0-9_-]{20,}',
    'aws_access_key': r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b',
    'github_token': r'(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}',
    'slack_token': r'xox[bporas]-[A-Za-z0-9-]{10,}',
    'slack_app_token': r'xapp-\d-[A-Za-z0-9-]{10,}',
    'stripe_key': r'(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}',
    'stripe_webhook_secret': r'whsec_[A-Za-z0-9]{20,}',
    'google_api_key': r'AIza[A-Za-z0-9_-]{35}',
    'google_oauth_secret': r'GOCSPX-[A-Za-z0-9_-]{20,}',
    'jwt': r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    # A complete PEM block. The newline after the header is required so prose
    # naming both markers in one sentence is not swallowed as a key.
    'private_key': (
        r'-----BEGIN[^-\n]*PRIVATE KEY(?: BLOCK)?-----\s*\n'
        r'[A-Za-z0-9+/=\s]*?'
        r'-----END[^-\n]*PRIVATE KEY(?: BLOCK)?-----'
    ),
    # A block with no END marker: take the header and the base64 lines under it,
    # and stop at the first line that is not body. Running to the end of the
    # input instead would delete whatever the user wrote after it.
    'private_key_body': r'-----BEGIN[^-\n]*PRIVATE KEY(?: BLOCK)?-----\s*\n(?:[A-Za-z0-9+/=]+\n?)+',
}
"""Credential shapes worth redacting, in the order they are applied.

Prefix-anchored, so false positives are rare. An AWS *secret* access key has no
distinctive shape -- it is 40 characters of base64 -- and is deliberately absent
rather than matched by a pattern that would also take ordinary text.
"""

DEFAULT_PII_PATTERNS: Mapping[str, str] = {
    # Before `credit_card`, whose digit groups would otherwise claim the middle
    # of a spaced IBAN and label it as a card number.
    # Counted in characters, not groups: the last group of a printed IBAN is
    # whatever is left over, so requiring groups of four drops it.
    'iban': r'\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){10,30}\b',
    # 4-4-4-4 and the 4-6-5 grouping American Express uses. Every match is
    # checked against the Luhn algorithm, which discards most runs of digits
    # that are not card numbers. Not all of them: roughly one in ten runs of
    # four consecutive years satisfies the checksum by chance.
    'credit_card': r'\b(?:\d{4}[ -]?\d{6}[ -]?\d{5}|(?:\d{4}[ -]?){3}\d{4})\b',
    'us_ssn': r'\b\d{3}[- ]\d{2}[- ]\d{4}\b',
    # The local part is bounded and preceded by a negative lookbehind: an
    # unbounded `+` lets a failed match restart at every interior offset, which
    # makes a pasted log quadratic and blocks the event loop for seconds.
    'email': (
        r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b'
    ),
}
"""Personal data shapes, in the order they are applied.

Narrower than the secret set, because these overlap ordinary text. `email`
matches anything shaped like an address, which includes `git@github.com` in a
shell command -- see the module docstring on where that matters.
"""

_REDACTED = '[redacted:{name}]'


def _luhn(digits: str) -> bool:
    """Whether a run of digits satisfies the Luhn checksum every card number does."""
    values = [int(character) for character in digits if character.isdigit()][::-1]
    total = sum(value if index % 2 == 0 else sum(divmod(value * 2, 10)) for index, value in enumerate(values))
    return len(values) >= 12 and total % 10 == 0


_VALIDATORS: Mapping[str, Callable[[str], bool]] = {'credit_card': _luhn}
"""Semantic checks that a shape alone cannot make. A name absent here is taken on its shape."""


def _compile(
    patterns: Mapping[str, str], only: Iterable[str] | None, extra: Mapping[str, str] | None
) -> tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...]:
    """The patterns a detector will run, as `(name, compiled, validator)` triples.

    A validator belongs to the built-in pattern it was written for, not to its
    name. A custom pattern supplied under a built-in name -- possible once
    `only` has dropped that built-in -- would otherwise be judged by a check
    written for different text and silently never match.
    """
    selected = dict(patterns) if only is None else {}
    if only is not None:
        for name in only:
            if name not in patterns:
                raise UserError(f'Unknown pattern {name!r}; available: {sorted(patterns)}.')
            selected[name] = patterns[name]
    if extra:
        clashes = sorted(set(extra) & set(selected))
        if clashes:
            raise UserError(
                f'`extra` would replace the built-in pattern(s) {clashes} rather than add to them. '
                'Rename them, or pass `only` to drop the built-in first.'
            )
        selected.update(extra)
    if not selected:
        raise UserError('This detector was given no patterns, so it would never match anything.')
    overridden = set(extra or ())
    return tuple(
        (name, re.compile(pattern), None if name in overridden else _VALIDATORS.get(name))
        for name, pattern in selected.items()
    )


def _redactor(
    compiled: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...], placeholder: str
) -> TextDetector:
    """A detector that rewrites every match and allows text with none."""

    def detect(text: str) -> GuardResult:
        cleaned = text
        for name, pattern, valid in compiled:
            substitution = placeholder.replace('{name}', name)

            def replace(match: re.Match[str], valid: Callable[[str], bool] | None = valid, out: str = substitution):
                # A function rather than a template string: `re.sub` reads
                # backreferences in a replacement, so a placeholder containing
                # `\g<0>` would re-emit the very text being redacted.
                return out if valid is None or valid(match.group()) else match.group()

            cleaned = pattern.sub(replace, cleaned)
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

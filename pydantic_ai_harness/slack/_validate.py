"""Checks shared by the places a caller hands this package a list of names."""

from __future__ import annotations

from collections.abc import Collection


def string_sequence(value: Collection[str], name: str) -> tuple[str, ...]:
    """`value` as a tuple, refusing a bare string.

    A string is a `Collection[str]`, so passing one where a list belongs type
    checks and then iterates per character. That reaches Slack as seven channels
    called `#`, `a`, `l` and so on, or an approver list of single letters that no
    real user id matches.
    """
    if isinstance(value, str):
        raise ValueError(
            f'{name} was given a single string, which would be one entry per character. '
            f'Pass a sequence: {name}=[{value!r}].'
        )
    return tuple(value)

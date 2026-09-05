"""The check the annotations cannot do."""

from __future__ import annotations


def reject_bare_string(value: object, name: str) -> None:
    """Refuse a single string where a list of them belongs.

    A string is a `Sequence[str]` and a `Collection[str]`, so those annotations
    accept one and then iterate it per character. This covers untyped callers
    and serialized configuration before it reaches the normalized policy.
    """
    if isinstance(value, str):
        raise ValueError(
            f'{name} was given a single string, which would be one entry per character. '
            f'Pass a list: {name}=[{value!r}].'
        )

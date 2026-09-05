"""The check the annotations cannot do."""

from __future__ import annotations


def reject_bare_string(value: object, name: str) -> None:
    """Refuse a single string where a list of them belongs.

    A string is a `Sequence[str]` and a `Collection[str]`, so those annotations
    accept one and then iterate it per character. The public annotations here are
    `list[str]`, which a type checker refuses. This covers the callers a type
    checker does not see: an agent spec, whose values reach `from_spec` as
    whatever the file said, and untyped Python.

    It matters most for `channels`, which is written into the model's
    instructions. Left alone, `'#alerts'` reaches the model as seven channels
    called `#`, `a`, `l` and so on, and nothing tells the person who set it.
    """
    if isinstance(value, str):
        raise ValueError(
            f'{name} was given a single string, which would be one entry per character. '
            f'Pass a list: {name}=[{value!r}].'
        )

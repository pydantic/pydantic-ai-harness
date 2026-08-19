"""Runtime validation shared by capability packages."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.exceptions import UserError


def reject_bare_str(class_name: str, fields: Sequence[tuple[str, Sequence[str], str]]) -> None:
    """Reject a bare string where a collection of names is meant.

    `str` satisfies `Sequence[str]`, so `denied_commands='rm'` type-checks and
    then splats into `['r', 'm']`: a denylist that stops matching `rm` while
    reading as configured. Each entry is `(field_name, value, noun)`, where the
    noun names what the collection holds in the error message.
    """
    for field_name, value, noun in fields:
        if isinstance(value, str):
            raise UserError(
                f'{class_name}.{field_name} takes a collection of {noun}, not a single string. '
                f'Pass [{value!r}] rather than {value!r}.'
            )

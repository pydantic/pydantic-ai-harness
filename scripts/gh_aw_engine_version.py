# /// script
# requires-python = ">=3.10"
# dependencies = ["packaging>=24", "pydantic>=2", "pyyaml>=6.0.2"]
# ///
"""Validate `gh-aw/pydantic.md` and print the `pydantic-ai-harness` version its engine pins.

GitHub Agentic Workflows resolves its `pydantic-ai` engine from that file on this
repository's `gh-aw-engine` branch, so the frontmatter is a published contract rather
than repo-local config: `engine.id` keys the entry in gh-aw's engine catalog, and
`engine.version` is the release the generated workflow installs from PyPI at run time.
A renamed field breaks the catalog entry, and a version that was never published breaks
every run of it at install time.

The lint job and the two jobs that verify a commit before `gh-aw-engine` advances to it
(`verify-gh-aw-engine` in `main.yml` and `verify` in `gh-aw-engine.yml`) call this, which
is why the checks live here rather than inlined three times as shell. The advancing jobs
check out nothing and never run it.

Inline dependency metadata, so `uv run --script scripts/gh_aw_engine_version.py` works
without a project sync: a verify job needs no other part of the harness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import yaml
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFINITION = Path(__file__).resolve().parent.parent / 'gh-aw' / 'pydantic.md'
ENGINE_REF = 'gh-aw-engine'


class _Engine(BaseModel):
    model_config = ConfigDict(extra='ignore')

    engine_id: Literal['pydantic-ai'] = Field(alias='id')
    # Not `str | float`: an unquoted `0.21.0` is a YAML float and `0.21` loses a
    # component on the way back to text, so the quoting is part of the contract.
    version: str

    @field_validator('version')
    @classmethod
    def _pep_440(cls, value: str) -> str:
        # gh-aw interpolates this into `pydantic-ai-harness[cli]==<version>`, and the
        # dispatch workflow interpolates it into a PyPI URL. Anything that is not a
        # version is a broken install for consumers, and a value carrying `/` or `?`
        # reaches a different PyPI endpoint than the one the check means to ask about.
        #
        # Rejected rather than normalized, both here and for the surrounding
        # whitespace `Version` would otherwise accept: gh-aw reads the same bytes this
        # file holds, so a guard that checks a cleaned-up copy is checking a string
        # that never ships.
        if value != value.strip():
            raise ValueError('must not be padded with whitespace, which gh-aw would install verbatim')
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ValueError(f'is not a PEP 440 version: {exc}') from exc
        return value


class _Frontmatter(BaseModel):
    model_config = ConfigDict(extra='ignore')

    engine: _Engine


def _frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError(f'{DEFINITION} must open with YAML frontmatter delimited by `---`.')
    closing = next((index for index, line in enumerate(lines[1:], start=1) if line == '---'), None)
    if closing is None:
        raise ValueError(f'{DEFINITION} has unclosed YAML frontmatter.')
    return '\n'.join(lines[1:closing])


def engine_version() -> str:
    """Return `engine.version` once the fields gh-aw and the release jobs read are known good."""
    parsed: object = yaml.safe_load(_frontmatter(DEFINITION.read_text(encoding='utf-8')))
    return _Frontmatter.model_validate(parsed).engine.version


def main() -> int:
    """Print the pinned version, or explain on stderr why the definition cannot be trusted."""
    parser = argparse.ArgumentParser(description='Validate the gh-aw engine definition.')
    parser.add_argument(
        '--expect',
        metavar='VERSION',
        help=f'also require `engine.version` to equal VERSION before `{ENGINE_REF}` may advance',
    )
    expected: str | None = parser.parse_args().expect

    try:
        version = engine_version()
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        print(f'{DEFINITION} is not a usable gh-aw engine definition: {exc}', file=sys.stderr)
        return 1

    if expected is not None and version != expected:
        print(
            f'{DEFINITION} pins `engine.version: {version}` but the release is {expected}, so '
            f'`{ENGINE_REF}` was not advanced and gh-aw keeps serving the definition pinned to '
            f'{version}. The package release itself is unaffected. Bump `engine.version` on main '
            f'and run the `{ENGINE_REF}` dispatch, or pin it before cutting the next tag.',
            file=sys.stderr,
        )
        return 1

    print(version)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""The documented AWS Lambda install must resolve cleanly.

`docs/aws-lambda.md` and `pydantic_ai_harness/aws_lambda/README.md` tell users to
`pip install "pydantic-ai-harness[aws-lambda,bedrock]"`. The `bedrock` extra is a pass-through
to `pydantic-ai-slim[bedrock]` (the `anthropic`/`cli` precedent), so the documented command is
only honest while the manifest defines it and the resolution carries slim's Bedrock boto3
floor. These checks resolve the documented install in an isolated environment, the way a
user's `pip install` builds it. Mirrors tests/experimental/acp/test_packaging.py.
"""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

# The install command both docs surfaces document (issue #844); test_docs_parity.py asserts
# the extras it names are defined in the manifest.
_DOCUMENTED_INSTALL = 'pydantic-ai-harness[aws-lambda,bedrock]'

# pydantic-ai-slim's bedrock extra floors boto3 at 1.42.63. The aws-lambda extra's own
# durable SDK floor (boto3>=1.42.1) does not, so the pass-through is what guarantees this.
_BEDROCK_BOTO3_FLOOR = (1, 42, 63)

_ROOT = Path(__file__).parents[2]


@cache
def _documented_install_resolution() -> subprocess.CompletedProcess[str]:
    """Resolve the documented install once per process in an isolated uv environment."""
    return subprocess.run(
        (
            'uv',
            'run',
            '--isolated',
            # --no-project keeps the repo's uv.lock out of the resolution, like a user's
            # `pip install`: the lowest-versions CI job re-locks `uv.lock` to lowest-direct
            # while pytest still runs under UV_LOCKED=1, so a project-aware run here would
            # be refused the re-lock and exit 2.
            '--no-project',
            '--no-progress',
            '--with',
            f'{_DOCUMENTED_INSTALL} @ file://{_ROOT}',
            'python',
            '-c',
            'import pydantic_ai_harness; from importlib.metadata import version; print(version("boto3"))',
        ),
        capture_output=True,
        check=False,
        cwd=_ROOT,
        text=True,
        timeout=600,
    )


def test_documented_install_command_resolves_without_unknown_extra_warning() -> None:
    resolved = _documented_install_resolution()
    output = resolved.stdout + resolved.stderr
    assert resolved.returncode == 0, output
    # The wordings seen while `bedrock` was missing (issue #844): uv and pip disagree.
    assert 'does not have an extra named' not in output
    assert 'does not provide the extra' not in output


def test_documented_install_resolves_boto3_at_or_above_bedrock_floor() -> None:
    resolved = _documented_install_resolution()
    assert resolved.returncode == 0, resolved.stdout + resolved.stderr
    version = tuple(int(part) for part in resolved.stdout.strip().splitlines()[-1].split('.'))
    assert version >= _BEDROCK_BOTO3_FLOOR, resolved.stdout

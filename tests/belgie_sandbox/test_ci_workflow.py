from __future__ import annotations

from pathlib import Path


def _workflow_lines(name: str = 'main.yml') -> list[str]:
    workflow = Path(__file__).parents[2] / '.github' / 'workflows' / name
    return workflow.read_text().splitlines()


def test_belgie_integration_is_scoped_to_execution_inputs() -> None:
    lines = _workflow_lines()
    changes_start = lines.index('  changes:')
    lint_start = lines.index('  lint:')
    changes_block = lines[changes_start:lint_start]

    assert '      belgie: ${{ steps.detect-belgie.outputs.belgie }}' in changes_block
    assert '      - id: detect-belgie' in changes_block
    for path in (
        'pydantic_ai_harness/belgie_sandbox',
        'tests/belgie_sandbox',
        'tests/conftest.py',
        'docs/belgie-sandbox.md',
        'Makefile',
        'pyproject.toml',
        'uv.lock',
        '.github/workflows/main.yml',
    ):
        assert any(line.strip().removesuffix(' \\') == path for line in changes_block)


def test_belgie_integration_runs_live_tier_and_gates_check() -> None:
    lines = _workflow_lines()
    job_start = lines.index('  belgie-integration:')
    localstack_start = lines.index('  localstack-integration:')
    job_block = lines[job_start:localstack_start]

    assert (
        "    if: ${{ always() && (github.ref_type == 'tag' || needs.changes.outputs.belgie == 'true') }}" in job_block
    )
    assert '      - run: uv sync --locked --group dev --extra belgie' in job_block
    assert '      - run: make integration-belgie' in job_block
    needs = next(line for line in lines if line.strip().startswith('needs: [') and 'coverage' in line)
    assert 'belgie-integration' in needs
    assert any('allowed-skips:' in line and 'belgie-integration' in line for line in lines)


def test_generic_ci_excludes_belgie_extra() -> None:
    workflow_lines = _workflow_lines()
    compat_lines = _workflow_lines('compat-test.yml')

    for line in (*workflow_lines, *compat_lines):
        if 'uv sync' in line and '--all-extras' in line:
            assert '--no-extra belgie' in line
    assert "            extras: '--all-extras --no-extra belgie'" in workflow_lines


def test_live_tests_are_opt_in_and_omitted_from_coverage() -> None:
    root = Path(__file__).parents[2]
    pyproject = (root / 'pyproject.toml').read_text().splitlines()
    makefile = (root / 'Makefile').read_text().splitlines()

    assert "addopts = ['-m', 'not belgie_live']" in pyproject
    assert "omit = ['tests/**/test_belgie_live.py', 'tests/**/test_modal_live.py']" in pyproject
    assert any(line.startswith('integration-belgie:') for line in makefile)
    assert any('pytest -m belgie_live tests/belgie_sandbox/test_belgie_live.py' in line for line in makefile)

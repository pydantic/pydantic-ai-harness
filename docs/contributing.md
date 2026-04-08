# Contributing

## Quick start

```bash
git clone https://github.com/pydantic/pydantic-harness
cd pydantic-harness
uv sync --all-groups
make test
```

## Rules

### Dependencies

**PRs that modify `pyproject.toml` or `uv.lock` are auto-closed by CI** unless you're a team member.

If you need a new dependency, [open an issue](https://github.com/pydantic/pydantic-harness/issues/new) explaining what you need and why. The team will evaluate and add it.

### Code quality

All code must pass:

```bash
make format    # ruff format (line-length=120, single quotes)
make lint      # ruff check (max-complexity=15)
make typecheck # pyright strict (no Any types)
make testcov   # pytest with 100% branch coverage
```

Run these before every commit.

### Testing

- Use `pydantic_ai.models.TestModel` -- no real API calls in tests
- Extend existing test files rather than creating new ones
- Test class naming: `TestCapabilityName`
- Test method naming: `test_<scenario>`

### Style

- Python 3.10+ target
- Single quotes for strings
- No type casting -- use type narrowing
- Docstrings use single backticks (markdown), not RST double backticks
- Comments should explain "why", not "what"

## Proposing a new capability

1. [Open a capability request issue](https://github.com/pydantic/pydantic-harness/issues/new?template=capability-request.yml)
2. Discuss the design in the issue
3. Once approved, implementation can begin (either by the team's AICA or by you)

## AICA workflow

This repository uses an automated development workflow:

1. Issues are labeled with `aica:write-plan` to trigger plan generation
2. The AICA opens a PR with `PLAN.md` for review
3. Humans review the plan, leave comments
4. `aica:update-plan` updates the plan based on feedback
5. `aica:implement-plan` implements the approved plan
6. The ralph loop handles review feedback iteratively

You can participate at any stage by commenting on PRs or issues. The AICA will incorporate your feedback.

## Building your own package

If your capability is too specialized for pydantic-harness, publish it as a standalone package. See the [`template/`](../template/) directory and [the philosophy doc](philosophy.md).

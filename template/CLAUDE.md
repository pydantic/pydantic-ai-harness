# My Capability

## What this is

A Pydantic AI capability package. Capabilities are `AbstractCapability` subclasses that hook into the agent graph via lifecycle hooks.

## Capabilities API reference

- https://ai.pydantic.dev/capabilities/ -- main docs
- https://ai.pydantic.dev/hooks/ -- lifecycle hooks reference
- https://ai.pydantic.dev/extensibility/ -- publishing packages

## Coding standards

- Python 3.10+, pyright strict, ruff (line-length=120, single quotes)
- 100% branch coverage required
- No `Any` types, no type casting
- Use `pydantic_ai.models.TestModel` for tests (no real API calls)

## Commands

```bash
make format     # ruff format
make lint       # ruff check
make typecheck  # pyright strict
make test       # pytest
make testcov    # pytest with coverage
```

Always run `make lint && make typecheck && make test` before committing.

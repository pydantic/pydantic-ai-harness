# Capability Package Template

This template provides everything you need to create a standalone Pydantic AI capability package.

## Setup

1. Copy this directory:
   ```bash
   cp -r template/ my-capability/
   cd my-capability/
   ```

2. Rename the package:
   - Rename `src/my_capability/` to `src/your_capability_name/`
   - Update `pyproject.toml`: change the package name, description, and source directory
   - Update imports in test files

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Implement your capability in `src/your_capability_name/capability.py`

5. Write tests:
   ```bash
   make test
   ```

6. Verify everything passes:
   ```bash
   make format && make lint && make typecheck && make test
   ```

## Publishing

Follow the [Pydantic AI publishing guide](https://ai.pydantic.dev/extensibility/#publishing-capability-packages):

- Use the naming convention `pydantic-ai-<name>` for your package
- Implement `get_serialization_name()` for spec/YAML support
- Implement `from_spec()` if your constructor takes non-serializable arguments

## Structure

```
src/my_capability/
  __init__.py          # public exports
  capability.py        # your AbstractCapability subclass
tests/
  conftest.py          # TestModel fixtures
  test_capability.py   # tests
Makefile               # format, lint, typecheck, test
pyproject.toml         # package configuration
CLAUDE.md              # context for AI assistants
```

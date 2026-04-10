# RepoContextInjection Capability

## Summary

Implements `RepoContextInjection`, an `AbstractCapability` that automatically discovers and injects repository convention files (AGENTS.md, CLAUDE.md, .cursorrules, etc.) into the agent's system prompt via `get_instructions()`.

## Design

### Discovery
- Walks from `root_dir` upward to filesystem root
- Checks each directory for files matching `file_patterns`
- Symlinks resolving to already-discovered files are deduplicated
- Files exceeding `max_total_chars` budget are silently skipped

### Injection
- Uses `get_instructions()` (called once at agent construction) to return a static string
- Each file's content is wrapped with `## Context from {path}`
- Returns `None` when no files are found (no instructions injected)

### Caching
- Discovery runs once on first access, result is cached for the lifetime of the capability instance
- No filesystem re-scanning on subsequent model requests

### Configuration
- `root_dir: str | Path` — starting directory (required)
- `file_patterns: tuple[str, ...]` — file names to search for (defaults to AGENTS.md, CLAUDE.md, .cursorrules, .github/copilot-instructions.md, CONVENTIONS.md, CODING_GUIDELINES.md)
- `max_total_chars: int` — limit on total injected context (default 100,000)

## Files

- `src/pydantic_harness/repo_context_injection.py` — capability implementation
- `src/pydantic_harness/__init__.py` — public export
- `tests/test_repo_context_injection.py` — 25 tests, 100% coverage

## References

- Issue: #64
- Prior art: Claude Code (CLAUDE.md), Cursor (.cursorrules), Open SWE (AGENTS.md)

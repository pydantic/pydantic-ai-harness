# pyright: reportPrivateUsage=false
"""Tests for pydantic_harness.repo_context_injection capability."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_harness.repo_context_injection import (
    DEFAULT_FILE_PATTERNS,
    RepoContextInjection,
    _discover_files,
    _DiscoveredFile,
    _format_context,
)

# ---------------------------------------------------------------------------
# _discover_files
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    def test_finds_agents_md(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('# Rules\nBe helpful.')
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        assert len(files) == 1
        assert files[0].content == '# Rules\nBe helpful.'

    def test_finds_multiple_patterns(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('agents content')
        (tmp_path / '.cursorrules').write_text('cursor content')
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        assert len(files) == 2
        contents = {f.content for f in files}
        assert contents == {'agents content', 'cursor content'}

    def test_finds_github_copilot_instructions(self, tmp_path: Path) -> None:
        github_dir = tmp_path / '.github'
        github_dir.mkdir()
        (github_dir / 'copilot-instructions.md').write_text('copilot rules')
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        assert len(files) == 1
        assert files[0].content == 'copilot rules'

    def test_walks_up_parent_directories(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('root context')
        child = tmp_path / 'subdir'
        child.mkdir()
        files = _discover_files(child, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        found_root = any(f.content == 'root context' for f in files)
        assert found_root

    def test_deduplicates_symlinks(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('the content')
        (tmp_path / 'CLAUDE.md').symlink_to(tmp_path / 'AGENTS.md')
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        assert len(files) == 1
        assert files[0].content == 'the content'

    def test_respects_max_total_chars(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('a' * 60)
        (tmp_path / 'CLAUDE.md').write_text('b' * 60)
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=80)
        # Only the first matching file should fit within the budget.
        assert len(files) == 1
        assert files[0].content == 'a' * 60

    def test_empty_directory(self, tmp_path: Path) -> None:
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        assert files == ()

    def test_skips_directories_with_matching_name(self, tmp_path: Path) -> None:
        # A directory named AGENTS.md should not be treated as a file.
        (tmp_path / 'AGENTS.md').mkdir()
        files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
        assert files == ()

    def test_custom_patterns(self, tmp_path: Path) -> None:
        (tmp_path / '.custom-rules').write_text('custom')
        files = _discover_files(tmp_path, ('.custom-rules',), max_total_chars=100_000)
        assert len(files) == 1
        assert files[0].content == 'custom'

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / 'AGENTS.md'
        f.write_text('content')
        f.chmod(0o000)
        try:
            files = _discover_files(tmp_path, DEFAULT_FILE_PATTERNS, max_total_chars=100_000)
            assert files == ()
        finally:
            f.chmod(0o644)


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_single_file(self) -> None:
        files = (_DiscoveredFile(path=Path('/repo/AGENTS.md'), content='Be helpful.'),)
        result = _format_context(files)
        assert result == '## Context from /repo/AGENTS.md\n\nBe helpful.'

    def test_multiple_files(self) -> None:
        files = (
            _DiscoveredFile(path=Path('/repo/AGENTS.md'), content='agents'),
            _DiscoveredFile(path=Path('/repo/.cursorrules'), content='cursor'),
        )
        result = _format_context(files)
        assert '## Context from /repo/AGENTS.md' in result
        assert '## Context from /repo/.cursorrules' in result
        assert 'agents' in result
        assert 'cursor' in result

    def test_empty(self) -> None:
        assert _format_context(()) == ''


# ---------------------------------------------------------------------------
# RepoContextInjection
# ---------------------------------------------------------------------------


class TestRepoContextInjection:
    def test_get_instructions_returns_context(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('# Guidelines\nFollow them.')
        cap: RepoContextInjection[None] = RepoContextInjection(root_dir=tmp_path)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert '# Guidelines' in instructions
        assert 'Follow them.' in instructions

    def test_get_instructions_returns_none_when_empty(self, tmp_path: Path) -> None:
        cap: RepoContextInjection[None] = RepoContextInjection(root_dir=tmp_path)
        assert cap.get_instructions() is None

    def test_caches_after_first_call(self, tmp_path: Path) -> None:
        f = tmp_path / 'AGENTS.md'
        f.write_text('original')
        cap: RepoContextInjection[None] = RepoContextInjection(root_dir=tmp_path)
        first = cap.get_instructions()
        f.write_text('modified')
        second = cap.get_instructions()
        assert first == second
        assert isinstance(first, str)
        assert 'original' in first

    def test_accepts_string_root_dir(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('content')
        cap: RepoContextInjection[None] = RepoContextInjection(root_dir=str(tmp_path))
        assert cap.get_instructions() is not None

    def test_custom_file_patterns(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('should be ignored')
        (tmp_path / '.my-rules').write_text('custom rules')
        cap: RepoContextInjection[None] = RepoContextInjection(
            root_dir=tmp_path,
            file_patterns=('.my-rules',),
        )
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'custom rules' in instructions
        assert 'should be ignored' not in instructions

    def test_custom_max_total_chars(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('x' * 200)
        cap: RepoContextInjection[None] = RepoContextInjection(root_dir=tmp_path, max_total_chars=50)
        # The file exceeds the limit, so nothing should be included.
        assert cap.get_instructions() is None

    def test_validation_empty_patterns(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='file_patterns must not be empty'):
            RepoContextInjection(root_dir=tmp_path, file_patterns=())

    def test_validation_non_positive_max_chars(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='max_total_chars must be positive'):
            RepoContextInjection(root_dir=tmp_path, max_total_chars=0)

    def test_symlink_deduplication(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('the real content')
        (tmp_path / 'CLAUDE.md').symlink_to(tmp_path / 'AGENTS.md')
        cap: RepoContextInjection[None] = RepoContextInjection(root_dir=tmp_path)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        # Content should appear exactly once.
        assert instructions.count('the real content') == 1

    def test_from_spec(self, tmp_path: Path) -> None:
        (tmp_path / 'AGENTS.md').write_text('spec content')
        cap = RepoContextInjection.from_spec(root_dir=tmp_path)
        assert cap.get_instructions() is not None

    def test_default_file_patterns_constant(self) -> None:
        assert 'AGENTS.md' in DEFAULT_FILE_PATTERNS
        assert 'CLAUDE.md' in DEFAULT_FILE_PATTERNS
        assert '.cursorrules' in DEFAULT_FILE_PATTERNS
        assert '.github/copilot-instructions.md' in DEFAULT_FILE_PATTERNS
        assert 'CONVENTIONS.md' in DEFAULT_FILE_PATTERNS
        assert 'CODING_GUIDELINES.md' in DEFAULT_FILE_PATTERNS


# ---------------------------------------------------------------------------
# Public API import
# ---------------------------------------------------------------------------


class TestPublicImport:
    def test_importable_from_package(self) -> None:
        from pydantic_harness import RepoContextInjection as Imported

        assert Imported is RepoContextInjection

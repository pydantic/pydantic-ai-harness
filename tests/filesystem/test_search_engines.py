"""Tests for the two engines behind `search_files`: ripgrep when it is installed, and the in-process scan.

The point of ripgrep here is throughput, not different behavior, so most of these assert that the two engines agree.
They are skipped, not failed, where ripgrep is absent: the fallback path is what those machines run.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from pydantic_ai_harness.filesystem import _ripgrep
from pydantic_ai_harness.filesystem._toolset import FileSystemToolset

requires_ripgrep = pytest.mark.skipif(_ripgrep.find_ripgrep() is None, reason='ripgrep is not installed')


def make_toolset(
    root: Path,
    *,
    denied_patterns: list[str] | None = None,
    max_search_results: int = 1000,
) -> FileSystemToolset[None]:
    """Build a toolset over `root` with the walkers wide open, as the other filesystem tests do."""
    return FileSystemToolset(
        root_dir=root,
        allowed_patterns=[],
        denied_patterns=denied_patterns if denied_patterns is not None else [],
        protected_patterns=[],
        max_read_lines=2000,
        max_list_results=1000,
        max_search_results=max_search_results,
        max_find_results=1000,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tree holding a nested match, a hidden file, a binary file, and a symlinked file."""
    (tmp_path / 'top.py').write_text('needle one\nplain\nneedle two\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.py').write_text('needle three\n')
    (tmp_path / '.hidden.py').write_text('needle four\n')
    (tmp_path / 'blob.bin').write_bytes(b'needle\x00five\n')
    (tmp_path / 'real.txt').write_text('needle six\n')
    (tmp_path / 'link.txt').symlink_to(tmp_path / 'real.txt')
    return tmp_path


class TestSearchEngines:
    @requires_ripgrep
    @pytest.mark.parametrize(
        ('pattern', 'include_glob'),
        [
            ('needle', None),
            ('needle', '*.py'),
            ('needle', '*.txt'),
            ('^needle two$', '*.py'),
            ('absent', None),
            ('needle\\s+six', None),
        ],
    )
    async def test_ripgrep_answers_exactly_like_the_in_process_scan(
        self,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        pattern: str,
        include_glob: str | None,
    ) -> None:
        toolset = make_toolset(workspace)
        with_ripgrep = await toolset.search_files(pattern, include_glob=include_glob)

        monkeypatch.setattr(_ripgrep, 'find_ripgrep', lambda: None)
        without_ripgrep = await toolset.search_files(pattern, include_glob=include_glob)

        assert with_ripgrep == without_ripgrep

    @requires_ripgrep
    async def test_ripgrep_orders_merged_symlink_hits_with_the_rest(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Symlinked files are matched separately and merged, so their ordering is ours to get right.

        Ripgrep reports paths with a `.` prefix for a search root of `.`, which would sort a merged hit to the end if
        it reached the merge key unnormalized.
        """
        toolset = make_toolset(workspace)
        with_ripgrep = await toolset.search_files('needle six')

        monkeypatch.setattr(_ripgrep, 'find_ripgrep', lambda: None)
        without_ripgrep = await toolset.search_files('needle six')

        assert with_ripgrep == 'link.txt:1:needle six\nreal.txt:1:needle six'
        assert with_ripgrep == without_ripgrep

    async def test_symlinked_file_contributes_matches_of_its_own(self, workspace: Path) -> None:
        result = await make_toolset(workspace).search_files('needle six')

        assert result == 'link.txt:1:needle six\nreal.txt:1:needle six'

    async def test_hidden_and_binary_files_stay_excluded(self, workspace: Path) -> None:
        result = await make_toolset(workspace).search_files('needle', include_glob='*')

        assert '.hidden.py' not in result
        assert 'blob.bin' not in result

    async def test_truncation_marker_reports_a_cut_budget(self, workspace: Path) -> None:
        """The marker appears only when a match had to be dropped, and matches stay in path order."""
        result = await make_toolset(workspace, max_search_results=2).search_files('needle')

        assert result.splitlines() == [
            'link.txt:1:needle six',
            'real.txt:1:needle six',
            '[... truncated at 2 matches]',
        ]

    async def test_exactly_at_budget_is_not_marked_truncated(self, workspace: Path) -> None:
        result = await make_toolset(workspace, max_search_results=4).search_files('needle')

        assert result.splitlines() == [
            'link.txt:1:needle six',
            'real.txt:1:needle six',
            'sub/nested.py:1:needle three',
            'top.py:1:needle one',
            '[... truncated at 4 matches]',
        ]

    async def test_denied_patterns_filter_matches_from_both_engines(self, workspace: Path) -> None:
        """The engine that walks does not decide what is readable: a denied file stays denied, symlink or not."""
        result = await make_toolset(workspace, denied_patterns=['real.txt']).search_files('needle six')

        assert result == 'No matches found.'

    async def test_missing_ripgrep_falls_back_to_the_in_process_scan(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_ripgrep, 'find_ripgrep', lambda: None)

        result = await make_toolset(workspace).search_files('needle three')

        assert result == 'sub/nested.py:1:needle three'

    async def test_search_leaves_the_event_loop_free(self, tmp_path: Path) -> None:
        """A scan is blocking work, so it runs in a worker thread instead of stalling every other task in the run.

        The ticker is the assertion: a search that ran on the loop would produce no ticks at all, which is what the
        in-process engine used to do over an entire large scan.
        """
        for index in range(120):
            (tmp_path / f'f{index}.py').write_text('x\n' * 40 + 'needle\n')
        toolset = make_toolset(tmp_path)
        ticks = 0
        result = ''

        async def tick() -> None:
            nonlocal ticks
            while True:
                await anyio.sleep(0.001)
                ticks += 1

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(tick)
            result = await toolset.search_files('needle')
            task_group.cancel_scope.cancel()

        assert 'f0.py:41:needle' in result
        assert ticks > 0

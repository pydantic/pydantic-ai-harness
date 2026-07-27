"""Shadow-git layer for file-level checkpoints.

`CheckpointStore` drives an ordinary `git` binary against a repository that lives
outside the project (`GIT_DIR`) while pointing its work tree at the project root
(`GIT_WORK_TREE`). Snapshots are commits in that shadow repository, so restoring
is a `git checkout` from it. The user's own `.git` is never read or written: the
shadow repo has its own `GIT_DIR`, its own committer identity, gpg signing off,
and no hooks. It works in projects that are not git repositories at all, because
the shadow repo is `git init`-ed on first use.

`.gitignore` semantics come for free: because `GIT_WORK_TREE` is the project root,
`git add -A` reads the project's own ignore files from the work tree, so ignored
paths (`node_modules/`, build output, ...) stay out of snapshots. Git also refuses
to add a nested `.git` directory, so the user's repository metadata is never
snapshotted; `info/exclude` in the shadow repo lists `.git/` as a belt-and-braces
second guard.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ['Checkpoint', 'CheckpointError', 'CheckpointStore']

# Passed to every git invocation so the shadow repo does not inherit anything from
# the user's environment that would change its behavior or touch their setup.
_CONFIG_FLAGS: tuple[str, ...] = (
    '-c',
    'commit.gpgsign=false',
    '-c',
    f'core.hooksPath={os.devnull}',
    '-c',
    'gc.auto=0',
    '-c',
    'core.autocrlf=false',
    '-c',
    'core.quotepath=false',
    '-c',
    'safe.directory=*',
)

_TOOL_TRAILER = 'Checkpoint-Tool:'
_RUN_TRAILER = 'Checkpoint-Run:'


class CheckpointError(RuntimeError):
    """A shadow-git operation failed (git missing, a command errored, an unknown id)."""


@dataclass(frozen=True)
class Checkpoint:
    """A single file snapshot, identified by its shadow-repo commit."""

    id: str
    """Short commit hash in the shadow repo. Pass it to `restore`."""

    time: datetime
    """Commit time of the snapshot (timezone-aware)."""

    tool_name: str | None
    """The mutating tool this checkpoint was taken *before*, or `None` for a manual snapshot."""

    files_changed: list[str]
    """Project-relative paths that changed since the previous checkpoint.

    For the first checkpoint this is every file the snapshot captured. Note this
    reflects the *previous* tool's effect (what accumulated in the work tree since
    the last snapshot), not what `tool_name` is about to do.
    """


def _project_slug(project_root: Path) -> str:
    """Stable per-project directory name: readable prefix plus a hash of the absolute path."""
    digest = hashlib.sha256(str(project_root).encode('utf-8')).hexdigest()[:12]
    name = re.sub(r'[^A-Za-z0-9._-]', '-', project_root.name)
    return f'{name}-{digest}'


def shadow_dir_for(project_root: Path, state_dir: Path) -> Path:
    """Return the shadow `GIT_DIR` for a project under a state directory."""
    return state_dir / 'checkpoints' / _project_slug(project_root)


@dataclass
class CheckpointStore:
    """Snapshot and restore a project's files through a shadow git repository.

    The store is stateless beyond its configuration -- every method reads the
    current state from the shadow repo on disk -- so it is cheap to build one per
    call and safe to use from several `Agent.run` calls against the same project.
    Concurrent runs share the shadow repo with last-writer-wins semantics: each
    snapshot commits the whole work tree as it looks at that instant.
    """

    project_root: Path
    """Absolute path to the project whose files are snapshotted (the git work tree)."""

    shadow_dir: Path
    """Absolute path to the shadow `GIT_DIR` (created on first use)."""

    committer_name: str = 'pydantic-ai-harness checkpoints'
    committer_email: str = 'noreply@pydantic.dev'

    _env: dict[str, str] = field(default_factory=dict[str, str], init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.shadow_dir = Path(self.shadow_dir).resolve()
        env = os.environ.copy()
        env.update(
            GIT_DIR=str(self.shadow_dir),
            GIT_WORK_TREE=str(self.project_root),
            GIT_AUTHOR_NAME=self.committer_name,
            GIT_AUTHOR_EMAIL=self.committer_email,
            GIT_COMMITTER_NAME=self.committer_name,
            GIT_COMMITTER_EMAIL=self.committer_email,
            # Isolate from the user's git config so their gpgsign / hooks / identity
            # never leak into shadow commits.
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
        )
        self._env = env

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ('git', *_CONFIG_FLAGS, *args),
            env=self._env,
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        if check and result.returncode != 0:
            joined = ' '.join(args)
            raise CheckpointError(f'shadow git `{joined}` failed: {result.stderr.strip() or result.stdout.strip()}')
        return result

    def ensure_initialized(self) -> None:
        """Create the shadow repo on first use. Idempotent."""
        if shutil.which('git') is None:  # pragma: no cover - environment without git
            raise CheckpointError('git executable not found; checkpoints require a `git` binary on PATH')
        if (self.shadow_dir / 'HEAD').exists():
            return
        try:
            self.shadow_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CheckpointError(f'could not create shadow repo at {self.shadow_dir}: {exc}') from exc
        self._git('init', '-q', str(self.shadow_dir))
        info_dir = self.shadow_dir / 'info'
        info_dir.mkdir(exist_ok=True)
        (info_dir / 'exclude').write_text('.git/\n', encoding='utf-8')

    def _has_head(self) -> bool:
        return self._git('rev-parse', '--verify', '--quiet', 'HEAD', check=False).returncode == 0

    def _index_matches_head(self) -> bool:
        # rc 0 == no staged difference from HEAD, i.e. the work tree is unchanged
        # since the last checkpoint, so a new snapshot would be an empty duplicate.
        return self._git('diff', '--cached', '--quiet', check=False).returncode == 0

    def snapshot(self, *, tool_name: str | None = None, run_id: str | None = None) -> Checkpoint:
        """Commit the current work tree as a checkpoint, or reuse the last one if nothing changed.

        Stages every non-ignored path (`git add -A`) and commits it. If the work
        tree is byte-identical to the most recent checkpoint, no commit is made and
        that checkpoint is returned instead (debounce -- no empty duplicates).
        """
        self.ensure_initialized()
        self._git('add', '-A')
        has_head = self._has_head()
        if has_head and self._index_matches_head():
            head = self.head()
            assert head is not None  # has_head is True
            return head
        args = ['commit', '--quiet', '--no-verify', '-m', _format_message(tool_name, run_id)]
        if not has_head:
            # First snapshot of an empty project: keep a baseline to restore to.
            args.append('--allow-empty')
        self._git(*args)
        head = self.head()
        assert head is not None  # just committed
        return head

    def head(self) -> Checkpoint | None:
        """The most recent checkpoint, or `None` if none has been taken."""
        if not self._has_head():
            return None
        sha = self._git('rev-parse', 'HEAD').stdout.strip()
        return self._checkpoint(sha)

    def list_checkpoints(self) -> list[Checkpoint]:
        """All checkpoints, oldest first."""
        if not self._has_head():
            return []
        shas = self._git('rev-list', '--reverse', 'HEAD').stdout.split()
        return [self._checkpoint(sha) for sha in shas]

    def restore(self, checkpoint_id: str, *, paths: Sequence[str] | None = None) -> None:
        """Restore files from a checkpoint into the work tree.

        With `paths=None` every path recorded in the checkpoint is restored;
        otherwise only the given project-relative paths. This overwrites files that
        existed at the checkpoint and re-creates ones deleted since; it does not
        remove files created after the checkpoint (a plain `git checkout`).
        """
        self.ensure_initialized()
        resolved = self._git('rev-parse', '--verify', '--quiet', f'{checkpoint_id}^{{commit}}', check=False)
        if resolved.returncode != 0:
            raise CheckpointError(f'unknown checkpoint id: {checkpoint_id!r}')
        sha = resolved.stdout.strip()
        targets = list(paths) if paths is not None else ['.']
        self._git('checkout', sha, '--', *targets)

    def _checkpoint(self, sha: str) -> Checkpoint:
        # `%ct` is the committer date as a Unix timestamp: parsing it is independent of
        # the git version's ISO rendering (some emit a `Z` suffix that Python < 3.11 rejects).
        info = self._git('show', '-s', '--format=%h%n%ct%n%B', sha).stdout
        short, timestamp, body = info.split('\n', 2)
        # `--root` makes the first (parentless) checkpoint list every file it captured
        # instead of an empty diff.
        files = self._git('diff-tree', '--root', '--no-commit-id', '--name-only', '-r', sha).stdout.split('\n')
        return Checkpoint(
            id=short.strip(),
            time=datetime.fromtimestamp(int(timestamp.strip()), tz=timezone.utc),
            tool_name=_parse_trailer(body, _TOOL_TRAILER),
            files_changed=[f for f in (line.strip() for line in files) if f],
        )


def _format_message(tool_name: str | None, run_id: str | None) -> str:
    subject = f'checkpoint before {tool_name}' if tool_name else 'checkpoint'
    lines = [subject, '']
    if tool_name:
        lines.append(f'{_TOOL_TRAILER} {tool_name}')
    if run_id:
        lines.append(f'{_RUN_TRAILER} {run_id}')
    return '\n'.join(lines)


def _parse_trailer(body: str, key: str) -> str | None:
    for line in body.splitlines():
        if line.startswith(key):
            return line[len(key) :].strip()
    return None

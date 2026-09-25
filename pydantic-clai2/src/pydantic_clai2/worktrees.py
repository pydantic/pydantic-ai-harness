"""Git worktrees for isolated CLI workspaces."""

import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def create_worktree(*, name: str) -> Path:
    """Create a checkout in `.worktrees` on a new `clai/<name>` branch."""
    name = name or f'worktree-{uuid4().hex[:8]}'
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) is None:
        raise ValueError('Worktree names must start with a letter or digit and contain only letters, digits, - or _.')
    try:
        root = Path(_git('rev-parse', '--show-toplevel'))
        exclude = Path(_git('rev-parse', '--git-path', 'info/exclude'))
        contents = exclude.read_bytes() if exclude.exists() else b''
        path = root / '.worktrees' / name
        branch = f'clai/{name}'
        _git('branch', branch, 'HEAD')
        try:
            _git('worktree', 'add', '--', str(path), branch)
        except (OSError, subprocess.CalledProcessError) as exc:
            try:
                _git('branch', '-d', '--', branch)
            except (OSError, subprocess.CalledProcessError) as cleanup:
                raise ValueError(
                    f'Cannot create worktree at {path}: {exc}. Branch {branch} could not be removed: {cleanup}'
                ) from cleanup
            raise
        if b'/.worktrees/' not in contents.splitlines():
            try:
                exclude.parent.mkdir(parents=True, exist_ok=True)
                with exclude.open('ab') as file:
                    file.write(b'\n/.worktrees/\n')
            except OSError as exc:
                raise ValueError(f'Worktree kept at {path}, but could not update Git excludes: {exc}') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'Cannot create worktree: {exc.stderr.strip()}') from exc
    except OSError as exc:
        raise ValueError(f'Cannot create worktree: {exc}') from exc
    return path


def offer_worktree_cleanup(created: Path | None = None) -> None:
    """Clean up a linked worktree after interactive shutdown without discarding work.

    A worktree created by this run (`created`) that is left without changes or new commits is removed
    with its branch, without asking. Any other linked worktree is only removed on confirmation, and
    its branch is kept.
    """
    if not sys.stdin.isatty():
        return
    try:
        root = Path(_git('rev-parse', '--show-toplevel')).resolve()
        common = Path(_git('rev-parse', '--git-common-dir')).resolve()
        if Path(_git('rev-parse', '--absolute-git-dir')).resolve() == common:
            return
        disposable = _disposable_branch(root, created)
    except (OSError, subprocess.CalledProcessError):
        return
    if disposable is not None:
        if _remove(root, common):
            try:
                # `-D` because `_disposable_branch` proved every commit is reachable from another ref,
                # while `-d` would also demand a merge into the main checkout's current branch.
                _git('-C', str(common), 'branch', '-D', '--', disposable)
            except (OSError, subprocess.CalledProcessError) as exc:
                print(f'Removed worktree {root}. Branch {disposable} kept: {_detail(exc)}', file=sys.stderr)
            else:
                print(f'Removed unchanged worktree {root} and branch {disposable}.')
        return
    try:
        answer = input(f'Remove worktree {root}? The branch will be kept. [y/N] ')
    except (EOFError, KeyboardInterrupt):
        answer = ''
        print()
    if answer.strip().lower() not in ('y', 'yes'):
        print(f'Worktree kept at {root}.')
    elif _remove(root, common):
        print(f'Removed worktree {root}. Branch kept.')


def _disposable_branch(root: Path, created: Path | None) -> str | None:
    """Return the branch of the worktree this run created if deleting it loses nothing, else `None`."""
    if created is None or created.resolve() != root:
        return None
    branch = f'clai/{created.name}'
    if _git('branch', '--show-current') != branch or _git('status', '--porcelain', '--untracked-files=all'):
        return None
    # Commits reachable from HEAD but from no other branch, tag, or remote would be lost with the branch.
    unique = _git('rev-list', '-n1', 'HEAD', '--not', f'--exclude={branch}', '--branches', '--tags', '--remotes')
    return None if unique else branch


def _remove(root: Path, common: Path) -> bool:
    original = Path.cwd()
    try:
        # Run outside the checkout so successful removal leaves a valid working directory.
        os.chdir(common.parent)
        _git('-C', str(common), 'worktree', 'remove', '--', str(root))
    except (OSError, subprocess.CalledProcessError) as exc:
        os.chdir(original)
        print(f'Worktree kept at {root}: {_detail(exc)}', file=sys.stderr)
        return False
    return True


def _detail(exc: OSError | subprocess.CalledProcessError) -> str:
    return exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.removesuffix('\n')

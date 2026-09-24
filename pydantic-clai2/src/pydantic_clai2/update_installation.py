"""Conservative update advice; no installers or subprocesses are invoked."""

import os
import shlex
import subprocess
import sys
from importlib.metadata import Distribution
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError


def update_guidance(installed: Distribution, *, prefix: Path | None = None) -> str:
    """Prefer source metadata and tool receipts over guessing from the launch directory."""
    prefix = prefix if prefix is not None else Path(sys.prefix)
    try:
        if installed.read_text('direct_url.json'):
            return 'Source/direct install: update from your original source, then restart CLAI2.'
        if (prefix / 'uv-receipt.toml').is_file() and prefix.name == 'pydantic-clai2':
            return 'Run: uv tool upgrade pydantic-clai2'
        pipx = prefix / 'pipx_metadata.json'
        if pipx.is_file():
            data = TypeAdapter(dict[str, JsonValue]).validate_json(pipx.read_bytes())
            main = data.get('main_package')
            if isinstance(main, dict) and main.get('package') == 'pydantic-clai2':
                return f'Run: pipx upgrade {_shell_command([prefix.name])}'
            return 'Injected pipx install: update pydantic-clai2 with your original pipx inject command.'
        installer = (installed.read_text('INSTALLER') or '').strip()
        if installer == 'pip':
            return f'Run: {_shell_command([sys.executable, "-m", "pip", "install", "--upgrade", "pydantic-clai2"])}'
        if installer == 'uv':
            return (
                'If using uvx: uvx --upgrade --from pydantic-clai2 clai2. '
                'Otherwise update pydantic-clai2 in its original uv project/environment.'
            )
    except (OSError, ValidationError):
        pass
    return 'Update pydantic-clai2 with your original installer, then restart CLAI2.'


def _shell_command(args: list[str]) -> str:
    return subprocess.list2cmdline(args) if os.name == 'nt' else shlex.join(args)

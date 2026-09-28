"""Keep Logfire write credentials in the same OS store as model credentials."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .credential_store import load_codex_credentials, save_codex_credentials


class LogfireCredentials(BaseModel):
    """The SDK credential fields needed to route telemetry to the selected project."""

    model_config = ConfigDict(hide_input_in_errors=True)
    token: str = Field(min_length=1, repr=False)
    logfire_api_url: HttpUrl


def logfire_directory() -> Path:
    """Resolve only an absolute user config path, never a checkout-relative one."""
    root = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
    if not root.is_absolute():
        root = Path.home() / '.config'
    return root / 'pydantic-clai2' / 'logfire'


def load_logfire_credentials() -> LogfireCredentials | None:
    """Migrate legacy SDK credentials only after the credential store retains them."""
    directory = logfire_directory()
    value = load_codex_credentials(account='logfire', fallback=directory.parent / 'credentials-logfire.json')
    if value is not None:
        return LogfireCredentials.model_validate_json(value)
    legacy = directory / 'logfire_credentials.json'
    if not legacy.is_file():
        return None
    credentials = LogfireCredentials.model_validate_json(legacy.read_text(encoding='utf-8'))
    save_logfire_credentials(credentials)
    legacy.unlink()
    return credentials


def save_logfire_credentials(credentials: LogfireCredentials) -> None:
    """Use keyring, or the existing private-file fallback when no backend exists."""
    save_codex_credentials(
        account='logfire',
        value=credentials.model_dump_json(),
        fallback=logfire_directory().parent / 'credentials-logfire.json',
    )

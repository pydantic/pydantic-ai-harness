"""Events emitted by the filesystem capability.

Each event carries two path fields inside the active workspace:

- `path`: normalized and relative to the emitting filesystem's root, never an
  absolute host path, so it is safe to echo to the model or a UI.
- `root_dir`: the absolute workspace root the `path` is relative to. Consumers
  that share the workspace can resolve the location through `ctx.workspace`;
  it is not necessarily a host filesystem path.
"""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

FILE_SYSTEM_EVENTS = 'file_system'


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    """A text file was read successfully."""

    path: str
    root_dir: str
    content_hash: str | None
    """Whole-file hash when the complete text was read; otherwise `None`."""


@dataclass(kw_only=True)
class DirectoryListedEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    """A directory was listed successfully."""

    path: str
    root_dir: str
    entry_count: int


@dataclass(kw_only=True)
class FileWrittenEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    """A text file was written or edited successfully."""

    path: str
    root_dir: str
    content_hash: str

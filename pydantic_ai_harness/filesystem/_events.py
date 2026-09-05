"""Events emitted by the filesystem capability."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

FILE_SYSTEM_EVENTS = 'file_system'


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    """A text file was read successfully."""

    path: str
    content_hash: str


@dataclass(kw_only=True)
class DirectoryListedEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    """A directory was listed successfully."""

    path: str
    entry_count: int


@dataclass(kw_only=True)
class FileWrittenEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    """A text file was written or edited successfully."""

    path: str
    content_hash: str

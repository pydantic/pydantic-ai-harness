from __future__ import annotations

import stat
from pathlib import Path

import anyio
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pydantic_ai_harness.slack import FileConversationStore, InMemoryConversationStore

pytestmark = pytest.mark.anyio


def _history() -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest(parts=[UserPromptPart(content='ship it')]),
        ModelResponse(parts=[TextPart(content='shipped')]),
    ]


class TestInMemoryConversationStore:
    async def test_round_trips_history(self) -> None:
        store = InMemoryConversationStore()
        assert list(await store.load('k')) == []
        await store.save('k', _history())
        assert len(await store.load('k')) == 2

    async def test_load_returns_a_copy(self) -> None:
        store = InMemoryConversationStore()
        await store.save('k', _history())
        loaded = await store.load('k')
        assert isinstance(loaded, list)
        loaded.clear()
        assert len(await store.load('k')) == 2


class TestFileConversationStore:
    async def test_survives_a_new_store_over_the_same_directory(self, tmp_path: Path) -> None:
        await FileConversationStore(tmp_path).save('T1:C1:1.1', _history())
        reopened = await FileConversationStore(tmp_path).load('T1:C1:1.1')
        assert [type(message).__name__ for message in reopened] == ['ModelRequest', 'ModelResponse']

    async def test_unknown_key_loads_empty(self, tmp_path: Path) -> None:
        assert list(await FileConversationStore(tmp_path).load('missing')) == []

    async def test_loading_from_an_absent_directory_does_not_create_it(self, tmp_path: Path) -> None:
        directory = tmp_path / 'not-created'
        assert list(await FileConversationStore(directory).load('missing')) == []
        assert not directory.exists()

    async def test_creates_the_directory_on_first_save(self, tmp_path: Path) -> None:
        store = FileConversationStore(tmp_path / 'nested' / 'deeper')
        await store.save('k', _history())
        assert len(await store.load('k')) == 2

    async def test_save_replaces_rather_than_appends(self, tmp_path: Path) -> None:
        store = FileConversationStore(tmp_path)
        await store.save('k', _history())
        await store.save('k', _history()[:1])
        assert len(await store.load('k')) == 1

    async def test_saves_leave_no_temporary_file_behind(self, tmp_path: Path) -> None:
        # Two saves for one key must not share a temporary name, or one can rename
        # the other's half-written bytes into place.
        store = FileConversationStore(tmp_path)
        await store.save('k', _history())
        await store.save('k', _history())
        assert sorted(path.suffix for path in tmp_path.iterdir()) == ['.json']

    async def test_keys_with_path_characters_do_not_escape_the_directory(self, tmp_path: Path) -> None:
        store = FileConversationStore(tmp_path)
        await store.save('../../etc/passwd', _history())
        assert [path.parent for path in tmp_path.rglob('*.json')] == [tmp_path]

    async def test_history_is_readable_only_by_its_owner(self, tmp_path: Path) -> None:
        # These files hold whole conversations, so a umask of 022 leaving them
        # world-readable in a home directory is not good enough.
        directory = tmp_path / 'history'
        await FileConversationStore(directory).save('k', _history())
        saved = next(directory.iterdir())
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600

    async def test_rejects_an_existing_directory_accessible_to_group_or_others(self, tmp_path: Path) -> None:
        directory = tmp_path / 'history'
        directory.mkdir(mode=0o777)
        directory.chmod(0o777)
        with pytest.raises(PermissionError, match='accessible by group or others'):
            await FileConversationStore(directory).save('k', _history())
        assert stat.S_IMODE(directory.stat().st_mode) == 0o777

    async def test_rejects_an_existing_directory_before_reading(self, tmp_path: Path) -> None:
        directory = tmp_path / 'history'
        directory.mkdir()
        directory.chmod(0o777)
        with pytest.raises(PermissionError, match='accessible by group or others'):
            await FileConversationStore(directory).load('k')
        assert stat.S_IMODE(directory.stat().st_mode) == 0o777

    async def test_a_failed_write_leaves_no_transcript_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def explode(self: anyio.Path, data: bytes) -> int:
            raise OSError('no space left on device')

        monkeypatch.setattr(anyio.Path, 'write_bytes', explode)
        with pytest.raises(OSError, match='no space left'):
            await FileConversationStore(tmp_path).save('k', _history())
        assert list(tmp_path.iterdir()) == []

    async def test_a_failed_replacement_preserves_previous_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileConversationStore(tmp_path)
        await store.save('k', _history())

        async def explode(self: anyio.Path, target: Path) -> None:
            raise OSError('replace failed')

        monkeypatch.setattr(anyio.Path, 'replace', explode)
        with pytest.raises(OSError, match='replace failed'):
            await store.save('k', _history()[:1])
        assert len(await store.load('k')) == 2
        assert sorted(path.suffix for path in tmp_path.iterdir()) == ['.json']

    async def test_cancellation_after_write_cleans_temp_and_preserves_previous_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileConversationStore(tmp_path)
        await store.save('k', _history())
        original_write = anyio.Path.write_bytes

        async def write_then_cancel(self: anyio.Path, data: bytes) -> int:
            result = await original_write(self, data)
            cancel_scope.cancel()
            return result

        monkeypatch.setattr(anyio.Path, 'write_bytes', write_then_cancel)
        with anyio.CancelScope() as cancel_scope:
            await store.save('k', _history()[:1])
        assert len(await store.load('k')) == 2
        assert sorted(path.suffix for path in tmp_path.iterdir()) == ['.json']

    async def test_expands_a_user_relative_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        store = FileConversationStore('~/agent-history')
        await store.save('k', _history())
        assert (tmp_path / 'agent-history').is_dir()

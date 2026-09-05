from __future__ import annotations

import stat
from pathlib import Path

import anyio
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pydantic_ai_harness.slack import FileConversationStore, InMemoryConversationStore, SlackThread

pytestmark = pytest.mark.anyio


def _history() -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest(parts=[UserPromptPart(content='ship it')]),
        ModelResponse(parts=[TextPart(content='shipped')]),
    ]


class TestConversationKey:
    def test_includes_team_when_set(self) -> None:
        # Two workspaces can hold the same channel id, so history has to stay apart.
        assert SlackThread(channel_id='C1', thread_ts='1.1', team_id='T1').key == 'T1:C1:1.1'

    def test_omits_team_when_absent(self) -> None:
        assert SlackThread(channel_id='C1', thread_ts='1.1').key == 'C1:1.1'

    def test_a_channel_with_no_thread_keys_on_the_channel(self) -> None:
        assert SlackThread(channel_id='C1').key == 'C1'

    def test_threads_in_one_channel_stay_separate(self) -> None:
        first = SlackThread(channel_id='C1', thread_ts='1.1')
        assert first.key != SlackThread(channel_id='C1', thread_ts='2.2').key


class TestInMemoryConversationStore:
    async def test_round_trips_history(self) -> None:
        store = InMemoryConversationStore()
        assert list(await store.load('k')) == []
        await store.save('k', _history())
        assert len(await store.load('k')) == 2

    async def test_load_returns_a_copy(self) -> None:
        store = InMemoryConversationStore()
        await store.save('k', _history())
        loaded = list(await store.load('k'))
        loaded.clear()
        assert len(await store.load('k')) == 2

    async def test_delete_is_forgiving(self) -> None:
        store = InMemoryConversationStore()
        await store.save('k', _history())
        await store.delete('k')
        await store.delete('k')
        assert list(await store.load('k')) == []


class TestFileConversationStore:
    async def test_survives_a_new_store_over_the_same_directory(self, tmp_path: Path) -> None:
        await FileConversationStore(tmp_path).save('T1:C1:1.1', _history())
        reopened = await FileConversationStore(tmp_path).load('T1:C1:1.1')
        assert [type(message).__name__ for message in reopened] == ['ModelRequest', 'ModelResponse']

    async def test_unknown_key_loads_empty(self, tmp_path: Path) -> None:
        assert list(await FileConversationStore(tmp_path).load('missing')) == []

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

    async def test_tightens_an_existing_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / 'history'
        directory.mkdir(mode=0o777)
        directory.chmod(0o777)
        await FileConversationStore(directory).save('k', _history())
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700

    async def test_tightens_an_existing_directory_before_reading(self, tmp_path: Path) -> None:
        directory = tmp_path / 'history'
        store = FileConversationStore(directory)
        await store.save('k', _history())
        directory.chmod(0o777)
        assert len(await store.load('k')) == 2
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700

    async def test_a_failed_write_leaves_no_transcript_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def explode(self: anyio.Path, data: bytes) -> int:
            raise OSError('no space left on device')

        monkeypatch.setattr(anyio.Path, 'write_bytes', explode)
        with pytest.raises(OSError, match='no space left'):
            await FileConversationStore(tmp_path).save('k', _history())
        assert list(tmp_path.iterdir()) == []

    async def test_delete_is_forgiving(self, tmp_path: Path) -> None:
        store = FileConversationStore(tmp_path)
        await store.save('k', _history())
        await store.delete('k')
        await store.delete('k')
        assert list(await store.load('k')) == []

    async def test_expands_a_user_relative_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        store = FileConversationStore('~/agent-history')
        await store.save('k', _history())
        assert (tmp_path / 'agent-history').is_dir()

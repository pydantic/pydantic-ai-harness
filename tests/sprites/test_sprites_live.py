"""Opt-in smoke test against a real Sprite.

The fake-backed tests cover integration-owned branching. This test verifies the
SDK assumptions that a fake cannot: real create and label retention, process
execution, the shared command/filesystem view, in-Sprite timeout enforcement,
bounded output, and final deletion.

Run locally only with explicit permission to create and delete a Sprite:

`PYDANTIC_AI_HARNESS_SPRITES_LIVE=1 uv run pytest -m sprites_live tests/sprites/test_sprites_live.py`
"""

from __future__ import annotations

import os

import pytest

from pydantic_ai_harness.sprites import SpriteSandboxError, SpriteSandboxSession, SpriteSandboxUnavailableError

_TOKEN = os.getenv('SPRITE_TOKEN')
_LIVE_ENABLED = os.getenv('PYDANTIC_AI_HARNESS_SPRITES_LIVE') == '1'

pytestmark = [
    pytest.mark.sprites_live,
    pytest.mark.skipif(
        not _LIVE_ENABLED or not _TOKEN,
        reason='requires PYDANTIC_AI_HARNESS_SPRITES_LIVE=1 and SPRITE_TOKEN',
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run the billed live smoke test once, using the SDK's primary asyncio backend."""
    return 'asyncio'


async def test_real_sprite_execution_filesystem_limits_and_lifecycle() -> None:
    """Exercise one real owned Sprite, then confirm the ownership-safe cleanup deleted it."""
    assert _TOKEN is not None
    name: str | None = None
    async with SpriteSandboxSession(token=_TOKEN) as session:
        name = session.sprite_name
        assert name is not None

        command = await session.exec(
            'printf stdout; printf stderr >&2; exit 3',
            timeout=15,
            max_output_bytes=1024,
        )
        assert command.output == 'stdoutstderr'
        assert command.returncode == 3

        await session.write_bytes('harness-live/nested.txt', b'from-filesystem\n')
        via_shell = await session.exec('cat harness-live/nested.txt', timeout=15, max_output_bytes=1024)
        assert via_shell.output == 'from-filesystem\n'
        assert await session.read_bytes('harness-live/nested.txt', max_bytes=1024) == b'from-filesystem\n'
        with pytest.raises(SpriteSandboxError, match='exceeded the 4-byte read limit'):
            await session.read_bytes('harness-live/nested.txt', max_bytes=4)

        await session.write_bytes('harness-live/second.txt', b'second\n')
        listing = await session.list_files('harness-live', max_entries=1, max_output_bytes=1024)
        assert len(listing.entries) == 1
        assert listing.truncated is True

        bounded = await session.exec('python3 -c "print(\'x\' * 5000)"', timeout=15, max_output_bytes=256)
        assert bounded.truncated is True
        assert len(bounded.output.encode()) <= 256

        literal_timeout_marker = await session.exec(
            "printf '\\n__PYDANTIC_AI_SPRITE_TIMEOUT__\\n'", timeout=15, max_output_bytes=1024
        )
        assert literal_timeout_marker.timed_out is False
        assert literal_timeout_marker.output == '\n__PYDANTIC_AI_SPRITE_TIMEOUT__\n'

        await session.exec('printf profile-noise > "$HOME/.bash_profile"', timeout=15, max_output_bytes=1024)
        after_profile = await session.exec('printf clean', timeout=15, max_output_bytes=1024)
        assert after_profile.output == 'clean'

        timed_out = await session.exec('printf before-timeout; sleep 30', timeout=1, max_output_bytes=1024)
        assert timed_out.output == 'before-timeout'
        assert timed_out.timed_out is True

    with pytest.raises(SpriteSandboxUnavailableError):
        async with SpriteSandboxSession(token=_TOKEN, sprite_name=name):
            pass

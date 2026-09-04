"""Tests for shared sandbox provider helpers."""

from __future__ import annotations

import anyio
import pytest

from pydantic_ai_harness._sandbox_provider import absolute_path, cleanup_call, raise_after_cleanup

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(None, None), ('/', '/'), ('/work/../repo', '/repo'), ('/work//src', '/work/src')],
)
def test_absolute_path(value: str | None, expected: str | None) -> None:
    assert absolute_path('workdir', value) == expected


def test_absolute_path_rejects_relative() -> None:
    with pytest.raises(ValueError, match='workdir must be an absolute sandbox path'):
        absolute_path('workdir', 'repo')


async def test_cleanup_call_returns_none() -> None:
    async def call() -> object:
        return object()

    assert await cleanup_call(call, timeout=1) is None


async def test_cleanup_call_returns_error() -> None:
    expected = RuntimeError('failed')

    async def call() -> object:
        raise expected

    assert await cleanup_call(call, timeout=1) is expected


async def test_cleanup_call_returns_timeout() -> None:
    async def call() -> object:
        await anyio.sleep_forever()

    assert isinstance(await cleanup_call(call, timeout=0.01), TimeoutError)


async def test_cleanup_call_survives_external_cancellation() -> None:
    started = anyio.Event()
    release = anyio.Event()
    outcomes: list[Exception | TimeoutError | None] = []

    async def call() -> object:
        started.set()
        await release.wait()
        return object()

    async def worker() -> None:
        outcomes.append(await cleanup_call(call, timeout=1))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker)
        await started.wait()
        task_group.cancel_scope.cancel()
        release.set()
    assert outcomes == [None]


async def test_raise_after_cleanup_prefers_cancellation() -> None:
    # Inside a cancelled scope the checkpoint delivers the cancellation, so the cleanup
    # error is never raised: the scope exits having caught its own cancellation.
    with anyio.CancelScope() as scope:
        scope.cancel()
        await raise_after_cleanup(RuntimeError('cleanup failed'))
    assert scope.cancelled_caught is True

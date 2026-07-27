"""Tests for the `StalenessTracker` capability.

End-to-end behavior is driven through `Agent(..., capabilities=[...])` with a
`FunctionModel` whose scripted steps make file read/write tool calls and, between
steps, mutate the files on disk to simulate concurrent change. Each step captures
the messages the model was actually handed, so we can assert the ephemeral
`<system-reminder>` notice reached the model without ever entering the durable
message history. Focused unit tests cover the extractor, LRU, and notice edges.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.staleness import StalenessTracker

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


Step = Callable[[list[ModelMessage]], ModelResponse]


def read_file(file_path: str) -> str:
    return Path(file_path).read_text()


def write_file(file_path: str, content: str) -> str:
    Path(file_path).write_text(content)
    return 'wrote'


def noop() -> str:
    return 'ok'


def _run(tracker: StalenessTracker[None], steps: Sequence[Step]) -> tuple[Agent[None, str], list[list[ModelMessage]]]:
    """Build an agent that plays `steps` in order, capturing the messages seen each request."""
    seen: list[list[ModelMessage]] = []
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        i = state['i']
        state['i'] += 1
        return steps[i](messages)

    agent = Agent(
        FunctionModel(fn),
        deps_type=type(None),
        capabilities=[tracker],
        tools=[read_file, write_file, noop],
    )
    return agent, seen


def _reminder_in(messages: list[ModelMessage]) -> str | None:
    """Return the text of any ephemeral list-content reminder a request carries.

    The tracker is the only capability here that appends list-content `UserPromptPart`s,
    so such a part is always its `<system-reminder>`; return its joined string fragments.
    """
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                    return ''.join(item for item in part.content if isinstance(item, str))
    return None


def _tool(name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, args)])


def _final() -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


class TestEndToEnd:
    async def test_external_modification_produces_notice(self, tmp_path: Path) -> None:
        """read -> external edit -> the next request carries a staleness notice."""
        f = tmp_path / 'foo.py'
        f.write_text('original')

        def step1(_: list[ModelMessage]) -> ModelResponse:
            # Simulate a concurrent writer changing the file after we observed it.
            f.write_text('EXTERNALLY CHANGED, LONGER')
            return _tool('noop', {})

        tracker = StalenessTracker[None]()
        agent, seen = _run(tracker, [lambda _: _tool('read_file', {'file_path': str(f)}), step1, lambda _: _final()])
        result = await agent.run('go')

        assert result.output == 'done'
        assert _reminder_in(seen[0]) is None  # nothing observed yet
        assert _reminder_in(seen[1]) is None  # edit happens during this step
        notice = _reminder_in(seen[2])
        assert notice is not None
        assert 'Files changed on disk since you last read them' in notice
        assert 'foo.py' in notice
        assert 'Re-read before relying on their contents.' in notice

    async def test_self_write_is_not_staleness(self, tmp_path: Path) -> None:
        """read -> the agent's own write to the same file -> no notice."""
        f = tmp_path / 'foo.py'
        f.write_text('original')

        tracker = StalenessTracker[None]()
        agent, seen = _run(
            tracker,
            [
                lambda _: _tool('read_file', {'file_path': str(f)}),
                lambda _: _tool('write_file', {'file_path': str(f), 'content': 'rewritten by me'}),
                lambda _: _final(),
            ],
        )
        await agent.run('go')

        assert all(_reminder_in(msgs) is None for msgs in seen)

    async def test_deletion_produces_notice(self, tmp_path: Path) -> None:
        """read -> external deletion -> a 'deleted' notice."""
        f = tmp_path / 'gone.py'
        f.write_text('here')

        def step1(_: list[ModelMessage]) -> ModelResponse:
            os.remove(f)
            return _tool('noop', {})

        tracker = StalenessTracker[None]()
        agent, seen = _run(tracker, [lambda _: _tool('read_file', {'file_path': str(f)}), step1, lambda _: _final()])
        await agent.run('go')

        notice = _reminder_in(seen[2])
        assert notice is not None
        assert 'Files deleted since you last read them: ' in notice
        assert 'gone.py' in notice

    async def test_notice_never_enters_durable_history(self, tmp_path: Path) -> None:
        """The ephemeral notice reaches the model but not the stored message history."""
        f = tmp_path / 'foo.py'
        f.write_text('original')

        def step1(_: list[ModelMessage]) -> ModelResponse:
            f.write_text('changed and longer than before')
            return _tool('noop', {})

        tracker = StalenessTracker[None]()
        agent, seen = _run(tracker, [lambda _: _tool('read_file', {'file_path': str(f)}), step1, lambda _: _final()])
        result = await agent.run('go')

        assert _reminder_in(seen[2]) is not None  # the model saw it
        stored = result.all_messages()
        assert _reminder_in(stored) is None  # but history is clean

    async def test_notice_keeps_firing_until_reread(self, tmp_path: Path) -> None:
        """A changed file stays flagged every request until the model re-reads it."""
        f = tmp_path / 'foo.py'
        f.write_text('original')

        def step1(_: list[ModelMessage]) -> ModelResponse:
            f.write_text('changed once, now longer')
            return _tool('noop', {})

        tracker = StalenessTracker[None]()
        agent, seen = _run(
            tracker,
            [
                lambda _: _tool('read_file', {'file_path': str(f)}),
                step1,
                lambda _: _tool('noop', {}),  # request 3 sees notice, ignores it
                lambda _: _tool('read_file', {'file_path': str(f)}),  # request 4 re-reads
                lambda _: _final(),  # request 5 is clean again
            ],
        )
        await agent.run('go')

        assert _reminder_in(seen[2]) is not None
        assert _reminder_in(seen[3]) is not None  # still stale, not yet re-read
        assert _reminder_in(seen[4]) is None  # re-read refreshed the record

    async def test_tracks_across_run_with_root_configured(self, tmp_path: Path) -> None:
        """A run with `root` set still tracks and flags an externally changed file."""
        (tmp_path / 'src').mkdir()
        f = tmp_path / 'src' / 'foo.py'
        f.write_text('original')

        def step1(_: list[ModelMessage]) -> ModelResponse:
            f.write_text('externally rewritten, longer')
            return _tool('noop', {})

        tracker = StalenessTracker[None](root=tmp_path)
        agent, seen = _run(
            tracker,
            [lambda _: _tool('read_file', {'file_path': str(f)}), step1, lambda _: _final()],
        )
        await agent.run('go')
        notice = _reminder_in(seen[2])
        assert notice is not None
        assert 'foo.py' in notice


class TestForRunIsolation:
    async def test_for_run_returns_fresh_ledger(self) -> None:
        base = StalenessTracker[None]()
        run1 = await base.for_run(MagicMock())
        run2 = await base.for_run(MagicMock())

        run1._observations[Path('/a')] = MagicMock()
        assert run2._observations == {}
        assert base._observations == {}
        assert run1 is not run2

    async def test_for_run_preserves_config(self) -> None:
        base = StalenessTracker[None](max_tracked=5, max_listed=2, root=Path('/repo'))
        run = await base.for_run(MagicMock())
        assert run.max_tracked == 5
        assert run.max_listed == 2
        assert run.root == Path('/repo')


class TestExtraction:
    def _extract(self, tracker: StalenessTracker[None], name: str, args: Mapping[str, Any]) -> Sequence[str]:
        return tracker._extract_paths(name, args)

    def test_default_prefers_file_path_then_path(self) -> None:
        tracker = StalenessTracker[None]()
        assert self._extract(tracker, 'read_file', {'file_path': 'a.py'}) == ['a.py']
        assert self._extract(tracker, 'read', {'path': 'b.py'}) == ['b.py']

    def test_unknown_tool_extracts_nothing(self) -> None:
        tracker = StalenessTracker[None]()
        assert self._extract(tracker, 'grep', {'path': 'a.py'}) == []

    def test_missing_or_nonstring_arg_extracts_nothing(self) -> None:
        tracker = StalenessTracker[None]()
        assert self._extract(tracker, 'read_file', {}) == []
        assert self._extract(tracker, 'read_file', {'file_path': 123}) == []

    def test_string_arg_override(self) -> None:
        tracker = StalenessTracker[None](track={'open': 'filename'})
        assert self._extract(tracker, 'open', {'filename': 'a.py'}) == ['a.py']
        assert self._extract(tracker, 'open', {'filename': ''}) == []

    def test_glob_pattern_match(self) -> None:
        tracker = StalenessTracker[None](track={'read*': 'file_path'})
        assert self._extract(tracker, 'read_bytes', {'file_path': 'a.py'}) == ['a.py']

    def test_callable_track_value(self) -> None:
        tracker = StalenessTracker[None](track={'patch': lambda args: list(args.get('targets', []))})
        assert self._extract(tracker, 'patch', {'targets': ['a.py', 'b.py']}) == ['a.py', 'b.py']

    def test_path_extractor_replaces_track(self) -> None:
        def extractor(tool_name: str, args: Mapping[str, Any]) -> Sequence[str]:
            return [args['p']] if tool_name == 'x' else []

        tracker = StalenessTracker[None](path_extractor=extractor)
        assert self._extract(tracker, 'x', {'p': 'a.py'}) == ['a.py']
        assert self._extract(tracker, 'read_file', {'file_path': 'a.py'}) == []


class TestResolve:
    def test_absolute_path_is_returned_as_is(self) -> None:
        tracker = StalenessTracker[None]()
        assert tracker._resolve('/etc/hosts') == Path('/etc/hosts')

    def test_relative_uses_root_when_set(self) -> None:
        tracker = StalenessTracker[None](root=Path('/repo'))
        assert tracker._resolve('src/a.py') == Path('/repo/src/a.py')

    def test_relative_uses_cwd_when_root_none(self) -> None:
        tracker = StalenessTracker[None]()
        assert tracker._resolve('a.py') == Path.cwd() / 'a.py'


class TestObserveAndNotice:
    def test_missing_file_is_not_tracked(self, tmp_path: Path) -> None:
        tracker = StalenessTracker[None]()
        tracker._observe(str(tmp_path / 'nope.py'))
        assert tracker._observations == {}

    def test_lru_eviction_past_max_tracked(self, tmp_path: Path) -> None:
        tracker = StalenessTracker[None](max_tracked=3)
        paths = [tmp_path / f'f{i}.py' for i in range(5)]
        for p in paths:
            p.write_text('x')
            tracker._observe(str(p))
        assert len(tracker._observations) == 3
        keys = list(tracker._observations)
        assert paths[0].resolve() not in keys  # oldest evicted
        assert paths[4].resolve() in keys  # newest kept

    def test_reobserving_refreshes_lru_position(self, tmp_path: Path) -> None:
        tracker = StalenessTracker[None](max_tracked=2)
        a, b, c = (tmp_path / n for n in ('a.py', 'b.py', 'c.py'))
        for p in (a, b, c):
            p.write_text('x')
        tracker._observe(str(a))
        tracker._observe(str(b))
        tracker._observe(str(a))  # touch a again -> b is now oldest
        tracker._observe(str(c))  # evicts b, not a
        keys = list(tracker._observations)
        assert a.resolve() in keys
        assert c.resolve() in keys
        assert b.resolve() not in keys

    def test_notice_none_when_nothing_changed(self, tmp_path: Path) -> None:
        f = tmp_path / 'a.py'
        f.write_text('x')
        tracker = StalenessTracker[None]()
        tracker._observe(str(f))
        assert tracker._notice() is None

    def test_notice_detects_same_size_mtime_change(self, tmp_path: Path) -> None:
        """A same-size rewrite is caught via mtime even when the byte count is identical."""
        f = tmp_path / 'a.py'
        f.write_text('abcd')
        tracker = StalenessTracker[None]()
        tracker._observe(str(f))
        obs = next(iter(tracker._observations.values()))
        f.write_text('wxyz')  # same size, different content
        os.utime(f, ns=(obs.mtime_ns + 1_000_000_000, obs.mtime_ns + 1_000_000_000))
        notice = tracker._notice()
        assert notice is not None
        assert 'a.py' in notice

    def test_notice_caps_changed_list(self, tmp_path: Path) -> None:
        tracker = StalenessTracker[None](max_listed=2)
        files = [tmp_path / f'f{i}.py' for i in range(5)]
        for f in files:
            f.write_text('x')
            tracker._observe(str(f))
        for f in files:
            f.write_text('changed and now definitely longer')
        notice = tracker._notice()
        assert notice is not None
        assert '(+3 more)' in notice

    def test_notice_reports_changed_and_deleted_together(self, tmp_path: Path) -> None:
        changed = tmp_path / 'changed.py'
        deleted = tmp_path / 'deleted.py'
        changed.write_text('x')
        deleted.write_text('y')
        tracker = StalenessTracker[None]()
        tracker._observe(str(changed))
        tracker._observe(str(deleted))
        changed.write_text('now much longer than before')
        os.remove(deleted)
        notice = tracker._notice()
        assert notice is not None
        assert 'Files changed on disk' in notice
        assert 'changed.py' in notice
        assert 'Files deleted since you last read them' in notice
        assert 'deleted.py' in notice


class TestWrapModelRequest:
    async def _run_hook(
        self, tracker: StalenessTracker[None], messages: list[ModelMessage]
    ) -> tuple[list[ModelMessage], ModelResponse]:
        captured: dict[str, list[ModelMessage]] = {}

        async def handler(rc: ModelRequestContext) -> ModelResponse:
            captured['messages'] = list(rc.messages)
            return ModelResponse(parts=[TextPart('ok')])

        ctx = ModelRequestContext(
            model=TestModel(),
            messages=messages,
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        response = await tracker.wrap_model_request(MagicMock(), request_context=ctx, handler=handler)
        return captured['messages'], response

    async def test_no_notice_leaves_request_untouched(self) -> None:
        tracker = StalenessTracker[None]()
        original = ModelRequest(parts=[UserPromptPart('hello')])
        seen, response = await self._run_hook(tracker, [original])
        assert seen[-1] is original  # not replaced
        assert isinstance(response.parts[0], TextPart)

    async def test_notice_appended_to_last_model_request(self, tmp_path: Path) -> None:
        f = tmp_path / 'a.py'
        f.write_text('x')
        tracker = StalenessTracker[None]()
        tracker._observe(str(f))
        f.write_text('now longer than before')
        original = ModelRequest(parts=[UserPromptPart('hello')])
        seen, _ = await self._run_hook(tracker, [original])
        assert _reminder_in(seen) is not None
        assert seen[-1] is not original  # replaced with an augmented copy
        assert len(original.parts) == 1  # the original object is untouched

    async def test_notice_skipped_when_last_is_not_a_request(self, tmp_path: Path) -> None:
        """Defensive: if the tail isn't a ModelRequest, inject nothing but still call the handler."""
        f = tmp_path / 'a.py'
        f.write_text('x')
        tracker = StalenessTracker[None]()
        tracker._observe(str(f))
        f.write_text('now longer than before')
        tail = ModelResponse(parts=[TextPart('prior')])
        seen, response = await self._run_hook(tracker, [tail])
        assert _reminder_in(seen) is None
        assert isinstance(response.parts[0], TextPart)


class TestSerialization:
    def test_opts_out_of_spec_construction(self) -> None:
        assert StalenessTracker.get_serialization_name() is None

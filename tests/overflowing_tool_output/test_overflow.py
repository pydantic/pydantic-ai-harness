"""Tests for pydantic_ai_harness.overflowing_tool_output."""

from __future__ import annotations

import dataclasses
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.overflowing_tool_output import (
    GREP_TOOL_NAME,
    Band,
    LocalFileStore,
    OverflowingToolOutput,
    Passthrough,
    Spill,
    Summarize,
    Truncate,
    TruncationStrategy,
)
from pydantic_ai_harness.overflowing_tool_output._capability import (
    READ_TOOL_NAME,
    _build_spill_preview,
    _grep_slice,
    _handle_key,
    _head_tail_preview,
    _read_slice,
    _select_action,
    _Unit,
    _with_handles,
)
from pydantic_ai_harness.overflowing_tool_output._payload import (
    is_binary,
    json_sketch,
    measure,
    strip_ansi,
    to_bytes,
    to_text,
    truncate_text,
)
from pydantic_ai_harness.overflowing_tool_output._store import _safe_segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(*, run_id: str | None = 'run-1', retry: int = 0, model: Any = None) -> Any:
    """Build a minimal RunContext-like object for testing the hook directly."""

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage
        run_id: str | None
        retry: int
        tool_call_id: str | None = 'call-1'
        model: Any = dataclasses.field(default_factory=_FakeModel)
        deps: None = None

    ctx = _FakeCtx(usage=RunUsage(), run_id=run_id, retry=retry)
    if model is not None:
        ctx.model = model
    return ctx


def _fixed_model(text: str) -> FunctionModel:
    """A `FunctionModel` whose single text response is `text` (no tool calls)."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(respond)


def _call(tool_name: str = 'big_tool', tool_call_id: str = 'call-1') -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args='{}', tool_call_id=tool_call_id)


def _tool_def(name: str = 'big_tool') -> ToolDefinition:
    return ToolDefinition(name=name)


async def _run(cap: OverflowingToolOutput[object], result: Any, *, ctx: Any = None, tool_name: str = 'big_tool') -> Any:
    return await cap.after_tool_execute(
        ctx if ctx is not None else _make_ctx(),
        call=_call(tool_name),
        tool_def=_tool_def(tool_name),
        args={},
        result=result,
    )


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# ---------------------------------------------------------------------------
# _payload helpers
# ---------------------------------------------------------------------------


class TestPayloadHelpers:
    def test_strip_ansi(self):
        assert strip_ansi('\x1b[31mred\x1b[0m') == 'red'

    def test_is_binary(self):
        assert is_binary(b'x') is True
        assert is_binary(bytearray(b'x')) is True
        assert is_binary('x') is False

    def test_to_bytes_variants(self):
        assert to_bytes('hi') == b'hi'
        assert to_bytes(memoryview(b'mv')) == b'mv'
        assert to_bytes(bytearray(b'ba')) == b'ba'
        assert to_bytes({'a': 1}) == b'{"a":1}'

    def test_to_text_variants(self):
        assert to_text('hi') == 'hi'
        assert to_text({'a': 1}) == '{"a":1}'

    def test_measure_chars_and_tokens(self):
        assert measure('x' * 100, over_tokens=False, tokenizer=None) == 100
        assert measure('x' * 100, over_tokens=True, tokenizer=None) == 25
        assert measure('abcd', over_tokens=True, tokenizer=lambda s: len(s)) == 4

    def test_json_sketch_mapping(self):
        assert json_sketch({'a': 1, 'b': 'x'}) == "{'a': int, 'b': str}"

    def test_json_sketch_mapping_truncated(self):
        big = {f'k{i}': i for i in range(12)}
        assert json_sketch(big).endswith('... (12 keys)}')

    def test_json_sketch_sequence(self):
        assert json_sketch([1, 2, 3]) == '[3 items of int]'

    def test_json_sketch_empty_sequence(self):
        assert json_sketch([]) == '[0 items of empty]'

    def test_json_sketch_scalar(self):
        assert json_sketch(42) == ''
        assert json_sketch('plain') == ''

    def test_truncate_under_limit(self):
        assert truncate_text('short', 100, TruncationStrategy.head_tail) == 'short'

    def test_truncate_head(self):
        out = truncate_text('a' * 100, 10, TruncationStrategy.head)
        assert out.startswith('aaaaaaaaaa')
        # Lossy truncation leaves a handle-less marker naming the omitted span.
        assert 'last 90 of 100 chars omitted' in out
        assert 're-run the tool' in out

    def test_truncate_tail(self):
        out = truncate_text('a' * 100, 10, TruncationStrategy.tail)
        assert out.endswith('aaaaaaaaaa')
        assert 'first 90 of 100 chars omitted' in out

    def test_truncate_head_tail(self):
        out = truncate_text('a' * 100, 10, TruncationStrategy.head_tail)
        assert 'from the middle omitted' in out


# ---------------------------------------------------------------------------
# Store: write/read, S1 hardening
# ---------------------------------------------------------------------------


class TestStore:
    def test_safe_segment(self):
        assert _safe_segment('a b!@#') == 'a_b_'
        assert _safe_segment('') == '_'
        assert _safe_segment('..') == '_'
        assert _safe_segment('.') == '_'
        assert _safe_segment('ok-1.2') == 'ok-1.2'

    def test_default_root(self):
        store = LocalFileStore()
        assert store._root.name == 'pyai_harness_overflow'

    async def test_write_read_roundtrip(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path / 'store')
        handle = await store.write('run-1/call-1.0', b'payload')
        assert handle == 'run-1/call-1.0'
        assert await store.read(handle) == b'payload'

    async def test_root_created_0700(self, tmp_path: Path):
        root = tmp_path / 'store'
        store = LocalFileStore(base_dir=root)
        await store.write('run/c.0', b'x')
        assert oct(root.stat().st_mode & 0o777) == '0o700'

    async def test_empty_key(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path / 'store')
        handle = await store.write('', b'data')
        assert await store.read(handle) == b'data'

    async def test_read_missing_raises(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path / 'store')
        with pytest.raises(OSError):
            await store.read('nope/x.0')

    async def test_dotdot_handle_stays_in_root(self, tmp_path: Path):
        # `_safe_segment` neutralizes `..`, so the read resolves inside the root and 404s
        # rather than escaping.
        store = LocalFileStore(base_dir=tmp_path / 'store')
        await store.write('run/c.0', b'inside')
        with pytest.raises(OSError):
            await store.read('../c.0')

    async def test_symlink_escape_rejected(self, tmp_path: Path):
        secret = tmp_path / 'secret.txt'
        secret.write_bytes(b'top secret')
        root = tmp_path / 'store'
        store = LocalFileStore(base_dir=root)
        await store.write('run/c.0', b'inside')  # creates the root
        (root / 'evil').symlink_to(secret)
        with pytest.raises(PermissionError, match='outside the store root'):
            await store.read('evil')


# ---------------------------------------------------------------------------
# Store: opt-in TTL cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_prune_removes_old_keeps_new(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path, cleanup_after=timedelta(seconds=1))
        old = tmp_path / 'old.bin'
        old.write_bytes(b'x')
        new = tmp_path / 'new.bin'
        new.write_bytes(b'y')
        (tmp_path / 'sub').mkdir()  # a directory rglob yields -- must be skipped
        past = time.time() - 100
        os.utime(old, (past, past))

        store._prune_sync()

        assert not old.exists()
        assert new.exists()

    def test_run_prune_swallows_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = LocalFileStore(base_dir=tmp_path, cleanup_after=timedelta(seconds=1))

        def boom() -> None:
            raise OSError('disk gone')

        monkeypatch.setattr(store, '_prune_sync', boom)
        with pytest.warns(UserWarning, match='cleanup failed'):
            store._run_prune()

    def test_schedule_none_when_disabled(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        assert store._schedule_cleanup() is None

    def test_schedule_starts_thread(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path, cleanup_after=timedelta(seconds=1))
        (tmp_path / 'f.bin').write_bytes(b'z')
        thread = store._schedule_cleanup()
        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive()

    async def test_write_schedules_cleanup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = LocalFileStore(base_dir=tmp_path / 'store', cleanup_after=timedelta(seconds=1))
        scheduled: list[int] = []
        monkeypatch.setattr(store, '_schedule_cleanup', lambda: scheduled.append(1))
        await store.write('run/c.0', b'data')
        assert scheduled == [1]


# ---------------------------------------------------------------------------
# Capability construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_band_is_spill_then_truncate(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput()
        assert len(cap._bands) == 1
        action = cap._bands[0].action
        assert isinstance(action, Spill)
        assert isinstance(action.then, Truncate)

    def test_bands_sorted_descending(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10, action=Truncate()), Band(over=100, action=Spill())]
        )
        assert [b.over for b in cap._bands] == [100, 10]

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match='non-negative'):
            OverflowingToolOutput(bands=[Band(over=-1, action=Passthrough())])

    def test_provided_store_used(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(store=store)
        assert cap._store is store

    def test_per_tool_prepared(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            per_tool={'read_file': [Band(over=5, action=Truncate())]}
        )
        assert 'read_file' in cap._per_tool


# ---------------------------------------------------------------------------
# Passthrough / filtering / guards
# ---------------------------------------------------------------------------


class TestPassthrough:
    async def test_read_tool_exempt(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate(max_chars=2))])
        out = await _run(cap, 'x' * 100, tool_name=READ_TOOL_NAME)
        assert out == 'x' * 100

    async def test_tool_filter_skips_unmatched(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=1, action=Truncate(max_chars=2))], tool_filter=['other']
        )
        out = await _run(cap, 'x' * 100)
        assert out == 'x' * 100

    async def test_callable_filter(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=1, action=Truncate(max_chars=2))],
            tool_filter=lambda ctx, td: td.name == 'big_tool',
        )
        out = await _run(cap, 'x' * 100)
        assert isinstance(out, str) and 'omitted' in out

    async def test_below_threshold_passthrough(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1000, action=Truncate())])
        out = await _run(cap, 'small')
        assert out == 'small'

    async def test_exception_result_passthrough(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate(max_chars=2))])
        err = ValueError('boom')
        assert await _run(cap, err) is err


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    async def test_truncates_text(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10, action=Truncate(max_chars=20, strategy=TruncationStrategy.head))]
        )
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str) and out.startswith('a' * 20)

    async def test_strip_ansi_applied(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=5, action=Truncate(max_chars=1000))], strip_ansi=True
        )
        out = await _run(cap, '\x1b[31m' + 'red text ' * 10 + '\x1b[0m')
        assert isinstance(out, str) and '\x1b[' not in out

    async def test_binary_truncate_falls_back_to_passthrough(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate())])
        data = b'\x00\x01' * 100
        assert await _run(cap, data) == data

    async def test_tool_return_envelope_preserved(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=10, action=Truncate(max_chars=20))])
        out = await _run(cap, ToolReturn(return_value='a' * 100, content='note', metadata={'k': 1}))
        assert isinstance(out, ToolReturn)
        assert out.content == 'note'
        assert out.metadata == {'k': 1}


# ---------------------------------------------------------------------------
# Spill
# ---------------------------------------------------------------------------


class TestSpill:
    async def test_spill_roundtrip(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill(preview_chars=20))], store=store
        )
        text = 'line\n' * 1000
        out = await _run(cap, text)
        assert isinstance(out, ToolReturn)
        assert isinstance(out.return_value, str) and 'too large' in out.return_value
        handle = out.metadata['overflow_handle']
        assert handle == 'run-1/call-1.0'
        assert await store.read(handle) == text.encode('utf-8')

    async def test_spill_binary_verbatim(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Spill())], store=store)
        data = b'\x00\xff' * 100
        out = await _run(cap, data)
        assert isinstance(out, ToolReturn)
        assert 'binary' in out.return_value  # type: ignore[operator]
        assert await store.read(out.metadata['overflow_handle']) == data

    async def test_spill_structured_includes_sketch(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, {'rows': list(range(1000)), 'ok': True})
        assert isinstance(out, ToolReturn)
        assert 'shape:' in out.return_value  # type: ignore[operator]

    async def test_spill_failure_falls_back_to_truncate(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill(then=Truncate(max_chars=15)))], store=_BrokenStore()
        )
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str) and 'omitted' in out

    async def test_spill_failure_no_fallback_returns_original(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill())], store=_BrokenStore()
        )
        out = await _run(cap, 'a' * 100)
        assert out == 'a' * 100

    async def test_handle_distinct_per_retry(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=5, action=Spill())], store=store)
        out0 = await _run(cap, 'a' * 100, ctx=_make_ctx(retry=0))
        out1 = await _run(cap, 'b' * 100, ctx=_make_ctx(retry=1))
        assert out0.metadata['overflow_handle'] != out1.metadata['overflow_handle']  # type: ignore[union-attr]

    async def test_spill_merges_existing_metadata(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, ToolReturn(return_value='a' * 100, metadata={'orig': True}))
        assert isinstance(out, ToolReturn)
        assert out.metadata['orig'] is True
        assert 'overflow_handle' in out.metadata


class _BrokenStore:
    """An `OverflowStore` whose writes always fail (for fallback tests)."""

    async def write(self, key: str, data: bytes) -> str:
        raise OSError('disk full')

    async def read(self, handle: str) -> bytes:  # pragma: no cover - never reached
        raise FileNotFoundError(handle)


# ---------------------------------------------------------------------------
# C1: model-visible ToolReturn.content is reduced too
# ---------------------------------------------------------------------------


class TestContentReduction:
    async def test_large_content_spilled(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=100, action=Spill(preview_chars=20))], store=store
        )
        out = await _run(cap, ToolReturn(return_value='small', content='C' * 5000))
        assert isinstance(out, ToolReturn)
        assert out.return_value == 'small'  # small return_value untouched
        assert isinstance(out.content, str) and 'too large' in out.content
        handle = out.metadata['overflow_content_handle']
        assert await store.read(handle) == ('C' * 5000).encode('utf-8')

    async def test_large_content_truncated(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=10, action=Truncate(max_chars=20))])
        out = await _run(cap, ToolReturn(return_value='small', content='C' * 200))
        assert isinstance(out, ToolReturn)
        assert isinstance(out.content, str) and 'omitted' in out.content

    async def test_both_value_and_content_reduced(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=50, action=Spill())], store=store)
        out = await _run(cap, ToolReturn(return_value='v' * 500, content='c' * 500))
        assert isinstance(out, ToolReturn)
        assert out.metadata['overflow_handle'] != out.metadata['overflow_content_handle']

    async def test_nontext_content_warns_and_passes_through(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=10, action=Truncate())])
        content = ['x' * 5000, BinaryContent(data=b'\x00', media_type='application/octet-stream')]
        original = ToolReturn(return_value='small', content=content)
        with pytest.warns(UserWarning, match='non-text content'):
            out = await _run(cap, original)
        assert out is original

    async def test_nontext_content_passthrough_action_no_warn(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Passthrough())])
        content = ['x' * 5000, BinaryContent(data=b'\x00', media_type='application/octet-stream')]
        original = ToolReturn(return_value='small', content=content)
        out = await _run(cap, original)  # Passthrough action -> no warning, returned unchanged
        assert out is original

    async def test_small_nontext_content_no_warn(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=10_000, action=Truncate())])
        content = ['tiny', BinaryContent(data=b'\x00', media_type='application/octet-stream')]
        original = ToolReturn(return_value='small', content=content)
        out = await _run(cap, original)
        assert out is original


# ---------------------------------------------------------------------------
# Summarize (M1: assert model + usage, not just a wholesale mock)
# ---------------------------------------------------------------------------


class TestSummarize:
    async def test_custom_sync_summarizer(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(summarize=lambda name, text: f'{name}:{len(text)}'))]
        )
        out = await _run(cap, 'x' * 100)
        assert out.endswith('big_tool:100')

    async def test_custom_async_summarizer(self):
        async def summ(name: str, text: str) -> str:
            return f'async:{len(text)}'

        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(summarize=summ))]
        )
        out = await _run(cap, 'x' * 100)
        assert out.endswith('async:100')

    async def test_summary_carries_explicit_marker(self):
        # The summary replaces the real tool output, so it must be marked as a harness
        # stand-in the same way every other elision path is -- not passed off as the result.
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(summarize=lambda name, text: 'THE GIST'))]
        )
        out = await _run(cap, 'x' * 100)
        assert out.startswith('[Tool output too large (100 chars); summarized by harness.')
        assert 're-run the tool for the full output' in out
        assert out.endswith('\nTHE GIST')

    async def test_inherited_model_and_usage(self):
        # model=None resolves to ctx.model, and the call threads usage=ctx.usage.
        ctx = _make_ctx(model=_fixed_model('THE SUMMARY'))
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=5, action=Summarize())])
        out = await _run(cap, 'x' * 100, ctx=ctx)
        assert out.endswith('THE SUMMARY')
        assert ctx.usage.requests == 1

    async def test_explicit_model_overrides_ctx(self):
        ctx = _make_ctx(model=_fixed_model('FROM CTX MODEL'))
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(model=_fixed_model('FROM EXPLICIT MODEL')))]
        )
        out = await _run(cap, 'x' * 100, ctx=ctx)
        assert out.endswith('FROM EXPLICIT MODEL')
        assert ctx.usage.requests == 1

    async def test_binary_summarize_falls_back(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=1, action=Summarize(then=Passthrough()))]
        )
        data = b'\x00' * 100
        assert await _run(cap, data) == data

    async def test_summarize_failure_falls_back(self):
        def boom(name: str, text: str) -> str:
            raise RuntimeError('model down')

        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(summarize=boom, then=Truncate(max_chars=10)))]
        )
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str) and 'omitted' in out


# ---------------------------------------------------------------------------
# Passthrough action + per-tool + band selection
# ---------------------------------------------------------------------------


class TestActionsAndSelection:
    async def test_passthrough_action(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Passthrough())])
        assert await _run(cap, 'x' * 100) == 'x' * 100

    async def test_per_tool_replaces_bands(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=1, action=Truncate(max_chars=5))],
            per_tool={'big_tool': [Band(over=100_000, action=Truncate())]},
        )
        # global band would truncate, but per_tool threshold is huge -> passthrough
        assert await _run(cap, 'x' * 100) == 'x' * 100

    def test_select_action_no_match(self):
        assert _select_action([Band(over=100, action=Passthrough())], 50) is None

    def test_select_action_first_match(self):
        bands = [Band(over=100, action=Spill()), Band(over=10, action=Truncate())]
        assert isinstance(_select_action(bands, 50), Truncate)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternals:
    def test_handle_key_defaults(self):
        ctx = _make_ctx(run_id=None, retry=2)
        ctx.tool_call_id = None
        key = _handle_key(ctx, ToolCallPart(tool_name='t', args='{}', tool_call_id=''))
        assert key == 'run/call.2'

    def test_handle_key_suffix(self):
        key = _handle_key(_make_ctx(), ToolCallPart(tool_name='t', args='{}', tool_call_id='c'), '.content')
        assert key.endswith('.content')

    def test_with_handles_non_mapping(self):
        meta = _with_handles('not-a-mapping', 'h/1.0', 42)
        assert meta == {'overflow_handle': 'h/1.0', 'overflow_bytes': 42}

    def test_with_handles_content_only(self):
        meta = _with_handles({'orig': 1}, None, 0, 'h/1.0.content')
        assert meta == {'orig': 1, 'overflow_content_handle': 'h/1.0.content'}

    def test_head_tail_preview_under(self):
        assert _head_tail_preview('short', 1000, 'h/1.0') == 'short'

    def test_head_tail_preview_over(self):
        out = _head_tail_preview('a' * 100, 10, 'h/1.0')
        assert 'omitted' in out
        # The spilled middle is retrievable, so the marker names both query tools.
        assert 'grep_tool_result' in out and 'read_tool_result' in out

    def test_build_spill_preview_tokens_unit(self):
        unit = _Unit(binary=False, text='x' * 100, data=b'x' * 100, value='x' * 100, suffix='')
        assert 'tokens' in _build_spill_preview('h/1.0', unit, 20, over_tokens=True)


# ---------------------------------------------------------------------------
# read_tool_result / _read_slice (C2 bounds + literal pattern)
# ---------------------------------------------------------------------------


class TestReadBack:
    async def test_read_slice_basic(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=3, from_end=False, pattern=None)
        assert 'line 0' in out and 'line 2' in out and 'line 3' not in out

    async def test_read_slice_from_end(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=2, from_end=True, pattern=None)
        assert 'line 49' in out and 'line 48' in out

    async def test_read_slice_literal_pattern(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'apple\nbanana\navocado\ncherry')
        out = await _read_slice(store, 'h/1.0', offset=0, limit=200, from_end=False, pattern='av')
        assert 'avocado' in out and 'apple' not in out and 'banana' not in out

    async def test_read_slice_pattern_is_literal_not_regex(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'plain line\n^anchored')
        # A regex metacharacter is matched literally, so it cannot trigger backtracking.
        out = await _read_slice(store, 'h/1.0', offset=0, limit=200, from_end=False, pattern='^a')
        assert 'anchored' in out and 'plain line' not in out

    async def test_read_slice_offset_negative(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='offset'):
            await _read_slice(store, 'h/1.0', offset=-1, limit=10, from_end=False, pattern=None)

    async def test_read_slice_limit_too_small(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='limit'):
            await _read_slice(store, 'h/1.0', offset=0, limit=0, from_end=False, pattern=None)

    async def test_read_slice_limit_clamped(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'l{i}' for i in range(2000)).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=10_000, from_end=False, pattern=None)
        assert out.count('\n') <= 1_000  # clamped to the line cap

    async def test_read_slice_output_capped(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', ('x' * 60_000).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=10, from_end=False, pattern=None)
        # The over-cap tail is dropped with an explicit inline marker, not silently.
        assert 'more bytes of this view omitted' in out
        assert len(out) < 60_000

    async def test_read_slice_missing_handle(self, tmp_path: Path):
        # A missing/wrong handle returns a guiding message (it does NOT raise): a bad
        # handle must not consume a retry and escalate to a fatal UnexpectedModelBehavior.
        store = LocalFileStore(base_dir=tmp_path)
        out = await _read_slice(store, 'missing/1.0', offset=0, limit=10, from_end=False, pattern=None)
        assert 'No stored tool result' in out
        assert 're-run the original tool' in out
        # The store's error (which can carry the resolved filesystem path) is not leaked.
        assert str(tmp_path) not in out

    async def test_get_toolset_registers_read_tool(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'hello\nworld')
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(store=store)
        toolset = cap.get_toolset()
        assert toolset is not None
        tool = toolset.tools[READ_TOOL_NAME]  # type: ignore[union-attr]
        out = await tool.function(_make_ctx(), 'h/1.0')  # type: ignore[attr-defined]
        assert 'hello' in out


# ---------------------------------------------------------------------------
# read_tool_result byte-range mode
# ---------------------------------------------------------------------------


class TestReadBytes:
    async def test_read_bytes_basic(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'0123456789')
        out = await _read_slice(store, 'h/1.0', offset=2, limit=4, from_end=False, pattern=None, unit='bytes')
        assert '2345' in out
        assert 'bytes 2-6 of 10' in out

    async def test_read_bytes_from_end(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'0123456789')
        out = await _read_slice(store, 'h/1.0', offset=0, limit=3, from_end=True, pattern=None, unit='bytes')
        assert '789' in out and '456' not in out

    async def test_read_bytes_out_of_range_empty(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'short')
        out = await _read_slice(store, 'h/1.0', offset=1000, limit=10, from_end=False, pattern=None, unit='bytes')
        # An offset past the end yields an empty window and just the header, never an error.
        assert 'bytes 5-5 of 5' in out

    async def test_read_bytes_rejects_pattern(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='pattern'):
            await _read_slice(store, 'h/1.0', offset=0, limit=10, from_end=False, pattern='x', unit='bytes')

    async def test_read_bytes_output_capped(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', ('y' * 60_000).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=60_000, from_end=False, pattern=None, unit='bytes')
        # limit is clamped to the byte cap, so the returned view stays bounded.
        assert len(out) < 60_000

    async def test_line_header_names_pattern(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'apple\nbanana\navocado')
        out = await _read_slice(store, 'h/1.0', offset=0, limit=200, from_end=False, pattern='av')
        assert "matching 'av'" in out


# ---------------------------------------------------------------------------
# grep_tool_result / _grep_slice
# ---------------------------------------------------------------------------


class TestGrep:
    async def test_grep_matches_with_line_numbers(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _grep_slice(store, 'h/1.0', pattern='line 7', context_lines=0, max_matches=20, is_regex=False)
        # 1-based line numbers: 'line 7' is index 7, so line 8.
        assert '8: line 7' in out
        assert '1 match(es)' in out

    async def test_grep_context_lines(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'a\nb\nMATCH\nc\nd')
        out = await _grep_slice(store, 'h/1.0', pattern='MATCH', context_lines=1, max_matches=20, is_regex=False)
        assert '2- b' in out and '3: MATCH' in out and '4- c' in out
        assert '1- a' not in out and '5- d' not in out

    async def test_grep_max_matches_cap(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', ('hit\n' * 100).encode('utf-8'))
        out = await _grep_slice(store, 'h/1.0', pattern='hit', context_lines=0, max_matches=5, is_regex=False)
        # Exactly max_matches shown, and the header flags that more exist.
        assert out.count(': hit') == 5
        assert 'first 5 of 5+' in out

    async def test_grep_no_match(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'apple\nbanana')
        out = await _grep_slice(store, 'h/1.0', pattern='zzz', context_lines=2, max_matches=20, is_regex=False)
        assert 'no matches for' in out and '2 line(s)' in out

    async def test_grep_separate_groups(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', ('X\n' + 'y\n' * 10 + 'X').encode('utf-8'))
        out = await _grep_slice(store, 'h/1.0', pattern='X', context_lines=0, max_matches=20, is_regex=False)
        # Two matches far apart -> two groups separated by a '--' line.
        assert '\n--\n' in out

    async def test_grep_regex(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'error 500\nok 200\nerror 404')
        out = await _grep_slice(store, 'h/1.0', pattern=r'error \d+', context_lines=0, max_matches=20, is_regex=True)
        assert 'error 500' in out and 'error 404' in out and 'ok 200' not in out

    async def test_grep_literal_metachar_not_regex(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'plain\na.b\naxb')
        # With is_regex=False the '.' is literal, so 'axb' must not match.
        out = await _grep_slice(store, 'h/1.0', pattern='a.b', context_lines=0, max_matches=20, is_regex=False)
        assert 'a.b' in out and 'axb' not in out

    async def test_grep_invalid_regex(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='Invalid regular expression'):
            await _grep_slice(store, 'h/1.0', pattern='(unclosed', context_lines=0, max_matches=20, is_regex=True)

    async def test_grep_empty_pattern(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='must not be empty'):
            await _grep_slice(store, 'h/1.0', pattern='', context_lines=0, max_matches=20, is_regex=False)

    async def test_grep_negative_context(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='context_lines'):
            await _grep_slice(store, 'h/1.0', pattern='x', context_lines=-1, max_matches=20, is_regex=False)

    async def test_grep_max_matches_too_small(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='max_matches'):
            await _grep_slice(store, 'h/1.0', pattern='x', context_lines=0, max_matches=0, is_regex=False)

    async def test_grep_missing_handle(self, tmp_path: Path):
        # Consistent with read: a bad handle returns guidance, it does not raise.
        store = LocalFileStore(base_dir=tmp_path)
        out = await _grep_slice(store, 'missing/1.0', pattern='x', context_lines=2, max_matches=20, is_regex=False)
        assert 'No stored tool result' in out
        assert str(tmp_path) not in out

    async def test_grep_output_capped(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join('hit ' + 'x' * 500 for _ in range(200)).encode('utf-8'))
        out = await _grep_slice(store, 'h/1.0', pattern='hit', context_lines=0, max_matches=200, is_regex=False)
        assert 'more bytes of this view omitted' in out

    async def test_get_toolset_registers_grep_tool(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'alpha\nbeta\ngamma')
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(store=store)
        toolset = cap.get_toolset()
        assert toolset is not None
        tool = toolset.tools[GREP_TOOL_NAME]  # type: ignore[union-attr]
        out = await tool.function(_make_ctx(), 'h/1.0', 'beta')  # type: ignore[attr-defined]
        assert '2: beta' in out

    async def test_grep_tool_result_exempt_from_reduction(self):
        # A large grep result must not itself be spilled or truncated (it would recurse).
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate(max_chars=2))])
        out = await _run(cap, 'x' * 100, tool_name=GREP_TOOL_NAME)
        assert out == 'x' * 100


# ---------------------------------------------------------------------------
# Uniform elision markers + determinism
# ---------------------------------------------------------------------------


class TestElisionMarkers:
    async def test_spill_marker_names_both_query_tools(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill(preview_chars=20))], store=store
        )
        out = await _run(cap, 'line\n' * 500)
        assert isinstance(out, ToolReturn) and isinstance(out.return_value, str)
        text = out.return_value
        # The stored-output banner and the middle-elision marker both name grep + read.
        assert 'stored as' in text
        assert GREP_TOOL_NAME in text and READ_TOOL_NAME in text
        assert 'omitted' in text

    async def test_truncate_marker_has_no_handle(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=10, action=Truncate(max_chars=20))])
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str)
        # Lossy path: no handle, no query-tool names, explicit re-run guidance instead.
        assert 're-run the tool' in out
        assert GREP_TOOL_NAME not in out and READ_TOOL_NAME not in out

    async def test_spill_preview_is_deterministic(self, tmp_path: Path):
        # Two identical spills (same run/call/retry) produce byte-identical previews:
        # no timestamps or other run-varying data, for cache-prefix stability.
        text = 'payload line\n' * 500

        async def spill_once() -> Any:
            store = LocalFileStore(base_dir=tmp_path / os.urandom(8).hex())
            cap: OverflowingToolOutput[object] = OverflowingToolOutput(
                bands=[Band(over=10, action=Spill(preview_chars=40))], store=store
            )
            return await _run(cap, text, ctx=_make_ctx(run_id='run-x', retry=0))

        first = await spill_once()
        second = await spill_once()
        assert isinstance(first, ToolReturn) and isinstance(second, ToolReturn)
        assert first.return_value == second.return_value
        assert first.metadata['overflow_handle'] == second.metadata['overflow_handle']

    async def test_truncate_is_deterministic(self):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(bands=[Band(over=10, action=Truncate(max_chars=30))])
        first = await _run(cap, 'z' * 200)
        second = await _run(cap, 'z' * 200)
        assert first == second


# ---------------------------------------------------------------------------
# Agent-path integration
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    async def test_spill_persists_in_history(self, tmp_path: Path, anyio_backend: str):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=100, action=Spill(preview_chars=50))], store=store
        )
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> str:
            return 'data line\n' * 500

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        spilled = [p for p in returns if p.tool_name == 'big_tool']
        assert spilled
        part = spilled[0]
        assert isinstance(part.content, str) and 'too large' in part.content
        assert part.metadata is not None and 'overflow_handle' in part.metadata
        assert await store.read(part.metadata['overflow_handle']) == ('data line\n' * 500).encode('utf-8')

    async def test_small_output_untouched(self, tmp_path: Path, anyio_backend: str):
        cap: OverflowingToolOutput[object] = OverflowingToolOutput(
            bands=[Band(over=10_000, action=Spill())], store=LocalFileStore(base_dir=tmp_path)
        )
        agent = Agent(TestModel(call_tools=['small_tool']), capabilities=[cap])

        @agent.tool_plain
        def small_tool() -> str:
            return 'tiny'

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        small = [p for p in returns if p.tool_name == 'small_tool']
        assert small and small[0].content == 'tiny'

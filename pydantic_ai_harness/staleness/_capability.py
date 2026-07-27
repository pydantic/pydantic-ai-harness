"""StalenessTracker: tell the model when tracked files changed since it read them.

A model cannot perceive elapsed time or concurrent change. Once it reads a file it acts
on that snapshot forever, even if another agent, a build step, or the user edits the file
underneath it. This capability records `(path, mtime, size)` whenever the model reads or
writes a file, and -- before each model request -- re-stats the tracked set and injects an
*ephemeral* notice naming any file that changed or was deleted since it was last observed.

The notice rides in the per-request message tail behind a `CachePoint`, exactly like
`Planning`'s plan reminder: it reaches the model but is never written back to the durable
message history, so the cached prefix stays byte-stable and no stale notices accumulate.
The re-stat is stat-only (no hashing) over a bounded, LRU-capped set, so it stays cheap.

The agent's own writes are not staleness: because observation happens in
`after_tool_execute` -- after the write lands -- a write records the post-write
`(mtime, size)`, so the file the agent just wrote never flags itself.

Which tool calls count as file reads/writes is data-driven and easily changed via the
`track` mapping (tool-name pattern -> the arg holding the path, or a callable) plus the
`path_extractor` escape hatch. This is a deliberately revisable default, not a contract.
"""

from __future__ import annotations

import fnmatch
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext


PathExtractor = Callable[[str, Mapping[str, Any]], Sequence[str]]
"""Map a `(tool_name, args)` pair to the file paths that call read or wrote."""

TrackValue = str | Callable[[Mapping[str, Any]], Sequence[str]]
"""How to pull path(s) out of one tool's args: an arg name, or a callable over the args."""


def _arg_paths(*names: str) -> Callable[[Mapping[str, Any]], Sequence[str]]:
    """Build a `track` value that returns the first present, non-empty string arg among `names`."""

    def extract(args: Mapping[str, Any]) -> Sequence[str]:
        for name in names:
            value = args.get(name)
            if isinstance(value, str) and value:
                return [value]
        return []

    return extract


_COMMON_PATH_ARG = _arg_paths('file_path', 'path')
"""Default extractor for the built-in file tools: prefer `file_path`, fall back to `path`."""

DEFAULT_TRACK: dict[str, TrackValue] = {
    name: _COMMON_PATH_ARG for name in ('read', 'read_file', 'write', 'write_file', 'edit', 'apply_patch')
}
"""Default tool-name-pattern -> path-arg mapping.

Covers the common read/write/edit tool names, each reading whichever of `file_path` /
`path` the call carries. Patterns are matched with `fnmatch` (so `read*` works). Override
per host: `StalenessTracker(track={'open_file': 'filename'})`.
"""


@dataclass
class _Observation:
    """What the agent last saw of a file: where it is, how the model named it, and its stat."""

    resolved: Path
    display: str
    mtime_ns: int
    size: int


def _stat(path: Path) -> tuple[int, int] | None:
    """Return `(mtime_ns, size)` for `path`, or `None` if it is missing or unreadable."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _truncate(names: Sequence[str], cap: int) -> str:
    """Render `names` as a comma-separated list, capped with a `(+N more)` suffix."""
    if len(names) <= cap:
        return ', '.join(names)
    shown = ', '.join(names[:cap])
    return f'{shown} (+{len(names) - cap} more)'


@dataclass
class StalenessTracker(AbstractCapability[AgentDepsT]):
    """Tell the model when files it read have changed on disk since it read them.

    Records `(path, mtime, size)` after every file read/write the model makes, then before
    each model request re-stats the tracked set and injects an ephemeral notice naming any
    file that changed or was deleted underneath it -- e.g.::

        Files changed on disk since you last read them: src/foo.py, tests/test_foo.py.
        Re-read before relying on their contents.

    The notice is delivered as a `<system-reminder>` in the per-request message tail (behind
    a `CachePoint`), so it reaches the model without entering the durable message history --
    the cached prefix stays byte-stable and notices never pile up. The check is stat-only
    (no hashing) over an LRU-capped set, so it stays cheap.

    The agent's own writes are never flagged: observation happens after the write lands, so
    a self-write records the post-write stat and matches on the next check.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.staleness import StalenessTracker

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[StalenessTracker()])
    ```

    Which tool calls count as file reads/writes is a deliberately revisable default: the
    `track` mapping and `path_extractor` escape hatch make it data-driven, so a host with
    differently-named tools reconfigures it without subclassing.
    """

    track: dict[str, TrackValue] = field(default_factory=lambda: dict(DEFAULT_TRACK))
    """Tool-name pattern -> how to find the path in that tool's args.

    Keys are `fnmatch` patterns matched against the tool name. Each value is either the name
    of the arg holding the path (a `str`), or a callable `(args) -> paths` for tools that
    carry the path differently. Defaults cover `read`/`read_file`/`write`/`write_file`/
    `edit`/`apply_patch`, each reading `file_path` or `path`. Replace freely.
    """

    path_extractor: PathExtractor | None = None
    """Full escape hatch: `(tool_name, args) -> paths`. When set, it replaces `track`
    entirely and decides for every tool which paths (if any) a call touched."""

    root: Path | None = None
    """Base directory for resolving relative paths. `None` uses the process working
    directory. Only affects file identity for stat-ing; the notice shows the path as the
    model named it."""

    max_tracked: int = 200
    """Cap on the number of files tracked at once. Least-recently-observed files are evicted
    past this bound, keeping the per-request re-stat cheap."""

    max_listed: int = 10
    """Cap on how many names the notice lists (per changed / deleted group) before it
    summarizes the rest as `(+N more)`."""

    cache_ttl: Literal['5m', '1h'] = '5m'
    """TTL for the `CachePoint` placed before the ephemeral notice."""

    notice_tag: str = 'system-reminder'
    """Tag wrapping the notice, aligning with the harness `<system-reminder>` convention."""

    _observations: OrderedDict[Path, _Observation] = field(
        default_factory=OrderedDict[Path, '_Observation'], init=False, repr=False, compare=False
    )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> StalenessTracker[AgentDepsT]:
        """Return a fresh per-run instance with its own observation ledger (config preserved)."""
        return replace(self)

    def _extract_paths(self, tool_name: str, args: Mapping[str, Any]) -> Sequence[str]:
        """Resolve the file paths a tool call touched, via `path_extractor` or `track`."""
        if self.path_extractor is not None:
            return self.path_extractor(tool_name, args)
        for pattern, how in self.track.items():
            if fnmatch.fnmatchcase(tool_name, pattern):
                if isinstance(how, str):
                    value = args.get(how)
                    return [value] if isinstance(value, str) and value else []
                return how(args)
        return []

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        base = self.root if self.root is not None else Path.cwd()
        return base / candidate

    def _observe(self, raw: str) -> None:
        """Record (or refresh) one file's observed stat, honoring the LRU bound."""
        resolved = self._resolve(raw)
        stat = _stat(resolved)
        if stat is None:
            return
        mtime_ns, size = stat
        key = resolved.resolve()
        self._observations.pop(key, None)
        self._observations[key] = _Observation(resolved=resolved, display=raw, mtime_ns=mtime_ns, size=size)
        while len(self._observations) > self.max_tracked:
            self._observations.popitem(last=False)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """After a file read/write lands, record its post-call `(mtime, size)`."""
        for raw in self._extract_paths(call.tool_name, args):
            self._observe(raw)
        return result

    def _notice(self) -> str | None:
        """Re-stat the tracked set and render a notice for changed/deleted files, or `None`."""
        changed: list[str] = []
        deleted: list[str] = []
        for obs in self._observations.values():
            stat = _stat(obs.resolved)
            if stat is None:
                deleted.append(obs.display)
            elif stat != (obs.mtime_ns, obs.size):
                changed.append(obs.display)
        if not changed and not deleted:
            return None
        sentences: list[str] = []
        if changed:
            sentences.append(f'Files changed on disk since you last read them: {_truncate(changed, self.max_listed)}.')
        if deleted:
            sentences.append(f'Files deleted since you last read them: {_truncate(deleted, self.max_listed)}.')
        sentences.append('Re-read before relying on their contents.')
        body = ' '.join(sentences)
        return f'<{self.notice_tag}>{body}</{self.notice_tag}>'

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Append the staleness notice as an ephemeral, cache-safe tail reminder.

        Runs after core has persisted the durable history; the per-request message list it
        mutates is never written back, so the notice reaches the model but never enters
        `ctx.messages`. The `CachePoint` sits before the notice, keeping it outside the
        cached region.
        """
        notice = self._notice()
        if notice is not None:
            messages = request_context.messages
            last = messages[-1]
            if isinstance(last, ModelRequest):
                part = UserPromptPart(content=[CachePoint(ttl=self.cache_ttl), notice])
                messages[-1] = replace(last, parts=[*last.parts, part])
        return await handler(request_context)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Opt out of spec construction: the `track`/`path_extractor` config holds callables."""
        return None

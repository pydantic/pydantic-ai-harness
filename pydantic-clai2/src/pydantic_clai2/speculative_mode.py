"""Speculative CodeMode wiring, ported from Code Puppy's `enable_speculative_code_mode`.

Every tool the run has, plugin and MCP tools included, becomes a function inside harness
`CodeMode`'s `run_code` except `write_file` and `edit_file`, which stay native so their diffs
render as usual. `shell` folds in too, as Code Puppy's shell tool does: harness `CodeMode` keeps
tools marked with `code_arg_name` metadata native by default, so `SpeculativeExecution` clears
that marker on every tool it sandboxes. Without it, a coding turn is all native `shell` calls
and eager execution never starts a build or test while the model is still writing. Only the read-only trio may launch speculatively. That allowlist is the safety
contract: an early launch may run for a branch the snippet never takes, so it is reserved for
calls that are harmless to re-run or discard. Everything else waits for eager or normal
execution.

The sandbox mounts the working directory read-write at its real path and gets Monty's
`OSAccess` (isolated environment, host clock, in-memory scratch files). There is no network in
the sandbox; anything remote goes through a wrapped tool such as `shell`.
"""

import os
from dataclasses import dataclass, replace

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai_harness.code_mode import (
    CodeMode,
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallMissedEvent,
)
from pydantic_monty import MountDir, OSAccess

from .customization import read_clai_customization_guide
from .eager_timing import EagerExecutionCompletedEvent, EagerTiming
from .speculation import SpeculationCounters

SPECULATIVE_TOOLS = ('list_files', 'read_file', 'grep', read_clai_customization_guide.__name__)
"""Pure with respect to the workspace, so safe to start early, re-run, or discard.

The customization guide reads a file shipped inside the package and takes no arguments.
"""

NATIVE_TOOLS = frozenset({'write_file', 'edit_file'})
"""CLAI's equivalents of Code Puppy's native `create_file` and `replace_in_file`."""

GUIDANCE = """\
Speculative execution is on. Use `write_file` and `edit_file` as native tools,
outside `run_code`; they are not available as functions inside the sandbox.
Every other tool, including `shell`, is an async function inside `run_code`, a
persistent sandboxed Python REPL. Call `run_code` with a Python snippet to use them; do
not attempt to call those functions as native tools.

The sandbox also has direct capabilities, no function call needed:

- The workspace is mounted read-write at its real absolute path: use
  `pathlib.Path` to read, write, glob, and stat project files directly.
- Environment variables (isolated), in-memory scratch files, and the real
  clock (`time` module) work.
- There is NO network in the sandbox: anything remote goes through a
  function like `shell` (e.g. `curl`).

Use raw Python strings for regex patterns so backslashes are not invalid escapes.

How to work. The runtime watches your code AS YOU WRITE IT and starts
eligible calls before the snippet is finished, so the SHAPE of your code
determines how fast it runs:

1. Emit small, flat statements, one per line. A read call (`list_files`,
   `read_file`, `grep`) starts executing the moment its line is complete,
   when its call has all-literal keyword arguments:

       hits = await grep(pattern="SpeculationCounters")
       src = await read_file(path="src/pydantic_clai2/speculation.py")

   Each such line runs while you are still writing the lines below it.
   Computing arguments forfeits that speculative head start. Literal
   calls can also be detected inside expressions or across multiple lines.
2. Go BIG in one `run_code` call. Do not split work across many small
   snippets: every extra round trip to the model wastes the runway that
   makes early execution pay. 60-100 lines with ten, twenty, thirty tool
   calls in a single snippet is the fast path: every additional literal
   read line is another call already running while you write the rest.
3. Front-load the reads: open every snippet with the literal read lines,
   one per line, then process the results with plain Python below them.
   Read the files you MIGHT need, not just the one you are sure of; an
   unused result costs nothing you were not already spending on generation.
4. Never introduce a variable just to pass it: `q = "x"` followed by
   `grep(pattern=q)` runs cold; `grep(pattern="x")` runs early. Repeat the
   literal even if it feels less DRY; here, DRY loses to speed.
5. Writes and shell commands never speculate, but eager execution can
   execute them as soon as their statements close, before generation ends.
   Put a slow build or test command on its own early line so it runs while
   you write the rest of the snippet.
   Obtain required approval BEFORE emitting a side-effectful statement;
   later code cannot undo it. Run independent calls concurrently with
   `await asyncio.gather(...)` (positional awaitables only; no other
   task-creation APIs exist in the sandbox).
6. Keep mutable state small and local: assign results to short fresh
   names, keep processing blocks brief, and never rebind a name a pending
   call's line already used. `print(...)` what matters and make the
   snippet's final expression the value you want returned.
7. State persists between snippets within a run: variables and functions
   carry over. Do not re-fetch what you already hold. Prefer the read
   functions over `pathlib` for discovery (they start early); use
   `pathlib` for surgical follow-ups on paths you already hold.
8. Read before you write, and verify after you change: re-read the file
   or run the tests in a follow-up snippet.
"""


def _sandboxed(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
    return tool_def.name not in NATIVE_TOOLS


@dataclass
class SpeculativeExecution(AbstractCapability[AgentDepsT]):
    """Fold code tools in, teach the snippet shape, stream tool arguments, and count outcomes."""

    counters: SpeculationCounters

    def get_instructions(self) -> str:
        """Code Puppy's guidance, with CLAI's tool and argument names."""
        return GUIDANCE

    def get_model_settings(self) -> AnthropicModelSettings:
        """Anthropic buffers a tool call's input by default, which leaves eager execution no runway.

        Other providers stream tool arguments already and ignore the setting.
        """
        return {'anthropic_eager_input_streaming': True}

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Clear `code_arg_name` so `CodeMode` sandboxes `shell` like any other tool.

        Core runs this inside every capability's wrapper toolset, so `CodeMode` sees the result.
        """
        return [
            replace(tool_def, metadata={k: v for k, v in tool_def.metadata.items() if k != 'code_arg_name'})
            if tool_def.metadata and 'code_arg_name' in tool_def.metadata and _sandboxed(ctx, tool_def)
            else tool_def
            for tool_def in tool_defs
        ]

    async def on_event(self, ctx: RunContext[AgentDepsT], *, event: AgentStreamEvent) -> None:
        """Consume telemetry without retaining generated code or rendering anything."""
        counters = self.counters
        if isinstance(event, EagerExecutionCompletedEvent):
            counters.eager_ms += max(0.0, event.saved_ms)
        elif isinstance(event, SpeculativeCallClaimedEvent):
            counters.hits += 1
            if event.ready_at_claim:
                counters.speculative_ms += max(0.0, event.elapsed_ms)
        elif isinstance(event, SpeculativeCallMissedEvent):
            counters.misses += 1
        elif isinstance(event, SpeculativeCallEvictedEvent):
            counters.wasted += 1


def speculative_capabilities(counters: SpeculationCounters) -> 'list[AbstractCapability[AgentDepsT]]':
    """Fold tools into `run_code` with eager execution and read-only speculation."""
    workspace = os.getcwd()
    return [
        CodeMode(
            tools=_sandboxed,
            speculate=SPECULATIVE_TOOLS,
            # Eager runs each streamed statement as it closes; speculation launches the
            # read-only calls beyond that frontier, and the eager feed claims them.
            eager=True,
            mount=MountDir(virtual_path=workspace, host_path=workspace, mode='read-write'),
            os_access=OSAccess(),
        ),
        EagerTiming(),
        SpeculativeExecution(counters),
    ]

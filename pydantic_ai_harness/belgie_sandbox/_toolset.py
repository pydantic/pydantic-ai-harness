"""Model-facing toolset for Belgie Sandbox."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness.belgie_sandbox._session import (
    DEFAULT_MAX_OLD_GENERATION_SIZE_MB,
    DEFAULT_TIMEOUT,
    BelgieSandboxExecutionError,
    BelgieSandboxSession,
    BelgieSandboxTimeoutError,
)

RUN_TYPESCRIPT_TOOL_NAME = 'run_typescript'
DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024

_TOOL_DESCRIPTION = """\
Run a complete JavaScript, TypeScript, or TSX module in Belgie's embedded Deno sandbox.

Export a callable function, preferably `export default async function run() { ... }` or \
`export function run() { ... }`. The exported function is called without arguments. Return \
the JSON-serializable value you want sent back; console output is not captured.

This is Deno, not Node.js. The owned runtime denies host files, environment variables, subprocesses, \
FFI, and system information. Remote package imports and `fetch` are unavailable unless the host \
explicitly enables their Belgie Sandbox options. Caller-supplied runtimes define their own permissions. \
External agent tools are not callable from inside this sandbox.
"""


class BelgieSandboxToolset(FunctionToolset[AgentDepsT]):
    """One `run_typescript` tool backed by a run-scoped Belgie session."""

    def __init__(
        self,
        *,
        allow_package_imports: bool,
        allow_network: bool,
        enable_rendering: bool,
        max_old_generation_size_mb: int | None,
        timeout: float,
        max_output_bytes: int,
        max_retries: int,
        toolset_id: str,
        session: BelgieSandboxSession | None = None,
        _run_scoped: bool = False,
    ) -> None:
        super().__init__(max_retries=max_retries, sequential=True, id=toolset_id)
        self._allow_package_imports = allow_package_imports
        self._allow_network = allow_network
        self._enable_rendering = enable_rendering
        self._max_old_generation_size_mb = max_old_generation_size_mb
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._external_session = session
        self._session: BelgieSandboxSession | None = None
        self._run_scoped = _run_scoped
        self._entered = False

        self.add_function(
            self.run_typescript,
            name=RUN_TYPESCRIPT_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            metadata={'code_arg_name': 'code', 'code_arg_language': 'typescript'},
            sequential=True,
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset so concurrent runs never share an owned runtime."""
        return BelgieSandboxToolset[AgentDepsT](
            allow_package_imports=self._allow_package_imports,
            allow_network=self._allow_network,
            enable_rendering=self._enable_rendering,
            max_old_generation_size_mb=self._max_old_generation_size_mb,
            timeout=self._timeout,
            max_output_bytes=self._max_output_bytes,
            max_retries=self.max_retries if self.max_retries is not None else 0,
            toolset_id=self.id or 'belgie_sandbox',
            session=self._external_session,
            _run_scoped=True,
        )

    async def __aenter__(self) -> Self:
        """Mark the per-run clone active and validate a caller-owned session."""
        if not self._run_scoped:
            return self
        self._entered = True
        if self._external_session is not None:
            if not self._external_session.is_open:
                self._entered = False
                raise BelgieSandboxExecutionError(
                    'The injected Belgie sandbox session is not open. '
                    'Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close only a session created by this run."""
        session = self._session
        if session is not None and self._external_session is None:
            await session.close()
        self._session = None
        self._entered = False

    async def _require_session(self) -> BelgieSandboxSession:
        if not self._entered:
            raise BelgieSandboxExecutionError('The Belgie sandbox toolset is not active in an agent run.')
        if self._session is not None:
            return self._session
        session = BelgieSandboxSession(
            allow_package_imports=self._allow_package_imports,
            allow_network=self._allow_network,
            enable_rendering=self._enable_rendering,
            max_old_generation_size_mb=self._max_old_generation_size_mb,
        )
        await session.__aenter__()
        self._session = session
        return session

    async def run_typescript(
        self,
        code: Annotated[
            str,
            Field(description='Complete JavaScript, TypeScript, or TSX `belgie.Script` module source.'),
        ],
    ) -> ToolReturn:
        """Run a TypeScript-family module in the configured Belgie sandbox."""
        session = await self._require_session()
        try:
            result = await session.run_script(code, timeout=self._timeout)
        except (BelgieSandboxExecutionError, BelgieSandboxTimeoutError) as error:
            raise ModelRetry(str(error)) from error

        try:
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        except (TypeError, ValueError) as error:
            raise ModelRetry(f'Belgie script returned an invalid JSON value: {error}') from error
        output_bytes = len(encoded)
        if output_bytes > self._max_output_bytes:
            raise ModelRetry(
                f'Belgie script returned {output_bytes} UTF-8 bytes, exceeding the '
                f'{self._max_output_bytes}-byte limit. Return a smaller value or summary.'
            )

        return ToolReturn(
            return_value=result,
            metadata={
                'belgie_sandbox': True,
                'code_language': 'typescript',
                'output_bytes': output_bytes,
            },
        )


__all__ = [
    'DEFAULT_MAX_OLD_GENERATION_SIZE_MB',
    'DEFAULT_MAX_OUTPUT_BYTES',
    'DEFAULT_TIMEOUT',
    'RUN_TYPESCRIPT_TOOL_NAME',
    'BelgieSandboxToolset',
]

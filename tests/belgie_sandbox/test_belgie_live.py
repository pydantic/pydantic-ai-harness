"""Offline smoke tests against the real embedded Belgie runtime."""

from __future__ import annotations

import sys

import pytest

from pydantic_ai_harness.belgie_sandbox import BelgieSandboxExecutionError, BelgieSandboxSession

pytestmark = [pytest.mark.anyio, pytest.mark.belgie_live]


@pytest.mark.skipif(
    sys.version_info < (3, 12) or sys.version_info >= (3, 15),
    reason='Belgie supports Python 3.12-3.14',
)
async def test_real_runtime_executes_typescript_and_denies_host_access() -> None:
    pytest.importorskip('belgie')

    async with BelgieSandboxSession() as session:
        result = await session.run_script(
            """
export default function run(): { total: number; label: string } {
  const values: number[] = [20, 22];
  return { total: values.reduce((sum, value) => sum + value, 0), label: "typescript" };
}
"""
        )
        assert result == {'total': 42, 'label': 'typescript'}

        denied_sources = [
            'export default () => Deno.env.get("HOME");',
            'export default () => Deno.readTextFile("/etc/passwd");',
            'export default () => new Deno.Command("echo").outputSync();',
            'export default async () => await fetch("https://example.com");',
            'import value from "https://example.com/value.ts"; export default () => value;',
        ]
        for source in denied_sources:
            with pytest.raises(BelgieSandboxExecutionError):
                await session.run_script(source)

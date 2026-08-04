"""Offline smoke tests against the real embedded Belgie runtime.

Run with `make integration-belgie`.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from pydantic_ai_harness.belgie_sandbox import BelgieSandboxExecutionError, BelgieSandboxSession

pytestmark = [pytest.mark.anyio, pytest.mark.belgie_live]


@pytest.mark.skipif(
    sys.version_info < (3, 12) or sys.version_info >= (3, 15),
    reason='Belgie supports Python 3.12-3.14',
)
async def test_real_runtime_executes_typescript_and_denies_host_access(tmp_path: Path) -> None:
    host_file = tmp_path / 'host-secret.txt'
    host_secret = 'host-file-secret-content'
    host_file.write_text(host_secret, encoding='utf-8')

    module_secret = 'host-module-secret-content'
    module_file = tmp_path / 'service-account.json'
    module_file.write_text(json.dumps({'token': module_secret}), encoding='utf-8')

    hits: list[str] = []

    class _ProbeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = HTTPServer(('127.0.0.1', 0), _ProbeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
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

            with pytest.raises(BelgieSandboxExecutionError, match='Requires env access'):
                await session.run_script('export default () => Deno.env.get("HOME");')

            with pytest.raises(BelgieSandboxExecutionError, match='Requires read access') as read_info:
                await session.run_script(f'export default () => Deno.readTextFile({json.dumps(str(host_file))});')
            assert host_secret not in str(read_info.value)

            with pytest.raises(BelgieSandboxExecutionError, match='Requires run access'):
                await session.run_script(
                    f'export default () => new Deno.Command({json.dumps(str(host_file))}).outputSync();'
                )

            with pytest.raises(BelgieSandboxExecutionError, match='Requires net access'):
                await session.run_script(f'export default async () => await fetch("http://127.0.0.1:{port}/probe");')
            assert hits == []

        # Module-graph denials can leave the embedded worker unable to accept
        # another script, so probe them in fresh sessions.
        file_import = (
            f'import credentials from {json.dumps(module_file.as_uri())} with {{ type: "json" }};\n'
            'export default () => credentials.token;'
        )
        async with BelgieSandboxSession() as session:
            with pytest.raises(BelgieSandboxExecutionError, match='Requires read access') as module_info:
                await session.run_script(file_import)
            assert module_secret not in str(module_info.value)

        async with BelgieSandboxSession() as session:
            with pytest.raises(BelgieSandboxExecutionError, match='--no-remote'):
                await session.run_script(
                    f'import value from "http://127.0.0.1:{port}/mod.ts"; export default () => value;'
                )
            assert hits == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

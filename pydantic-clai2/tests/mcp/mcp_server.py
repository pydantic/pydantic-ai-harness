"""A real local MCP server used by the plugin integration tests."""

import json
import os
import sys
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP

server = FastMCP('clai-test')


@server.tool()
def context() -> str:
    sys.stderr.write('MCP stderr sentinel\n')
    sys.stderr.flush()
    return json.dumps({'cwd': str(Path.cwd()), 'token': os.environ.get('CLAI_MCP_TOKEN'), 'pid': os.getpid()})


@server.tool()
async def wait() -> str:
    async with await anyio.connect_tcp('127.0.0.1', int(os.environ['CLAI_MCP_READY_PORT'])) as stream:
        for byte in str(os.getpid()).encode():
            await stream.send(bytes((byte,)))
        await anyio.sleep_forever()
    return 'unreachable'  # pragma: no cover -- the test cancels this tool while it is waiting.


if __name__ == '__main__':
    Path(os.environ['CLAI_MCP_PID_FILE']).write_text(str(os.getpid()))
    server.run(transport='stdio')
